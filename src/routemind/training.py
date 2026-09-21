"""Single-GPU QLoRA. Imports GPU libraries only for explicit training/tokenization."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from pydantic import Field

from .data import assert_disjoint, load_records
from .prompting import pad_features, training_example
from .schema import Contract


class TrainConfig(Contract):
    model_name: str = "Qwen/Qwen3-32B"
    train_file: str = "data/processed/train.jsonl"
    validation_file: str = "data/processed/validation.jsonl"
    test_file: str = "data/processed/test.jsonl"
    output_dir: str = "outputs/qwen3-32b"
    max_length: int = Field(default=8192, ge=128)
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.0001, gt=0)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    seed: int = 42


def config_from(path: str) -> TrainConfig:
    return TrainConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_records(config: TrainConfig, allow_unreviewed: bool = False):
    train = load_records(config.train_file)
    val = load_records(config.validation_file)
    test = load_records(config.test_file)
    assert_disjoint(train, val, test)
    if any(r.review_status == "rejected" for r in train + val):
        raise ValueError("rejected records cannot be used for training or validation")
    if not allow_unreviewed and any(r.review_status != "approved" for r in train + val):
        raise ValueError("training requires approved records; --allow-unreviewed is for pipeline experiments only")
    return train, val, test


def preflight(config: TrainConfig, allow_unreviewed: bool = False) -> dict:
    train, val, test = prepare_records(config, allow_unreviewed)
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": config.model_dump(), "packages": packages,
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "allow_unreviewed": allow_unreviewed,
        "gpu_validated": False,
        "note": "Offline preflight does not load weights, check CUDA compatibility or guarantee sufficient VRAM.",
    }


def run_training(config: TrainConfig, allow_unreviewed: bool = False, resume: str | None = None):
    train, val, _ = prepare_records(config, allow_unreviewed)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("output directory is non-empty; choose another or explicitly resume a checkpoint")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("v0.1 supports one GPU; distributed QLoRA is not configured")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training")
    set_seed(config.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Format before allocating model weights; over-length records fail explicitly.
    train_data = [training_example(tokenizer, r.input, r.target, config.max_length) for r in train]
    val_data = [training_example(tokenizer, r.input, r.target, config.max_length) for r in val]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, quantization_config=quantization, torch_dtype=dtype,
        device_map={"": torch.cuda.current_device()}, trust_remote_code=False, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    ))

    def collate(features):
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_features(features, tokenizer.pad_token_id).items()}

    args = TrainingArguments(
        output_dir=str(output), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.03,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, report_to="none", seed=config.seed, data_seed=config.seed,
        prediction_loss_only=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                      data_collator=collate, processing_class=tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    manifest = preflight(config, allow_unreviewed)
    manifest.update({
        "gpu_validated": True, "training_completed": False, "gpu": torch.cuda.get_device_name(),
        "data_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                        (("train", config.train_file), ("validation", config.validation_file), ("test", config.test_file))},
        "model_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    })
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics = trainer.train(resume_from_checkpoint=resume).metrics
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    manifest.update(training_completed=True, train_metrics=metrics)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
