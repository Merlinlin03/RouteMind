"""Raw text generation backends; business logic lives outside the model."""
from __future__ import annotations

from .prompting import prompt_ids
from .schema import FeedbackRequest


class InputTooLong(ValueError):
    pass


class TransformersBackend:

    def __init__(self, model_name: str, adapter: str | None = None,
                 max_input_tokens: int = 6144, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for the real inference backend")
        self.model_name, self.adapter = model_name, adapter
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.torch = torch
        if adapter:
            from peft import PeftConfig
            adapter_config = PeftConfig.from_pretrained(adapter)
            if adapter_config.base_model_name_or_path != model_name:
                raise ValueError("adapter base model must match model_name exactly")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_name, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map={"": torch.cuda.current_device()},
            trust_remote_code=False, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, request: FeedbackRequest) -> str:
        ids = prompt_ids(self.tokenizer, request)
        if len(ids) > self.max_input_tokens:
            raise InputTooLong("input exceeds configured token limit; no silent truncation")
        tensor = self.torch.tensor([ids], device=self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tensor, attention_mask=self.torch.ones_like(tensor),
                max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
