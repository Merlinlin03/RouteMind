from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import load_records, split_file, validate_records, write_jsonl
from .prompting import messages_for
from .schema import Understanding


def main():
    parser = argparse.ArgumentParser(description="RouteMind data / QLoRA / evaluation / API scripts")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("source")
    merge = sub.add_parser("merge", help="Merge seed, generated and imported annotation files before splitting")
    merge.add_argument("sources", nargs="+")
    merge.add_argument("--output", required=True)
    split = sub.add_parser("split")
    split.add_argument("source")
    split.add_argument("--output", default="data/processed")
    split.add_argument("--seed", type=int, default=42)
    fmt = sub.add_parser("format", help="Export readable SFT messages without loading a tokenizer")
    fmt.add_argument("source")
    fmt.add_argument("--output", required=True)
    schema = sub.add_parser("schema")
    schema.add_argument("--output", default="outputs/understanding.schema.json")
    for name in ("preflight", "train", "tokenize-check"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/qwen3_32b.json")
        command.add_argument("--allow-unreviewed", action="store_true")
        if name == "train":
            command.add_argument("--resume")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8020)
    predict = sub.add_parser("predict")
    predict.add_argument("source")
    predict.add_argument("--output", required=True)
    predict.add_argument("--model", default="Qwen/Qwen3-32B")
    predict.add_argument("--adapter")
    predict.add_argument("--max-input-tokens", type=int, default=6144)
    predict.add_argument("--max-new-tokens", type=int, default=2048)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("gold")
    evaluate.add_argument("predictions")
    evaluate.add_argument("--output", default="outputs/evaluation.json")
    compare = sub.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("adapter")
    synth = sub.add_parser("synthesize", help="Explicit remote API call; may incur provider charges")
    synth.add_argument("source")
    synth.add_argument("--output", required=True)
    synth.add_argument("--limit", type=int, default=6)
    synth.add_argument("--base-url", default=os.getenv("SYNTHESIS_BASE_URL", ""))
    synth.add_argument("--model", default=os.getenv("SYNTHESIS_MODEL", ""))
    args = parser.parse_args()
    if args.command == "validate":
        records = load_records(args.source)
        result = {"valid_records": len(records), "languages": sorted({r.target.language for r in records}),
                  "pending_review": sum(r.review_status == "pending" for r in records)}
    elif args.command == "merge":
        if Path(args.output).exists():
            raise ValueError("choose a new merged dataset path")
        records = [r for source in args.sources for r in load_records(source)]
        validate_records(records)
        write_jsonl(args.output, records)
        result = {"merged": len(records), "output": args.output}
    elif args.command == "split":
        result = split_file(args.source, args.output, args.seed)
    elif args.command == "format":
        records = load_records(args.source)
        write_jsonl(args.output, ({"id": r.id, "scenario_group_id": r.scenario_group_id,
            "messages": messages_for(r.input) + [{"role": "assistant", "content": r.target.model_dump_json()}],
            "review_status": r.review_status, "synthetic": r.synthetic} for r in records))
        result = {"formatted": len(records), "note": "Training CLI reads original DatasetRecord files, not this inspection export."}
    elif args.command == "schema":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(Understanding.model_json_schema(), indent=2), encoding="utf-8")
        result = {"output": str(path)}
    elif args.command in {"preflight", "train", "tokenize-check"}:
        from .training import config_from, preflight, prepare_records, run_training
        config = config_from(args.config)
        if args.command == "preflight":
            result = preflight(config, args.allow_unreviewed)
        elif args.command == "tokenize-check":
            from transformers import AutoTokenizer
            from .prompting import training_example
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=False)
            train, val, _ = prepare_records(config, args.allow_unreviewed)
            lengths = [len(training_example(tokenizer, r.input, r.target, config.max_length)["input_ids"])
                       for r in train + val]
            result = {"tokenized": len(lengths), "max_tokens": max(lengths), "weights_loaded": False}
        else:
            run_training(config, args.allow_unreviewed, args.resume)
            result = {"training_completed": True, "output": config.output_dir}
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("routemind.api:create_app", factory=True, host=args.host, port=args.port, workers=1)
        return
    elif args.command == "predict":
        from .backends import TransformersBackend
        from .evaluation import generate_predictions
        records = load_records(args.source)
        backend = TransformersBackend(
            args.model, args.adapter, args.max_input_tokens, args.max_new_tokens)
        generate_predictions(records, backend, args.output)
        result = {"predictions": len(records), "model": backend.model_name}
    elif args.command == "evaluate":
        from .evaluation import evaluate_files
        report = evaluate_files(args.gold, args.predictions, args.output)
        result = {"report": args.output, "samples": report["samples"]}
    elif args.command == "compare":
        from .evaluation import compare_reports
        result = compare_reports(args.baseline, args.adapter)
    else:
        from .synthesis import synthesize
        result = synthesize(args.source, args.output, args.limit, args.base_url, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
