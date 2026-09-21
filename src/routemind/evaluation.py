"""Evaluate saved raw generations; invalid/missing outputs stay in the denominator."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import Field

from .data import load_records, read_jsonl, write_jsonl
from .schema import Contract, DatasetRecord, Understanding, evidence_items, validate_evidence


class Prediction(Contract):
    id: str
    model: str
    adapter: str | None
    raw_output: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    error: str | None


def generate_predictions(records: list[DatasetRecord], backend, output: str):
    """Write incrementally, never send target labels to inference."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            start, raw, error = time.perf_counter(), "", None
            try:
                raw = backend.generate(record.input)
            except Exception as exc:
                error = type(exc).__name__
            prediction = Prediction(id=record.id, model=backend.model_name, adapter=backend.adapter,
                raw_output=raw, latency_ms=(time.perf_counter() - start) * 1000, error=error)
            handle.write(prediction.model_dump_json() + "\n")
            handle.flush()


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tokens(result: Understanding | None, field: str) -> set:
    if result is None:
        return set()
    if field == "intent":
        return {i.intent for i in result.issues}
    if field == "aspect":
        return {(i.intent, i.aspect) for i in result.issues}
    if field == "status":
        return {(i.intent, i.aspect, i.status) for i in result.issues}
    if field == "sentiment":
        return {(i.intent, i.aspect, i.sentiment) for i in result.issues}
    if field == "slots":
        return {(i.intent, i.aspect, k, v) for i in result.issues
                for k, v in i.slots.model_dump().items() if v is not None}
    if field == "evidence":
        return {(e.message_id, e.quote) for e in evidence_items(result)}
    if field == "conditions":
        return {(c.intent, c.condition) for c in result.request_conditions}
    if field == "corrections":
        return {(c.field, c.old_value, c.new_value) for c in result.corrections}
    if field == "ad_fields":
        return {(i.intent, i.aspect, k, getattr(i, k)) for i in result.issues
                for k in ("ad_format", "ad_placement", "advertised_claim", "reported_experience")
                if getattr(i, k) is not None}
    if field == "missing":
        return set(result.missing_information)
    if field == "urgency":
        return {s.kind for s in result.urgency_signals}
    raise ValueError(field)


def evaluate(records: list[DatasetRecord], predictions: list[Prediction]) -> dict:
    if not records:
        raise ValueError("empty gold dataset")
    ids = [p.id for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) - {r.id for r in records}:
        raise ValueError("duplicate or unknown prediction ids")
    identities = {(p.model, p.adapter) for p in predictions}
    if len(identities) > 1:
        raise ValueError("evaluate one model/adapter per file")
    by_id = {p.id: p for p in predictions}
    validator = Draft202012Validator(Understanding.model_json_schema())
    n, json_ok, schema_ok, evidence_ok, primary_ok, exact, sentiment_ok, language_ok = len(records), 0, 0, 0, 0, 0, 0, 0
    human_tp = human_fp = human_fn = human_correct = 0
    fields = ("intent", "aspect", "status", "sentiment", "slots", "evidence", "conditions", "corrections", "ad_fields", "missing", "urgency")
    counts = {key: [0, 0, 0] for key in fields}
    intent_counts: dict[str, list[int]] = {}
    errors, latency = [], []
    for record in records:
        prediction, result, reason = by_id.get(record.id), None, None
        if prediction is None:
            reason = "missing_prediction"
        elif prediction.error:
            reason = "generation_error"
        else:
            try:
                obj = json.loads(prediction.raw_output)
                json_ok += 1
            except (ValueError, TypeError):
                reason = "invalid_json"
            if reason is None:
                if list(validator.iter_errors(obj)):
                    reason = "invalid_schema"
                else:
                    try:
                        result = Understanding.model_validate(obj)
                        schema_ok += 1
                    except ValueError:
                        reason = "invalid_semantics"
            if reason is None:
                try:
                    validate_evidence(result, record.input)
                    evidence_ok += 1
                except ValueError:
                    result, reason = None, "invalid_evidence"
        if prediction is not None:
            latency.append(prediction.latency_ms)
        gold = record.target
        primary_ok += int(result is not None and result.primary_intent == gold.primary_intent)
        language_ok += int(result is not None and result.language == gold.language)
        sentiment_ok += int(result is not None and tokens(result, "sentiment") == tokens(gold, "sentiment"))
        same = result is not None and result.model_dump() == gold.model_dump()
        exact += int(same)
        if not same:
            errors.append({"id": record.id, "language": gold.language,
                           "reason": reason or "field_mismatch",
                           "raw_output": prediction.raw_output if prediction else None})
        for field in fields:
            expected, actual = tokens(gold, field), tokens(result, field)
            for index, value in enumerate((len(expected & actual), len(actual - expected), len(expected - actual))):
                counts[field][index] += value
            if field == "intent":
                for label in expected | actual:
                    entry = intent_counts.setdefault(label, [0, 0, 0])
                    entry[0 if label in expected & actual else 1 if label in actual else 2] += 1
        human_actual = result.explicit_human_request if result else False
        human_tp += int(gold.explicit_human_request and human_actual)
        human_fp += int(not gold.explicit_human_request and human_actual)
        human_fn += int(gold.explicit_human_request and not human_actual)
        human_correct += int(result is not None and human_actual == gold.explicit_human_request)
    label_scores = {label: prf(*count) for label, count in sorted(intent_counts.items())}
    macro_values = [s["f1"] for s in label_scores.values() if s["f1"] is not None]
    return {
        "samples": n, "predictions": len(predictions),
        "identity": list(next(iter(identities))) if identities else None,
        "json_parse_rate": json_ok / n, "schema_valid_rate": schema_ok / n,
        "evidence_valid_rate": evidence_ok / n, "primary_intent_accuracy": primary_ok / n,
        "language_accuracy": language_ok / n, "sentiment_set_accuracy": sentiment_ok / n,
        "exact_match_rate": exact / n,
        "set_metrics": {key: prf(*count) for key, count in counts.items()},
        "intent_macro_f1": statistics.mean(macro_values) if macro_values else None,
        "intent_per_label": label_scores,
        "explicit_human_request": {**prf(human_tp, human_fp, human_fn), "accuracy": human_correct / n},
        "latency_ms": {"count": len(latency), "mean": statistics.mean(latency) if latency else None,
                       "p95": sorted(latency)[math.ceil(0.95 * len(latency)) - 1] if latency else None},
        "errors": errors,
    }


def evaluate_files(gold_path: str, prediction_path: str, output: str) -> dict:
    records = load_records(gold_path)
    predictions = [Prediction.model_validate(row) for row in read_jsonl(prediction_path)]
    report = evaluate(records, predictions)
    report["by_language"] = {}
    for lang in sorted({r.target.language for r in records}):
        subset = [r for r in records if r.target.language == lang]
        selected = {r.id for r in subset}
        subreport = evaluate(subset, [p for p in predictions if p.id in selected])
        subreport.pop("errors")
        report["by_language"][lang] = subreport
    report["gold_sha256"] = hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()
    report["dataset"] = {"synthetic_count": sum(r.synthetic for r in records),
                         "approved_count": sum(r.review_status == "approved" for r in records)}
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest.with_suffix(".errors.jsonl"), report.pop("errors"))
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def compare_reports(baseline: str, adapter: str) -> dict:
    base, tuned = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (baseline, adapter)]
    if base["gold_sha256"] != tuned["gold_sha256"]:
        raise ValueError("comparison requires the same gold dataset")
    keys = ("primary_intent_accuracy", "json_parse_rate", "schema_valid_rate", "evidence_valid_rate", "exact_match_rate")
    return {"baseline": base["identity"], "adapter": tuned["identity"],
            "absolute_delta": {k: tuned[k] - base[k] for k in keys}}
