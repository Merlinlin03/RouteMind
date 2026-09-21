"""Explicit, bounded API synthesis. Outputs always require review."""
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx

from .data import load_records, validate_records, write_jsonl
from .schema import DatasetRecord, validate_evidence


def synthesize(source: str, output: str, limit: int, base_url: str, model: str):
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    key = os.environ.get("SYNTHESIS_API_KEY")
    if not key or not model or not base_url:
        raise ValueError("set SYNTHESIS_API_KEY, model and base URL explicitly")
    destination = Path(output)
    if destination.exists():
        raise ValueError("choose a new output path; existing annotations must not be overwritten")
    seeds, generated = load_records(source), []
    batch_id = uuid4().hex[:12]
    with httpx.Client(timeout=120) as client:
        for index in range(limit):
            seed = seeds[index % len(seeds)]
            response = client.post(base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"}, json={
                    "model": model, "temperature": 0.7, "max_tokens": 4096,
                    "messages": [{"role": "system", "content":
                        "Generate one synthetic multilingual feedback annotation as JSON. "
                        "Paraphrase the seed in the same language; preserve its semantic challenge. "
                        "Use fictional data only, never add unsupported facts. Every evidence quote "
                        "must occur verbatim in the referenced new message. Output a DatasetRecord "
                        "matching this schema: " + json.dumps(DatasetRecord.model_json_schema())},
                        {"role": "user", "content": seed.model_dump_json()}],
                })
            response.raise_for_status()
            row = DatasetRecord.model_validate(json.loads(response.json()["choices"][0]["message"]["content"]))
            # Model-supplied provenance must never promote synthetic data to reviewed real data.
            if row.target.language != seed.target.language:
                raise ValueError("generated language differs from the requested seed language")
            row.id = f"synthetic-{batch_id}-{index:06d}-{seed.id}"
            row.scenario_group_id = seed.scenario_group_id
            row.input.request_id = row.id
            row.synthetic = True
            row.review_status = "pending"
            row.source = f"synthetic:{model}"
            validate_evidence(row.target, row.input)
            generated.append(row)
    validate_records(seeds + generated)
    write_jsonl(destination, generated)
    return {"written": len(generated), "review_status": "pending", "output": str(destination)}
