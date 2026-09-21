"""Offline validation and grouped splitting. No model or GPU required."""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path

from .schema import DatasetRecord, validate_evidence


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fingerprint(record: DatasetRecord) -> str:
    text = json.dumps([(m.role, unicodedata.normalize("NFKC", " ".join(m.content.split())).casefold())
                       for m in record.input.messages], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def load_records(path: str | Path) -> list[DatasetRecord]:
    records = [DatasetRecord.model_validate(row) for row in read_jsonl(path)]
    validate_records(records)
    return records


def validate_records(records: list[DatasetRecord]) -> None:
    if not records:
        raise ValueError("empty dataset")
    ids, texts = set(), set()
    for record in records:
        validate_evidence(record.target, record.input)
        if record.id in ids or fingerprint(record) in texts:
            raise ValueError(f"duplicate record id or conversation: {record.id}")
        ids.add(record.id)
        texts.add(fingerprint(record))


def assert_disjoint(*datasets: list[DatasetRecord]) -> None:
    seen_ids, seen_groups, seen_texts = set(), set(), set()
    for records in datasets:
        ids = {r.id for r in records}
        groups = {r.scenario_group_id for r in records}
        texts = {fingerprint(r) for r in records}
        if seen_ids & ids or seen_groups & groups or seen_texts & texts:
            raise ValueError("split leakage: shared id, scenario group or conversation")
        seen_ids |= ids
        seen_groups |= groups
        seen_texts |= texts


def split_records(records: list[DatasetRecord], seed: int = 42) -> dict[str, list[DatasetRecord]]:
    groups = sorted({r.scenario_group_id for r in records})
    if len(groups) < 3:
        raise ValueError("at least three scenario groups required")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * 0.15))
    n_val = max(1, round(len(groups) * 0.15))
    assigned = {g: "test" if i < n_test else "validation" if i < n_test + n_val else "train"
                for i, g in enumerate(groups)}
    result = {name: [r for r in records if assigned[r.scenario_group_id] == name]
              for name in ("train", "validation", "test")}
    assert_disjoint(*result.values())
    return result


def split_file(source: str, output: str, seed: int) -> dict:
    records = load_records(source)
    splits = split_records(records, seed)
    for name, rows in splits.items():
        write_jsonl(Path(output) / f"{name}.jsonl", rows)
    manifest = {
        "seed": seed, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "groups": {name: sorted({r.scenario_group_id for r in rows}) for name, rows in splits.items()},
        "note": "Group separation does not establish real-world dataset representativeness.",
    }
    (Path(output) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
