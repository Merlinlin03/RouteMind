from pathlib import Path

import pytest

from routemind.data import load_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def records():
    return load_records(ROOT / "data/seeds/feedback.jsonl")


@pytest.fixture
def by_id(records):
    return {r.id: r for r in records}
