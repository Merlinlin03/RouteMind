import json

import pytest

from routemind.evaluation import Prediction, compare_reports, evaluate


def prediction(row, raw=None, **kwargs):
    return Prediction(id=row.id, model="test-double", adapter=None,
                      raw_output=raw if raw is not None else row.target.model_dump_json(),
                      latency_ms=1.0, error=None, **kwargs)


def test_invalid_and_missing_predictions_stay_in_denominator(by_id):
    rows = [by_id["human-en"], by_id["human-es"], by_id["human-pt"]]
    report = evaluate(rows, [prediction(rows[0]), prediction(rows[1], "not JSON")])
    assert report["json_parse_rate"] == pytest.approx(1 / 3)
    assert report["primary_intent_accuracy"] == pytest.approx(1 / 3)
    assert report["explicit_human_request"]["recall"] == pytest.approx(1 / 3)
    assert report["set_metrics"]["intent"]["fn"] == 2
    assert len(report["errors"]) == 2


def test_schema_rate_separate_from_json_rate(by_id):
    row = by_id["premium-en"]
    report = evaluate([row], [prediction(row, "{}")])
    assert report["json_parse_rate"] == 1
    assert report["schema_valid_rate"] == 0


def test_evidence_failure_does_not_count_as_correct_intent(by_id):
    row = by_id["premium-en"]
    obj = row.target.model_dump()
    obj["issues"][0]["evidence_spans"][0]["quote"] = "invented"
    report = evaluate([row], [prediction(row, json.dumps(obj))])
    assert report["schema_valid_rate"] == 1
    assert report["evidence_valid_rate"] == 0
    assert report["primary_intent_accuracy"] == 0


def test_duplicate_predictions_rejected(by_id):
    row = by_id["premium-en"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([row], [prediction(row), prediction(row)])


def test_absent_positive_class_is_not_claimed_perfect(by_id):
    row = by_id["premium-en"]
    report = evaluate([row], [prediction(row)])
    assert report["explicit_human_request"]["recall"] is None


def test_report_comparison_requires_same_testset(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"gold_sha256": "one"}))
    b.write_text(json.dumps({"gold_sha256": "two"}))
    with pytest.raises(ValueError, match="same gold"):
        compare_reports(str(a), str(b))
