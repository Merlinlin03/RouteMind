import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from routemind.adapters import to_growthtriage_labels, to_opspilot_context
from routemind.schema import FeedbackRequest, Understanding, parse_output, validate_evidence


def test_all_seed_schemas_and_evidence(records):
    validator = Draft202012Validator(Understanding.model_json_schema())
    assert {r.target.language for r in records} >= {"en", "es", "pt", "id", "hi", "ar"}
    for row in records:
        validator.validate(row.target.model_dump())
        validate_evidence(row.target, row.input)
        assert row.synthetic and row.review_status == "pending"


def test_hallucinated_evidence_is_rejected(by_id):
    row = by_id["premium-en"]
    obj = row.target.model_dump()
    obj["issues"][0]["evidence_spans"][0]["quote"] = "The backend verified payment."
    with pytest.raises(ValueError, match="exact substring"):
        parse_output(json.dumps(obj), row.input)


@pytest.mark.parametrize("field", ["next_node", "execute_action", "campaign_id"])
def test_model_cannot_emit_business_commands(by_id, field):
    obj = by_id["premium-en"].target.model_dump()
    obj[field] = "invented"
    with pytest.raises(ValidationError):
        Understanding.model_validate(obj)


def test_human_request_needs_user_evidence(by_id):
    row = by_id["human-en"]
    obj = row.target.model_dump()
    obj["human_request_evidence"] = []
    with pytest.raises(ValidationError):
        Understanding.model_validate(obj)


def test_request_ids_and_roles(by_id):
    obj = by_id["premium-en"].input.model_dump()
    obj["messages"].append(obj["messages"][0].copy())
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate(obj)
    obj["messages"] = [{"id": "a", "role": "assistant", "content": "hello"}]
    with pytest.raises(ValidationError):
        FeedbackRequest.model_validate(obj)


def test_opspilot_adapter_is_context_not_turn_plan(by_id):
    context = to_opspilot_context(by_id["conditional_refund-en"].target)
    assert "commands" not in context and "task" not in context
    assert context["slot_candidates"] == []
    assert context["routemind_semantics"]["issues"][1]["status"] == "conditional"
    human = to_opspilot_context(by_id["human-en"].target)
    assert "start_flow" not in json.dumps(human)


def test_no_country_inferred_from_language(by_id):
    context = to_opspilot_context(by_id["premium-es"].target)
    assert "country" not in json.dumps(context)


def test_growthtriage_legacy_mapping_and_precedence(by_id):
    mixed = by_id["frequency-en"].target.model_copy(deep=True)
    mixed.issues += by_id["promise-en"].target.issues
    labels = to_growthtriage_labels({"one": mixed, "two": by_id["frequency-en"].target,
                                    "three": by_id["human-en"].target})
    assert labels == {"one": "promise_mismatch", "two": "repeat_exposure", "three": "other"}
    mixed.issues[1].status = "resolved"
    assert to_growthtriage_labels({"one": mixed})["one"] == "repeat_exposure"
