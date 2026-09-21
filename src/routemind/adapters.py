"""Pure compatibility adapters. No planning, state mutation or action execution."""
from .schema import Understanding


def to_opspilot_context(result: Understanding) -> dict:
    candidates = []
    allowed = {"payment_order_id", "app_version", "device_model"}
    for issue in result.issues:
        if issue.status in {"negated", "resolved", "conditional"}:
            continue
        for field, value in issue.slots.model_dump().items():
            if field in allowed and value is not None:
                candidates.append({"field": field, "value": value,
                                   "evidence": [e.model_dump() for e in issue.evidence_spans]})
    return {"routemind_semantics": result.model_dump(), "slot_candidates": candidates,
            "note": "Candidates only. TurnPlanner validates state, selects flows and emits commands."}


def to_growthtriage_labels(results: dict[str, Understanding]) -> dict[str, str]:
    """Lossy legacy mapping. Promise mismatch wins over frequency for mixed feedback."""
    labels = {}
    for feedback_id, result in results.items():
        aspects = {i.aspect for i in result.issues if i.status == "unresolved"}
        labels[feedback_id] = ("promise_mismatch" if "ad_promise_mismatch" in aspects else
                               "repeat_exposure" if "ad_frequency" in aspects else "other")
    return labels
