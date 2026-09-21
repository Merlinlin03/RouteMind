"""One versioned contract shared by annotation, training, evaluation and serving."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Intent = Literal[
    "subscription_billing", "refund_request", "premium_not_activated",
    "crash_performance", "ad_complaint", "privacy_permissions", "login_issue",
    "general_feedback", "unknown",
]
Aspect = Literal[
    "billing", "refund", "premium", "crash", "performance", "ad_frequency",
    "ad_interruption", "ad_close_difficulty", "ad_promise_mismatch",
    "ad_reward_missing", "paid_still_ads", "privacy", "login", "other",
]
Language = Literal["en", "es", "pt", "id", "hi", "ar", "mixed", "unknown"]
SlotName = Literal[
    "payment_channel", "payment_order_id", "amount", "currency", "app_version",
    "device_model", "os", "time_expression", "product",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(Contract):
    id: str = Field(min_length=1, max_length=80)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class FeedbackRequest(Contract):
    request_id: str = Field(min_length=1, max_length=100)
    messages: list[Message] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def valid_conversation(self):
        if len({m.id for m in self.messages}) != len(self.messages):
            raise ValueError("duplicate message ids")
        if self.messages[-1].role != "user":
            raise ValueError("last message must be from the user")
        if sum(len(m.content) for m in self.messages) > 24000:
            raise ValueError("conversation exceeds 24000 characters")
        return self


class Evidence(Contract):
    message_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=1, max_length=12000)


class Slots(Contract):
    payment_channel: str | None
    payment_order_id: str | None
    amount: str | None
    currency: str | None
    app_version: str | None
    device_model: str | None
    os: str | None
    time_expression: str | None
    product: str | None


class Issue(Contract):
    intent: Intent
    aspect: Aspect
    sentiment: Literal["positive", "neutral", "negative", "mixed", "unknown"]
    status: Literal["unresolved", "resolved", "conditional", "negated", "unknown"]
    slots: Slots
    evidence_spans: list[Evidence] = Field(min_length=1, max_length=10)
    ad_format: str | None
    ad_placement: str | None
    advertised_claim: str | None
    reported_experience: str | None


class RequestCondition(Contract):
    intent: Intent
    condition: str = Field(min_length=1, max_length=1000)
    evidence_spans: list[Evidence] = Field(min_length=1, max_length=5)


class Correction(Contract):
    field: SlotName
    old_value: str | None
    new_value: str = Field(min_length=1, max_length=500)
    evidence_spans: list[Evidence] = Field(min_length=1, max_length=5)


class UrgencySignal(Contract):
    kind: Literal["blocked_access", "repeated_failure", "time_sensitive", "explicit_urgency"]
    evidence_spans: list[Evidence] = Field(min_length=1, max_length=5)


class Understanding(Contract):
    schema_version: Literal["1.0"]
    language: Language
    primary_intent: Intent
    issues: list[Issue] = Field(min_length=1, max_length=12)
    request_conditions: list[RequestCondition] = Field(max_length=12)
    corrections: list[Correction] = Field(max_length=12)
    explicit_human_request: bool
    human_request_evidence: list[Evidence] = Field(max_length=5)
    urgency_signals: list[UrgencySignal] = Field(max_length=8)
    missing_information: list[SlotName] = Field(max_length=10)

    @model_validator(mode="after")
    def consistent_labels(self):
        if self.primary_intent not in {i.intent for i in self.issues}:
            raise ValueError("primary intent must occur in issues")
        if self.explicit_human_request != bool(self.human_request_evidence):
            raise ValueError("human request flag and evidence disagree")
        if len(set(self.missing_information)) != len(self.missing_information):
            raise ValueError("duplicate missing fields")
        return self


class DatasetRecord(Contract):
    id: str = Field(min_length=1)
    scenario_group_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    synthetic: bool
    review_status: Literal["pending", "approved", "rejected"]
    input: FeedbackRequest
    target: Understanding


def evidence_items(result: Understanding) -> list[Evidence]:
    items = list(result.human_request_evidence)
    for collection in (result.issues, result.request_conditions, result.corrections, result.urgency_signals):
        for item in collection:
            items.extend(item.evidence_spans)
    return items


def validate_evidence(result: Understanding, request: FeedbackRequest) -> None:
    messages = {m.id: m for m in request.messages}
    for evidence in evidence_items(result):
        message = messages.get(evidence.message_id)
        if message is None or evidence.quote not in message.content:
            raise ValueError("evidence must be an exact substring of the referenced message")
    for evidence in result.human_request_evidence:
        if messages[evidence.message_id].role != "user":
            raise ValueError("human request evidence must come from the user")


def parse_output(raw: str, request: FeedbackRequest) -> Understanding:
    # No stripping code fences or repairing JSON: preserve raw-model compliance metrics.
    result = Understanding.model_validate(json.loads(raw))
    validate_evidence(result, request)
    return result
