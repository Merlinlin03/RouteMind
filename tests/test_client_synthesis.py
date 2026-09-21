import json

import httpx
import pytest

from routemind.client import RouteMindClient
from routemind.data import load_records, write_jsonl
from routemind.synthesis import synthesize


def test_client_validates_request_identity(by_id, monkeypatch):
    row = by_id["premium-en"]
    payload = {"request_id": "wrong", "model": "test-double", "adapter": None,
               "result": row.target.model_dump()}

    def post(url, **kwargs):
        assert kwargs["timeout"] == 120
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr("routemind.client.httpx.post", post)
    with pytest.raises(ValueError, match="request_id"):
        RouteMindClient().understand(row.input)
    payload["request_id"] = row.input.request_id
    assert RouteMindClient().understand(row.input).result == row.target


def test_client_propagates_service_errors(by_id, monkeypatch):
    monkeypatch.setattr("routemind.client.httpx.post", lambda url, **kwargs: httpx.Response(
        503, json={"detail": "model_not_ready"}, request=httpx.Request("POST", url)))
    with pytest.raises(httpx.HTTPStatusError):
        RouteMindClient().understand(by_id["premium-en"].input)


def test_synthesis_requires_key_before_any_request(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHESIS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYNTHESIS_API_KEY"):
        synthesize("not-opened.jsonl", str(tmp_path / "out.jsonl"), 1, "https://example.invalid/v1", "generator")


def test_generator_cannot_approve_data_or_change_group(by_id, tmp_path, monkeypatch):
    row = by_id["premium-en"]
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    write_jsonl(source, [row])
    generated = row.model_dump()
    generated.update(review_status="approved", synthetic=False, scenario_group_id="wrong-group")
    replacement = "Premium is still locked after I paid through Stripe."
    generated["input"]["messages"][0]["content"] = replacement
    generated["target"]["issues"][0]["evidence_spans"][0]["quote"] = replacement
    monkeypatch.setenv("SYNTHESIS_API_KEY", "test-only-not-a-real-key")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            assert kwargs["json"]["max_tokens"] == 4096
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "choices": [{"message": {"content": json.dumps(generated)}}]})

    monkeypatch.setattr("routemind.synthesis.httpx.Client", FakeClient)
    synthesize(str(source), str(output), 1, "https://example.invalid/v1", "test-generator")
    result = load_records(output)[0]
    assert result.synthetic is True and result.review_status == "pending"
    assert result.scenario_group_id == row.scenario_group_id
    assert result.source == "synthetic:test-generator"
    with pytest.raises(ValueError, match="existing annotations"):
        synthesize(str(source), str(output), 1, "https://example.invalid/v1", "test-generator")
