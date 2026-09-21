from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from routemind.api import create_app
from routemind.backends import InputTooLong


class FakeBackend:
    model_name = "test-double"
    adapter = None

    def __init__(self, raw):
        self.raw = raw

    def generate(self, request):
        return self.raw


def test_api_contract(by_id):
    with TestClient(create_app(FakeBackend(by_id["premium-en"].target.model_dump_json()))) as client:
        assert client.get("/health").json()["ready"] is True
        response = client.post("/v1/understand", json=by_id["premium-en"].input.model_dump())
        assert response.status_code == 200
        assert response.json()["result"]["primary_intent"] == "premium_not_activated"
        assert client.post("/v1/understand", json={"messages": []}).status_code == 422
        assert client.get("/schema").status_code == 200


def test_unready_backend_returns_unavailable(by_id, monkeypatch):
    def fail(*args):
        raise RuntimeError("model not installed")
    monkeypatch.setattr("routemind.api.TransformersBackend", fail)
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["ready"] is False
        assert client.post("/v1/understand", json=by_id["premium-en"].input.model_dump()).status_code == 503


def test_invalid_output_returns_error_and_releases_lock(by_id):
    row = by_id["premium-en"]
    backend = FakeBackend("{bad json")
    with TestClient(create_app(backend)) as client:
        assert client.post("/v1/understand", json=row.input.model_dump()).status_code == 502
        backend.raw = row.target.model_dump_json()
        assert client.post("/v1/understand", json=row.input.model_dump()).status_code == 200


def test_concurrent_gpu_work_is_rejected(by_id):
    row = by_id["premium-en"]
    entered, release = Event(), Event()

    class Blocking(FakeBackend):
        def generate(self, request):
            entered.set()
            assert release.wait(5)
            return self.raw

    with TestClient(create_app(Blocking(row.target.model_dump_json()))) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, "/v1/understand", json=row.input.model_dump())
            try:
                assert entered.wait(5)
                assert client.post("/v1/understand", json=row.input.model_dump()).status_code == 429
            finally:
                release.set()
            assert pending.result().status_code == 200


@pytest.mark.parametrize("exception,status", [(InputTooLong(), 413), (RuntimeError("secret"), 503)])
def test_generation_errors_are_bounded_and_sanitized(by_id, exception, status):
    class Broken(FakeBackend):
        def generate(self, request):
            raise exception

    with TestClient(create_app(Broken(""))) as client:
        response = client.post("/v1/understand", json=by_id["premium-en"].input.model_dump())
        assert response.status_code == status
        assert "secret" not in response.text
