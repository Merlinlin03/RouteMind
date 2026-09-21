"""Local single-worker API. Explicit readiness and one in-flight GPU generation."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from .backends import InputTooLong, TransformersBackend
from .schema import Contract, FeedbackRequest, Understanding, parse_output


class UnderstandResponse(Contract):
    request_id: str
    model: str
    adapter: str | None
    result: Understanding


def create_app(backend=None) -> FastAPI:
    injected = backend is not None
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        if not injected:
            try:
                app.state.backend = TransformersBackend(
                    os.getenv("ROUTEMIND_MODEL", "Qwen/Qwen3-32B"),
                    os.getenv("ROUTEMIND_ADAPTER") or None,
                    int(os.getenv("ROUTEMIND_MAX_INPUT_TOKENS", "6144")),
                    int(os.getenv("ROUTEMIND_MAX_NEW_TOKENS", "2048")),
                )
            except Exception as exc:
                # Do not expose filesystem paths, tokens or provider payloads through API errors.
                logging.getLogger(__name__).error("Backend initialization failed (%s)", type(exc).__name__)
                app.state.backend = None
        yield

    app = FastAPI(title="RouteMind", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend

    @app.get("/health")
    def health():
        engine = app.state.backend
        return {"alive": True, "ready": engine is not None,
                "model": engine.model_name if engine else None}

    @app.get("/schema")
    def schema():
        return Understanding.model_json_schema()

    @app.post("/v1/understand", response_model=UnderstandResponse)
    def understand(request: FeedbackRequest):
        engine = app.state.backend
        if engine is None:
            raise HTTPException(503, "model_not_ready")
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "inference_busy", headers={"Retry-After": "1"})
        try:
            try:
                raw = engine.generate(request)
            except InputTooLong:
                raise HTTPException(413, "input_token_limit_exceeded") from None
            except Exception:
                raise HTTPException(503, "inference_failed") from None
            try:
                result = parse_output(raw, request)
            except ValueError:
                raise HTTPException(502, "model_output_invalid") from None
            return UnderstandResponse(request_id=request.request_id, model=engine.model_name,
                                      adapter=engine.adapter, result=result)
        finally:
            lock.release()

    return app
