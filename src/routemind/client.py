import httpx

from .api import UnderstandResponse
from .schema import FeedbackRequest, validate_evidence


class RouteMindClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8020", timeout: float = 120):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def understand(self, request: FeedbackRequest) -> UnderstandResponse:
        response = httpx.post(f"{self.base_url}/v1/understand", json=request.model_dump(), timeout=self.timeout)
        response.raise_for_status()
        result = UnderstandResponse.model_validate(response.json())
        if result.request_id != request.request_id:
            raise ValueError("response request_id mismatch")
        validate_evidence(result.result, request)
        return result
