"""HTTP integration: provide a FeedbackRequest JSON file on stdin."""
import json
import os
import sys

from routemind.adapters import to_opspilot_context
from routemind.client import RouteMindClient
from routemind.schema import FeedbackRequest

if __name__ == "__main__":
    request = FeedbackRequest.model_validate(json.load(sys.stdin))
    response = RouteMindClient(os.getenv("ROUTEMIND_URL", "http://127.0.0.1:8020")).understand(request)
    print(json.dumps(to_opspilot_context(response.result), ensure_ascii=False))
    # Add this context to TurnPlanner prompt inputs; preserve existing flow/state validation.
