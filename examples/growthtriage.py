"""HTTP integration: stdin is an array of FeedbackRequest objects with unique request_ids."""
import json
import os
import sys

from routemind.adapters import to_growthtriage_labels
from routemind.client import RouteMindClient
from routemind.schema import FeedbackRequest

if __name__ == "__main__":
    requests = [FeedbackRequest.model_validate(row) for row in json.load(sys.stdin)]
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate request ids")
    client = RouteMindClient(os.getenv("ROUTEMIND_URL", "http://127.0.0.1:8020"))
    results = {r.request_id: client.understand(r).result for r in requests}
    print(json.dumps({"labels": to_growthtriage_labels(results),
                      "semantic_evidence": {key: value.model_dump() for key, value in results.items()}},
                     ensure_ascii=False))
    # Attach semantic_evidence separately; pass labels into the existing feedback pipeline.
