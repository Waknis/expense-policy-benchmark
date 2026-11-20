"""Replay the model's saved JSON and validate input digests without a model server."""

import hashlib
import json
from pathlib import Path

from expensebench.data import build_database, case_packet, load_cases
from expensebench.llm import parse_response, request_body


def test_all_recorded_model_requests_and_decisions_match_the_frozen_cases():
    root = Path(__file__).resolve().parents[1] / "results" / "predictions"
    cases = {row["case_id"]: row for row in load_cases()}
    conn = build_database()
    try:
        for method in ("llm_policy", "llm_evidence", "llm_finding"):
            predictions = [
                json.loads(line) for line in (root / f"{method}.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            assert len(predictions) == len(cases)
            assert {row["case_id"] for row in predictions} == cases.keys()
            for row in predictions:
                case = cases[row["case_id"]]
                assert (row["method"], row["split"]) == (method, case["split"])
                packet = case_packet(conn, row["case_id"])
                body = request_body(packet, method, model=row["model"])
                assert row["prompt_sha256"] == hashlib.sha256(body).hexdigest()
                assert (row["action"], row["evidence_ids"]) == parse_response(
                    row["raw"], row["finish_reason"], method
                )
    finally:
        conn.close()
