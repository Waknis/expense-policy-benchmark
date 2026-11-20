import json

import pytest

from expensebench.llm import messages_for, parse_response, write_split


def test_ablations_only_differ_in_visible_evidence():
    packet = {
        "case_id": "heldout-secret", "expense": {"id": "X", "amount_cents": 899},
        "policy": {"id": "P", "text": "Itemized receipts must be available."},
        "evidence": [{"id": "R", "description": "Itemized receipt."}],
        "gold": "SENTINEL",
    }
    basic, full = [messages_for(packet, m) for m in ("llm_policy", "llm_evidence")]
    assert basic[0] == full[0]
    full_visible = json.loads(full[1]["content"])
    assert full_visible.pop("evidence") == packet["evidence"]
    assert json.loads(basic[1]["content"]) == full_visible == {
        "policy": packet["policy"], "expense": packet["expense"]
    }


@pytest.mark.parametrize(("text", "reason"), [
    ("not JSON", "stop"),
    ('{"action":"approve","evidence_ids":[]}', "length"),
    ('{"action":"approve","evidence_ids":"P"}', "stop"),
    ('{"action":"unknown","evidence_ids":[]}', "stop"),
])
def test_parse_failure_remains_invalid(text, reason):
    assert parse_response(text, reason) == ("invalid", [])


def test_hallucinated_or_repeated_citations_are_preserved_for_evaluation():
    text = '{"action":"approve","evidence_ids":["P","unknown","P"]}'
    assert parse_response(text, "stop") == ("approve", ["P", "unknown", "P"])


def test_finding_variant_uses_identical_observations_and_requires_a_finding():
    packet = {"expense": {"id": "X"}, "policy": {"id": "P"}, "evidence": [{"id": "E"}]}
    full = messages_for(packet, "llm_evidence")
    finding = messages_for(packet, "llm_finding")
    assert finding[1] == full[1]
    assert finding[0]["content"].startswith(full[0]["content"])
    bare = '{"action":"review","evidence_ids":["P"]}'
    assert parse_response(bare, "stop", "llm_finding") == ("invalid", [])
    with_finding = '{"finding":"A receipt is missing.","action":"review","evidence_ids":["P"]}'
    assert parse_response(with_finding, "stop", "llm_finding") == ("review", ["P"])


def test_split_rerun_replaces_only_selected_split(tmp_path):
    path = tmp_path / "predictions.jsonl"
    dev = {"case_id": "a", "split": "dev", "action": "approve"}
    test = {"case_id": "b", "split": "test", "action": "reject"}
    write_split(path, [dev], "dev")
    write_split(path, [test], "test")
    write_split(path, [dict(dev, action="review")], "dev")
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        dict(dev, action="review"), test,
    ]
