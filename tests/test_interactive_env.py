import json
from copy import deepcopy

import pytest

from expensebench.interactive_env import ExpenseEnv, LOOKUP_COST, MAX_QUERIES, TOOLS, load_interactive_split
from expensebench.scoring import grade_cases


def episode():
    case = {
        "case_id": "c1", "family": "unit", "pair_id": "pair", "split": "train",
        "expense": {"id": "e1", "amount_cents": 6500},
        "policy": {"id": "p1", "text": "Require membership, then enforce a unit cap.",
                   "checks": [
                       {"tool": "registry", "field": "membership", "operator": "equals", "target": "member"},
                       {"tool": "ledger", "field": "units", "operator": "at_most", "target": 5},
                   ]},
        "tools": {
            "registry": {"id": "r1", "field": "membership", "value": "member", "description": "active membership"},
            "ledger": {"id": "l1", "field": "units", "value": 5, "description": "reserved units"},
            "document": {"id": "d1", "field": "format", "value": "invoice", "description": "attachment type"},
            "authorization": {"id": "a1", "field": "signer", "value": "staff", "description": "signer type"},
        },
    }
    gold = {"case_id": "c1", "action": "approve", "required_evidence": ["p1", "r1", "l1"]}
    return case, gold


def test_observations_hide_counterfactual_until_requested_and_reset_clears_history():
    left, left_gold = episode()
    right = deepcopy(left)
    right["case_id"] = "c2"
    right["tools"]["ledger"]["value"] = 6
    right_gold = {**left_gold, "case_id": "c2", "action": "reject"}
    a, b = ExpenseEnv(left, left_gold), ExpenseEnv(right, right_gold)
    obs = a.reset()
    assert obs == b.reset()
    assert set(obs) == {"expense", "policy", "tools", "records", "remaining_queries"}
    assert obs["tools"] == TOOLS
    assert obs["records"] == {}
    assert obs["remaining_queries"] == MAX_QUERIES
    obs["policy"]["checks"].clear()
    oa, reward, done, info = a.step("lookup:registry")
    assert oa == b.step("lookup:registry")[0]
    assert list(oa["records"]) == ["registry"]
    assert (reward, done, info) == (-LOOKUP_COST, False, {"reason": "retrieved", "source": "registry"})
    oa["records"]["registry"]["value"] = None
    assert a.step("lookup:ledger")[0]["records"]["registry"]["value"] == "member"
    assert b.step("lookup:ledger")[0]["records"]["ledger"]["value"] == 6
    assert a.reset() == b.reset()


def test_exact_retrieval_beats_citing_distractors_and_can_submit_explicit_subset():
    case, gold = episode()
    env = ExpenseEnv(case, gold)
    for distractor, explicit, expected_total in ((False, None, 1.9), (True, None, .35),
                                                 (True, ["p1", "r1", "l1"], 1.85)):
        sources = ("registry", "ledger", "document") if distractor else ("registry", "ledger")
        env.reset()
        total = sum(env.step(f"lookup:{source}")[1] for source in sources)
        _, reward, done, info = env.step("approve", explicit)
        assert total + reward == pytest.approx(expected_total)
        allowed = {"c1": {"p1", "r1", "l1", "d1", "a1", "e1"}}
        graded, = grade_cases([case], {"c1": gold}, [info["prediction"]], allowed)
        assert info["grade"] == {key: graded[key] for key in info["grade"]}
        assert set(info["grade"]) == {"action_correct", "evidence_exact", "evidence_sufficient", "joint_correct"}
        assert done
        with pytest.raises(RuntimeError, match="reset"):
            env.step("review")


@pytest.mark.parametrize(("action", "cost"), [("approve", -3), ("review", -1), ("reject", .5)])
def test_asymmetric_terminal_reward_and_unsupported_correct_decision(action, cost):
    case, gold = episode()
    case["tools"]["registry"]["value"] = "expired"
    gold.update(action="reject", required_evidence=["p1", "r1"])
    env = ExpenseEnv(case, gold)
    env.reset()
    _, reward, done, info = env.step(action)
    assert done and reward == cost
    assert info["prediction"]["evidence_ids"] == ["p1"]
    env.reset()
    env.step("lookup:registry")
    assert env.step("reject")[1] == 2.0


def test_budget_and_invalid_actions_cannot_hide_unsafe_episodes():
    case, gold = episode()
    env = ExpenseEnv(case, gold)
    with pytest.raises(RuntimeError):
        env.step("approve")
    env.reset()
    for source in TOOLS[:MAX_QUERIES]:
        obs, _, done, _ = env.step(f"lookup:{source}")
        assert not done
    assert obs["remaining_queries"] == 0
    _, reward, done, info = env.step("lookup:authorization")
    assert (reward, done, info["reason"]) == (-3.0, True, "lookup_budget_exhausted")
    assert info["prediction"]["action"] == "invalid"
    for first, second, citations, reason in (
        ("lookup:registry", "lookup:registry", None, "repeated_lookup"),
        (None, "lookup:document", ["p1"], "citations_on_lookup"),
        (None, "lookup:gold", None, "unknown_action"),
        (None, "approve", ["p1", "l1"], "invalid_citation"),
        (None, "approve", "p1", "invalid_citation"),
    ):
        env.reset()
        if first:
            env.step(first)
        _, reward, done, info = env.step(second, citations)
        assert done and reward == -3 and info["reason"] == reason
        assert info["prediction"] == {"case_id": "c1", "action": "invalid", "evidence_ids": []}


def test_split_loader_requires_only_selected_directory_and_rejects_duplicate_labels(tmp_path):
    case, gold = episode()
    directory = tmp_path / "train"
    directory.mkdir()
    (directory / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    labels = directory / "gold.jsonl"
    labels.write_text(json.dumps(gold) + "\n", encoding="utf-8")
    assert load_interactive_split("train", tmp_path) == ([case], {"c1": gold})
    labels.write_text((json.dumps(gold) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="uniquely"):
        load_interactive_split("train", tmp_path)
    with pytest.raises(ValueError, match="split must"):
        load_interactive_split("../test", tmp_path)
