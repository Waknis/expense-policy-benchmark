from expensebench.routing import METHOD_PRIORITY, always_review, citation_gate, receipt_rule, select_route


def packet():
    return {
        "case_id": "c1",
        "policy": {"id": "p1"},
        "expense": {"id": "e1", "amount_cents": 7500, "currency": "USD"},
        "evidence": [{"id": "r1", "kind": "receipt", "value": "itemized"}],
    }


def test_receipt_control_ceiling_currency_and_receipt():
    case = packet()
    assert receipt_rule(case)["action"] == "approve"
    case["expense"]["amount_cents"] += 1
    assert receipt_rule(case)["action"] == "review"
    case["expense"].update(amount_cents=100, currency="EUR")
    assert receipt_rule(case)["action"] == "review"
    case["expense"]["currency"] = "USD"
    case["evidence"][0]["value"] = "card slip"
    assert receipt_rule(case)["action"] == "review"
    assert always_review(case)["evidence_ids"] == []


def test_citation_gate_requires_policy_and_a_local_fact():
    case = packet()
    prediction = {"action": "approve", "evidence_ids": ["p1", "r1"]}
    assert citation_gate(case, prediction)["action"] == "approve"
    for citations in [["p1"], ["e1", "r1"], ["p1", "made-up"], ["p1", "p1"], None]:
        assert citation_gate(case, {**prediction, "evidence_ids": citations})["action"] == "review"
    assert citation_gate(case, {**prediction, "action": "reject"})["action"] == "reject"
    case["evidence"].extend({"id": str(i), "kind": "memo", "value": ""} for i in range(4))
    assert citation_gate(case, {**prediction, "evidence_ids": ["p1", "0", "1", "2", "3"]})["action"] == "review"


def test_selection_uses_development_error_constraint_and_declared_tie_break():
    summaries = {m: {"false_approvals": 0, "approvals": 0} for m in METHOD_PRIORITY}
    assert select_route(summaries) == "always_review"
    summaries["receipt_rule"].update(approvals=9, false_approvals=1)
    summaries["llm_evidence"].update(approvals=7)
    summaries["citation_gate"].update(approvals=7)
    assert select_route(summaries) == "citation_gate"
