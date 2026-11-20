"""Fixed expense controls and development-only selection of an approval route."""

METHOD_PRIORITY = ["citation_gate", "llm_evidence", "receipt_rule", "llm_policy", "always_review"]


def always_review(packet):
    return {"case_id": packet["case_id"], "action": "review", "evidence_ids": []}


def receipt_rule(packet):
    """A low-value, itemized-receipt control that does not interpret policy prose."""
    expense = packet["expense"]
    receipts = [item for item in packet["evidence"] if item["kind"] == "receipt"]
    affirmative = [r for r in receipts if r["value"] == "itemized"]
    approve = expense["currency"] == "USD" and expense["amount_cents"] <= 7500 and bool(affirmative)
    return {
        "case_id": packet["case_id"],
        "action": "approve" if approve else "review",
        "evidence_ids": [expense["id"], *(item["id"] for item in (affirmative or receipts))],
    }


def packet_ids(packet):
    return {packet["policy"]["id"], packet["expense"]["id"], *(item["id"] for item in packet["evidence"])}


def citation_gate(packet, prediction):
    """Intercept invalid outputs and approvals with unresolved or unselective citations."""
    action = prediction["action"]
    evidence = prediction["evidence_ids"]
    if action not in {"approve", "review", "reject"}:
        return {**always_review(packet), "gate_reason": "invalid_action"}
    decision = {"case_id": packet["case_id"], "action": action, "evidence_ids": evidence}
    if action != "approve":
        return {**decision, "gate_reason": "no_approval"}
    allowed = packet_ids(packet)
    policy_id = packet["policy"]["id"]
    cited = set(evidence) if isinstance(evidence, list) and all(isinstance(x, str) for x in evidence) else set()
    if policy_id in cited and 2 <= len(cited) <= 4 and cited <= allowed:
        return {**decision, "gate_reason": "passed"}
    return {**always_review(packet), "gate_reason": "unsupported_approval"}


def select_route(summaries):
    """Pick maximum coverage with zero false approvals on development, using the fixed tie-break."""
    eligible = [m for m in METHOD_PRIORITY if summaries[m]["false_approvals"] == 0]
    best = max(eligible, key=lambda m: summaries[m]["approvals"])
    return best if summaries[best]["approvals"] else "always_review"
