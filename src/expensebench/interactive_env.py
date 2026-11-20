"""A finite-horizon, partially observed expense-policy environment.

One episode reviews one claim. The public clause and transaction are visible
at reset. A lookup returns one underlying record. A final decision submits
citations to the same action-and-evidence grader used by the static benchmark.
"""

import json
from copy import deepcopy
from pathlib import Path

from .data import DEFAULT_DATA_DIR
from .scoring import ACTIONS as DECISIONS
from .scoring import grade_cases

TOOLS = ("registry", "ledger", "document", "authorization")
LOOKUPS = tuple(f"lookup:{source}" for source in TOOLS)
ACTIONS = LOOKUPS + DECISIONS
MAX_QUERIES = 3
LOOKUP_COST = 0.05


def load_interactive_split(split, data_dir=None):
    """Load just this split. An explicit data_dir is the interactive dataset root."""
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR / "interactive"
    directory = root / split
    with (directory / "cases.jsonl").open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    with (directory / "gold.jsonl").open(encoding="utf-8") as stream:
        labels = [json.loads(line) for line in stream if line.strip()]
    gold = {row["case_id"]: row for row in labels}
    case_ids = {case["case_id"] for case in cases}
    if (not cases or len(case_ids) != len(cases) or len(gold) != len(labels)
            or gold.keys() != case_ids or any(case["split"] != split for case in cases)):
        raise ValueError(f"Cases and labels must match uniquely within {split}.")
    return cases, gold


class ExpenseEnv:
    """Use reset() followed by step(action, evidence_ids=None) until done.

    A valid episode has at most three lookups and a terminal decision.
    The default terminal citation list contains the policy and all retrieved
    records. Passing a list instead lets the agent select its own visible IDs.
    The grader sees all records and rejects malformed reference annotations.
    The policy sees only the current observation and receives a scalar reward.
    """

    def __init__(self, case, gold):
        case_id = case["case_id"]
        if gold["case_id"] != case_id:
            raise ValueError("Case and reference IDs differ.")
        if set(case["tools"]) != set(TOOLS):
            raise ValueError("Each episode needs one record for each evidence tool.")
        self._case = case
        self._gold = gold
        self._allowed = {
            case["policy"]["id"], case["expense"]["id"],
            *(case["tools"][source]["id"] for source in TOOLS),
        }
        if len(self._allowed) != len(TOOLS) + 2:
            raise ValueError("Policy, expense and tool records must have distinct IDs within a claim.")
        self._done = True
        self._retrieved = {}

    def _observe(self):
        return {
            "expense": deepcopy(self._case["expense"]),
            "policy": deepcopy(self._case["policy"]),
            "tools": TOOLS,
            "records": deepcopy(self._retrieved),
            "remaining_queries": MAX_QUERIES - len(self._retrieved),
        }

    def reset(self):
        """Start this claim from a clean state. No supporting records are open."""
        self._done = False
        self._retrieved = {}
        return self._observe()

    def _finish(self, action, citations, reason):
        self._done = True
        case_id = self._case["case_id"]
        prediction = {"case_id": case_id, "action": action, "evidence_ids": list(citations)}
        grade, = grade_cases(
            [self._case], {case_id: self._gold}, [prediction], {case_id: self._allowed}
        )
        if grade["action_correct"]:
            reward = 2.0 if grade["evidence_exact"] else 0.5
        elif action in ("approve", "invalid"):
            reward = -3.0
        else:
            reward = -1.0
        info = {
            "reason": reason,
            "prediction": prediction,
            "grade": {name: grade[name] for name in
                      ("action_correct", "evidence_exact", "evidence_sufficient", "joint_correct")},
        }
        return self._observe(), reward, True, info

    def step(self, action, evidence_ids=None):
        """Return (next observation, scalar reward, done, info).

        A duplicate or fourth lookup, unknown action, or inaccessible citation
        ends the episode as an invalid prediction. A third distinct lookup
        succeeds and leaves the agent one step for its final decision.
        """
        if self._done:
            raise RuntimeError("Call reset() before taking another action.")
        if action in LOOKUPS:
            source = action.removeprefix("lookup:")
            if evidence_ids is not None:
                return self._finish("invalid", [], "citations_on_lookup")
            if source in self._retrieved:
                return self._finish("invalid", [], "repeated_lookup")
            if len(self._retrieved) >= MAX_QUERIES:
                return self._finish("invalid", [], "lookup_budget_exhausted")
            self._retrieved[source] = self._case["tools"][source]
            return self._observe(), -LOOKUP_COST, False, {"reason": "retrieved", "source": source}
        if action not in DECISIONS:
            return self._finish("invalid", [], "unknown_action")

        trace_ids = [self._case["policy"]["id"], *(record["id"] for record in self._retrieved.values())]
        if evidence_ids is None:
            evidence_ids = trace_ids
        else:
            visible = {*trace_ids, self._case["expense"]["id"]}
            if (not isinstance(evidence_ids, list)
                    or any(not isinstance(item, str) or item not in visible for item in evidence_ids)):
                return self._finish("invalid", [], "invalid_citation")
        return self._finish(action, evidence_ids, "decision")
