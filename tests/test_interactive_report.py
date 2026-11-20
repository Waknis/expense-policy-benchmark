import copy
import json
from pathlib import Path

import pytest

from expensebench.interactive_env import load_interactive_split
from expensebench.interactive_report import replay, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_replay_detects_changed_evidence_or_return():
    cases, labels = load_interactive_split("dev")
    cases = {c["case_id"]: c for c in cases}
    with (ROOT / "results/interactive/rollouts/dev.jsonl").open(encoding="utf-8") as source:
        row = next(r for r in map(json.loads, source) if r["stage"] == "after" and r["queries"] > 0)
    cid = row["case_id"]
    args = (cases[cid], labels[cid], row["actions"], row["seed"])
    assert replay(*args, recorded=row)["joint"]
    edited = copy.deepcopy(row)
    edited["steps"][0]["record"]["id"] = "another-record"
    with pytest.raises(ValueError, match="Retrieved record mismatch"):
        replay(*args, recorded=edited)
    with pytest.raises(ValueError, match="Saved rollout does not reproduce"):
        replay(*args, recorded={**row, "episode_return": 999})


def test_counterfactual_pairs_must_come_from_the_same_seed():
    row = {"seed": 11, "family": "f", "pair_id": "m", "correct": True}
    with pytest.raises(ValueError, match="both members"):
        summarize([row, {**row, "seed": 23}])
