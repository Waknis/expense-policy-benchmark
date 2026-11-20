"""Check split isolation, observable counterfactuals and the authored verifier labels."""

from collections import Counter, defaultdict
from itertools import combinations
import json
import operator
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1] / "data" / "interactive"
SPLITS = {"train": (8, 1536), "dev": (3, 144), "test": (6, 288)}
COMPARATORS = {"equals": operator.eq, "at_most": operator.le, "at_least": operator.ge}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_interactive_fixture_regeneration_and_split_locality(tmp_path):
    for split in SPLITS:
        out = tmp_path / split
        subprocess.run([sys.executable, str(ROOT / "generate.py"), "--split", split,
                        "--output-dir", str(out)], check=True)
        assert {p.name for p in out.iterdir()} == {split}
        for name in ("cases.jsonl", "gold.jsonl"):
            assert (out / split / name).read_bytes() == (ROOT / split / name).read_bytes()


def test_interactive_policy_disjointness_and_counterfactuals():
    split_identifiers = {}
    v1_families = {r["family"] for r in read_rows(ROOT.parent / "cases.jsonl")}
    for split, (n_families, n_cases) in SPLITS.items():
        cases = read_rows(ROOT / split / "cases.jsonl")
        gold = {row["case_id"]: row for row in read_rows(ROOT / split / "gold.jsonl")}
        families = {row["family"] for row in cases}
        policy_ids = {row["policy"]["id"] for row in cases}
        assert len(families) == n_families
        assert families.isdisjoint(v1_families)
        assert len(cases) == len(gold) == n_cases
        assert {row["case_id"] for row in cases} == set(gold)
        assert Counter(row["action"] for row in gold.values()) == dict.fromkeys(
            ["approve", "review", "reject"], n_cases // 3)
        pairs = defaultdict(list)
        for row in cases:
            assert row["split"] == split
            pairs[row["pair_id"]].append(row)
        split_identifiers[split] = (set(gold), set(pairs), families, policy_ids,
                                   {r["policy"]["text"] for r in cases})
        for arms in pairs.values():
            assert len(arms) == 2
            a, b = arms
            assert (a["policy"], a["family"], a["expense"]) == (b["policy"], b["family"], b["expense"])
            assert gold[a["case_id"]]["action"] != gold[b["case_id"]]["action"]
            assert a["tools"].keys() == b["tools"].keys()
            changed = [
                (tool, key) for tool in a["tools"] for key in a["tools"][tool]
                if a["tools"][tool][key] != b["tools"][tool][key]
            ]
            assert len(changed) == 1
            assert changed[0][1] == "value"
            assert changed[0][0] in {c["tool"] for c in a["policy"]["checks"]}
    for first, second in combinations(split_identifiers.values(), 2):
        assert all(a.isdisjoint(b) for a, b in zip(first, second, strict=True))


def test_all_interactive_actions_and_minimal_evidence_follow_visible_policy():
    # This verifier reads the frozen packets. The generator labels by the sampled
    # latent case states. A mismatch in the construction or thresholds fails here.
    for split in SPLITS:
        cases = read_rows(ROOT / split / "cases.jsonl")
        gold = {r["case_id"]: r for r in read_rows(ROOT / split / "gold.jsonl")}
        for row in cases:
            policy, records = row["policy"], row["tools"]
            assert set(records) == {"registry", "ledger", "document", "authorization"}
            assert len({rec["id"] for rec in records.values()}) == 4
            assert len({c["tool"] for c in policy["checks"]}) == 2
            used = [policy["id"]]
            action = "approve"
            for check in policy["checks"]:
                source = records[check["tool"]]
                assert check["field"] == source["field"]
                used.append(source["id"])
                if source["value"] is None:
                    action = "review"
                    break
                if not COMPARATORS[check["operator"]](source["value"], check["target"]):
                    action = "reject"
                    break
            expected = gold[row["case_id"]]
            assert (action, used) == (expected["action"], expected["required_evidence"])
