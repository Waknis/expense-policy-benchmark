"""Checks on the frozen fixture, its measurement units, and its inference boundary."""

import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import closing
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from expensebench.data import build_database, case_packet, load_cases, load_gold

ROOT = Path(__file__).resolve().parents[1]


def test_generator_recreates_frozen_files(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "data/generate.py"), "--output-dir", str(tmp_path)], check=True
    )
    for name in ("cases", "gold", "observations", "policies"):
        name = name + ".jsonl"
        assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes()


def test_case_and_pair_integrity():
    cases = load_cases()
    gold = load_gold()
    assert len(cases) == len(gold) == len({row["case_id"] for row in cases}) == 120
    assert Counter(row["split"] for row in cases) == {"dev": 48, "test": 72}
    families = {
        split: {row["family"] for row in cases if row["split"] == split} for split in ("dev", "test")
    }
    assert len(families["dev"]) == 4 and len(families["test"]) == 6
    assert families["dev"].isdisjoint(families["test"])
    pairs = defaultdict(list)
    labels = defaultdict(Counter)
    with closing(build_database()) as db:
        for row in cases:
            cid = row["case_id"]
            packet = case_packet(db, cid)
            label = gold[cid]
            assert set(packet) == {"case_id", "expense", "policy", "evidence"}
            assert packet["policy"]["id"] == row["policy_id"]
            assert packet["expense"]["id"] == row["expense_id"]
            assert packet["policy"]["effective_from"] <= packet["expense"]["spent_at"][:10]
            references = [row["policy_id"], row["expense_id"], *(r["id"] for r in packet["evidence"])]
            assert len(references) == len(set(references))
            assert row["policy_id"] in label["required_evidence"]
            assert set(label["required_evidence"]) <= set(references)
            assert label["rationale"].strip()
            labels[row["family"]][label["action"]] += 1
            pairs[row["pair_id"]].append((row, packet, label["action"]))
    assert all(count == {"approve": 4, "review": 4, "reject": 4} for count in labels.values())
    assert len(pairs) == 60
    for members in pairs.values():
        assert len(members) == 2
        (left, first, action_a), (right, second, action_b) = members
        assert left["family"] == right["family"] and left["split"] == right["split"]
        assert action_a != action_b
        assert first["expense"] == second["expense"] and first["policy"] == second["policy"]
        changes = []
        for a, b in zip(first["evidence"], second["evidence"], strict=True):
            assert all(a[k] == b[k] for k in ("id", "kind", "description"))
            if a["value"] != b["value"]:
                changes.append(a["kind"])
        assert len(changes) == 1


def test_database_contains_only_observables_and_is_query_only(tmp_path):
    database = tmp_path / "records.sqlite"
    with closing(build_database(database)) as db:
        assert {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {
            "expenses",
            "evidence",
            "policies",
        }
        with pytest.raises(KeyError):
            case_packet(db, "' OR '1'='1")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM evidence")
    with pytest.raises(FileExistsError):
        build_database(database)


def test_reference_thresholds_and_unavailable_records():
    """Independently recalculate held-out currency conversions and effective-date limits."""
    gold = load_gold()
    with closing(build_database()) as db:
        for row in load_cases():
            if row["family"] not in {"equipment_allowance_revision", "foreign_meal_rate"}:
                continue
            packet = case_packet(db, row["case_id"])
            expense = packet["expense"]
            evidence = {entry["kind"]: entry for entry in packet["evidence"]}
            fx = row["family"] == "foreign_meal_rate"
            decisive = evidence["purchase_date_fx" if fx else "half_year_paid_cents"]
            refs = {packet["policy"]["id"], decisive["id"]}
            if decisive["value"] is None:
                action = "review"
            else:
                refs.add(expense["id"])
                if fx:
                    cents = Decimal(expense["amount_cents"]) * Decimal(decisive["value"])
                    allowed = cents.quantize(Decimal(1), rounding=ROUND_HALF_UP) <= 6000
                else:
                    cap = 30000 if expense["spent_at"][:10] < "2025-07-01" else 40000
                    allowed = expense["amount_cents"] + decisive["value"] <= cap
                action = "approve" if allowed else "reject"
            label = gold[row["case_id"]]
            assert label["action"] == action
            assert set(label["required_evidence"]) == refs
