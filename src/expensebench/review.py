"""Export a blinded review packet or measure agreement between two completed reviews."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .data import build_database, case_packet, load_cases

ACTIONS = ("approve", "review", "reject")


def export(path, seed):
    if path.exists():
        raise ValueError(f"Choose a new review file; {path} already exists.")
    groups = defaultdict(lambda: defaultdict(list))
    for case in load_cases():
        if case["split"] == "test":
            groups[case["family"]][case["pair_id"]].append(case["case_id"])
    rng = random.Random(seed)
    ids = []
    for family in sorted(groups):
        pairs = groups[family]
        ids.extend(sorted(pairs[rng.choice(sorted(pairs))]))
    rng.shuffle(ids)
    db = build_database()
    try:
        records = [
            {**case_packet(db, case_id), "action": "", "evidence_ids": [], "reason": ""}
            for case_id in ids
        ]
    finally:
        db.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ratings(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key, action = record["case_id"], record["action"]
        if key in rows or action not in ACTIONS:
            raise ValueError(f"Duplicate case ID or missing/invalid action in {path}: {key}")
        rows[key] = action
    return rows


def agreement(left, right):
    if not left or left.keys() != right.keys():
        raise ValueError("Both reviewers must label the same nonempty case set.")
    n = len(left)
    a, b = Counter(left.values()), Counter(right.values())
    differences = [key for key in sorted(left) if left[key] != right[key]]
    observed = (n - len(differences)) / n
    chance = sum(a[label] * b[label] for label in ACTIONS) / (n * n)
    return {"n": n, "agreements": n - len(differences), "agreement": observed,
            "cohens_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
            "disagreed_cases": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packet = commands.add_parser("export")
    packet.add_argument("path", type=Path)
    packet.add_argument("--seed", type=int, default=745)
    agree = commands.add_parser("agree")
    agree.add_argument("left", type=Path)
    agree.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.path, args.seed)
    else:
        print(json.dumps(agreement(read_ratings(args.left), read_ratings(args.right)), indent=2))


if __name__ == "__main__":
    main()
