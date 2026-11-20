"""Print only model-visible packets for the predeclared audit sample."""

from collections import defaultdict
from hashlib import sha256
import json

from expensebench.data import build_database, case_packet, load_cases


def digest(*parts):
    return sha256("\0".join(("expensebench-blind-audit-v1", *parts)).encode("utf-8")).hexdigest()


def sample_cases(cases):
    families = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if case["split"] == "test":
            families[case["family"]][case["pair_id"]].append(case["case_id"])
    picked = []
    for family, pairs in families.items():
        pair = min(pairs, key=lambda key: digest(family, key))
        if len(pairs[pair]) != 2:
            raise ValueError("Expected a matched pair.")
        picked.extend(pairs[pair])
    return sorted(picked, key=digest)


if __name__ == "__main__":
    connection = build_database()
    try:
        for case_id in sample_cases(load_cases()):
            print(json.dumps(case_packet(connection, case_id), sort_keys=True))
    finally:
        connection.close()
