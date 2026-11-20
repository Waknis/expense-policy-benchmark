"""Replay the frozen annotations against the initial and released references."""

import hashlib
import json
from pathlib import Path

from expensebench.data import build_database, case_packet, load_cases, load_gold
from expensebench.scoring import grade_cases, summarize
from sample import sample_cases

HERE = Path(__file__).resolve().parent
REVIEWER_PACKET_SHA = "ff933df4834b215b28dbc9a4eaeeea966597d5d5b72dea132c5e1c9b01162ff3"
ANNOTATION_SHA = "1197fbf1e6691dbe8329c6a932f3f760aced82a686dd3a3683b6683c91cde5fc"


def agreement(cases, labels, predictions, allowed):
    grades = grade_cases(cases, labels, predictions, allowed)
    totals = summarize(grades)
    return {
        "cases": totals["n_cases"],
        "actions_agree": totals["actions_correct"],
        "exact_evidence_sets": totals["evidence_exact_count"],
        "sufficient_evidence_sets": totals["evidence_sufficient_count"],
        "confusion": totals["confusion"],
        "differences": [
            {
                "case_id": g["case_id"],
                "reference_action": g["expected_action"], "reviewer_action": g["outcome"],
                "extra_citations": sorted(set(g["cited_evidence"]) - set(g["required_evidence"])),
                "missing_citations": sorted(set(g["required_evidence"]) - set(g["cited_evidence"])),
            }
            for g in grades if not g["joint_correct"]
        ],
    }


def main():
    cases = load_cases()
    ids = sample_cases(cases)
    connection = build_database()
    try:
        packets = [case_packet(connection, case_id) for case_id in ids]
    finally:
        connection.close()
    release_text = "".join(json.dumps(packet, sort_keys=True) + "\n" for packet in packets)
    review_text = (HERE / "reviewer_inputs.jsonl").read_text(encoding="utf-8")
    reviewed_packets = [json.loads(line) for line in review_text.splitlines()]
    packet_sha = hashlib.sha256(review_text.encode("utf-8")).hexdigest()
    annotation_text = (HERE / "predictions.jsonl").read_text(encoding="utf-8")
    annotation_sha = hashlib.sha256(annotation_text.encode("utf-8")).hexdigest()
    if (packet_sha, annotation_sha) != (REVIEWER_PACKET_SHA, ANNOTATION_SHA):
        raise ValueError("The packets shown to the reviewer or the frozen annotations changed.")
    predictions = [json.loads(line) for line in annotation_text.splitlines()]
    if any([row["case_id"] for row in source] != ids for source in (predictions, reviewed_packets)):
        raise ValueError("Audit IDs must match the declared sample and order.")
    excluded = [p["case_id"] for p, old in zip(packets, reviewed_packets, strict=True) if p != old]
    predictions = [row for row in predictions if row["case_id"] not in excluded]
    allowed = {
        p["case_id"]: {p["expense"]["id"], p["policy"]["id"], *(d["id"] for d in p["evidence"])}
        for p in packets
    }
    by_id = {case["case_id"]: case for case in cases}
    picked = [by_id[i] for i in ids if i not in excluded]
    initial = [json.loads(line) for line in (HERE / "original_reference.jsonl").read_text(encoding="utf-8").splitlines()]
    initial_gold = {row["case_id"]: row for row in initial}
    output = {
        "reviewer_packet_sha256": packet_sha, "annotation_sha256": annotation_sha,
        "release_packet_sha256": hashlib.sha256(release_text.encode("utf-8")).hexdigest(),
        "excluded_changed_packets": excluded, "eligible_families": sorted({c["family"] for c in picked}),
        "original_full_label_sha256": "584c429efc3b610fc40c09de8f853cec20c094cae39bcb4dff326f1805e3ff1a",
        "original": agreement(picked, initial_gold, predictions, allowed),
        "release": agreement(picked, load_gold(), predictions, allowed),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
