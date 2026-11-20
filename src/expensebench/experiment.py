"""Freeze a development routing decision, then reproduce the held-out comparison."""

import argparse
import hashlib
import json
from pathlib import Path

from .data import build_database, case_packet, load_cases, load_gold
from .routing import METHOD_PRIORITY, always_review, citation_gate, packet_ids, receipt_rule, select_route
from .scoring import compare_systems, score_system


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(data):
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_predictions(path, cases, split):
    expected = {c["case_id"] for c in cases}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] != split:
            continue
        case_id = row["case_id"]
        if case_id in rows:
            raise ValueError(f"Duplicate {case_id} in {path}")
        rows[case_id] = row
    if set(rows) != expected:
        raise ValueError(f"Incomplete or unexpected {split} predictions in {path}: {set(rows) ^ expected}")
    return [rows[c["case_id"]] for c in cases]


def assemble(split, source):
    cases = sorted((c for c in load_cases() if c["split"] == split), key=lambda c: c["case_id"])
    gold = load_gold()
    labels = {c["case_id"]: gold[c["case_id"]] for c in cases}
    db = build_database()
    try:
        packets = [case_packet(db, c["case_id"]) for c in cases]
    finally:
        db.close()
    evidence = {p["case_id"]: packet_ids(p) for p in packets}
    models = {m: read_predictions(source / f"{m}.jsonl", cases, split) for m in ("llm_policy", "llm_evidence")}
    predictions = {
        "always_review": [always_review(p) for p in packets],
        "receipt_rule": [receipt_rule(p) for p in packets],
        **models,
        "citation_gate": [citation_gate(p, row) for p, row in zip(packets, models["llm_evidence"], strict=True)],
    }
    return cases, labels, evidence, predictions, packets


def development_selection(source):
    cases, labels, evidence, predictions, packets = assemble("dev", source)
    summaries = {m: score_system(cases, labels, rows, evidence) for m, rows in predictions.items()}
    return {
        "split": "dev",
        "case_count": len(cases),
        "families": sorted({c["family"] for c in cases}),
        "winner": select_route(summaries),
        "priority": METHOD_PRIORITY,
        "input_sha256": digest({"cases": cases, "packets": packets, "labels": labels}),
        "prediction_sha256": {m: digest(rows) for m, rows in predictions.items()},
        "systems": summaries,
    }


def heldout_comparison(source, lock, out, replicates):
    with lock.open(encoding="utf-8") as stream:
        selected = json.load(stream)
    if development_selection(source) != selected:
        raise ValueError("Development inputs, predictions or selection differ from the frozen decision.")
    cases, labels, evidence, predictions, packets = assemble("test", source)
    metrics = compare_systems(cases, labels, predictions, evidence, seed=1729, replicates=replicates)
    metrics["selected_on_development"] = selected["winner"]
    metrics["split"] = "test"
    metrics["input_sha256"] = digest({"cases": cases, "packets": packets, "labels": labels})
    write_json(out / "heldout.json", metrics)
    # A row for every decision, including the receipt heuristic and gate, for easy independent audit.
    timeline = []
    for idx, case in enumerate(cases):
        observations = {}
        for name, rows in predictions.items():
            row = rows[idx]
            observations[name] = {k: row[k] for k in ("action", "evidence_ids", "gate_reason") if k in row}
        timeline.append({**case, "gold": labels[case["case_id"]], "decisions": observations})
    (out / "heldout_decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in timeline),
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["select", "report"])
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    parser.add_argument("--lock", type=Path, default=Path("results/development_selection.json"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.phase == "select":
        decision = development_selection(args.predictions)
        if args.lock.exists() and json.loads(args.lock.read_text(encoding="utf-8")) != decision:
            parser.error("Existing selection differs. Use a separate --lock path for a new experiment.")
        write_json(args.lock, decision)
        print(f"Selected {decision['winner']} using {decision['case_count']} development cases.")
    else:
        report = heldout_comparison(args.predictions, args.lock, args.out, args.replicates)
        print(f"Held-out results written to {args.out}. Frozen route: {report['selected_on_development']}")


if __name__ == "__main__":
    main()
