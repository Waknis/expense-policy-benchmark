"""Grade actions and evidence, and compare systems on the same policy families."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

ACTIONS = ("approve", "review", "reject")
OUTCOMES = (*ACTIONS, "invalid", "missing")

# Numerator and denominator columns of a summary. Paired inference uses pooled
# counts after resampling whole policy families, preserving both cases in a pair.
COMPARISON_METRICS = {
    "action_accuracy": ("actions_correct", "n_cases"),
    "approval_coverage": ("approvals", "n_cases"),
    "false_approval_risk": ("false_approvals", "approvals"),
    "joint_accuracy": ("joint_correct", "n_cases"),
    "pair_accuracy": ("pairs_both_correct", "n_pairs"),
}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def index_predictions(predictions):
    """Reject duplicate IDs rather than silently choosing a winning attempt."""
    if isinstance(predictions, Mapping):
        rows = list(predictions.values())
        if any(row.get("case_id") != key for key, row in predictions.items()):
            raise ValueError("Prediction map keys must match case_id.")
    else:
        rows = predictions
    index = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in index:
            raise ValueError(f"Missing or duplicate prediction case_id: {case_id!r}")
        index[case_id] = row
    return index


def grade_cases(cases, gold, predictions, allowed_evidence):
    """Return one outcome per requested case, including missing or malformed outputs."""
    index = index_predictions(predictions)
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id in evaluation cases.")
    extra = set(index).difference(case_ids)
    if extra:
        raise ValueError(f"Predictions outside selected cases: {sorted(extra)}")
    rows = []
    for case in cases:
        case_id = case["case_id"]
        label = gold[case_id]
        reference = label["action"]
        required = set(label["required_evidence"])
        allowed = set(allowed_evidence[case_id])
        if reference not in ACTIONS or not required or not required <= allowed:
            raise ValueError(f"Invalid reference action or evidence for {case_id}.")
        prediction = index.get(case_id)
        action = prediction.get("action") if prediction is not None else None
        citations = prediction.get("evidence_ids") if prediction is not None else None
        valid = (
            action in ACTIONS and isinstance(citations, list)
            and all(isinstance(item, str) and item.strip() for item in citations)
        )
        outcome = action if valid else ("missing" if prediction is None else "invalid")
        cited = set(citations) if valid else set()
        invented = cited - allowed
        exact = valid and cited == required
        rows.append({
            "case_id": case_id, "family": case["family"], "pair_id": case["pair_id"],
            "expected_action": reference, "outcome": outcome,
            "action_correct": outcome == reference,
            "required_evidence": sorted(required), "cited_evidence": sorted(cited),
            "unknown_evidence": sorted(invented),
            "evidence_exact": exact,
            "evidence_sufficient": valid and not invented and required <= cited,
            "joint_correct": outcome == reference and exact,
        })
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Cannot score an empty case set.")
    size = len(rows)
    counts = Counter(row["outcome"] for row in rows)
    confusion = {action: dict.fromkeys(OUTCOMES, 0) for action in ACTIONS}
    pairs = defaultdict(list)
    for row in rows:
        confusion[row["expected_action"]][row["outcome"]] += 1
        pairs[row["family"], row["pair_id"]].append(row)
    if any(len(group) > 2 for group in pairs.values()):
        raise ValueError("A matched pair must contain at most two cases.")
    complete = [group for group in pairs.values() if len(group) == 2]
    switches = [group for group in complete if group[0]["expected_action"] != group[1]["expected_action"]]
    # An invalid result opposed to an action is never counted as responding to a changed fact.
    predicted_switches = sum(
        a["outcome"] in ACTIONS and b["outcome"] in ACTIONS and a["outcome"] != b["outcome"]
        for a, b in switches
    )
    approvals = counts["approve"]
    mistakes = approvals - confusion["approve"]["approve"]
    unsafe_cases = sum(sum(confusion[action].values()) for action in ("review", "reject"))
    exact = sum(row["evidence_exact"] for row in rows)
    sufficient = sum(row["evidence_sufficient"] for row in rows)
    result = {
        "n_cases": size, "actions_correct": sum(row["action_correct"] for row in rows),
        "action_counts": {key: counts[key] for key in OUTCOMES}, "confusion": confusion,
        "approvals": approvals, "false_approvals": mistakes,
        "nonapprovable_cases": unsafe_cases,
        "false_approval_rate_among_nonapprovable": _ratio(mistakes, unsafe_cases),
        "review_rate": counts["review"] / size,
        "fallback_review_rate": (counts["review"] + counts["invalid"] + counts["missing"]) / size,
        "evidence_exact_count": exact, "evidence_exact_accuracy": exact / size,
        "evidence_sufficient_count": sufficient, "evidence_sufficient_accuracy": sufficient / size,
        "unknown_evidence_cases": sum(bool(row["unknown_evidence"]) for row in rows),
        "joint_correct": sum(row["joint_correct"] for row in rows),
        "n_pairs": len(complete),
        "pairs_both_correct": sum(a["action_correct"] and b["action_correct"] for a, b in complete),
        "n_action_change_pairs": len(switches), "predicted_changes": predicted_switches,
        "pair_change_rate": _ratio(predicted_switches, len(switches)),
    }
    for metric, (numerator, denominator) in COMPARISON_METRICS.items():
        result[metric] = _ratio(result[numerator], result[denominator])
    return result


def score_system(cases, gold, predictions, allowed_evidence):
    return summarize(grade_cases(cases, gold, predictions, allowed_evidence))


def _quantile(sorted_values, p):
    at = (len(sorted_values) - 1) * p
    low = math.floor(at)
    return sorted_values[low] + (at - low) * (sorted_values[math.ceil(at)] - sorted_values[low])


def compare_systems(cases, gold, predictions_by_system, allowed_evidence, *, seed=1729, replicates=10000):
    """Paired percentile intervals resample the same policy families for every system."""
    if replicates < 1 or not predictions_by_system:
        raise ValueError("At least one system and one bootstrap replicate are required.")
    families = sorted({case["family"] for case in cases})
    by_family = {family: {} for family in families}
    overall = {}
    for name, predictions in predictions_by_system.items():
        rows = grade_cases(cases, gold, predictions, allowed_evidence)
        overall[name] = summarize(rows)
        for family in families:
            by_family[family][name] = summarize([row for row in rows if row["family"] == family])

    rng = random.Random(seed)
    draws = [
        rng.choices(range(len(families)), k=len(families)) for _ in range(replicates)
    ]
    comparisons = []
    for left, right in itertools.combinations(sorted(overall), 2):
        metrics = {}
        for metric, (num, den) in COMPARISON_METRICS.items():
            family_stats = [
                (by_family[f][left][num], by_family[f][left][den],
                 by_family[f][right][num], by_family[f][right][den])
                for f in families
            ]
            deltas = []
            for selection in draws:
                ln, ld, rn, rd = map(sum, zip(*(family_stats[i] for i in selection)))
                if ld and rd:
                    deltas.append(ln / ld - rn / rd)
            deltas.sort()
            lhs, rhs = overall[left][metric], overall[right][metric]
            metrics[metric] = {
                "delta": lhs - rhs if lhs is not None and rhs is not None else None,
                "ci95": ([_quantile(deltas, .025), _quantile(deltas, .975)]
                         if deltas and len(families) >= 2 else None),
                "defined_replicates": len(deltas),
            }
        comparisons.append({"left": left, "right": right, "metrics": metrics})
    return {
        "systems": overall, "families": by_family, "paired_comparisons": comparisons,
        "bootstrap": {
            "unit": "policy family", "family_count": len(families),
            "replicates": replicates, "seed": seed, "interval": "paired percentile 95%",
            "delta": "left minus right", "aggregation": "pooled case or pair counts",
            "interpretation": "Descriptive sensitivity to the sampled synthetic families. "
                              "Undefined ratio draws are excluded and counted. "
                              "Intervals do not establish a production risk bound.",
        },
    }


def main(argv=None):
    from expensebench.data import build_database, case_packet, load_cases, load_gold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    all_cases = load_cases(data_dir=args.data_dir)
    all_ids = {case["case_id"] for case in all_cases}
    cases = [case for case in all_cases if case["split"] == args.split]
    selected = {case["case_id"] for case in cases}
    methods = {}
    for spec in args.pred:
        name, sep, filename = spec.partition("=")
        if not sep or not name or not filename or name in methods:
            parser.error("Each --pred must give a unique NAME=FILE.")
        with Path(filename).open(encoding="utf-8") as stream:
            indexed = index_predictions(json.loads(line) for line in stream if line.strip())
        if set(indexed) - all_ids:
            parser.error(f"{name} contains unknown case IDs.")
        methods[name] = [row for key, row in indexed.items() if key in selected]
    conn = build_database(data_dir=args.data_dir)
    try:
        allowed = {}
        for case_id in selected:
            packet = case_packet(conn, case_id)
            allowed[case_id] = {packet["expense"]["id"], packet["policy"]["id"]}
            allowed[case_id].update(record["id"] for record in packet["evidence"])
    finally:
        conn.close()
    report = compare_systems(
        cases, load_gold(data_dir=args.data_dir), methods, allowed,
        seed=args.seed, replicates=args.replicates,
    )
    report["split"] = args.split
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
