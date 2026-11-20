"""Replay interactive decisions, run reference policies and summarize a locked comparison."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from .interactive_env import ExpenseEnv, load_interactive_split
from .interactive_train import SEEDS

STAGES = ("before", "after")
RATIOS = {
    "mean_return": ("return_sum", "episodes"),
    "joint_accuracy": ("joint_correct", "episodes"),
    "action_accuracy": ("actions_correct", "episodes"),
    "mean_lookups": ("lookups", "episodes"),
    "approval_coverage": ("approvals", "episodes"),
    "false_approval_risk": ("false_approvals", "approvals"),
    "pair_accuracy": ("pairs_both_correct", "pairs"),
}


def public_rule(obs):
    """Execute the written, ordered checks; look up a record before using its value."""
    for check in obs["policy"]["checks"]:
        tool = check["tool"]
        if tool not in obs["records"]:
            return f"lookup:{tool}"
        record = obs["records"][tool]
        if record["field"] != check["field"]:
            raise ValueError("Policy check and returned record disagree on field.")
        value, target = record["value"], check["target"]
        if value is None:
            return "review"
        op = check["operator"]
        if op == "equals":
            passed = value == target
        elif op == "at_most":
            passed = value <= target
        elif op == "at_least":
            passed = value >= target
        else:
            raise ValueError(f"Unknown policy operator {op}.")
        if not passed:
            return "reject"
    return "approve"


def run_reference(case, gold, policy):
    env = ExpenseEnv(case, gold)
    obs = env.reset()
    actions = []
    while True:
        action = policy(obs)
        actions.append(action)
        obs, _, done, _ = env.step(action)
        if done:
            return actions


def replay(case, gold, actions, seed, *, recorded=None):
    """Execute the recorded actions from a fresh reset and independently recover the score."""
    env = ExpenseEnv(case, gold)
    env.reset()
    rewards = []
    done = False
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if recorded is not None:
            step = recorded["steps"][i]
            if (step["action"], step["reward"]) != (action, reward):
                raise ValueError(f"Step mismatch on {case['case_id']} at {i}")
            if not done and step.get("record") != obs["records"][action.removeprefix("lookup:")]:
                raise ValueError(f"Retrieved record mismatch on {case['case_id']} at {i}")
        if done and i != len(actions) - 1:
            raise ValueError(f"Actions after termination on {case['case_id']}")
    if not done:
        raise ValueError(f"Unfinished episode on {case['case_id']}")
    total = math.fsum(rewards)
    queries = len(obs["records"])
    prediction = info["prediction"]
    grade = info["grade"]
    if recorded is not None and not (
        math.isclose(total, recorded["episode_return"], abs_tol=1e-9)
        and len(recorded["steps"]) == len(actions) and queries == recorded["queries"]
        and prediction == recorded["prediction"] and grade == recorded["grade"]
    ):
        raise ValueError(f"Saved rollout does not reproduce on {case['case_id']}")
    return {
        "case_id": case["case_id"], "family": case["family"], "pair_id": case["pair_id"],
        "seed": seed, "action": prediction["action"], "reference": gold["action"],
        "joint": grade["joint_correct"], "correct": grade["action_correct"],
        "exact": grade["evidence_exact"], "lookups": queries, "return": total,
    }


def summarize(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["seed"], row["family"], row["pair_id"]].append(row["correct"])
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Expected both members of each matched pair for every policy.")
    counts = {
        "episodes": len(rows), "return_sum": round(math.fsum(row["return"] for row in rows), 8),
        "joint_correct": sum(row["joint"] for row in rows), "actions_correct": sum(row["correct"] for row in rows),
        "evidence_exact": sum(row["exact"] for row in rows), "lookups": sum(row["lookups"] for row in rows),
        "approvals": sum(row["action"] == "approve" for row in rows),
        "false_approvals": sum(row["action"] == "approve" and row["reference"] != "approve" for row in rows),
        "pairs": len(pairs), "pairs_both_correct": sum(all(pair) for pair in pairs.values()),
        "dispositions": dict(sorted(Counter(row["action"] for row in rows).items())),
    }
    for metric, (num, den) in RATIOS.items():
        counts[metric] = counts[num] / counts[den] if counts[den] else None
    return counts


def family_intervals(by_family, overall, draws=10000, seed=1729):
    """Resample paired families with all their cases and training seeds kept together."""
    rng = random.Random(seed)
    units = [by_family[k] for k in sorted(by_family)]
    delta_samples = {k: [] for k in RATIOS}
    for _ in range(draws):
        sample = rng.choices(units, k=len(units))
        for metric, (num, den) in RATIOS.items():
            before_den = sum(g["before"][den] for g in sample)
            after_den = sum(g["after"][den] for g in sample)
            if before_den and after_den:
                delta_samples[metric].append(
                    sum(g["after"][num] for g in sample) / after_den
                    - sum(g["before"][num] for g in sample) / before_den
                )
    intervals = {}
    for key, values in delta_samples.items():
        values.sort()
        ci = []
        for q in (.025, .975):
            if not values:
                break
            at = (len(values) - 1) * q
            low = math.floor(at)
            ci.append(values[low] + (at - low) * (values[math.ceil(at)] - values[low]))
        before, after = overall["before"][key], overall["after"][key]
        intervals[key] = {"delta": after - before if after is not None and before is not None else None,
                          "ci95": ci or None, "defined_draws": len(values)}
    return {"unit": "policy family", "draws": draws, "seed": seed, "after_minus_before": intervals}


def build_report(split, rollout_path):
    cases, labels = load_interactive_split(split)
    indexed = {c["case_id"]: c for c in cases}
    observed = set()
    methods = defaultdict(list)
    with rollout_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = row["case_id"], row["seed"], row["stage"]
            cid, seed, stage = key
            if key in observed or cid not in indexed or seed not in SEEDS or stage not in STAGES:
                raise ValueError(f"Unexpected rollout key {key}")
            observed.add(key)
            methods[stage].append(replay(indexed[cid], labels[cid], row["actions"], seed, recorded=row))
    if len(observed) != len(cases) * len(SEEDS) * len(STAGES):
        raise ValueError("Initial/final coverage incomplete; every seed must see every case.")
    for case in cases:
        label = labels[case["case_id"]]
        for name, reference in (("always_review", lambda obs: "review"), ("public_rule", public_rule)):
            actions = run_reference(case, label, reference)
            methods[name].append(replay(case, label, actions, None))
    overall = {name: summarize(rows) for name, rows in sorted(methods.items())}
    families = sorted({c["family"] for c in cases})
    by_family = {
        f: {name: summarize([r for r in rows if r["family"] == f]) for name, rows in sorted(methods.items())}
        for f in families
    }
    by_seed = {
        str(seed): {stage: summarize([r for r in methods[stage] if r["seed"] == seed]) for stage in STAGES}
        for seed in SEEDS
    }
    return {"split": split, "cases": len(cases), "families": families, "training_seeds": list(SEEDS),
            "overall": overall, "by_seed": by_seed, "by_family": by_family,
            "paired_uncertainty": family_intervals(by_family, overall)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.split, args.rollouts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified and summarized {report['cases']} cases across {len(report['training_seeds'])} seeds.")


if __name__ == "__main__":
    main()
