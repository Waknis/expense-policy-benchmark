"""Verify the inputs and checkpoints fixed before the held-out rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "results/interactive/freeze.json"


def manifest():
    paths = [
        "protocol/interactive-rl.md", "data/interactive/generate.py",
        "src/expensebench/scoring.py", "src/expensebench/interactive_env.py",
        "src/expensebench/interactive_agent.py", "src/expensebench/interactive_train.py",
        "src/expensebench/interactive_report.py",
        "results/interactive/training/config.json", "results/interactive/rollouts/dev.jsonl",
    ]
    for split in ("train", "dev", "test"):
        for filename in ("cases", "gold"):
            paths.append(f"data/interactive/{split}/{filename}.jsonl")
    config = json.loads((ROOT / "results/interactive/training/config.json").read_text(encoding="utf-8"))
    for seed in config["seeds"]:
        for stage in ("before", "after"):
            paths.append(f"results/interactive/training/checkpoints/seed-{seed}-{stage}.json")
        paths.append(f"results/interactive/training/logs/seed-{seed}.jsonl")
    return {
        "hash": "SHA-256, LF-normalized UTF-8 source and JSON",
        "checkpoint_selection": "initial and final checkpoint for every predeclared seed",
        "files": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for p in sorted(paths)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create a new lock file once.")
    args = parser.parse_args()
    current = manifest()
    if args.create:
        with LOCK.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(current, indent=2) + "\n")
    elif current != json.loads(LOCK.read_text(encoding="utf-8")):
        raise ValueError("Interactive experiment differs from its frozen inputs.")
    print(f"Verified {len(current['files'])} frozen inputs and training artifacts.")


if __name__ == "__main__":
    main()
