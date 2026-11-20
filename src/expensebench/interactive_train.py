"""Train from interaction rewards or replay fixed greedy policies on a named split."""

import argparse
import json
from pathlib import Path
import random

from .interactive_agent import Actor, ENTROPY_WEIGHT, LEARNING_RATE, Reinforce
from .interactive_env import ExpenseEnv, load_interactive_split


SEEDS = (11, 23, 37, 53, 71)
UPDATES = 384
BATCH_SIZE = 64
DEFAULT_ROOT = Path("results/interactive")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def environments(split, data_dir):
    # The environment owns the private record store and reference annotation.
    # Neither enters the policy, optimizer or action-selection routines.
    cases, labels = load_interactive_split(split, data_dir)
    return [
        (case["case_id"], ExpenseEnv(case, labels[case["case_id"]]))
        for case in sorted(cases, key=lambda c: c["case_id"])
    ]


def sample_episode(actor, environment, rng):
    observation = environment.reset()
    episode = []
    while True:
        action, sampling = actor.choose(observation, rng)
        observation, reward, done, _ = environment.step(action)
        episode.append((sampling, reward))
        if done:
            return episode


def checkpoint_path(directory, seed, stage):
    return directory / "checkpoints" / f"seed-{seed}-{stage}.json"


def train_seed(training_envs, seed, output):
    actor = Actor.initialize(seed)
    actor.save(checkpoint_path(output, seed, "before"))
    learner = Reinforce(actor)
    rng = random.Random(100_000 + seed)
    logs = []
    for update in range(1, UPDATES + 1):
        batch = []
        for _ in range(BATCH_SIZE):
            environment = rng.choice(training_envs)[1]
            batch.append(sample_episode(actor, environment, rng))
        learner.update(batch)
        returns = [sum(r for _, r in episode) for episode in batch]
        logs.append({
            "seed": seed,
            "update": update,
            "episodes": update * BATCH_SIZE,
            "batch_mean_return": sum(returns) / BATCH_SIZE,
            "batch_mean_queries": sum(len(ep) - 1 for ep in batch) / BATCH_SIZE,
        })
    actor.save(checkpoint_path(output, seed, "after"))
    write_lines(output / "logs" / f"seed-{seed}.jsonl", logs)


def greedy_episode(actor, env, case_id, seed, stage):
    observation = env.reset()
    steps = []
    while True:
        action, _ = actor.choose(observation)
        observation, reward, done, info = env.step(action)
        step = {"action": action, "reward": reward}
        if not done:
            step["record"] = observation["records"][info["source"]]
        steps.append(step)
        if done:
            return {
                "case_id": case_id,
                "seed": seed,
                "stage": stage,
                "actions": [s["action"] for s in steps],
                "steps": steps,
                "queries": len(steps) - 1,
                "episode_return": round(sum(s["reward"] for s in steps), 10),
                "prediction": info["prediction"],
                "grade": info["grade"],
            }


def replay(split, data_dir, models, output):
    envs = environments(split, data_dir)

    def records():
        for seed in SEEDS:
            for stage in ("before", "after"):
                actor = Actor.load(checkpoint_path(models, seed, stage))
                for case_id, env in envs:
                    yield greedy_episode(actor, env, case_id, seed, stage)

    write_lines(output, records())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run the locked schedule on train only.")
    train.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "training")
    train.add_argument("--data-dir", type=Path)
    evaluate = commands.add_parser("replay", help="Evaluate stored policies without updating weights.")
    evaluate.add_argument("--models", type=Path, default=DEFAULT_ROOT / "training")
    evaluate.add_argument("--data-dir", type=Path)
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument("--output", type=Path, help="JSONL destination (default results/interactive/rollouts/SPLIT.jsonl)")
    args = parser.parse_args(argv)
    if args.command == "replay":
        destination = args.output or DEFAULT_ROOT / "rollouts" / f"{args.split}.jsonl"
        replay(args.split, args.data_dir, args.models, destination)
        print(f"Saved {args.split} greedy rollouts to {destination}.")
        return
    envs = environments("train", args.data_dir)
    schedule = {
        "seeds": list(SEEDS), "batch_size": BATCH_SIZE, "updates": UPDATES, "discount": 1,
        "learning_rate": LEARNING_RATE, "entropy_weight": ENTROPY_WEIGHT,
        "algorithm": "Monte Carlo REINFORCE with return-to-go",
        "sampling": "uniform train cases with replacement; stochastic legal actions",
        "init": "independent Gaussian logits, standard deviation 0.05",
        "optimizer": "Adam ascent, beta1=0.9, beta2=0.999, epsilon=1e-8",
        "baseline": "per-observed-state exponential average of earlier batch returns, step 0.1",
        "selection": "last completed update; no early stopping or checkpoint selection",
        "n_train_cases": len(envs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(schedule, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for seed in SEEDS:
        train_seed(envs, seed, args.output_dir)
        print(f"Finished {UPDATES * BATCH_SIZE} episodes for seed {seed}.", flush=True)


if __name__ == "__main__":
    main()
