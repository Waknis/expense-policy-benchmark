"""Run both frozen policies on one claim and show the records they chose to inspect."""

import argparse

from expensebench.interactive_agent import Actor
from expensebench.interactive_env import ExpenseEnv, load_interactive_split
from expensebench.interactive_train import DEFAULT_ROOT, SEEDS, checkpoint_path, greedy_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", help="A case ID from the requested split.")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=11)
    args = parser.parse_args()
    cases, labels = load_interactive_split(args.split)
    case = next((c for c in cases if c["case_id"] == args.case_id), None)
    if case is None:
        parser.error(f"{args.case_id} is absent from {args.split}.")
    env = ExpenseEnv(case, labels[args.case_id])
    print(case["case_id"], case["family"], "\n" + env.reset()["policy"]["text"])
    for stage in ("before", "after"):
        actor = Actor.load(checkpoint_path(DEFAULT_ROOT / "training", args.seed, stage))
        trace = greedy_episode(actor, env, args.case_id, args.seed, stage)
        print(f"\n{stage} training")
        for step in trace["steps"]:
            print(f"  {step['action']}  (reward {step['reward']:+g})")
            if "record" in step:
                record = step["record"]
                print(f"    {record['id']}: {record['field']} = {record['value']!r}")
        print("  Cited:", ", ".join(trace["prediction"]["evidence_ids"]))
        print(f"  Episode return {trace['episode_return']:+g}, exact decision and evidence: "
              f"{trace['grade']['joint_correct']}")


if __name__ == "__main__":
    main()
