# Interactive expense review: frozen evaluation plan

## Question

Can a policy trained only on feedback from the expense grader learn which evidence sources to open, when to stop, and which disposition to submit under new written policies? The primary comparison is the same randomly initialized agent before and after reinforcement learning. The target is a structured decision policy. The original Qwen experiment is a separate fixed-model benchmark.

## Task and reward

A case starts with a public expense, a written policy, its two ordered machine-readable checks, and four named evidence tools. Each lookup reveals one case-local record and costs 0.05 reward. The first unresolved or violated check is decisive. When both pass, approval requires evidence for both. The environment allows at most three distinct lookups before a terminal decision is required. An invalid action, repeated lookup, or attempt to cite hidden evidence ends the episode at a penalty.

The terminal grader checks the action and the set of cited records. A correct action with exactly the required evidence earns +2. A correct action with imperfect citations earns +0.5. A false approval earns −3 and another incorrect decision earns −1. An invalid action earns −3. Lookup costs apply in addition. Every non-approval would still need human handling in an operational approval workflow.

## Training

Use on-policy REINFORCE with return-to-go, a running state baseline, discount 1, entropy coefficient 0.025, and Adam learning rate 0.025. Each of five seeds (11, 23, 37, 53, 71) gets 384 updates of 64 episodes, sampling only training cases. Preserve both the untrained parameters and the final parameters. Evaluate greedy actions with a fixed tie break. Use the final checkpoint for every seed; development results do not select a checkpoint or seed.

The policy receives derived comparison states for already opened records and the source names specified in the clause. Policy, family, pair and case identifiers, reference outcomes, hidden records and expected citations are excluded from its decision features. The agent shares weights across six action roles (first source, second source, other source, and three dispositions). Weights and biases start from independent normal draws with standard deviation 0.05. It cites the policy and all inspected records. Excess lookups can therefore reduce both the episode return and exact-evidence credit. A general environment caller may submit its own visible citation subset.

## Held-out measurement

Use fresh policy families disjoint from the original benchmark and from interactive train/development data. Keep the same case, environment, reward, and action budget for before and after. Record every action, citation and reward, including failures. Freeze fixture hashes, configuration, implementation and all initial/final checkpoint hashes in version control before authorizing held-out evaluation.

Primary outcomes are mean episode return and joint correctness (correct disposition with the exact required evidence). Also report action accuracy, incorrect approvals with their denominators, lookups, complete matched-pair success, per-seed results, and differences by held-out family. Compare against always requesting human review and an executable reference strategy that reads the public checks and incurs the same lookup costs. The reference strategy uses no hidden record before lookup.

For differences, pair each trained seed with its own initialization on the identical case. Aggregate at the family level and use a paired family bootstrap (10,000 draws, seed 1729), keeping all arms and training seeds together within a draw. Report seed variability separately. These intervals describe sensitivity to the particular synthetic families. They are not estimates of financial risk in a production queue.
