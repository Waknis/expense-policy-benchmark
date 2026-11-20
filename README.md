# Expense Policy Benchmark

When can an expense reviewer safely hand an approval to a model? This benchmark compares decisions on matched claims: the policy and transaction stay fixed while one supporting fact changes. A later approval timestamp, an unsettled reimbursement, or a missing exchange rate can change the correct action from approve to review or reject.

The project is independent portfolio research inspired by Ramp's public discussion of disagreement and escalation in expense review. All policies, transactions and labels are synthetic. [Ramp's financial benchmarks](https://builders.ramp.com/post/financial-benchmarks)

One development pair uses the same $150 course invoice, dated July 14, for a course starting July 16. In case `C-8019e2393`, the registrar confirmed enrollment on July 15, so the policy says to approve. In `C-c121eeaaf`, that one date moves to July 18 and the correct action becomes reject. The receipt alone cannot distinguish them.

## Experiment

- **120 cases in 60 matched pairs.** Four policy families are used for development and six different families for the final test. Each family has four cases for each of the three actions.
- **One model, three conditions.** Qwen2.5-7B-Instruct sees the expense and policy, then the same inputs with records from SQLite, or the full records with a request for one factual finding. A fixed receipt rule, an always-review baseline and a citation check put those results in operational terms.
- **A frozen routing decision.** Among the five originally eligible methods, select the one that approves the most development claims with zero observed false approvals. Refer every other claim to a person. The secondary factual-finding condition tests a prompt change on identical information.
- **Inspectable grading.** Scores cover three-way decisions, both members of each pair, false approvals and cited records. Each response can be traced to its original model output and exact request digest.

## Reproduce

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m expensebench.experiment report --out .cache/replay
```

The published outputs can be rescored on CPU without model access. The [reproduction guide](reports/reproduction.md) includes instructions for testing a new reviewer and measuring independent human agreement. The [dataset card](data/README.md) describes the policy families and label construction. The [experiment protocol](protocol/experiment.md) records the decision rule.
