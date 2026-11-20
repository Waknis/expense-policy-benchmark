# Reproducing the comparison

From a clone of this repository, use Python 3.11 or newer. SQLite is part of the Python standard library. The scoring path is CPU-only and runs without network access once the package is installed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m expensebench.experiment select --lock .cache/replay/dev.json
python -m expensebench.experiment report --lock .cache/replay/dev.json --out .cache/replay
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. GitHub Actions installs the wheel and compares the reconstructed development lock, held-out metrics, and per-case decisions against the committed files. The replay tests also reconstruct each request digest from its case packet and parse its raw model response back into a decision.

## Inspect a case

```python
from expensebench.data import build_database, case_packet, load_cases

case_id = load_cases()[0]["case_id"]
db = build_database()
try:
    print(case_packet(db, case_id))
finally:
    db.close()
```

The returned packet is exactly what the full-evidence condition can see, aside from the administrative case ID. The policy-only condition also drops the evidence rows. The SQLite database holds no labels. `data/gold.jsonl` and `data/cases.jsonl` are separate evaluator files.

## Try another reviewer

For a fresh blind human review, export a packet and give identical copies to two reviewers. Each person fills in `action`, `evidence_ids`, and a short `reason` independently. Actions use the exact values `approve`, `review`, or `reject`. The packet includes the expense, applicable policy and evidence in shuffled case order. Use a fresh file path for each export.

```bash
python -m expensebench.review export .cache/review/blank.jsonl
python -m expensebench.review agree reviewer-one.jsonl reviewer-two.jsonl
```

The agreement command aligns cases by ID and reports raw three-way agreement, Cohen's kappa when defined, and the IDs of disagreements. The sample contains one matched pair per held-out family. Report the denominator and discuss disagreements before using the rubric operationally.

## Evaluate another system

Produce JSONL decisions with `case_id`, `action` and `evidence_ids`. Keep all examples from each matched pair together. To score a candidate alongside a saved model with the standalone grader:

```bash
python -m expensebench.scoring --split test \
  --pred candidate=my-decisions.jsonl \
  --pred grounded=results/predictions/llm_evidence.jsonl \
  --out .cache/comparison.json
```

The grader counts missing and malformed decisions as failures and rejects duplicate case IDs. It reports whether cited IDs include all annotated evidence without invented IDs, and whether the citation set exactly matches the minimal reference. It never calls an LLM judge.

Regenerate the plot with `python -m pip install -e '.[plot]'` and `python reports/plot_results.py`.
