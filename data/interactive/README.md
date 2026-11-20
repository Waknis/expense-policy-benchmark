# Interactive expense-policy data

Each episode starts with an expense and one employer policy. A case contains four records from separate systems. The environment releases a record when the agent queries its corresponding tool. The reference answer consists of an action and supporting record IDs. These are synthetic exercises in applying a supplied policy. The amounts, employers, policies, records and labels were constructed for the benchmark.

| Split | Policy families | Policy editions | Matched pairs | Cases |
| --- | ---: | ---: | ---: | ---: |
| Training | 8 | 24 | 768 | 1,536 |
| Development | 3 | 9 | 72 | 144 |
| Fresh test | 6 | 18 | 144 | 288 |

Training covers fleet bicycle service, translation delivery, field office day passes, shipment insurance, badge replacement, projector delivery, archive collection and media destruction. Development covers instrument calibration, specimen storage and visitor interpretation. The fresh test families are outage generator hire, solvent return, dependent care during work travel, warranty repair intake, accessible sign installation and independent machine inspection.

Families, policy IDs, policy texts, case IDs and pair IDs are disjoint across splits. The test families were authored for this release and differ from all ten v1 benchmark families. Each family has three editions with different numeric thresholds. The test policies share the ordered rule language and tool vocabulary used for training. Two of the six test families use a sequence of tool/operator combinations absent from training. This split measures transfer to new policies within a known workflow.

## Decision and evidence

The human-readable clause and its two ordered machine-readable checks are both public at reset. The four evidence sources are a registry, ledger, document and authorization system. Each policy names two sources. The other two return irrelevant records such as an invoice format or a previous month's line count. String comparisons test equality. Numeric comparisons include the boundary. A null value means the requested record lacks a usable answer.

Apply the checks in their written order. Reject at the first known violation, refer for review at the first missing value, or approve after both checks pass. The required citations are the policy ID and the IDs of the records for the evaluated prefix. A decisive first check needs one record. Any disposition that depends on the second check needs both. Two-thirds of cases require both records. The visibility of the written checks makes tool selection and routing learnable without natural-language parsing. The reference labels live only in the separate gold files.

In one training pair the registry identifies a company fleet bicycle and the service cap is $95.00. The invoice in `C-0c5c1ff54851` shows $95.00, an eligible claim. In `C-4db40d23c4a0` the service subtotal is $102.75, requiring rejection. The merchant, face value of the expense, edition, record IDs and all other evidence are identical. The invoice field is the single intervention.

## Reading and rebuilding

Each split has its own `cases.jsonl` and `gold.jsonl`. Open only the split needed. The case manifest includes administrative pair and family IDs and the hidden tool store. These are inputs to the environment; an acting policy should receive the public reset observation and its lookup results. The cases and references are published for replay. Use a new private split for claims based on tuning after inspecting the released test traces.

```bash
python data/interactive/generate.py --split train --output-dir .cache/interactive-fixtures
python data/interactive/generate.py --output-dir .cache/interactive-all
python -m pytest tests/test_interactive_data.py -q
```

Counterfactual pairs keep the expense, policy and record IDs fixed and change exactly one relevant record value. The three reference actions and all three unordered action transitions are balanced within each family. The generator assigns labels from the sampled case states. A separate test derives every label and its minimal evidence directly from the observable predicate and record values. Generation and scoring require no outside labeling service. No independent finance professional adjudicated these scenarios.

The matched arms share context, and the many cases within a family share policy structure, so they are correlated. A small number of synthetic families, uniform action proportions and explicitly structured constraints do not estimate real-world financial risk. The included cases also do not test policy retrieval, open-ended document reading, OCR, negotiations with submitters or real expense execution.
