# Dataset card

Version 1 contains 120 fictional expenses organized into 60 matched pairs. Four policy families supply 48 development cases. Six different families supply 72 test cases. Every family has four approvals, four reviews, and four rejections. Each pair changes exactly one evidence value and changes the correct action. Within a pair the transaction, applicable policy, merchant, evidence record IDs, and other values stay fixed. Case IDs and sort order derive from hashes; the agents receive no pair relationships, splits or labels.

The transactions, merchants, policies, and reference decisions were authored for this benchmark. They contain no customer records or proprietary Ramp policies. Labels come from constructed scenarios and their explicit clauses. This collection has no independent human finance expert labels.

## Task boundaries

Each case asks for a decision under **one supplied clause**. Other expense controls are outside the case. The policy text defines which source is authoritative, when a known violation requires rejection, and which missing fact requires review. For example, an itemized receipt alone cannot prove that a travel booking was preapproved. Some evidence is intentionally irrelevant or covers a different date or period. Null means the specified record value is unavailable. Money is expressed in minor currency units and all timestamps are UTC.

| Split | Family | Policy judgment |
| --- | --- | --- |
| Development | Hosted meal allowance | Apply a USD 45 cap per confirmed attendee. |
| Development | Local transport | Distinguish verified business journeys from commutes and unresolved destinations. |
| Development | Rail contract | Use current registry status for a reseller rather than last year's designation. |
| Development | Course enrollment | Compare the confirmed enrollment date to the course start date. |
| Test | Lodging room rate | Check the room-only rate per night against the destination zone cap. |
| Test | Purchase preapproval | Compare manager authorization time to the purchase time for equipment above USD 150. |
| Test | Prior reimbursement | Match receipt identifiers and reconcile the earlier payment status. |
| Test | Foreign meal rate | Check the inclusive USD cap using the currency rate on the charge date. |
| Test | Equipment allowance revision | Apply the effective half-year limit and include previous reimbursements. |
| Test | Airfare cabin exception | Use the ticket cabin and longest airborne segment under an explicit exception. |

## Artifacts and provenance

| File | Contents and permitted use |
| --- | --- |
| `observations.jsonl` | Observable expense and evidence records, with stable IDs scoped to each case. |
| `policies.jsonl` | Effective dates and full policy clauses. The SQLite adapter loads these and the observations for inference. |
| `cases.jsonl` | Evaluator manifest with split, case, pair, family, policy, and expense identifiers. |
| `gold.jsonl` | Evaluator-only actions, required supporting record IDs including the clause, and rationales. |
| `generate.py` | Authoring source for the fixed policies, paired interventions, labels, and IDs. Keep it outside agent context. |

From the repository root, run `python data/generate.py --output-dir <new-directory>` to regenerate byte-identical JSONL files. The adapter exposes one case at a time using parameterized SQLite lookups. The development and held-out sets are open for reproduction. After examining test labels or traces, use a fresh private holdout to measure subsequent improvements.

Each family uses six deliberately related pairs. This design tests sensitivity to specific facts. The artificial label balance, limited policy coverage and shared scenario templates make these frequencies unsuitable for estimating customer traffic or production error rates.

The required references follow the final labels described in [the evidence correction](ERRATA.md).
