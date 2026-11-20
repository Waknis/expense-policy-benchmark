# Expense-policy review experiment

## Research question

When does providing a claim's underlying records improve the decision, and which claims could a reviewer delegate? The unit of evaluation is one expense. Matched cases change one material fact. Development and test use different policy families. Every case and policy is fictional.

## Frozen comparisons

| Method | Accessible information | Output |
| --- | --- | --- |
| `always_review` | None | Refer every claim to a person |
| `receipt_rule` | Expense amount and receipt evidence | Approve inexpensive documented claims, refer others |
| `llm_policy` | Expense and applicable written policy | Approve, review, or reject with citations |
| `llm_evidence` | Same expense and policy plus records fetched from SQLite | Same instructions, model, and answer format |
| `citation_gate` | Full-evidence model decision and IDs available in the packet | Refer an approval if its citations fail the checks below |

The receipt heuristic uses a USD 75 inclusive ceiling, requires an affirmative receipt record and never rejects. It is a deliberately simple fixed rule without company-specific policy logic. The citation gate accepts an approval only if it cites the applicable policy, at least one claim-specific fact (expense or evidence ID), no unknown IDs and at most four distinct IDs. These checks validate references. A genuine record can still be misinterpreted.

## Decision rule

Use development cases to choose the method with the highest automatic-approval coverage among methods with **zero observed false approvals**. A false approval is any approval of a gold review or reject case. The always-review fallback is always eligible. Break ties in descending order of preference: citation gate, full-evidence model, fixed receipt rule, policy-only model, always review. If all eligible methods have zero approvals, choose always review. A zero-error development sample provides no production risk guarantee.

Freeze the winner and the development prediction hashes before producing a scored test comparison. Run all methods on the same held-out cases. Report each denominator, the full-evidence minus policy-only accuracy difference, uncertainty grouped by policy family, citation quality, and representative failures. A review is a valid outcome. For routing, every non-approval (including a model rejection) would go to a human. The benchmark also measures whether the three-way action itself was correct.

The two LLM conditions isolate access to evidence. They receive neither labels nor case-family metadata. Records are retrieved deterministically; autonomous tool selection is outside this experiment. The unit tests, data and saved model responses can be inspected and rescored without model inference.
