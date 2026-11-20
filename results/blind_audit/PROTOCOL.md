# Blinded audit of reference labels

This audit uses a separate Codex assistant to apply the written policies to a fixed subset. The reviewer helped implement the scoring code using hand-built unit fixtures. During annotation, the reviewer read the benchmark schema and model-visible case packets. The reference decisions, generated system predictions and dataset generator were excluded from that review. This is a model-assisted consistency check. No finance professional or human rater participated.

## Sampling

Select one complete matched pair per test family. Within each family, sort pair IDs by SHA-256 of the UTF-8 string `expensebench-blind-audit-v1` + NUL + family + NUL + pair ID and take the first. Shuffle the selected cases by SHA-256 of the same salt + NUL + case ID. The reviewer sees the expense, policy and evidence records without family, pair or split metadata. Matched cases remain correlated.

`reviewer_inputs.jsonl` preserves the exact packets originally shown, recovered from the review transcript. With the package installed, run the following from the repository root to reconstruct the sampled packets from the released dataset.

```sh
python results/blind_audit/sample.py
```

## Review

For each case, select `approve`, `review` or `reject` using the supplied policy. A hard violation takes the action prescribed by that policy. Escalate when the policy requires more information or a judgment call. Cite the expense, policy and any material evidence records used to reach the action. Record a short rationale.

Freeze the decisions before opening the reference labels. Report action agreement, disagreements and citation-set agreement separately. With only one model reviewer and one pair per family, any agreement rate applies only to these selected cases.

## Version check

The reviewer saw 12 packets during data preparation. Two duplicate-reimbursement packets (`C-19aac709b` and `C-52f3f1115`) changed before the benchmark freeze. The first file digest was taken after annotation and identified the updated release packets. A comparison of the original transcript with the release identified the two differences. We excluded those cases from both versions of the agreement results and retained their original annotations for inspection. The 10 eligible packets match the release field for field. They cover five policy families.

`original_reference.jsonl` preserves the initial reference subset for the evidence-label comparison. The release comparison reads the repository's current reference labels. Replay the audit with `python results/blind_audit/grade.py`. The script validates the inputs and annotations, identifies changed packets and recomputes both comparisons.

## Agreement on the ten eligible packets

The reviewer and released key agreed on four approvals, three rejections and three reviews. Every citation set contained the documents required by the key and cited only available records. Five sets matched the released key exactly. The five others included the expense record as additional context, as directed in the reviewer instructions. Exact-set scores therefore depend in part on the citation convention.

The initial key gave six exact citation matches. A subsequent evidence-only correction removed a redundant expense ID for a missing half-year total, accounting for the difference in this sample. All ten action labels stayed the same. The complete case-level differences appear in `agreement.json`.
