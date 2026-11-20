# Secondary comparison on identical evidence

This addendum was specified before inspecting any model scores. Every matched pair has the same expense and policy in both arms. The policy-only condition therefore measures the cost of withholding records, and a deterministic policy-only response cannot distinguish the arms. Comparing two designs that both see the records answers a different question.

Run the same model a third time (`llm_finding`), using the same full packet and asking for one short factual finding in addition to the action and supporting evidence IDs. Compare its outcomes, paired-case accuracy, false approvals, citations, and completion tokens with the direct full-evidence condition. The additional text is an output-format and prompting change. This design cannot isolate the effects of those two changes.

The original development routing rule and eligible methods remain as specified in `experiment.md`. The additional condition is evaluated separately from that selection.
