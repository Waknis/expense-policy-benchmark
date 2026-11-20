-- Each row of audit_rows contains one full held-out decision record as JSON.
WITH decisions AS (
    SELECT
        json_extract(record, '$.pair_id') AS pair_id,
        json_extract(record, '$.gold.action') AS reference,
        system.key AS method,
        json_extract(system.value, '$.action') AS action
    FROM audit_rows, json_each(record, '$.decisions') AS system
),
matched AS (
    SELECT
        method, pair_id, count(*) AS n_cases,
        sum(action = reference) AS correct,
        sum(action = 'approve') AS approvals,
        sum(action = 'approve' AND reference <> 'approve') AS false_approvals
    FROM decisions
    GROUP BY method, pair_id
)
SELECT
    method, sum(n_cases) AS n_cases, sum(correct) AS actions_correct,
    sum(approvals) AS approvals, sum(false_approvals) AS false_approvals,
    sum(n_cases = 2 AND correct = 2) AS pairs_both_correct
FROM matched
GROUP BY method
ORDER BY method;
