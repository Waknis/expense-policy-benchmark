"""Small examples with hand-computable numerators, denominators and paired deltas."""

import unittest

from expensebench.scoring import compare_systems, grade_cases, score_system


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"case_id": key, "family": f, "pair_id": f}
            for f, keys in (("travel", ("a", "b")), ("meals", ("c", "d"))) for key in keys
        ]
        self.gold = {
            key: {"action": action, "required_evidence": ["policy", key]}
            for key, action in zip("abcd", ("approve", "reject", "review", "approve"))
        }
        self.allowed = {key: ["policy", key, "other"] for key in "abcd"}
        self.perfect = [
            {"case_id": key, "action": row["action"], "evidence_ids": row["required_evidence"]}
            for key, row in self.gold.items()
        ]

    def score(self, predictions):
        return score_system(self.cases, self.gold, predictions, self.allowed)

    def test_numerators_and_different_risk_denominators(self):
        rows = [
            {"case_id": "a", "action": "approve", "evidence_ids": ["policy", "a", "other"]},
            {"case_id": "b", "action": "approve", "evidence_ids": ["policy", "b", "invented"]},
            {"case_id": "c", "action": "review", "evidence_ids": []},
            {"case_id": "d", "action": "approve", "evidence_ids": ["d", "policy"]},
        ]
        scored = self.score(rows)
        self.assertEqual(scored["approvals"], 3)
        self.assertEqual(scored["false_approvals"], 1)
        self.assertEqual(scored["false_approval_risk"], 1 / 3)
        self.assertEqual(scored["false_approval_rate_among_nonapprovable"], 1 / 2)
        self.assertEqual(scored["approval_coverage"], 3 / 4)
        self.assertEqual(scored["action_accuracy"], 3 / 4)
        self.assertEqual(scored["joint_accuracy"], 1 / 4)
        self.assertEqual(scored["evidence_sufficient_accuracy"], 2 / 4)
        self.assertEqual(scored["unknown_evidence_cases"], 1)
        self.assertEqual(scored["pairs_both_correct"], 1)
        self.assertEqual(scored["pair_change_rate"], 1 / 2)
        self.assertEqual(scored["confusion"]["reject"]["approve"], 1)

    def test_invalid_and_missing_route_to_review_and_receive_zero_credit(self):
        rows = [
            {"case_id": "a", "action": "approve", "evidence_ids": "policy"},
            {"case_id": "b", "action": "escalate", "evidence_ids": ["policy", "b"]},
            self.perfect[2],
        ]
        scored = self.score(rows)
        self.assertEqual(scored["action_counts"]["missing"], 1)
        self.assertEqual(scored["action_counts"]["invalid"], 2)
        self.assertEqual(scored["review_rate"], 1 / 4)
        self.assertEqual(scored["fallback_review_rate"], 1)
        self.assertIsNone(scored["false_approval_risk"])
        self.assertEqual(scored["action_accuracy"], 1 / 4)
        self.assertEqual(scored["pairs_both_correct"], 0)
        self.assertEqual(scored["pair_change_rate"], 0)

    def test_duplicate_or_foreign_predictions_do_not_silently_pass(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.score([self.perfect[0], self.perfect[0]])
        with self.assertRaisesRegex(ValueError, "outside selected"):
            self.score([dict(self.perfect[0], case_id="other-id")])

    def test_bootstrap_resamples_whole_families_for_all_methods(self):
        # A wins the travel family; B wins the meals family. A paired draw of
        # the two clusters gives only {-1,0,+1} as possible accuracy deltas.
        systems = {"A": self.perfect[:2], "B": self.perfect[2:], "C": self.perfect[:2]}
        kwargs = dict(replicates=200, seed=72)
        report = compare_systems(self.cases, self.gold, systems, self.allowed, **kwargs)
        self.assertEqual(report, compare_systems(self.cases, self.gold, systems, self.allowed, **kwargs))
        ab, ac, _ = report["paired_comparisons"]
        self.assertEqual(ab["metrics"]["action_accuracy"], {
            "delta": 0, "ci95": [-1, 1], "defined_replicates": 200,
        })
        self.assertEqual(ac["metrics"]["action_accuracy"]["ci95"], [0, 0])
        self.assertLess(ab["metrics"]["false_approval_risk"]["defined_replicates"], 200)
        self.assertEqual(report["families"]["travel"]["A"]["pair_accuracy"], 1)

    def test_single_family_or_no_approvals_never_imply_certain_risk(self):
        report = compare_systems(self.cases[:2], self.gold, {"A": [], "B": []}, self.allowed, replicates=10)
        metrics = report["paired_comparisons"][0]["metrics"]
        self.assertEqual(metrics["false_approval_risk"], {
            "delta": None, "ci95": None, "defined_replicates": 0,
        })
        self.assertIsNone(metrics["action_accuracy"]["ci95"])
        rows = grade_cases(self.cases[:1], self.gold, self.perfect[:1], self.allowed)
        self.assertEqual(rows[0]["outcome"], "approve")
        scored = score_system(self.cases[:1], self.gold, self.perfect[:1], self.allowed)
        self.assertIsNone(scored["pair_accuracy"])
        self.assertIsNone(scored["false_approval_rate_among_nonapprovable"])


if __name__ == "__main__":
    unittest.main()
