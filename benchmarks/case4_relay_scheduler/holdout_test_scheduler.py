import unittest
from scheduler import dispatch_batch
from precision import weighted_score


class TestRelaySchedulerHoldout(unittest.TestCase):
    def test_compensated_cancellation(self):
        # Holdout numerical stability check: naive float addition yields 0.0 instead of 1.0
        score = weighted_score([1e16, 1.0, -1e16], [1.0, 1.0, 1.0])
        self.assertEqual(score, 1.0)

    def test_zero_budget_and_non_mutating_output(self):
        tasks = [{"id": "x1", "tokens": 10, "metrics": [1.0], "weights": [1.0]}]
        self.assertEqual(dispatch_batch(tasks, max_tokens=0), [])
        selected = dispatch_batch(tasks, max_tokens=10)
        selected[0]["tokens"] = 999
        self.assertEqual(tasks[0]["tokens"], 10)


if __name__ == "__main__":
    unittest.main()
