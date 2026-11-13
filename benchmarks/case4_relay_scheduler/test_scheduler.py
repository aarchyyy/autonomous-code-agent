import unittest
from scheduler import dispatch_batch
from precision import weighted_score


class TestRelayScheduler(unittest.TestCase):
    def test_priority_and_budget(self):
        tasks = [
            {"id": "t1", "tokens": 60, "metrics": [0.5, 0.5], "weights": [1.0, 1.0]},
            {"id": "t2", "tokens": 50, "metrics": [2.0, 1.0], "weights": [1.0, 1.0]},
            {"id": "t3", "tokens": 50, "metrics": [1.0, 1.0], "weights": [1.0, 1.0]},
        ]
        selected = dispatch_batch(tasks, max_tokens=100)
        self.assertEqual([t["id"] for t in selected], ["t2", "t3"])

    def test_tie_break_by_id(self):
        tasks = [
            {"id": "job_b", "tokens": 40, "metrics": [1.0], "weights": [2.0]},
            {"id": "job_a", "tokens": 40, "metrics": [1.0], "weights": [2.0]},
        ]
        selected = dispatch_batch(tasks, max_tokens=40)
        self.assertEqual([t["id"] for t in selected], ["job_a"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            dispatch_batch([], max_tokens=-1)
        with self.assertRaises(ValueError):
            dispatch_batch([{"id": "bad", "tokens": 0, "metrics": [1.0], "weights": [1.0]}], max_tokens=10)
        with self.assertRaises(ValueError):
            weighted_score([1.0, float("nan")], [1.0, 1.0])
        with self.assertRaises(ValueError):
            weighted_score([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
