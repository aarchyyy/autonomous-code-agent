import unittest

from src.evaluator import run_fault_injection_http_sim, run_redteam_suite


class TestEvaluatorSuites(unittest.TestCase):
    def test_redteam_8_probes_all_defended(self):
        rt = run_redteam_suite()
        self.assertEqual(rt["probes_total"], 8)
        self.assertEqual(rt["probes_defended"], 8, msg=rt)

    def test_fault_injection_http_unseen_generalization_sim(self):
        sim = run_fault_injection_http_sim()
        self.assertEqual(sim["problems_total"], 4)
        self.assertEqual(sim["problems_passed"], 4, msg=sim)
        self.assertGreaterEqual(sim.get("http_400_fallbacks", 0), 1)
        u2 = next(r for r in sim["results"] if r["problem_id"] == "U2_token_bucket_limiter")
        self.assertGreaterEqual(u2["telemetry"]["context_compactions"], 1)


if __name__ == "__main__":
    unittest.main()
