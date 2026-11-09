import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent import CodeIterationAgent
from src.llm_backend import DeterministicMockLLM

class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "../benchmarks/case2_vector_scale"))

    def test_vector_scale_case_convergence(self):
        # Reset kernel.py with broken code
        broken_code = "def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):\n    return [a + b for a, b in zip(tensor_a, tensor_b)]\n"
        kernel_path = os.path.join(self.workspace, "kernel.py")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(broken_code)

        agent = CodeIterationAgent(
            target_file="kernel.py",
            test_file="test_kernel.py",
            workspace_root=self.workspace,
            max_iterations=3,
            llm_backend=DeterministicMockLLM()
        )

        result = agent.run_loop()
        self.assertTrue(result["success"])
        self.assertLessEqual(result["iterations"], 3)
        self.assertGreater(len(result["history"]), 0)

if __name__ == "__main__":
    unittest.main()
