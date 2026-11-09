import unittest
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tools import ToolBox

class TestToolBox(unittest.TestCase):
    def setUp(self):
        self.tools = ToolBox(workspace_root="/tmp/agent_test_workspace")
        os.makedirs("/tmp/agent_test_workspace", exist_ok=True)

    def test_file_read_write(self):
        self.tools.write_source("sample.py", "x = 42\n")
        content = self.tools.read_source("sample.py")
        self.assertEqual(content, "x = 42\n")

    def test_syntax_validation(self):
        valid, msg = self.tools.validate_syntax("def foo():\n    return 10\n")
        self.assertTrue(valid)

        invalid, err = self.tools.validate_syntax("def broken(:\n    pass")
        self.assertFalse(invalid)
        self.assertIn("SyntaxError", err)

    def test_diff_computation(self):
        orig = "a = 1\nb = 2\n"
        patched = "a = 1\nb = 3\n"
        diff = self.tools.compute_diff(orig, patched, filename="test.py")
        self.assertIn("-b = 2", diff)
        self.assertIn("+b = 3", diff)

if __name__ == "__main__":
    unittest.main()
