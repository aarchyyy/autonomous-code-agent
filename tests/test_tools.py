import os
import tempfile
import unittest

from src.tools import ToolBox


class TestToolBox(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_tools_test_")
        self.ws = self.tmp_ctx.name
        self.tools = ToolBox(workspace_root=self.ws)

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def test_file_read_write(self):
        self.tools.write_source("sample.py", "x = 42\n")
        content = self.tools.read_source("sample.py")
        self.assertEqual(content, "x = 42\n")

    def test_syntax_validation(self):
        valid, _ = self.tools.validate_syntax("def foo():\n    return 10\n")
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

    def test_react_tools_list_search_patch(self):
        self.tools.write_source("pkg/mod.py", "def calc(x):\n    return x + 1\n")
        files = self.tools.list_files()
        self.assertIn("pkg/mod.py", files)

        hits = self.tools.search_code("def calc")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["file"], "pkg/mod.py")

        patch_res = self.tools.dispatch_tool(
            "patch_file",
            {"filepath": "pkg/mod.py", "old_text": "return x + 1", "new_text": "return x * 2"},
        )
        self.assertTrue(patch_res["ok"])
        self.assertIn("return x * 2", self.tools.read_source("pkg/mod.py"))

        bad_patch = self.tools.dispatch_tool(
            "patch_file",
            {"filepath": "pkg/mod.py", "old_text": "return x * 2", "new_text": "return x *"},
        )
        self.assertFalse(bad_patch["ok"])
        self.assertIn("ASTPolicyError", bad_patch["error"])
        # File remains intact after AST rejection
        self.assertIn("return x * 2", self.tools.read_source("pkg/mod.py"))


if __name__ == "__main__":
    unittest.main()
