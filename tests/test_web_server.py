import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from run_web import AgentDashboardHandler, _DashboardServer


class TestWebServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _DashboardServer(("127.0.0.1", 0), AgentDashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_index_and_config_endpoints(self):
        with urllib.request.urlopen(f"{self.base}/", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("ACS-Agent", html)

        with urllib.request.urlopen(f"{self.base}/api/config", timeout=5) as resp:
            cfg = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(cfg["cases"]), 4)
            self.assertIn("kunlun-gateway", cfg["presets"])
            self.assertIn("ollama", cfg["presets"])

    def test_sse_single_job_streaming(self):
        req = urllib.request.Request(
            f"{self.base}/api/jobs",
            data=json.dumps({"kind": "single", "case": "case2", "mode": "mock"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            job_id = json.loads(resp.read().decode("utf-8"))["job_id"]

        with urllib.request.urlopen(f"{self.base}/api/jobs/{job_id}/events", timeout=10) as stream:
            raw = stream.read().decode("utf-8")
        self.assertIn("event: started", raw)
        self.assertIn("event: step", raw)
        self.assertIn("event: done", raw)
        self.assertIn('"success": true', raw)

    def test_sse_relay_job_streaming(self):
        req = urllib.request.Request(
            f"{self.base}/api/jobs",
            data=json.dumps({"kind": "relay", "mode": "mock"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            job_id = json.loads(resp.read().decode("utf-8"))["job_id"]

        with urllib.request.urlopen(f"{self.base}/api/jobs/{job_id}/events", timeout=15) as stream:
            raw = stream.read().decode("utf-8")
        self.assertIn("event: started", raw)
        self.assertIn("event: relay_event", raw)
        self.assertIn("event: done", raw)
        self.assertIn("HANDOVER_REPORT.md", raw)


if __name__ == "__main__":
    unittest.main()
