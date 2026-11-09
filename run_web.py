#!/usr/bin/env python3
"""
WebUI Server for Autonomous Code Synthesis & Verification Agent.
Built entirely with Python standard library (http.server) - zero third-party dependencies required.
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.agent import CodeIterationAgent
from src.llm_backend import create_llm_backend

CASES = {
    "case1": {
        "name": "Case 1: SpMM CSR Operator",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case1_spmm_boundary"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: CSR Dynamic Sparse Matrix Multiplication
def spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols):
    result = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
    for r in range(num_rows):
        start = row_ptr[r]
        end = row_ptr[r + 1]
        for k in range(start, end):
            c = col_indices[k]
            val = values[k]
            for j in range(num_cols):
                result[r][j] += val * dense_b[c][j]
    return result
"""
    },
    "case2": {
        "name": "Case 2: Vector Scale Operator",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case2_vector_scale"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: Vector Scale Operator
def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):
    return [a + b for a, b in zip(tensor_a, tensor_b)]
"""
    },
    "case3": {
        "name": "Case 3: GQA KV-Cache Broadcast",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case3_kv_cache_broadcast"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: GQA KV-Cache Broadcast
def gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2):
    return kv_tensor
"""
    }
}

class AgentDashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "web/index.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return
        
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        case_id = query.get("case", ["case2"])[0]
        case_info = CASES.get(case_id, CASES["case2"])

        if parsed.path == "/api/reset":
            target_path = os.path.join(case_info["workspace"], case_info["target"])
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(case_info["initial_code"])

            self._send_json({"status": "ok", "message": f"Reset {case_id} to broken initial code."})
            return

        if parsed.path == "/api/run":
            # Reset first to ensure deterministic reproduction
            target_path = os.path.join(case_info["workspace"], case_info["target"])
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(case_info["initial_code"])

            agent = CodeIterationAgent(
                target_file=case_info["target"],
                test_file=case_info["test"],
                workspace_root=case_info["workspace"],
                max_iterations=4,
                llm_backend=create_llm_backend()
            )
            result = agent.run_loop()
            self._send_json(result)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, data: dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

def run_server(port: int = 8765):
    server = HTTPServer(("127.0.0.1", port), AgentDashboardHandler)
    print("=" * 80)
    print(f"🚀 ACS-Agent Dashboard running at: http://127.0.0.1:{port}")
    print(f"   Open your browser to interactively inspect the night-relay TDD loop!")
    print(f"   Press Ctrl+C to terminate the server.")
    print("=" * 80)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
        server.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACS-Agent Web Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="HTTP Server Port (default: 8765)")
    args = parser.parse_args()
    run_server(port=args.port)
