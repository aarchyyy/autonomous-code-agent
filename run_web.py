#!/usr/bin/env python3
"""
WebUI Server for Autonomous Code Synthesis & Verification Agent (ACS-Agent).
Built with Python standard library (ThreadingHTTPServer + SSE streaming) — zero third-party deps.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.agent import CodeIterationAgent
from src.benchmarks_catalog import (
    BENCHMARK_CASES,
    inject_defective_starter,
    stage_benchmarks_copy,
)
from src.llm_backend import HARDWARE_PRESETS, create_llm_backend, probe_llm_endpoint
from src.relay import RelayOrchestrator, RelayPlan

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _create_backend_from_payload(payload: Dict[str, Any]):
    mode = payload.get("mode", "auto")
    preset = payload.get("preset") or None
    base_url = payload.get("base_url") or None
    api_key = payload.get("api_key") or None
    model = payload.get("model") or None
    return create_llm_backend(
        mode=mode,
        preset=preset,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _push_job_event(job_id: str, event_type: str, data: Dict[str, Any]) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is not None:
        job["queue"].put({"type": event_type, "data": data})


def _run_single_job(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        backend = _create_backend_from_payload(payload)
        case_id = payload.get("case", "case2")
        custom_ws = (payload.get("workspace") or "").strip()

        if custom_ws:
            workspace_root = os.path.realpath(custom_ws)
            writable = [f.strip() for f in (payload.get("writable_files") or "").split(",") if f.strip()]
            if not writable:
                writable = ["kernel.py"]
            test_file = (payload.get("test_file") or "test_kernel.py").strip()
            holdout = (payload.get("holdout_file") or "").strip() or None
            spec = (payload.get("spec") or "").strip() or None
            agent_mode = payload.get("agent_mode", "auto")
            strict_ast = bool(payload.get("strict_ast", False))
        else:
            case_info = BENCHMARK_CASES.get(case_id, BENCHMARK_CASES["case2"])
            workspace_root = inject_defective_starter(case_id)
            writable = case_info["writable_files"]
            test_file = case_info["test"]
            holdout = case_info.get("holdout")
            spec = case_info.get("spec")
            agent_mode = case_info.get("agent_mode", "auto")
            strict_ast = case_info.get("strict_ast", False)

        _push_job_event(job_id, "started", {
            "kind": "single",
            "backend_mode": "deterministic_mock" if backend.is_mock else "live_llm",
            "backend_label": backend.backend_label,
            "model": backend.model,
            "workspace": workspace_root,
        })

        agent = CodeIterationAgent(
            target_file=writable[0],
            writable_files=writable,
            test_file=test_file,
            holdout_test_file=holdout,
            workspace_root=workspace_root,
            max_iterations=int(payload.get("max_iterations", 4)),
            llm_backend=backend,
            task_spec=spec,
            agent_mode=agent_mode,
            strict_ast=strict_ast,
        )

        result = agent.run_loop(
            callback=lambda step: _push_job_event(job_id, "step", step)
        )
        _push_job_event(job_id, "done", result)
    except Exception as exc:
        _push_job_event(job_id, "error", {"error": str(exc)})
    finally:
        _push_job_event(job_id, "eof", {})


def _run_relay_job(job_id: str, payload: Dict[str, Any]) -> None:
    tmp_stage = None
    try:
        backend = _create_backend_from_payload(payload)
        plan_path = (payload.get("plan_path") or "").strip() or os.path.join(
            BASE_DIR, "benchmarks/night_relay_plan.json"
        )
        plan = RelayPlan.from_json_file(plan_path)
        if payload.get("wall_budget"):
            plan.max_total_seconds = int(payload["wall_budget"])

        custom_ws = (payload.get("workspace") or "").strip()
        if custom_ws:
            workspace_root = os.path.realpath(custom_ws)
        else:
            tmp_stage = tempfile.TemporaryDirectory(prefix="acs_web_relay_")
            workspace_root = stage_benchmarks_copy(tmp_stage.name)

        output_dir = os.path.join(BASE_DIR, ".acs_runs", f"web_{job_id[:8]}")
        _push_job_event(job_id, "started", {
            "kind": "relay",
            "project_name": plan.project_name,
            "total_tasks": len(plan.tasks),
            "backend_mode": "deterministic_mock" if backend.is_mock else "live_llm",
            "backend_label": backend.backend_label,
            "model": backend.model,
        })

        orchestrator = RelayOrchestrator(
            workspace_root=workspace_root,
            plan=plan,
            llm_backend=backend,
            output_dir=output_dir,
            enable_git_branch=False,
        )
        summary = orchestrator.run(
            event_callback=lambda ev: _push_job_event(job_id, "relay_event", ev)
        )
        _push_job_event(job_id, "done", summary)
    except Exception as exc:
        _push_job_event(job_id, "error", {"error": str(exc)})
    finally:
        if tmp_stage is not None:
            tmp_stage.cleanup()
        _push_job_event(job_id, "eof", {})


class AgentDashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(BASE_DIR, "web/index.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path == "/api/config":
            cases_meta = [
                {
                    "id": cid,
                    "name": info["name"],
                    "short_name": info["short_name"],
                    "writable_files": info["writable_files"],
                    "test": info["test"],
                    "holdout": info.get("holdout"),
                    "agent_mode": info.get("agent_mode", "auto"),
                    "spec": info.get("spec", ""),
                }
                for cid, info in BENCHMARK_CASES.items()
            ]
            has_env_live = bool(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("KUNLUN_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            )
            self._send_json({
                "cases": cases_meta,
                "presets": HARDWARE_PRESETS,
                "env_live_configured": has_env_live,
                "env_model": os.getenv("LLM_MODEL", ""),
            })
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/events"):
            parts = parsed.path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "Job not found"}, status=404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            q: queue.Queue = job["queue"]
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    except OSError:
                        break
                if msg["type"] == "eof":
                    break
                frame = f"event: {msg['type']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"
                try:
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    break
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        payload = self._read_json_body()

        if parsed.path == "/api/probe":
            preset = payload.get("preset")
            base_url = payload.get("base_url") or (
                HARDWARE_PRESETS[preset]["base_url"] if preset in HARDWARE_PRESETS else "http://127.0.0.1:8000/v1"
            )
            api_key = payload.get("api_key") or None
            self._send_json(probe_llm_endpoint(base_url, api_key=api_key))
            return

        if parsed.path == "/api/reset":
            case_id = payload.get("case") or query.get("case", ["case2"])[0]
            if case_id not in BENCHMARK_CASES:
                case_id = "case2"
            inject_defective_starter(case_id)
            self._send_json({
                "status": "ok",
                "case": case_id,
                "message": f"Reset {case_id} to defective starter code.",
            })
            return

        if parsed.path == "/api/jobs":
            job_id = str(uuid.uuid4())
            with JOBS_LOCK:
                JOBS[job_id] = {"queue": queue.Queue()}
            kind = payload.get("kind", "single")
            target_fn = _run_relay_job if kind == "relay" else _run_single_job
            t = threading.Thread(target=target_fn, args=(job_id, payload), daemon=True)
            t.start()
            self._send_json({"job_id": job_id})
            return

        if parsed.path == "/api/run":
            case_id = payload.get("case") or query.get("case", ["case2"])[0]
            case_info = BENCHMARK_CASES.get(case_id, BENCHMARK_CASES["case2"])
            ws = inject_defective_starter(case_id)
            backend = _create_backend_from_payload(payload)
            agent = CodeIterationAgent(
                target_file=case_info["target"],
                writable_files=case_info["writable_files"],
                test_file=case_info["test"],
                holdout_test_file=case_info.get("holdout"),
                workspace_root=ws,
                max_iterations=int(payload.get("max_iterations", 4)),
                llm_backend=backend,
                task_spec=case_info.get("spec"),
                agent_mode=case_info.get("agent_mode", "auto"),
                strict_ast=case_info.get("strict_ast", False),
            )
            self._send_json(agent.run_loop())
            return

        self.send_error(404, "Endpoint not found")


class _DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(port: int = 8765) -> None:
    server = _DashboardServer(("127.0.0.1", port), AgentDashboardHandler)
    print("=" * 80)
    print(f"  ACS-Agent Dashboard running at: http://127.0.0.1:{port}")
    print("  Features: Real-Time SSE Stream | Single-Task ReAct | 24h Night Relay Workbench")
    print("  Hardware Presets: Kunlun GPT-Factory One / Ollama (Ryzen AI MAX+ 395) / vLLM / Cloud")
    print("  Press Ctrl+C to terminate the server.")
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
