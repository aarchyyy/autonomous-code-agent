"""
Checkpoint, Best-State Rollback, and Git Relay Branch Manager.

Ensures autonomous overnight runs never corrupt working code:
- FileSnapshotManager captures exact contents of authorized writable files before
  each iteration and tracks the highest-scoring passing state for automatic rollback.
- GitRelayManager optionally isolates overnight autonomous commits onto a dedicated
  `acs-relay/<run_id>` branch with structured commit trailers.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .sandbox import PathGuard


@dataclass
class CheckpointState:
    label: str
    score: Tuple[int, int]
    files: Dict[str, Optional[str]]
    state_hash: str


class FileSnapshotManager:
    """In-memory atomic snapshot & rollback manager for authorized workspace files."""

    def __init__(self, workspace_root: str, tracked_files: Iterable[str]):
        self.workspace_root = os.path.realpath(workspace_root)
        self.guard = PathGuard(self.workspace_root)
        self.tracked_files: List[str] = []
        for f in tracked_files:
            _, rel = self.guard.canonicalize(f)
            if rel not in self.tracked_files:
                self.tracked_files.append(rel)
        self.checkpoints: Dict[str, CheckpointState] = {}
        self.best_checkpoint: Optional[CheckpointState] = None

    def capture_state(self, extra_files: Iterable[str] = ()) -> Tuple[Dict[str, Optional[str]], str]:
        files_to_read = list(self.tracked_files)
        for f in extra_files:
            _, rel = self.guard.canonicalize(f)
            if rel not in files_to_read:
                files_to_read.append(rel)

        snapshot: Dict[str, Optional[str]] = {}
        h = hashlib.sha256()
        for rel in sorted(files_to_read):
            abs_path = os.path.join(self.workspace_root, rel)
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as fp:
                    content = fp.read()
                snapshot[rel] = content
                h.update(rel.encode("utf-8"))
                h.update(b"\x00")
                h.update(content.encode("utf-8"))
            else:
                snapshot[rel] = None
                h.update(rel.encode("utf-8"))
                h.update(b"\x01MISSING")
        return snapshot, h.hexdigest()[:16]

    def current_hash(self) -> str:
        _, state_hash = self.capture_state()
        return state_hash

    def save_checkpoint(self, label: str, score: Tuple[int, int] = (0, 0)) -> CheckpointState:
        files, state_hash = self.capture_state()
        cp = CheckpointState(label=label, score=score, files=files, state_hash=state_hash)
        self.checkpoints[label] = cp
        if self.best_checkpoint is None or score > self.best_checkpoint.score:
            self.best_checkpoint = cp
        return cp

    def update_best_if_better(self, label: str, score: Tuple[int, int]) -> bool:
        files, state_hash = self.capture_state()
        cp = CheckpointState(label=label, score=score, files=files, state_hash=state_hash)
        self.checkpoints[label] = cp
        if self.best_checkpoint is None or score > self.best_checkpoint.score:
            self.best_checkpoint = cp
            return True
        return False

    def restore(self, cp: CheckpointState) -> List[str]:
        restored: List[str] = []
        for rel, content in cp.files.items():
            abs_path, _ = self.guard.canonicalize(rel)
            if content is None:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    restored.append(f"removed:{rel}")
            else:
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as fp:
                    fp.write(content)
                restored.append(f"restored:{rel}")
        return restored

    def restore_checkpoint(self, label: str) -> List[str]:
        if label not in self.checkpoints:
            raise KeyError(f"Unknown checkpoint: {label}")
        return self.restore(self.checkpoints[label])

    def restore_best(self) -> Optional[List[str]]:
        if self.best_checkpoint is None:
            return None
        return self.restore(self.best_checkpoint)


class GitRelayManager:
    """Optional Git integration creating isolated night-relay branches and milestone commits."""

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.realpath(workspace_root)
        self.repo_root = self._detect_repo_root()
        self.relay_branch: Optional[str] = None
        self.original_branch: Optional[str] = None

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root or self.workspace_root,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _detect_repo_root(self) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                return os.path.realpath(res.stdout.strip())
        except Exception:
            pass
        return None

    @property
    def is_available(self) -> bool:
        return self.repo_root is not None

    def start_relay_branch(self, run_id: str) -> Optional[str]:
        if not self.is_available:
            return None
        cur = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if cur.returncode == 0:
            self.original_branch = cur.stdout.strip()
        branch = f"acs-relay/{run_id}"
        res = self._run_git(["checkout", "-b", branch])
        if res.returncode == 0:
            self.relay_branch = branch
            return branch
        return None

    def commit_milestone(self, files: List[str], message: str) -> Optional[str]:
        if not self.is_available or not files:
            return None
        abs_files = [
            os.path.relpath(os.path.realpath(os.path.join(self.workspace_root, f)), self.repo_root)
            for f in files
        ]
        add_res = self._run_git(["add", "--", *abs_files])
        if add_res.returncode != 0:
            return None

        diff_check = self._run_git(["diff", "--cached", "--quiet"])
        if diff_check.returncode == 0:
            # Nothing staged to commit
            return None

        commit_res = self._run_git([
            "-c", "user.name=ACS-Relay-Agent",
            "-c", "user.email=acs-relay@localhost",
            "commit",
            "-m", message,
        ])
        if commit_res.returncode != 0:
            return None

        rev = self._run_git(["rev-parse", "--short", "HEAD"])
        return rev.stdout.strip() if rev.returncode == 0 else "committed"

    def head_sha(self) -> Optional[str]:
        if not self.is_available:
            return None
        rev = self._run_git(["rev-parse", "--short", "HEAD"])
        return rev.stdout.strip() if rev.returncode == 0 else None
