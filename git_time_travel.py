"""
Phase 6, Step 3 - Git Time-Travel Checkpoint & 3-Strike Auto-Rollback Engine (`git_time_travel.py`)
===================================================================================================
Protects Feral's workspace during multi-hour autonomous coding runs and powers the Time-Travel
Checkpoint timeline in the Phase 7 Desktop Command Center UI:

  1. Automatic Git Checkpoint Snapshots (`create_checkpoint(label, verified_green=True)`):
     Stages all workspace files (`git add -A`) and creates a commit with a structured label,
     recording the full 40-character SHA, short SHA, changed files list, and green/unverified status.
  2. 3-Strike Consecutive Failure Watchdog (`record_outcome(passed, detail)`):
     - On `passed=True`: resets `consecutive_strikes = 0` and automatically snapshots a new
       `verified_green=True` checkpoint commit (`last_green_sha`).
     - On `passed=False`: increments `consecutive_strikes` (`1/3`, `2/3`, `3/3`).
     - When `consecutive_strikes >= max_strikes` (3 strikes), automatically executes
       `git reset --hard <last_green_sha>` + `git clean -fd`, restoring the entire working tree
       to the last known-good state and returning a recovery directive to Feral.
  3. One-Click UI Time-Travel (`rollback_to_checkpoint(sha_or_id)`):
     Allows the user or self-healing orchestrator to jump to any recorded checkpoint SHA
     and inspect the `git diff` between any two checkpoints.
"""
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feral_core import ROOT


@dataclass
class GitCheckpoint:
    checkpoint_id: int
    sha: str
    short_sha: str
    label: str
    verified_green: bool
    files_changed: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class GitTimeTravelManager:
    """
    Manages Git checkpoint commits, diff inspection, and 3-strike automatic `git reset --hard`
    rollbacks inside a target workspace directory.
    """

    def __init__(self, workspace_root: Path, max_strikes: int = 3):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_strikes = max(1, int(max_strikes))

        self.consecutive_strikes: int = 0
        self.strike_history: List[str] = []
        self.checkpoints: List[GitCheckpoint] = []
        self.rollback_events: List[Dict[str, Any]] = []
        self._cp_counter = 0

        self._ensure_git_repo()

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git"] + args,
            cwd=str(self.workspace_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

    def _ensure_git_repo(self) -> None:
        git_dir = self.workspace_root / ".git"
        if not git_dir.exists():
            self._run_git(["init"])
        self._run_git(["config", "user.name", "HighonWheat"])
        self._run_git(["config", "user.email", "HighonWheat@users.noreply.github.com"])

    def get_head_sha(self) -> Tuple[str, str]:
        res = self._run_git(["rev-parse", "HEAD"])
        sha = res.stdout.strip()
        short = sha[:7] if sha else ""
        return sha, short

    def create_checkpoint(self, label: str, verified_green: bool = True) -> GitCheckpoint:
        """
        Stages all workspace modifications and records a Git checkpoint commit.
        """
        status_res = self._run_git(["status", "--porcelain"])
        changed_files = [
            line[3:].strip()
            for line in status_res.stdout.splitlines()
            if len(line) > 3
        ]

        self._run_git(["add", "-A"])
        tag_prefix = "[GREEN]" if verified_green else "[SNAPSHOT]"
        commit_msg = f"{tag_prefix} {label}"
        self._run_git(["commit", "-m", commit_msg, "--allow-empty"])

        sha_res = self._run_git(["rev-parse", "HEAD"])
        sha = sha_res.stdout.strip()
        short_sha = sha[:7]

        self._cp_counter += 1
        cp = GitCheckpoint(
            checkpoint_id=self._cp_counter,
            sha=sha,
            short_sha=short_sha,
            label=label,
            verified_green=bool(verified_green),
            files_changed=changed_files,
            timestamp=time.time(),
        )
        self.checkpoints.append(cp)
        return cp

    def last_green_checkpoint(self) -> Optional[GitCheckpoint]:
        for cp in reversed(self.checkpoints):
            if cp.verified_green:
                return cp
        return self.checkpoints[-1] if self.checkpoints else None

    def rollback_to_checkpoint(self, target_sha_or_id: Any, reason: str = "manual_rollback") -> Dict[str, Any]:
        """
        Hard-resets the working tree (`git reset --hard <sha>` + `git clean -fd`) to the
        specified checkpoint ID or commit SHA.
        """
        target_cp: Optional[GitCheckpoint] = None
        if isinstance(target_sha_or_id, int) or str(target_sha_or_id).isdigit():
            cid = int(target_sha_or_id)
            for cp in self.checkpoints:
                if cp.checkpoint_id == cid:
                    target_cp = cp
                    break
        else:
            prefix = str(target_sha_or_id).strip()
            for cp in self.checkpoints:
                if cp.sha.startswith(prefix) or cp.short_sha == prefix:
                    target_cp = cp
                    break

        if target_cp is None:
            raise KeyError(f"Checkpoint `{target_sha_or_id}` not found in timeline")

        before_sha = self._run_git(["rev-parse", "HEAD"]).stdout.strip()[:7]
        self._run_git(["reset", "--hard", target_cp.sha])
        self._run_git(["clean", "-fd"])
        after_sha = self._run_git(["rev-parse", "HEAD"]).stdout.strip()[:7]

        event = {
            "event_id": len(self.rollback_events) + 1,
            "from_short_sha": before_sha,
            "to_checkpoint_id": target_cp.checkpoint_id,
            "to_short_sha": after_sha,
            "to_label": target_cp.label,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.rollback_events.append(event)
        self.consecutive_strikes = 0
        self.strike_history.clear()
        return event

    def record_outcome(
        self,
        passed: bool,
        detail: str,
        checkpoint_label_on_pass: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tracks consecutive test/execution outcomes:
          - If `passed=True`: resets strikes to 0 and creates a new green checkpoint if requested.
          - If `passed=False`: increments `consecutive_strikes`. On Strike `max_strikes` (3/3),
            automatically rolls back the workspace to `last_green_checkpoint()`.
        """
        if passed:
            self.consecutive_strikes = 0
            self.strike_history.clear()
            cp = None
            if checkpoint_label_on_pass:
                cp = self.create_checkpoint(label=checkpoint_label_on_pass, verified_green=True)
            return {
                "passed": True,
                "strikes": 0,
                "max_strikes": self.max_strikes,
                "rolled_back": False,
                "checkpoint": asdict(cp) if cp else None,
            }

        self.consecutive_strikes += 1
        self.strike_history.append(detail)

        if self.consecutive_strikes >= self.max_strikes:
            green_cp = self.last_green_checkpoint()
            if green_cp is None:
                raise RuntimeError("3 strikes reached, but no green checkpoint exists to roll back to!")
            failed_attempts = list(self.strike_history)
            rb_event = self.rollback_to_checkpoint(
                green_cp.checkpoint_id,
                reason=f"3_strike_auto_rollback ({'; '.join(failed_attempts)})",
            )
            directive = (
                f"[TIME-TRAVEL AUTO-ROLLBACK TRIGGERED: 3/3 Strikes] "
                f"Reverted working tree to verified green Checkpoint #{green_cp.checkpoint_id} "
                f"(`{green_cp.short_sha}` - {green_cp.label}). "
                f"Do NOT repeat the failed edits: {failed_attempts}"
            )
            return {
                "passed": False,
                "strikes": self.max_strikes,
                "max_strikes": self.max_strikes,
                "rolled_back": True,
                "rollback_event": rb_event,
                "recovery_directive": directive,
            }

        return {
            "passed": False,
            "strikes": self.consecutive_strikes,
            "max_strikes": self.max_strikes,
            "rolled_back": False,
            "detail": detail,
        }

    def get_diff_against_checkpoint(self, checkpoint_id_or_sha: Any) -> str:
        """Returns `git diff <sha>` comparing the current working tree against a checkpoint."""
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id_or_sha or cp.sha.startswith(str(checkpoint_id_or_sha)):
                res = self._run_git(["diff", cp.sha])
                return res.stdout.strip()
        return ""

    def get_timeline_snapshot(self) -> Dict[str, Any]:
        head_sha = self._run_git(["rev-parse", "HEAD"]).stdout.strip()[:7]
        return {
            "head_short_sha": head_sha,
            "consecutive_strikes": self.consecutive_strikes,
            "max_strikes": self.max_strikes,
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "rollback_events": list(self.rollback_events),
        }


def run_workspace_pytest(workspace: Path) -> Tuple[bool, str]:
    res = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    combined = (res.stdout + "\n" + res.stderr).strip()
    last_line = combined.splitlines()[-1] if combined else ""
    return res.returncode == 0, last_line


def main():
    test_ws = ROOT / "feral_sandbox"
    if test_ws.exists():
        shutil.rmtree(test_ws)
    (test_ws / "src").mkdir(parents=True, exist_ok=True)
    (test_ws / "tests").mkdir(parents=True, exist_ok=True)

    calc_file = test_ws / "src" / "tax_engine.py"
    calc_file.write_text(
        "def compute_net_income(gross: float, tax_rate: float = 0.20) -> float:\n"
        "    return round(gross * (1.0 - tax_rate), 2)\n",
        encoding="utf-8",
    )
    (test_ws / "tests" / "test_tax.py").write_text(
        "import unittest\n"
        "from src.tax_engine import compute_net_income\n\n"
        "class TestTaxEngine(unittest.TestCase):\n"
        "    def test_standard_bracket(self):\n"
        "        self.assertEqual(compute_net_income(1000.0, 0.20), 800.0)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )

    try:
        print("1. Initializing `GitTimeTravelManager` & Creating Verified Green Baseline Checkpoint...")
        tt = GitTimeTravelManager(workspace_root=test_ws, max_strikes=3)
        ok0, msg0 = run_workspace_pytest(test_ws)
        assert ok0, f"Baseline test failed: {msg0}"
        cp1 = tt.create_checkpoint("Baseline passing tax_engine.py", verified_green=True)
        print(f"   - Checkpoint #{cp1.checkpoint_id} Created : `{cp1.short_sha}` ({cp1.label}) | Unit Tests: {msg0}")

        print("\n2. Simulating 3 Consecutive Bad Edits That Break Unit Tests (3-Strike Watchdog)...")
        bad_variants = [
            ("Strike 1: subtracts tax_rate directly from gross", "    return round(gross - tax_rate, 2)\n"),
            ("Strike 2: multiplies gross by tax_rate instead of (1 - tax_rate)", "    return round(gross * tax_rate, 2)\n"),
            ("Strike 3: divides gross by tax_rate", "    return round(gross / tax_rate, 2)\n"),
        ]

        outcome = {}
        for idx, (desc, bad_line) in enumerate(bad_variants, 1):
            calc_file.write_text(
                "def compute_net_income(gross: float, tax_rate: float = 0.20) -> float:\n" + bad_line,
                encoding="utf-8",
            )
            passed, test_out = run_workspace_pytest(test_ws)
            outcome = tt.record_outcome(passed=passed, detail=f"{desc} -> {test_out}")
            print(
                f"   - Attempt {idx} ({desc}) -> Test: {test_out} | "
                f"Strikes: {outcome['strikes']}/{outcome['max_strikes']} | Rolled Back: {outcome['rolled_back']}"
            )

        assert outcome.get("rolled_back") is True
        print(f"\n3. Verifying Automatic `git reset --hard` Time-Travel Restoration...")
        print(f"   - Recovery Directive Emitted    : {outcome['recovery_directive']}")

        post_rb_ok, post_rb_msg = run_workspace_pytest(test_ws)
        restored_code = calc_file.read_text(encoding="utf-8")
        snap = tt.get_timeline_snapshot()
        print(
            f"   - Post-Rollback HEAD & Tests    : HEAD=`{snap['head_short_sha']}` (matches Checkpoint #1 `{cp1.short_sha}`) | "
            f"Unit Tests: {post_rb_msg} (passed={post_rb_ok}) | Active Strikes Reset: {snap['consecutive_strikes']}/3"
        )
        assert post_rb_ok is True and snap["head_short_sha"] == cp1.short_sha
        assert "gross * (1.0 - tax_rate)" in restored_code

    finally:
        if test_ws.exists():
            shutil.rmtree(test_ws, ignore_errors=True)

    print("\nPHASE 6 STEP 3 (`git_time_travel.py`): PASS -> Git checkpointing and 3-strike automatic `git reset --hard` rollback verified!")


if __name__ == "__main__":
    main()
