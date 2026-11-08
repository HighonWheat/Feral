"""
Phase 6, Step 4 - Duplicate Tool-Call Loop Watchdog & Temperature-Bump Interrupt (`loop_watchdog.py`)
=====================================================================================================
Prevents Feral from getting stuck in infinite tool-calling loops during unattended overnight
missions by combining canonical tool-call fingerprinting, sliding-window loop detection,
system-interrupt injection, and dynamic temperature escalation:

  1. Canonical Tool-Call Fingerprinting (`fingerprint_tool_call(name, arguments)`):
     Normalizes Windows/POSIX file paths (`\\` -> `/`), trims whitespace, and hashes sorted
     JSON keys via SHA-256 so cosmetically different duplicate calls share the same signature.
  2. Dual-Mode Loop Detection (`LoopWatchdog.inspect_tool_call`):
     - Consecutive Identical Loop (`A -> A`): triggers when the same fingerprint appears
       `repeat_threshold` (default=2) times in a row.
     - Alternating Ping-Pong Oscillation (`A -> B -> A -> B`): triggers when the last 4 calls
       oscillate between two fingerprints (`fp[-4] == fp[-2]` and `fp[-3] == fp[-1]`).
  3. Dynamic Temperature Bump & Automatic Cooldown:
     - When a loop is intercepted, escalates `recommended_temperature` (`0.00 -> 0.45 -> 0.65`)
       to break greedy decoding traps and returns a `[LOOP WATCHDOG INTERRUPT]` directive.
     - As soon as Feral emits a novel tool call on the following turn, automatically cools
       `recommended_temperature` back to `base_temperature` (`0.00`) for deterministic patching.
"""
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from feral_core import ROOT, SYSTEM_FERAL


def _normalize_value(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip().replace("\r\n", "\n")
        # Normalize path separators for path-like strings without newlines
        if "\n" not in s and ("\\" in s or "/" in s) and not s.startswith(("http://", "https://")):
            s = re.sub(r"[\\/]+", "/", s)
        return s
    if isinstance(val, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, list):
        return [_normalize_value(x) for x in val]
    return val


def fingerprint_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Computes a canonical 16-char SHA-256 hex fingerprint for `(name, arguments)`."""
    payload = {
        "name": name.strip(),
        "arguments": _normalize_value(arguments or {}),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class WatchdogVerdict:
    blocked: bool
    loop_type: str  # "none" | "consecutive_duplicate" | "ping_pong_oscillation"
    fingerprint: str
    repeat_count: int
    recommended_temperature: float
    interrupt_message: str
    timestamp: float = field(default_factory=time.time)


class LoopWatchdog:
    """
    Monitors Feral's tool-call stream, intercepts consecutive or ping-pong loops,
    injects corrective system interrupts, and modulates sampling temperature.
    """

    def __init__(
        self,
        repeat_threshold: int = 2,
        base_temperature: float = 0.0,
        bump_step: float = 0.45,
        max_temperature: float = 0.75,
        window_size: int = 16,
    ):
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.base_temperature = float(base_temperature)
        self.bump_step = float(bump_step)
        self.max_temperature = float(max_temperature)
        self.window_size = max(6, int(window_size))

        self.current_temperature: float = self.base_temperature
        self.history: List[Dict[str, Any]] = []
        self.interrupt_events: List[WatchdogVerdict] = []

    def _suggest_alternative(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "apply_diff_patch":
            path = arguments.get("path", "<file>")
            return (
                f"Call `read_file_lines` on `{path}` to inspect the exact current lines and indentation "
                f"before attempting another `apply_diff_patch`."
            )
        if tool_name == "run_powershell":
            return (
                "Inspect the previous error output carefully and switch to a native Windows PowerShell "
                "cmdlet or verify file paths with `ast_repo_map` / `read_file_lines`."
            )
        if tool_name == "read_file_lines":
            return "You already read this exact line range. Proceed to `apply_diff_patch` or read a different range."
        return "Choose a different tool or change the arguments to make forward progress."

    def inspect_tool_call(self, name: str, arguments: Dict[str, Any]) -> WatchdogVerdict:
        """
        Inspects a proposed `(name, arguments)` tool call before execution.
        Returns a `WatchdogVerdict` indicating whether execution should be blocked and
        what sampling temperature should be used on the next turn.
        """
        fp = fingerprint_tool_call(name, arguments)

        # Count consecutive trailing occurrences of `fp`
        consecutive = 1
        for past in reversed(self.history):
            if past["fingerprint"] == fp:
                consecutive += 1
            else:
                break

        # Check for A -> B -> A -> B ping-pong oscillation across the last 3 calls + current call
        is_ping_pong = False
        if len(self.history) >= 3:
            fp_minus_3 = self.history[-3]["fingerprint"]
            fp_minus_2 = self.history[-2]["fingerprint"]
            fp_minus_1 = self.history[-1]["fingerprint"]
            if fp == fp_minus_2 and fp_minus_1 == fp_minus_3 and fp != fp_minus_1:
                is_ping_pong = True

        self.history.append(
            {
                "name": name,
                "arguments": arguments,
                "fingerprint": fp,
                "timestamp": time.time(),
            }
        )
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size :]

        if consecutive >= self.repeat_threshold or is_ping_pong:
            loop_kind = "consecutive_duplicate" if consecutive >= self.repeat_threshold else "ping_pong_oscillation"
            if self.current_temperature <= self.base_temperature + 1e-4:
                self.current_temperature = min(self.max_temperature, self.base_temperature + self.bump_step)
            else:
                self.current_temperature = min(self.max_temperature, round(self.current_temperature + 0.20, 2))

            hint = self._suggest_alternative(name, arguments)
            interrupt_msg = (
                f"ERROR: [LOOP WATCHDOG INTERRUPT - {loop_kind.upper()}] "
                f"You repeated `{name}({json.dumps(arguments)})` (fingerprint `{fp}`, count={consecutive}). "
                f"Execution blocked to prevent an infinite loop; sampling temperature bumped to {self.current_temperature:.2f}. "
                f"REQUIRED ACTION: {hint}\n[exit_code=1]"
            )
            verdict = WatchdogVerdict(
                blocked=True,
                loop_type=loop_kind,
                fingerprint=fp,
                repeat_count=consecutive,
                recommended_temperature=self.current_temperature,
                interrupt_message=interrupt_msg,
            )
            self.interrupt_events.append(verdict)
            return verdict

        # Novel tool call -> cool temperature back down to deterministic baseline
        self.current_temperature = self.base_temperature
        return WatchdogVerdict(
            blocked=False,
            loop_type="none",
            fingerprint=fp,
            repeat_count=consecutive,
            recommended_temperature=self.current_temperature,
            interrupt_message="",
        )

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "current_temperature": self.current_temperature,
            "base_temperature": self.base_temperature,
            "history_len": len(self.history),
            "interrupt_count": len(self.interrupt_events),
            "recent_interrupts": [asdict(v) for v in self.interrupt_events[-5:]],
        }


def main():
    print("1. Testing Canonical Tool-Call Fingerprinting Across Path/Whitespace Variants...")
    fp1 = fingerprint_tool_call("read_file_lines", {"path": "src\\billing.py", "start_line": 1, "end_line": 20})
    fp2 = fingerprint_tool_call("read_file_lines", {"end_line": 20, "path": "src/billing.py ", "start_line": 1})
    print(f"   - Variant 1 (`src\\\\billing.py`) : `{fp1}`")
    print(f"   - Variant 2 (`src/billing.py `) : `{fp2}` (exact match={fp1 == fp2})")
    assert fp1 == fp2

    print("\n2. Testing Consecutive Duplicate Loop Detection (`A -> A`) & Dynamic Temperature Escalation...")
    wd = LoopWatchdog(repeat_threshold=2, base_temperature=0.0, bump_step=0.45)
    call_a = {"path": "src/paginator.py", "search_block": "start = page * per_page", "replace_block": "start = (page - 1) * per_page"}

    v1 = wd.inspect_tool_call("apply_diff_patch", call_a)
    v2 = wd.inspect_tool_call("apply_diff_patch", call_a)
    print(f"   - Call 1 (`apply_diff_patch`)   : blocked={v1.blocked}, temp={v1.recommended_temperature:.2f}")
    print(f"   - Call 2 (Duplicate `A -> A`)   : blocked={v2.blocked}, loop_type={v2.loop_type}, temp={v2.recommended_temperature:.2f}")
    assert not v1.blocked and v2.blocked and v2.loop_type == "consecutive_duplicate"
    assert v2.recommended_temperature == 0.45

    print("\n3. Testing Alternating Ping-Pong Oscillation (`A -> B -> A -> B`) & Cooldown on Novel Call...")
    wd_pp = LoopWatchdog(repeat_threshold=3, base_temperature=0.0, bump_step=0.45)
    _ = wd_pp.inspect_tool_call("ast_repo_map", {"directory": "src"})
    _ = wd_pp.inspect_tool_call("ast_repo_map", {"directory": "tests"})
    _ = wd_pp.inspect_tool_call("ast_repo_map", {"directory": "src"})
    v_pp = wd_pp.inspect_tool_call("ast_repo_map", {"directory": "tests"})
    v_novel = wd_pp.inspect_tool_call("read_file_lines", {"path": "src/paginator.py", "start_line": 1, "end_line": 10})
    print(f"   - Call 4 (`A -> B -> A -> B`)   : blocked={v_pp.blocked}, loop_type={v_pp.loop_type}, temp={v_pp.recommended_temperature:.2f}")
    print(f"   - Call 5 (Novel Tool Call)      : blocked={v_novel.blocked}, temp cooled back to {v_novel.recommended_temperature:.2f}")
    assert v_pp.blocked and v_pp.loop_type == "ping_pong_oscillation"
    assert not v_novel.blocked and v_novel.recommended_temperature == 0.0

    print("\n4. Live RTX 5070 Ti Verification: Breaking a Stuck Patch Loop with `FeralLLMClient` on `cuda:0`...")
    from feral_llm_client import FeralLLMClient

    llm = FeralLLMClient()
    llm.update_context_panel(temperature=v2.recommended_temperature)

    stuck_messages = [
        {"role": "system", "content": SYSTEM_FERAL},
        {"role": "user", "content": "Fix the off-by-one bug in src/paginator.py."},
        {
            "role": "assistant",
            "content": (
                "<think>\nGuess the search_block in src/paginator.py without reading it first.\n</think>\n"
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/paginator.py", "search_block": "return items[page:]", "replace_block": "return items[page-1:]"}}</tool_call>'
            ),
        },
        {
            "role": "user",
            "content": "<tool_response>\nERROR: SEARCH block not found in src/paginator.py (file untouched)\n[exit_code=1]\n</tool_response>",
        },
        {
            "role": "assistant",
            "content": (
                "<think>\nTry the exact same apply_diff_patch again.\n</think>\n"
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/paginator.py", "search_block": "return items[page:]", "replace_block": "return items[page-1:]"}}</tool_call>'
            ),
        },
        {
            "role": "user",
            "content": f"<tool_response>\n{v2.interrupt_message}\n</tool_response>",
        },
    ]

    recovery_turn = llm.generate_turn(stuck_messages, temperature=v2.recommended_temperature)
    print(f"   - Watchdog Interrupt Injected   : {v2.interrupt_message.splitlines()[0][:105]}...")
    print(
        f"   - Model Recovery Turn ({recovery_turn.tokens_per_sec:5.2f} tok/s, temp={v2.recommended_temperature:.2f}) -> "
        f"{json.dumps(recovery_turn.tool_call)}"
    )
    assert recovery_turn.is_valid_tool_call and recovery_turn.tool_call is not None
    assert recovery_turn.tool_call["name"] == "read_file_lines"

    print("\nPHASE 6 STEP 4 (`loop_watchdog.py`): PASS -> Duplicate & ping-pong loop detection, temperature bump, and live model recovery verified!")


if __name__ == "__main__":
    main()
