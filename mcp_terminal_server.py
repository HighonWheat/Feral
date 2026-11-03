"""
Phase 5, Step 2 - Persistent Stateful PowerShell MCP Server (`mcp_terminal_server.py`)
======================================================================================
Provides Feral with a persistent, self-healing Windows PowerShell session exposed via both a
direct Python API and Model Context Protocol (MCP) JSON-RPC 2.0 (`tools/list`, `tools/call`):

  1. Stateful Persistence Across Tool Calls (`PersistentPowerShellSession`):
     Maintains a single long-lived `powershell.exe -NoLogo -NoProfile` process with UTF-8 pipes.
     Working directory changes (`Set-Location` / `cd`), session environment variables (`$env:VAR`),
     and PowerShell shell variables (`$port = 8080`) persist across turns.
  2. Deterministic Sentinel Framing & Error Unwrapping:
     Wraps each command so native `ErrorRecord` exceptions and cmdlet failures emit clean
     `ERROR: <message>` lines and set `[exit_code=1]`, followed by a unique sentinel line
     carrying the post-command `$PWD.Path` and exit code.
  3. Watchdog Timeout & Auto-Respawn Recovery:
     Reads stdout via a non-blocking queue thread. If a command hangs past `timeout_sec`
     (e.g., `Start-Sleep` or an infinite loop), the server terminates the hung process tree,
     respawns a clean PowerShell process in the last known working directory with tracked
     `$env:` variables restored, and returns `[exit_code=124]`.
  4. Rolling Console Log for the Autonomous Agents Board:
     Stores structured command telemetry (`command`, `output`, `exit_code`, `cwd`, `elapsed_ms`)
     for live streaming into the Phase 7 Odysseus-style Autonomous Agents Board & Terminal Pane.
"""
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from feral_core import ROOT

BLOCKED_PATTERNS = [
    r"\bFormat-Volume\b",
    r"\bStop-Computer\b",
    r"\bRestart-Computer\b",
    r"\bClear-Disk\b",
    r"\bRemove-Item\s+.*C:\\Windows",
]


@dataclass
class TerminalCommandRecord:
    """Telemetry entry for the Autonomous Agents Board rolling console log."""
    cmd_id: int
    command: str
    output: str
    exit_code: int
    cwd: str
    elapsed_ms: float
    timed_out: bool
    timestamp: float


class PersistentPowerShellSession:
    """
    Long-lived stateful Windows PowerShell session with per-command timeout and auto-recovery.
    """

    def __init__(
        self,
        workspace_root: Path = ROOT,
        default_timeout_sec: float = 30.0,
        max_history: int = 200,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.current_cwd = self.workspace_root
        self.default_timeout_sec = float(default_timeout_sec)
        self.max_history = max_history

        self.tracked_env: Dict[str, str] = {}
        self.history: List[TerminalCommandRecord] = []
        self._cmd_counter = 0
        self._lock = threading.Lock()

        self._proc: Optional[subprocess.Popen] = None
        self._out_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._start_process()

    def _start_process(self) -> None:
        """Spawns a persistent UTF-8 PowerShell process in `self.current_cwd`."""
        self.close()
        self._out_queue = queue.Queue()

        init_cmd = (
            "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "$OutputEncoding = [System.Text.Encoding]::UTF8; "
            "$ProgressPreference = 'SilentlyContinue'; "
            "function prompt { '' }"
        )
        self._proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "-",
            ],
            cwd=str(self.current_cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _reader(pipe, q):
            try:
                for line in iter(pipe.readline, ""):
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)

        self._reader_thread = threading.Thread(
            target=_reader,
            args=(self._proc.stdout, self._out_queue),
            daemon=True,
        )
        self._reader_thread.start()

        # Send UTF-8 & silent prompt bootstrap plus any tracked environment variables
        assert self._proc.stdin is not None
        self._proc.stdin.write(init_cmd + "\n")
        for k, v in self.tracked_env.items():
            safe_v = v.replace("'", "''")
            self._proc.stdin.write(f"$env:{k} = '{safe_v}'\n")
        self._proc.stdin.flush()

    def close(self) -> None:
        """Terminates the underlying PowerShell process tree cleanly."""
        if self._proc is not None:
            pid = self._proc.pid
            try:
                if self._proc.poll() is None:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _check_safety(self, command: str) -> Optional[str]:
        for pat in BLOCKED_PATTERNS:
            if re.search(pat, command, flags=re.I):
                return f"ERROR: Command blocked by Feral safety guard (matched `{pat}`).\n[exit_code=126]"
        return None

    def _track_env_assignments(self, command: str) -> None:
        """Captures simple `$env:NAME = 'value'` or `$env:NAME = \"value\"` assignments for crash recovery."""
        for m in re.finditer(r"\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", command):
            self.tracked_env[m.group(1)] = m.group(3)

    def run_powershell(self, command: str, timeout_sec: Optional[float] = None) -> str:
        """
        Executes `command` in the persistent PowerShell session and returns:
            `<clean stdout/stderr>\n[exit_code=<N>]`
        """
        blocked = self._check_safety(command)
        if blocked is not None:
            return blocked

        timeout = self.default_timeout_sec if timeout_sec is None else float(timeout_sec)

        with self._lock:
            self._cmd_counter += 1
            cmd_id = self._cmd_counter
            if self._proc is None or self._proc.poll() is not None:
                self._start_process()

            self._track_env_assignments(command)

            # Drain any stale lines in queue before sending command
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    break

            token = uuid.uuid4().hex[:12]
            sentinel_prefix = f"__FERAL_DONE_{token}__"

            # Base64-encode the user script block in UTF-8 so multiline commands, quotes,
            # and braces execute inside the persistent session scope (`.` dot-sourced!)
            cmd_b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
            payload = (
                f"$global:__feral_err = $false; $global:LASTEXITCODE = 0; $Error.Clear(); "
                f"$__feral_code = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{cmd_b64}')); "
                f"$__feral_sb = [ScriptBlock]::Create($__feral_code); "
                f"try {{ . $__feral_sb 2>&1 | ForEach-Object {{ "
                f"if ($_ -is [System.Management.Automation.ErrorRecord]) {{ "
                f"if ($_.FullyQualifiedErrorId -like 'NativeCommandError*') {{ "
                f"if ($null -ne $_.TargetObject) {{ \"$($_.TargetObject)\" }} "
                f"}} else {{ $global:__feral_err = $true; \"ERROR: $($_.Exception.Message)\" }} "
                f"}} else {{ $_ }} "
                f"}} | Out-String -Stream -Width 160 }} catch {{ $global:__feral_err = $true; \"ERROR: $($_.Exception.Message)\" }}; "
                f"if ($global:__feral_err -or $Error.Count -gt 0) {{ $__ec = 1 }} "
                f"elseif ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {{ $__ec = $LASTEXITCODE }} "
                f"else {{ $__ec = 0 }}; "
                f"[Console]::WriteLine('{sentinel_prefix}|' + $__ec + '|' + (Get-Location).Path)\n"
            )

            t0 = time.perf_counter()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
            except Exception:
                self._start_process()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()

            collected_lines: List[str] = []
            exit_code = 0
            timed_out = False
            deadline = time.perf_counter() + timeout

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    line = self._out_queue.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    continue

                if line is None:
                    # Process exited unexpectedly
                    self._start_process()
                    exit_code = 1
                    break

                stripped = line.rstrip("\r\n")
                if sentinel_prefix in stripped:
                    before, _, after = stripped.partition(sentinel_prefix)
                    if before.strip():
                        collected_lines.append(before.rstrip())
                    parts = after.lstrip("|").split("|", 1)
                    try:
                        exit_code = int(parts[0])
                    except Exception:
                        exit_code = 1
                    if len(parts) > 1 and parts[1].strip():
                        new_cwd = Path(parts[1].strip())
                        if new_cwd.exists():
                            self.current_cwd = new_cwd
                    break
                else:
                    collected_lines.append(stripped)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if timed_out:
                # Kill hung process tree and auto-recover in `self.current_cwd` with `self.tracked_env`
                self._start_process()
                exit_code = 124
                body = (
                    "\n".join(collected_lines).strip()
                    + f"\nERROR: Command timed out after {timeout:.1f}s. Session auto-recovered in `{self.current_cwd}`."
                ).strip()
            else:
                # Trim leading/trailing blank lines from Out-String -Stream
                while collected_lines and not collected_lines[0].strip():
                    collected_lines.pop(0)
                while collected_lines and not collected_lines[-1].strip():
                    collected_lines.pop()
                body = "\n".join(collected_lines)

            formatted = f"{body}\n[exit_code={exit_code}]" if body else f"[exit_code={exit_code}]"

            rec = TerminalCommandRecord(
                cmd_id=cmd_id,
                command=command,
                output=formatted,
                exit_code=exit_code,
                cwd=str(self.current_cwd),
                elapsed_ms=elapsed_ms,
                timed_out=timed_out,
                timestamp=time.time(),
            )
            self.history.append(rec)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            return formatted


class MCPTerminalServer:
    """
    Model Context Protocol (MCP) JSON-RPC 2.0 Server wrapping `PersistentPowerShellSession`.
    Exposes `run_powershell` for Feral's tool-calling loop and console telemetry for the UI.
    """

    def __init__(self, workspace_root: Path = ROOT, default_timeout_sec: float = 30.0):
        self.session = PersistentPowerShellSession(
            workspace_root=workspace_root,
            default_timeout_sec=default_timeout_sec,
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "run_powershell",
                "description": (
                    "Execute a command inside Feral's persistent Windows 11 PowerShell session. "
                    "Working directory (`cd` / `Set-Location`), `$env:` variables, and shell variables "
                    "persist across calls."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "PowerShell command or script block to execute.",
                        },
                        "timeout_sec": {
                            "type": "number",
                            "description": "Optional timeout in seconds (default: 30).",
                        },
                    },
                    "required": ["command"],
                },
            }
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name != "run_powershell":
            raise ValueError(f"MCPTerminalServer does not provide tool `{name}`")
        cmd = str(arguments["command"])
        timeout = arguments.get("timeout_sec")
        return self.session.run_powershell(cmd, timeout_sec=timeout)

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles standard MCP JSON-RPC 2.0 requests (`tools/list`, `tools/call`, `terminal/status`)."""
        req_id = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        try:
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}
            if method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {}) or {}
                out = self.call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": out}],
                        "isError": not out.endswith("[exit_code=0]"),
                        "cwd": str(self.session.current_cwd),
                    },
                }
            if method == "terminal/status":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "cwd": str(self.session.current_cwd),
                        "tracked_env": dict(self.session.tracked_env),
                        "history_count": len(self.session.history),
                        "recent_logs": [asdict(r) for r in self.session.history[-10:]],
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    def close(self) -> None:
        self.session.close()


def main():
    test_root = ROOT / "feral_ps_sandbox"
    if test_root.exists():
        shutil.rmtree(test_root)
    (test_root / "sub_module").mkdir(parents=True, exist_ok=True)
    (test_root / "sub_module" / "hello.py").write_text("print('Hello from stateful sub_module!')\n", encoding="utf-8")

    print("1. Spawning Persistent Stateful PowerShell MCP Server (`MCPTerminalServer`)...")
    server = MCPTerminalServer(workspace_root=test_root, default_timeout_sec=10.0)
    try:
        # Test 1: Stateful working directory (`Set-Location`) persistence across separate calls
        print("\n2. Verifying Stateful Working Directory (`cd`) Persistence Across Separate Tool Calls...")
        res_cd = server.call_tool("run_powershell", {"command": "Set-Location sub_module; (Get-Location).Path"})
        res_py = server.call_tool("run_powershell", {"command": "python hello.py"})
        print(f"   - Call 1 (`Set-Location sub_module`) : {res_cd.splitlines()[0]} ({res_cd.splitlines()[-1]})")
        print(f"   - Call 2 (`python hello.py` in new cwd): {res_py.splitlines()[0]} ({res_py.splitlines()[-1]})")
        assert "Hello from stateful sub_module!" in res_py and res_py.endswith("[exit_code=0]")

        # Test 2: Stateful `$env:` and PowerShell shell variable persistence across calls
        print("\n3. Verifying `$env:` & Shell Variable (`$worker_count`) Persistence Across Tool Calls...")
        server.call_tool("run_powershell", {"command": "$env:FERAL_MODE = 'autonomous'; $worker_count = 16"})
        res_vars = server.call_tool("run_powershell", {"command": "\"mode=$($env:FERAL_MODE), workers=$worker_count\""})
        print(f"   - Call 4 (Read variables set in Call 3): {res_vars.splitlines()[0]} ({res_vars.splitlines()[-1]})")
        assert "mode=autonomous, workers=16" in res_vars and res_vars.endswith("[exit_code=0]")

        # Test 3: Native PowerShell ErrorRecord capture (`grep` failure -> clean ERROR + `[exit_code=1]`)
        print("\n4. Verifying PowerShell Cmdlet Failure Unwrapping (`grep` -> `ERROR:` + `[exit_code=1]`)...")
        res_err = server.call_tool("run_powershell", {"command": "grep -rn 'foo' ."})
        print(f"   - Call 5 (`grep` error capture)        : {res_err.splitlines()[0]} ({res_err.splitlines()[-1]})")
        assert "ERROR:" in res_err and res_err.endswith("[exit_code=1]")

        # Test 4: Watchdog timeout + automatic session respawn with `$env:FERAL_MODE` and `cwd` preserved
        print("\n5. Verifying Watchdog Timeout (`timeout_sec=1.2`) & Automatic State Recovery...")
        res_timeout = server.call_tool(
            "run_powershell",
            {"command": "Write-Output 'Starting long task...'; Start-Sleep -Seconds 10", "timeout_sec": 1.2},
        )
        res_after = server.call_tool(
            "run_powershell",
            {"command": "\"recovered_cwd=$(Split-Path -Leaf (Get-Location).Path), env=$($env:FERAL_MODE)\""},
        )
        print(f"   - Call 6 (Hung command auto-killed)    : {res_timeout.splitlines()[-2]} ({res_timeout.splitlines()[-1]})")
        print(f"   - Call 7 (Post-recovery state check)   : {res_after.splitlines()[0]} ({res_after.splitlines()[-1]})")
        assert res_timeout.endswith("[exit_code=124]")
        assert "recovered_cwd=sub_module, env=autonomous" in res_after

        # Test 5: MCP JSON-RPC 2.0 `tools/call` & `terminal/status` rolling console telemetry
        print("\n6. Verifying MCP JSON-RPC 2.0 (`tools/call` & `terminal/status`) for Autonomous Agents Board...")
        rpc_resp = server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "terminal/status",
        })
        status = rpc_resp["result"]
        print(f"   - Recorded Console History Entries     : {status['history_count']} commands | Active CWD: {Path(status['cwd']).name}")

    finally:
        server.close()
        if test_root.exists():
            shutil.rmtree(test_root, ignore_errors=True)

    print("\nPHASE 5 STEP 2 (`mcp_terminal_server.py`): PASS -> Stateful PowerShell persistence, error unwrapping, and timeout auto-recovery verified!")


if __name__ == "__main__":
    main()
