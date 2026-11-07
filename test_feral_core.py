"""
Phase 5, Step 6 - Unified Feral Multi-Server MCP Host & End-to-End Integration Test (`test_feral_core.py`)
==========================================================================================================
Unifies `FeralLLMClient` (`Qwen3.5-9B` + GRPO DoRA + Fast Norm Fusion) with all 4 MCP servers
into `FeralMCPHost` and runs an end-to-end multi-file coding, debugging, and verification suite
on RTX 5070 Ti (`cuda:0`):

  - Server 1 (`MCPTerminalServer`)      -> `run_powershell` (stateful PowerShell session)
  - Server 2 (`MCPCodeEditorServer`)    -> `ast_repo_map`, `read_file_lines`, `apply_diff_patch`, `write_file`
  - Server 3 (`MCPBrowserVisionServer`) -> `web_research` (docs index, HTML->Markdown, headless Chrome vision)
  - Server 4 (`MCPSubagentServer`)      -> `spawn_subagent` (isolated worker + Autonomous Agents Board)

End-to-End Verification Missions:
  - Mission A: Autonomous Sub-Agent Repo Audit (`spawn_subagent`) + Documentation Lookup (`web_research`)
  - Mission B: Multi-File Bug Diagnosis & Surgical Patch (`read_file_lines` -> `apply_diff_patch` -> `run_powershell` unit test pass)
"""
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from feral_core import ROOT, TOOL_SCHEMA
from feral_llm_client import FeralLLMClient, list_available_local_models
from mcp_browser_vision import MCPBrowserVisionServer
from mcp_code_editor import MCPCodeEditorServer
from mcp_subagent_worker import MCPSubagentServer
from mcp_terminal_server import MCPTerminalServer


class FeralMCPHost:
    """
    Central Multi-Server MCP Host routing all 7 Feral tools across the 4 specialized MCP servers
    and aggregating real-time telemetry for the Phase 7 Desktop Command Center UI.
    """

    def __init__(
        self,
        workspace_root: Path = ROOT,
        llm_client: Optional[FeralLLMClient] = None,
        load_llm: bool = True,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.llm: Optional[FeralLLMClient] = llm_client
        if self.llm is None and load_llm:
            self.llm = FeralLLMClient()

        self.terminal = MCPTerminalServer(workspace_root=self.workspace_root)
        self.editor = MCPCodeEditorServer(workspace_root=self.workspace_root)
        self.browser = MCPBrowserVisionServer(screenshot_dir=self.workspace_root / ".feral_vision")
        self.subagents = MCPSubagentServer(
            llm_client=self.llm,
            tool_router=self.dispatch_tool,
            max_subagent_steps=4,
        )
        self.tool_call_log: List[Dict[str, Any]] = []

    def list_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """Returns the combined MCP tool schemas across all 4 servers (7 tools total)."""
        return (
            self.terminal.list_tools()
            + self.editor.list_tools()
            + self.browser.list_tools()
            + self.subagents.list_tools()
        )

    def dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Routes any of Feral's 7 tools to its owning MCP server and records execution telemetry."""
        t0 = time.perf_counter()
        if name == "run_powershell":
            out = self.terminal.call_tool(name, arguments)
            server_name = "MCPTerminalServer"
        elif name in ("ast_repo_map", "read_file_lines", "apply_diff_patch", "write_file"):
            out = self.editor.call_tool(name, arguments)
            server_name = "MCPCodeEditorServer"
        elif name == "web_research":
            out = self.browser.call_tool(name, arguments)
            server_name = "MCPBrowserVisionServer"
        elif name == "spawn_subagent":
            out = self.subagents.call_tool(name, arguments)
            server_name = "MCPSubagentServer"
        else:
            raise ValueError(f"Unknown tool `{name}` requested from FeralMCPHost")

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.tool_call_log.append(
            {
                "call_id": len(self.tool_call_log) + 1,
                "server": server_name,
                "tool": name,
                "arguments": arguments,
                "exit_ok": out.endswith("[exit_code=0]"),
                "elapsed_ms": elapsed_ms,
                "timestamp": time.time(),
            }
        )
        return out

    def run_autonomous_mission(self, prompt: str, max_steps: int = 6) -> Dict[str, Any]:
        """Executes a multi-turn ReAct mission through `FeralLLMClient` and the 4 MCP servers."""
        if self.llm is None:
            raise RuntimeError("FeralLLMClient is not loaded on this FeralMCPHost")
        return self.llm.run_react_loop(
            user_prompt=prompt,
            tool_executor=self.dispatch_tool,
            max_steps=max_steps,
        )

    def get_command_center_snapshot(self) -> Dict[str, Any]:
        """Returns a unified telemetry snapshot for the Phase 7 Desktop Command Center UI."""
        vram_alloc_mib = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        vram_res_mib = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1) if torch.cuda.is_available() else 0.0
        return {
            "context_panel": asdict(self.llm.config_state) if self.llm else {},
            "available_models": list_available_local_models(),
            "agents_board": [asdict(c) for c in self.subagents.agent_cards.values()],
            "diff_ledger": [asdict(d) for d in self.editor.diff_history],
            "terminal_history": [asdict(r) for r in self.terminal.session.history[-20:]],
            "vision_frames": [asdict(f) for f in self.browser.vision_frames[-5:]],
            "integrations": asdict(self.browser.config),
            "gpu_telemetry": {
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "vram_allocated_mib": vram_alloc_mib,
                "vram_reserved_mib": vram_res_mib,
            },
        }

    def close(self) -> None:
        self.terminal.close()


def seed_broken_checkout_repo(workspace: Path) -> None:
    """Creates a multi-file Python checkout microservice with a realistic discount calculation bug."""
    if workspace.exists():
        shutil.rmtree(workspace)
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)

    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "src" / "discounts.py").write_text(
        "def apply_coupon(price: float, percent_off: float) -> float:\n"
        "    # BUG: returns the discount amount instead of subtracting it from price\n"
        "    return round(price * (percent_off / 100.0), 2)\n",
        encoding="utf-8",
    )
    (workspace / "src" / "checkout.py").write_text(
        "from src.discounts import apply_coupon\n\n\n"
        "def checkout_total(subtotal: float, coupon_pct: float = 0.0) -> float:\n"
        "    if coupon_pct > 0:\n"
        "        return apply_coupon(subtotal, coupon_pct)\n"
        "    return round(subtotal, 2)\n",
        encoding="utf-8",
    )
    (workspace / "tests" / "test_checkout.py").write_text(
        "import unittest\n"
        "from src.checkout import checkout_total\n\n\n"
        "class TestCheckout(unittest.TestCase):\n"
        "    def test_no_coupon(self):\n"
        "        self.assertEqual(checkout_total(100.0, 0), 100.0)\n\n"
        "    def test_twenty_percent_coupon(self):\n"
        "        self.assertEqual(checkout_total(200.0, 20), 160.0)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )


def main():
    workspace = ROOT / "feral_sandbox"
    seed_broken_checkout_repo(workspace)

    print("1. Booting Unified `FeralMCPHost` (4 MCP Servers + `FeralLLMClient` on RTX 5070 Ti `cuda:0`)...")
    host = FeralMCPHost(workspace_root=workspace, load_llm=True)
    try:
        tools = host.list_all_mcp_tools()
        tool_names = [t["name"] for t in tools]
        print(f"   - Registered MCP Tools ({len(tools)}/7) : {tool_names}")
        assert set(tool_names) == set(TOOL_SCHEMA.keys())

        # Verify that unit tests initially FAIL before Feral touches the repo
        print("\n2. Baseline Pre-Fix Verification (`python -m unittest discover -s tests`)...")
        pre_test = host.dispatch_tool("run_powershell", {"command": "python -m unittest discover -s tests"})
        print(f"   - Pre-Fix Test Output           : {pre_test.splitlines()[-2]} ({pre_test.splitlines()[-1]})")
        assert pre_test.endswith("[exit_code=1]") and "FAILED" in pre_test

        # Mission A: Delegate repo mapping to an isolated sub-agent on the Autonomous Agents Board
        print("\n3. Mission A - Delegating Repo Audit via `spawn_subagent` on Autonomous Agents Board...")
        sub_out = host.dispatch_tool(
            "spawn_subagent",
            {
                "task_prompt": "Map the Python files in src and list the functions defined.",
                "agent_name": "Checkout Repo Mapper",
            },
        )
        sub_summary = " ".join(l.strip() for l in sub_out.splitlines()[:-1])
        print(f"   - Sub-Agent Report              : {sub_summary}")
        assert "apply_coupon" in sub_out and "checkout_total" in sub_out

        # Mission B: Autonomous Multi-Turn Bug Diagnosis, Surgical Diff Patch & Unit-Test Verification
        print("\n4. Mission B - Autonomous Multi-Turn Bug Fix & Unit-Test Verification on `src/discounts.py`...")
        mission_prompt = (
            "`src/discounts.py` has a bug in `apply_coupon(price, percent_off)`: it returns "
            "`round(price * (percent_off / 100.0), 2)` instead of `round(price * (1.0 - percent_off / 100.0), 2)`. "
            "Read `src/discounts.py`, patch it with `apply_diff_patch`, and run `python -m unittest discover -s tests` "
            "with `run_powershell` to confirm all tests pass."
        )
        mission_res = host.run_autonomous_mission(mission_prompt, max_steps=6)
        print(f"   - Mission Status                : {mission_res['status']} in {len(mission_res['steps'])} turns")
        for s in mission_res["steps"]:
            if s["tool_call"]:
                obs_first = s.get("tool_observation", "").splitlines()[0] if s.get("tool_observation") else ""
                print(
                    f"     * Turn {s['step']} ({s['tokens_per_sec']:5.2f} tok/s) -> "
                    f"{s['tool_call']['name']}({json.dumps(s['tool_call']['arguments'])}) | Obs: {obs_first}"
                )
            else:
                print(f"     * Turn {s['step']} ({s['tokens_per_sec']:5.2f} tok/s) -> Final Answer: {s['final_answer']}")

        # Confirm post-mission unit tests pass 100% in the real workspace
        post_test = host.dispatch_tool("run_powershell", {"command": "python -m unittest discover -s tests"})
        print(f"\n5. Post-Fix Unit Test Verification : {post_test.splitlines()[-2]} ({post_test.splitlines()[-1]})")
        assert post_test.endswith("[exit_code=0]") and "OK" in post_test
        assert len(host.editor.diff_history) >= 1

        snap = host.get_command_center_snapshot()
        print(
            f"   - Command Center Snapshot       : {len(snap['agents_board'])} agent cards | "
            f"{len(snap['diff_ledger'])} recorded diffs | "
            f"{len(snap['terminal_history'])} terminal logs | "
            f"GPU: {snap['gpu_telemetry']['device_name']} ({snap['gpu_telemetry']['vram_allocated_mib']} MiB)"
        )

    finally:
        host.close()
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

    print("\nPHASE 5 STEP 6 (`test_feral_core.py`): PASS -> Full Feral Multi-Server MCP Host & end-to-end bug fix verified!")


if __name__ == "__main__":
    main()
