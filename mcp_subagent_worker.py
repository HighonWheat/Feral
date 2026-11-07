"""
Phase 5, Step 5 - Isolated Sub-Agent Worker & Autonomous Agents Board (`mcp_subagent_worker.py`)
================================================================================================
Implements Feral's `spawn_subagent(task_prompt)` tool and the backend orchestrator for the
Phase 7 Odysseus-inspired **Autonomous Agents Board**:

  1. Isolated Context Window Execution (`spawn_subagent(task_prompt)`):
     Spawns a fresh child conversation context (`child_messages`) that reuses the already-loaded
     `FeralLLMClient` on `cuda:0` (0 MiB extra model VRAM) and dispatches tool calls to the
     workspace MCP servers (`MCPCodeEditorServer`, `MCPTerminalServer`, `MCPBrowserVisionServer`)
     while blocking recursive `spawn_subagent` calls.
  2. Context-Window Compression:
     All intermediate child `<think>`, `<tool_call>`, and `<tool_response>` turns stay inside
     the isolated worker. Only the concise final report (`SUBAGENT REPORT: ...`) is returned to
     the parent agent, preventing parent KV-cache bloat.
  3. Autonomous Agents Board State (`AgentCardState`):
     Tracks every deployed agent card on the grid dashboard with:
       - `agent_id`, `name`, `goal`, `status` (`Idle`, `Running`, `Paused`, `Completed`, `Failed`)
       - `grid_position` (`{"col": int, "row": int}`) for drag-and-drop / grid orchestration
       - `console_log`: real-time rolling console lines (`[THINK]`, `[TOOL]`, `[OBS]`, `[RESULT]`)
       - `child_context_tokens` vs `parent_summary_tokens` compression telemetry.
"""
import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from feral_core import ROOT, SYSTEM_FERAL


@dataclass
class AgentCardState:
    """State model for each card on the Phase 7 Autonomous Agents Board."""
    agent_id: str
    name: str
    goal: str
    status: str = "Idle"  # Idle | Running | Paused | Completed | Failed
    grid_col: int = 0
    grid_row: int = 0
    steps_completed: int = 0
    max_steps: int = 4
    child_context_tokens: int = 0
    parent_summary_tokens: int = 0
    final_report: str = ""
    console_log: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def append_log(self, tag: str, message: str, max_lines: int = 120) -> None:
        ts = time.strftime("%H:%M:%S")
        clean = message.strip().replace("\r\n", "\n")
        for line in clean.splitlines() or [""]:
            self.console_log.append(f"[{ts}] [{tag}] {line[:180]}")
        if len(self.console_log) > max_lines:
            self.console_log = self.console_log[-max_lines:]
        self.updated_at = time.time()


class MCPSubagentServer:
    """
    Isolated Sub-Agent Worker & Autonomous Agents Board MCP Server.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_router: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        max_subagent_steps: int = 4,
    ):
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.max_subagent_steps = max_subagent_steps

        self.agent_cards: Dict[str, AgentCardState] = {}
        self._pause_events: Dict[str, threading.Event] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def create_agent_card(
        self,
        name: str,
        goal: str,
        grid_col: Optional[int] = None,
        grid_row: Optional[int] = None,
    ) -> AgentCardState:
        """Creates a new card on the Autonomous Agents Board in `Idle` status."""
        with self._lock:
            self._counter += 1
            aid = f"agent-{self._counter:02d}"
            idx = self._counter - 1
            col = (idx % 3) if grid_col is None else int(grid_col)
            row = (idx // 3) if grid_row is None else int(grid_row)

            card = AgentCardState(
                agent_id=aid,
                name=name,
                goal=goal,
                status="Idle",
                grid_col=col,
                grid_row=row,
                max_steps=self.max_subagent_steps,
            )
            card.append_log("INIT", f"Agent card `{name}` initialized with goal: {goal}")
            self.agent_cards[aid] = card

            run_gate = threading.Event()
            run_gate.set()  # Set = allowed to run; Cleared = Paused
            self._pause_events[aid] = run_gate
            return card

    def set_agent_status(self, agent_id: str, action: str) -> AgentCardState:
        """Controls an agent card from the Autonomous Agents Board (`pause`, `resume`)."""
        if agent_id not in self.agent_cards:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        card = self.agent_cards[agent_id]
        gate = self._pause_events[agent_id]

        if action == "pause":
            gate.clear()
            card.status = "Paused"
            card.append_log("CTRL", "Execution paused by user from Autonomous Agents Board.")
        elif action == "resume":
            if card.status == "Paused":
                card.status = "Running"
            gate.set()
            card.append_log("CTRL", "Execution resumed from Autonomous Agents Board.")
        else:
            raise ValueError(f"Unsupported control action: {action}")
        return card

    def move_agent_card(self, agent_id: str, grid_col: int, grid_row: int) -> AgentCardState:
        """Updates a card's grid coordinates after drag-and-drop on the Autonomous Agents Board."""
        if agent_id not in self.agent_cards:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        card = self.agent_cards[agent_id]
        card.grid_col = int(grid_col)
        card.grid_row = int(grid_row)
        card.append_log("GRID", f"Moved card to grid slot (col={card.grid_col}, row={card.grid_row})")
        return card

    def _estimate_tokens(self, text: str) -> int:
        if self.llm_client is not None and hasattr(self.llm_client, "tokenizer"):
            return len(self.llm_client.tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text.split()))

    def spawn_subagent(
        self,
        task_prompt: str,
        agent_name: str = "Sub-Agent Worker",
        existing_agent_id: Optional[str] = None,
    ) -> str:
        """
        Executes `task_prompt` inside a fresh, isolated child context window.
        Streams every `<think>`, `<tool_call>`, and `<tool_response>` into the card's
        rolling `console_log` and returns ONLY the concise final report to the parent.
        """
        if existing_agent_id and existing_agent_id in self.agent_cards:
            card = self.agent_cards[existing_agent_id]
            card.goal = task_prompt
        else:
            card = self.create_agent_card(name=agent_name, goal=task_prompt)

        gate = self._pause_events[card.agent_id]
        card.status = "Running"
        card.append_log("START", f"Launching isolated sub-agent context for: {task_prompt}")

        if self.llm_client is None or self.tool_router is None:
            card.status = "Failed"
            card.append_log("ERROR", "No LLM client or tool router attached to MCPSubagentServer.")
            return "ERROR: Sub-agent worker is not bound to an LLM client or tool router.\n[exit_code=1]"

        child_messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_FERAL},
            {"role": "user", "content": task_prompt},
        ]

        final_answer = ""
        try:
            for step_idx in range(1, self.max_subagent_steps + 1):
                # Honor Pause toggle from Autonomous Agents Board
                while not gate.is_set():
                    time.sleep(0.1)

                turn = self.llm_client.generate_turn(child_messages)
                child_messages.append({"role": "assistant", "content": turn.raw_text})
                card.steps_completed = step_idx

                if turn.think:
                    card.append_log("THINK", turn.think)

                if not turn.is_valid_tool_call or turn.tool_call is None:
                    final_answer = turn.final_answer or turn.raw_text
                    card.append_log("RESULT", final_answer)
                    card.status = "Completed"
                    break

                t_name = turn.tool_call["name"]
                t_args = turn.tool_call["arguments"]
                card.append_log("TOOL", f"{t_name}({json.dumps(t_args)})")

                # Block recursive sub-agent spawning inside a child worker
                if t_name == "spawn_subagent":
                    obs = "ERROR: Nested `spawn_subagent` calls are disabled inside a sub-agent worker. Complete the task directly.\n[exit_code=1]"
                else:
                    obs = self.tool_router(t_name, t_args)

                first_obs_line = obs.splitlines()[0] if obs else ""
                card.append_log("OBS", f"{first_obs_line} ({len(obs.splitlines())} lines)")
                child_messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})

            if not final_answer:
                final_answer = f"Completed {card.steps_completed} steps (see card {card.agent_id} log)."
                card.status = "Completed"

        except Exception as e:
            card.status = "Failed"
            card.append_log("ERROR", str(e))
            return f"ERROR: Sub-agent `{card.agent_id}` failed: {e}\n[exit_code=1]"

        # Measure context compression: full child history vs concise report returned to parent
        full_child_text = "\n".join(m["content"] for m in child_messages)
        report_body = f"SUBAGENT REPORT [{card.agent_id} | {card.name}]: {final_answer}"
        card.child_context_tokens = self._estimate_tokens(full_child_text)
        card.parent_summary_tokens = self._estimate_tokens(report_body)
        card.final_report = final_answer
        card.append_log(
            "DONE",
            f"Compressed {card.child_context_tokens} child tokens -> {card.parent_summary_tokens} parent tokens.",
        )

        return f"{report_body}\n[exit_code=0]"

    # ---------------- MCP JSON-RPC 2.0 Interface ----------------

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "spawn_subagent",
                "description": (
                    "Delegate an isolated subtask to a fresh-context worker on the Autonomous Agents Board; "
                    "returns only a concise report so the parent context window stays clean."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_prompt": {"type": "string"},
                        "agent_name": {"type": "string"},
                    },
                    "required": ["task_prompt"],
                },
            }
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name != "spawn_subagent":
            raise ValueError(f"MCPSubagentServer does not provide tool `{name}`")
        prompt = str(arguments["task_prompt"])
        agent_name = str(arguments.get("agent_name", "Sub-Agent Worker"))
        return self.spawn_subagent(task_prompt=prompt, agent_name=agent_name)

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        try:
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}
            if method == "tools/call":
                out = self.call_tool(params.get("name", ""), params.get("arguments", {}) or {})
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": out}],
                        "isError": not out.endswith("[exit_code=0]"),
                    },
                }
            if method == "subagent/board_state":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "cards": [asdict(c) for c in self.agent_cards.values()],
                    },
                }
            if method == "subagent/control":
                aid = str(params["agent_id"])
                act = str(params["action"])
                card = self.set_agent_status(aid, act)
                return {"jsonrpc": "2.0", "id": req_id, "result": asdict(card)}
            if method == "subagent/move_card":
                card = self.move_agent_card(
                    str(params["agent_id"]),
                    int(params["grid_col"]),
                    int(params["grid_row"]),
                )
                return {"jsonrpc": "2.0", "id": req_id, "result": asdict(card)}
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


def main():
    from feral_llm_client import FeralLLMClient
    from mcp_code_editor import MCPCodeEditorServer
    from mcp_terminal_server import MCPTerminalServer

    test_root = ROOT / "feral_sandbox"
    if test_root.exists():
        shutil.rmtree(test_root)
    (test_root / "src").mkdir(parents=True, exist_ok=True)
    (test_root / "src" / "inventory.py").write_text(
        "class Warehouse:\n"
        "    def __init__(self):\n"
        "        self.items = {}\n\n"
        "    def add_stock(self, sku: str, qty: int) -> int:\n"
        "        self.items[sku] = self.items.get(sku, 0) + qty\n"
        "        return self.items[sku]\n\n"
        "    def low_stock_skus(self, threshold: int = 5) -> list:\n"
        "        return [k for k, v in self.items.items() if v < threshold]\n",
        encoding="utf-8",
    )

    print("1. Loading `FeralLLMClient` + Workspace MCP Servers (`MCPCodeEditorServer`, `MCPTerminalServer`)...")
    llm = FeralLLMClient()
    editor = MCPCodeEditorServer(workspace_root=test_root)
    terminal = MCPTerminalServer(workspace_root=test_root)

    try:
        def route_workspace_tool(name: str, args: Dict[str, Any]) -> str:
            if name in ("ast_repo_map", "read_file_lines", "apply_diff_patch", "write_file"):
                return editor.call_tool(name, args)
            if name == "run_powershell":
                return terminal.call_tool(name, args)
            raise ValueError(f"Unsupported tool: {name}")

        subagent_srv = MCPSubagentServer(llm_client=llm, tool_router=route_workspace_tool, max_subagent_steps=4)

        print("\n2. Testing Autonomous Agents Board Card Creation, Pause/Resume & Drag-and-Drop Grid Positioning...")
        card_a = subagent_srv.create_agent_card(
            name="Repo Structure Auditor",
            goal="Map the Python classes and functions in src and summarize what Warehouse does.",
            grid_col=0,
            grid_row=0,
        )
        subagent_srv.set_agent_status(card_a.agent_id, "pause")
        assert card_a.status == "Paused"
        subagent_srv.set_agent_status(card_a.agent_id, "resume")
        subagent_srv.move_agent_card(card_a.agent_id, grid_col=2, grid_row=1)
        print(f"   - Card `{card_a.agent_id}` ({card_a.name}) | Grid Slot: ({card_a.grid_col}, {card_a.grid_row}) | Pre-Run Status: {card_a.status}")

        print("\n3. Running Live Isolated Sub-Agent Worker (`spawn_subagent`) on RTX 5070 Ti (`cuda:0`)...")
        report = subagent_srv.spawn_subagent(
            task_prompt="Map the Python symbols in src and report the methods of Warehouse.",
            agent_name="Repo Structure Auditor",
            existing_agent_id=card_a.agent_id,
        )
        print(f"   - Parent Report Returned        : {report.splitlines()[0]}")
        print(f"   - Card Final Status             : {card_a.status} ({card_a.steps_completed} steps)")
        print(
            f"   - Context Window Compression    : {card_a.child_context_tokens} child tokens -> "
            f"{card_a.parent_summary_tokens} parent tokens "
            f"({card_a.child_context_tokens / max(card_a.parent_summary_tokens, 1):.2f}x compression)"
        )
        print("   - Rolling Console Log (`card.console_log`):")
        for log_line in card_a.console_log:
            print(f"     | {log_line}")

        assert card_a.status == "Completed" and report.endswith("[exit_code=0]")
        assert card_a.child_context_tokens > card_a.parent_summary_tokens

    finally:
        terminal.close()
        if test_root.exists():
            shutil.rmtree(test_root, ignore_errors=True)

    print("\nPHASE 5 STEP 5 (`mcp_subagent_worker.py`): PASS -> Isolated sub-agent execution, context compression, and Autonomous Agents Board verified!")


if __name__ == "__main__":
    main()
