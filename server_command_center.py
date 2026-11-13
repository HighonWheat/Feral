"""
Phase 7, Step 1 - Feral Desktop Command Center FastAPI + WebSocket Backend (`server_command_center.py`)
=======================================================================================================
Unites all Phase 5 & Phase 6 engines into a single real-time FastAPI + WebSocket backend powering
the Odysseus-inspired Feral Desktop Command Center UI:

  1. Module 1 — The Chat Interface (Main Hub) & Collapsible Right "Context Panel":
     - `/api/chat`, `/api/context`, `/ws` (real-time token streaming via `TextIteratorStreamer`,
       collapsible `<think>` trace, `<tool_call>` badges, system prompt editor, temperature slider,
       max tokens config, and local model selector).
  2. Module 2 — The Autonomous Agents Board & Hierarchical Task DAG:
     - `/api/agents/create`, `/api/agents/{id}/deploy`, `/api/agents/{id}/control`, `/api/agents/{id}/move`,
       `/api/dag/node` (grid-based agent cards with `Idle`, `Running`, `Paused` controls and rolling
       real-time `[THINK]` / `[TOOL]` / `[OBS]` console logs).
  3. Module 3 — Knowledge Base & Vector Storage UI (ChromaDB / SQLite Vector Store):
     - `/api/kb/collections`, `/api/kb/upload`, `/api/kb/query` (drag-and-drop `.txt`, `.md`, `.pdf`
       ingestion into Knowledge Clusters with live `0% -> 100%` embedding progress bars and 384-dim
       cosine similarity search).
  4. Module 4 — Integrations Panel, Hardware Cookbook & Developer Workbenches:
     - `/api/integrations` (toggles & endpoints for SearXNG web search, local IMAP/SMTP email,
       CalDAV calendar sync, and self-healing watchdogs), live RTX 5070 Ti GPU/VRAM telemetry,
       Unified Diff Ledger with 1-click patch/Git rollback, Persistent PowerShell terminal, and
       Live Headless Chromium Vision Feed.
"""
import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from context_compactor import ContextCompactor, LocalVectorKnowledgeBase, build_minimal_pdf
from dag_orchestrator import DAGOrchestrator
from feral_core import ROOT, SYSTEM_FERAL
from feral_llm_client import (
    FeralLLMClient,
    GenerationConfigState,
    list_available_local_models,
    parse_feral_output,
)
from git_time_travel import GitTimeTravelManager
from loop_watchdog import LoopWatchdog
from test_feral_core import FeralMCPHost, seed_broken_checkout_repo

UI_DIR = ROOT / "ui"
SANDBOX_WORKSPACE = ROOT / "outputs" / "command_center_workspace"
DAG_DB_PATH = ROOT / "outputs" / "command_center_dag.sqlite"
KB_DB_PATH = ROOT / "outputs" / "command_center_kb.sqlite"


# ============================================================================
# 1. Integrations Panel & Hardware-Aware Cookbook Models (Module 4)
# ============================================================================
@dataclass
class IntegrationsPanelState:
    """State for Module 4: The Odysseus-inspired Integrations & Extensions Panel."""
    # Web Search Providers (SearXNG + Headless Chromium Vision)
    searxng_enabled: bool = False
    searxng_endpoint: str = "http://localhost:8080/search"
    headless_browser_enabled: bool = True
    vision_screenshots_enabled: bool = True

    # Local Email Client (IMAP / SMTP Bridge)
    email_enabled: bool = False
    imap_host: str = "127.0.0.1"
    imap_port: int = 1143
    smtp_host: str = "127.0.0.1"
    smtp_port: int = 1025
    email_account: str = "feral-agent@localhost.local"

    # Calendar Sync (CalDAV / Local ICS)
    calendar_sync_enabled: bool = False
    caldav_endpoint: str = "http://localhost:5232/feral/calendar.ics"
    calendar_poll_minutes: int = 15

    # Phase 6 Self-Healing Reliability Engines
    loop_watchdog_enabled: bool = True
    git_time_travel_enabled: bool = True
    auto_compaction_enabled: bool = True


def get_hardware_cookbook_telemetry() -> Dict[str, Any]:
    """
    Queries live NVIDIA RTX 5070 Ti (`cuda:0`) telemetry via PyTorch + `nvidia-smi`
    and computes Hardware-Aware Cookbook recommendations for local model execution.
    """
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    cap = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
    total_vram_mib = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1) if has_cuda else 0.0
    alloc_vram_mib = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1) if has_cuda else 0.0
    reserved_vram_mib = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1) if has_cuda else 0.0
    free_vram_mib = round(max(0.0, total_vram_mib - reserved_vram_mib), 1)

    gpu_util_pct = 0.0
    gpu_temp_c = 0.0
    gpu_power_w = 0.0
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = [p.strip() for p in smi.stdout.splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_util_pct = float(parts[0])
                gpu_temp_c = float(parts[1])
                gpu_power_w = float(parts[2])
    except Exception:
        pass

    gguf_path = ROOT / "outputs" / "feral-qwen3.5-9b-grpo-q4_k_m.gguf"
    gguf_size_mib = round(gguf_path.stat().st_size / (1024 ** 2), 1) if gguf_path.exists() else 3964.0

    return {
        "device_name": device_name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}" if has_cuda else "cpu",
        "architecture": "NVIDIA Blackwell" if cap[0] >= 12 else "CUDA GPU",
        "total_vram_mib": total_vram_mib,
        "vram_allocated_mib": alloc_vram_mib,
        "vram_reserved_mib": reserved_vram_mib,
        "vram_free_mib": free_vram_mib,
        "vram_utilization_pct": round((alloc_vram_mib / max(total_vram_mib, 1.0)) * 100.0, 1),
        "gpu_core_utilization_pct": gpu_util_pct,
        "gpu_temp_celsius": gpu_temp_c,
        "gpu_power_watts": gpu_power_w,
        "cookbook_recommendations": {
            "active_precision_recipe": "4-bit NF4 + DoubleQuant + BF16 Compute + Fast DoRA Norm Fusion (7.29x speedup)",
            "max_safe_context_tokens": 8192,
            "compactor_trigger_tokens": 3072,
            "max_zero_vram_subagents": 4,
            "gguf_artifact": {
                "filename": gguf_path.name,
                "size_mib": gguf_size_mib,
                "quantization": "Q4_K_M + Q8_0 Sensitive Projections",
                "ready": gguf_path.exists(),
            },
        },
    }


# ============================================================================
# 2. Request / Payload Schemas
# ============================================================================
class ContextPanelUpdateRequest(BaseModel):
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None


class ChatRequest(BaseModel):
    message: str
    max_steps: int = 5
    reset_session: bool = False


class CreateAgentCardRequest(BaseModel):
    name: str
    goal: str
    grid_col: Optional[int] = None
    grid_row: Optional[int] = None


class AgentControlRequest(BaseModel):
    action: str  # "pause" | "resume"


class AgentMoveRequest(BaseModel):
    grid_col: int
    grid_row: int


class CreateDAGNodeRequest(BaseModel):
    mission_id: str = "mission-default"
    node_id: str
    title: str
    prompt: str
    assigned_agent: str = "Feral-Core"
    depends_on: List[str] = []
    priority: int = 10


class CreateKBCollectionRequest(BaseModel):
    collection_id: str
    name: str
    description: str = ""


class KBJsonUploadRequest(BaseModel):
    collection_id: str
    filename: str
    content_text: Optional[str] = None
    content_base64: Optional[str] = None


class KBQueryRequest(BaseModel):
    collection_id: str
    query: str
    top_k: int = 3


class IntegrationsUpdateRequest(BaseModel):
    searxng_enabled: Optional[bool] = None
    searxng_endpoint: Optional[str] = None
    headless_browser_enabled: Optional[bool] = None
    vision_screenshots_enabled: Optional[bool] = None
    email_enabled: Optional[bool] = None
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    email_account: Optional[str] = None
    calendar_sync_enabled: Optional[bool] = None
    caldav_endpoint: Optional[str] = None
    calendar_poll_minutes: Optional[int] = None
    loop_watchdog_enabled: Optional[bool] = None
    git_time_travel_enabled: Optional[bool] = None
    auto_compaction_enabled: Optional[bool] = None


class TerminalCommandRequest(BaseModel):
    command: str


class VisionCaptureRequest(BaseModel):
    query: str = "local_preview"
    url: str = ""


class RollbackPatchRequest(BaseModel):
    patch_id: int


class RollbackGitRequest(BaseModel):
    checkpoint_id_or_sha: Any


# ============================================================================
# 3. Unified Command Center Orchestrator (`FeralCommandCenter`)
# ============================================================================
class FeralCommandCenter:
    """
    Central stateful controller for the Phase 7 Desktop Command Center UI.
    Coordinates `FeralMCPHost`, `DAGOrchestrator`, `ContextCompactor`,
    `LocalVectorKnowledgeBase`, `GitTimeTravelManager`, and `LoopWatchdog`.
    """

    def __init__(
        self,
        workspace_root: Path = SANDBOX_WORKSPACE,
        llm_client: Optional[FeralLLMClient] = None,
        load_llm: bool = True,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.exists() or not (self.workspace_root / "src").exists():
            seed_broken_checkout_repo(self.workspace_root)

        self.host = FeralMCPHost(
            workspace_root=self.workspace_root,
            llm_client=llm_client,
            load_llm=load_llm,
        )
        self.integrations = IntegrationsPanelState()
        self.dag = DAGOrchestrator(db_path=DAG_DB_PATH)
        self.kb = LocalVectorKnowledgeBase(db_path=KB_DB_PATH, embedding_dim=384)
        self.compactor = ContextCompactor(
            workspace_root=self.workspace_root,
            max_context_tokens=4096,
            trigger_ratio=0.75,
            keep_recent_messages=4,
            token_counter=(
                (lambda txt: len(self.host.llm.tokenizer.encode(txt, add_special_tokens=False)))
                if self.host.llm is not None
                else None
            ),
        )
        self.time_travel = GitTimeTravelManager(workspace_root=self.workspace_root, max_strikes=3)
        self.watchdog = LoopWatchdog(
            repeat_threshold=2,
            base_temperature=0.0,
            bump_step=0.45,
        )

        self.active_mission_id = "mission-default"
        self.chat_messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_FERAL}
        ]
        self.chat_ui_log: List[Dict[str, Any]] = []
        self._inference_lock = threading.Lock()
        self._ws_clients: Set[WebSocket] = set()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        self._seed_initial_state()

    def _seed_initial_state(self) -> None:
        """Seeds initial checkpoints, Autonomous Agents Board cards, DAG nodes, and Vector Collections."""
        if not self.time_travel.checkpoints:
            self.time_travel.create_checkpoint("Initial workspace baseline", verified_green=True)

        # Seed default Autonomous Agents Board cards if empty
        if not self.host.subagents.agent_cards:
            c1 = self.host.subagents.create_agent_card(
                name="Repo Architect",
                goal="Map Python AST symbols across `src/` and `tests/` and identify bug hotspots.",
                grid_col=0,
                grid_row=0,
            )
            c1.append_log("READY", "Awaiting deployment on `src/discounts.py` and `src/checkout.py`.")

            c2 = self.host.subagents.create_agent_card(
                name="Patch Surgeon",
                goal="Inspect failing unit tests, apply surgical diff patches, and verify exit_code=0.",
                grid_col=1,
                grid_row=0,
            )
            c2.append_log("READY", "Configured with pre-commit `ast.parse` syntax guard & 3-strike Git rollback.")

            c3 = self.host.subagents.create_agent_card(
                name="Research & Docs Scout",
                goal="Index technical specs into ChromaDB/SQLite collections and capture headless Chrome vision.",
                grid_col=2,
                grid_row=0,
            )
            c3.append_log("READY", "Connected to local vector knowledge base (384-dim L2 cosine retrieval).")

        # Seed default Mission DAG if empty
        try:
            graph = self.dag.get_dag_snapshot(self.active_mission_id)
        except Exception:
            graph = {"nodes": []}

        if not graph.get("nodes"):
            self.dag.create_mission(
                self.active_mission_id,
                goal="Diagnose, patch, and verify checkout discount microservice",
            )
            self.dag.add_node(
                self.active_mission_id,
                node_id="step-1-map",
                title="AST Repository Mapping",
                prompt="Run `ast_repo_map` on workspace to locate discount logic.",
                assigned_agent="Repo Architect",
                priority=1,
            )
            self.dag.add_node(
                self.active_mission_id,
                node_id="step-2-patch",
                title="Surgical Diff Patch on `src/discounts.py`",
                prompt="Fix `apply_coupon` in `src/discounts.py` so discount is subtracted from price.",
                assigned_agent="Patch Surgeon",
                depends_on=["step-1-map"],
                priority=2,
            )
            self.dag.add_node(
                self.active_mission_id,
                node_id="step-3-verify",
                title="Unit Test & Green Git Checkpoint",
                prompt="Run `python -m unittest discover -s tests` and snapshot green commit.",
                assigned_agent="Patch Surgeon",
                depends_on=["step-2-patch"],
                priority=3,
            )

        # Seed default Knowledge Base Collection if empty
        existing_cols = self.kb.list_collections()
        if not existing_cols:
            self.kb.create_collection(
                collection_id="feral-core-specs",
                name="Feral Architecture & Cookbook",
                description="Core engineering specifications for Qwen3.5-9B DoRA, GRPO rewards, and Phase 6 self-healing.",
            )
            docs_dir = ROOT / "outputs" / "kb_seed_docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            seed_md = docs_dir / "feral_system_architecture.md"
            seed_md.write_text(
                "# Feral Autonomous Engineering Architecture\n\n"
                "Feral runs `unsloth/Qwen3.5-9B-Base` with rank-32 DoRA adapters trained via SFT and GRPO "
                "on an NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Blackwell sm_120). Precomputed DoRA column "
                "norms (`dora_weight_norms.safetensors`) eliminate per-token 4-bit weight dequantization, "
                "accelerating generation from 1.93 tok/s to 14.10 tok/s (7.29x speedup).\n\n"
                "## Self-Healing Reliability Stack\n\n"
                "Phase 6 equips Feral with a persistent SQLite WAL Task DAG (`dag_orchestrator.py`), "
                "Context Auto-Compactor with `AGENT_STATE.md` memory pinning (`context_compactor.py`), "
                "3-strike Git Time-Travel automatic rollback (`git_time_travel.py`), and a canonical "
                "SHA-256 tool-call Loop Watchdog (`loop_watchdog.py`).\n",
                encoding="utf-8",
            )
            self.kb.ingest_document("feral-core-specs", seed_md, chunk_chars=360)

    def broadcast_event(self, payload: Dict[str, Any]) -> None:
        """Broadcasts a JSON event to all connected WebSocket clients if an event loop is active."""
        if not self._ws_clients or self._event_loop is None:
            return
        dead: List[WebSocket] = []
        for ws in list(self._ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_json(payload), self._event_loop)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    def get_full_snapshot(self) -> Dict[str, Any]:
        """Returns the complete state for all 4 Odysseus UI hubs + Developer Workbenches."""
        base_snap = self.host.get_command_center_snapshot()
        state_md_text = (
            self.compactor.state_file.read_text(encoding="utf-8")
            if self.compactor.state_file.exists()
            else self.compactor.state.to_markdown()
        )
        return {
            "timestamp": time.time(),
            "workspace_root": str(self.workspace_root),
            # 1. The Chat Interface (Main Hub) + Right Sidebar ("Context Panel")
            "chat_hub": {
                "messages": self.chat_ui_log[-60:],
                "context_panel": base_snap["context_panel"] or asdict(GenerationConfigState()),
                "available_models": base_snap["available_models"],
                "compaction": {
                    "active_tokens": self.compactor.count_messages_tokens(self.chat_messages),
                    "max_context_tokens": self.compactor.max_context_tokens,
                    "trigger_ratio": self.compactor.trigger_ratio,
                    "compaction_count": self.compactor.state.compaction_count,
                    "agent_state_markdown": state_md_text,
                },
            },
            # 2. The Autonomous Agents Board + Task DAG
            "agents_board": {
                "cards": base_snap["agents_board"],
                "dag": self.dag.get_dag_snapshot(self.active_mission_id),
            },
            # 3. Knowledge Base & Vector Storage UI (ChromaDB / SQLite Vector Store)
            "knowledge_base": {
                "collections": self.kb.list_collections(),
            },
            # 4. Integrations Panel + Hardware Cookbook
            "integrations_panel": {
                "config": asdict(self.integrations),
                "hardware_cookbook": get_hardware_cookbook_telemetry(),
            },
            # 5. Developer Workbenches (Diff Viewer, Git Time-Travel, Terminal, Vision, Watchdog)
            "workbenches": {
                "diff_ledger": base_snap["diff_ledger"],
                "git_time_travel": self.time_travel.get_timeline_snapshot(),
                "terminal_history": base_snap["terminal_history"],
                "vision_frames": base_snap["vision_frames"],
                "loop_watchdog": self.watchdog.get_status_snapshot(),
                "tool_call_log": self.host.tool_call_log[-30:],
            },
        }

    def execute_chat_mission(self, user_message: str, max_steps: int = 5, reset_session: bool = False) -> Dict[str, Any]:
        """
        Executes a user prompt or autonomous multi-step coding mission through `FeralLLMClient`,
        streaming tokens over WebSocket and enforcing `LoopWatchdog`, `ContextCompactor`, and
        `GitTimeTravelManager`.
        """
        if self.host.llm is None:
            raise RuntimeError("FeralLLMClient is not loaded on cuda:0")

        with self._inference_lock:
            if reset_session:
                self.chat_messages = [
                    {"role": "system", "content": self.host.llm.config_state.system_prompt}
                ]
                self.chat_ui_log.clear()
                self.watchdog.history.clear()
                self.watchdog.current_temperature = self.watchdog.base_temperature

            # Sync system prompt with Context Panel
            if self.chat_messages and self.chat_messages[0].get("role") == "system":
                self.chat_messages[0]["content"] = self.host.llm.config_state.system_prompt

            self.chat_messages.append({"role": "user", "content": user_message})
            user_entry = {
                "id": len(self.chat_ui_log) + 1,
                "role": "user",
                "content": user_message,
                "timestamp": time.time(),
            }
            self.chat_ui_log.append(user_entry)
            self.broadcast_event({"type": "chat_user", "entry": user_entry})

            steps_trace: List[Dict[str, Any]] = []
            final_answer = ""

            for step_idx in range(1, max_steps + 1):
                # 1. Auto-Compact context if enabled and above threshold
                if self.integrations.auto_compaction_enabled:
                    self.chat_messages, comp_meta = self.compactor.maybe_compact(self.chat_messages)
                    if comp_meta.get("compacted"):
                        self.broadcast_event({"type": "context_compacted", "meta": comp_meta})

                # 2. Determine sampling temperature (escalated if LoopWatchdog triggered)
                active_temp = (
                    max(self.host.llm.config_state.temperature, self.watchdog.current_temperature)
                    if self.integrations.loop_watchdog_enabled
                    else self.host.llm.config_state.temperature
                )

                # 3. Stream assistant tokens live
                gen = self.host.llm.stream_turn(self.chat_messages, temperature=active_temp)
                parsed_turn = None
                try:
                    while True:
                        ev = next(gen)
                        if ev.get("type") == "token":
                            self.broadcast_event(
                                {
                                    "type": "chat_stream_token",
                                    "step": step_idx,
                                    "section": ev.get("section", "answer"),
                                    "text": ev.get("text", ""),
                                }
                            )
                        elif ev.get("type") == "complete":
                            parsed_turn = ev["parsed"]
                except StopIteration as stop:
                    if parsed_turn is None and stop.value is not None:
                        parsed_turn = stop.value

                if parsed_turn is None:
                    raise RuntimeError("stream_turn completed without returning a ParsedFeralTurn")

                self.chat_messages.append({"role": "assistant", "content": parsed_turn.raw_text})

                step_record: Dict[str, Any] = {
                    "step": step_idx,
                    "think": parsed_turn.think,
                    "tool_call": parsed_turn.tool_call,
                    "final_answer": parsed_turn.final_answer,
                    "tokens_generated": parsed_turn.tokens_generated,
                    "tokens_per_sec": parsed_turn.tokens_per_sec,
                    "vram_allocated_mib": parsed_turn.vram_allocated_mib,
                    "temperature_used": active_temp,
                }

                # 4. If final answer (no tool call), record and finish
                if not parsed_turn.is_valid_tool_call or parsed_turn.tool_call is None:
                    final_answer = parsed_turn.final_answer or parsed_turn.raw_text
                    assistant_entry = {
                        "id": len(self.chat_ui_log) + 1,
                        "role": "assistant",
                        "think": parsed_turn.think,
                        "tool_call": None,
                        "tool_observation": None,
                        "content": final_answer,
                        "tokens_per_sec": parsed_turn.tokens_per_sec,
                        "vram_allocated_mib": parsed_turn.vram_allocated_mib,
                        "timestamp": time.time(),
                    }
                    self.chat_ui_log.append(assistant_entry)
                    steps_trace.append(step_record)
                    self.broadcast_event({"type": "chat_assistant", "entry": assistant_entry})
                    break

                # 5. Inspect tool call with LoopWatchdog before dispatching to MCP
                tc_name = parsed_turn.tool_call["name"]
                tc_args = parsed_turn.tool_call["arguments"]

                if self.integrations.loop_watchdog_enabled:
                    verdict = self.watchdog.inspect_tool_call(tc_name, tc_args)
                else:
                    verdict = None

                if verdict is not None and verdict.blocked:
                    obs = verdict.interrupt_message
                    step_record["watchdog_blocked"] = True
                    step_record["loop_type"] = verdict.loop_type
                else:
                    try:
                        obs = self.host.dispatch_tool(tc_name, tc_args)
                    except Exception as exc:
                        obs = f"ERROR: {type(exc).__name__}: {exc}\n[exit_code=1]"

                    # 6. If Git Time-Travel is enabled and a test or patch ran, track green/strike state
                    if self.integrations.git_time_travel_enabled and tc_name in ("apply_diff_patch", "write_file", "run_powershell"):
                        passed = obs.endswith("[exit_code=0]") and not obs.lstrip().startswith("ERROR:")
                        is_unittest = tc_name == "run_powershell" and "unittest" in str(tc_args.get("command", ""))
                        tt_res = self.time_travel.record_outcome(
                            passed=passed,
                            detail=f"{tc_name}({json.dumps(tc_args)})",
                            checkpoint_label_on_pass=(
                                f"Verified green after {tc_name}" if (passed and is_unittest) else None
                            ),
                        )
                        if tt_res.get("rolled_back"):
                            obs = f"{obs}\n{tt_res['recovery_directive']}"

                step_record["tool_observation"] = obs
                steps_trace.append(step_record)

                self.chat_messages.append(
                    {"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"}
                )

                assistant_entry = {
                    "id": len(self.chat_ui_log) + 1,
                    "role": "assistant",
                    "think": parsed_turn.think,
                    "tool_call": parsed_turn.tool_call,
                    "tool_observation": obs,
                    "content": "",
                    "tokens_per_sec": parsed_turn.tokens_per_sec,
                    "vram_allocated_mib": parsed_turn.vram_allocated_mib,
                    "timestamp": time.time(),
                }
                self.chat_ui_log.append(assistant_entry)
                self.broadcast_event({"type": "chat_assistant", "entry": assistant_entry})

            self.compactor.save_agent_state_md()
            return {
                "steps": steps_trace,
                "final_answer": final_answer,
                "total_steps": len(steps_trace),
                "snapshot": self.get_full_snapshot(),
            }

    def close(self) -> None:
        self.host.close()


# ============================================================================
# 4. FastAPI Application Factory (`create_command_center_app`)
# ============================================================================
def create_command_center_app(center: FeralCommandCenter) -> FastAPI:
    app = FastAPI(
        title="Feral Desktop Command Center API",
        description="Odysseus-inspired local autonomous AI engineering hub powered by Qwen3.5-9B + GRPO DoRA",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    UI_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

    @app.on_event("startup")
    async def _capture_loop() -> None:
        center._event_loop = asyncio.get_running_loop()

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> Any:
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse(
            "<html><body style='background:#0b0d12;color:#e2e8f0;font-family:monospace;padding:2rem;'>"
            "<h2>Feral Command Center Backend Online</h2>"
            "<p>API Snapshot: <a style='color:#38bdf8' href='/api/snapshot'>/api/snapshot</a></p>"
            "</body></html>"
        )

    @app.get("/api/snapshot")
    async def api_snapshot() -> Dict[str, Any]:
        return center.get_full_snapshot()

    # --- Module 1: Chat Hub & Collapsible Context Panel ---
    @app.post("/api/context")
    async def api_update_context(req: ContextPanelUpdateRequest) -> Dict[str, Any]:
        if center.host.llm is None:
            raise HTTPException(status_code=503, detail="LLM not loaded")
        updated = center.host.llm.update_context_panel(
            model_id=req.model_id,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
        )
        payload = asdict(updated)
        center.broadcast_event({"type": "context_updated", "context_panel": payload})
        return {"status": "ok", "context_panel": payload}

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: center.execute_chat_mission(
                user_message=req.message,
                max_steps=req.max_steps,
                reset_session=req.reset_session,
            ),
        )
        return result

    @app.post("/api/chat/clear")
    async def api_chat_clear() -> Dict[str, Any]:
        center.chat_messages = [
            {
                "role": "system",
                "content": center.host.llm.config_state.system_prompt if center.host.llm else SYSTEM_FERAL,
            }
        ]
        center.chat_ui_log.clear()
        center.watchdog.history.clear()
        center.watchdog.current_temperature = center.watchdog.base_temperature
        return {"status": "cleared", "snapshot": center.get_full_snapshot()}

    # --- Module 2: Autonomous Agents Board & Task DAG ---
    @app.post("/api/agents/create")
    async def api_create_agent(req: CreateAgentCardRequest) -> Dict[str, Any]:
        card = center.host.subagents.create_agent_card(
            name=req.name,
            goal=req.goal,
            grid_col=req.grid_col,
            grid_row=req.grid_row,
        )
        data = asdict(card)
        center.broadcast_event({"type": "agent_card_updated", "card": data})
        return {"status": "ok", "card": data}

    @app.post("/api/agents/{agent_id}/deploy")
    async def api_deploy_agent(agent_id: str) -> Dict[str, Any]:
        if agent_id not in center.host.subagents.agent_cards:
            raise HTTPException(status_code=404, detail=f"Unknown agent_id `{agent_id}`")
        card = center.host.subagents.agent_cards[agent_id]
        loop = asyncio.get_running_loop()
        with center._inference_lock:
            report = await loop.run_in_executor(
                None,
                lambda: center.host.subagents.spawn_subagent(
                    task_prompt=card.goal,
                    agent_name=card.name,
                    existing_agent_id=agent_id,
                ),
            )
        updated_card = asdict(center.host.subagents.agent_cards[agent_id])
        center.broadcast_event({"type": "agent_card_updated", "card": updated_card})
        return {"status": "ok", "report": report, "card": updated_card}

    @app.post("/api/agents/{agent_id}/control")
    async def api_control_agent(agent_id: str, req: AgentControlRequest) -> Dict[str, Any]:
        try:
            card = center.host.subagents.set_agent_status(agent_id, req.action)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        data = asdict(card)
        center.broadcast_event({"type": "agent_card_updated", "card": data})
        return {"status": "ok", "card": data}

    @app.post("/api/agents/{agent_id}/move")
    async def api_move_agent(agent_id: str, req: AgentMoveRequest) -> Dict[str, Any]:
        try:
            card = center.host.subagents.move_agent_card(agent_id, req.grid_col, req.grid_row)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        data = asdict(card)
        center.broadcast_event({"type": "agent_card_updated", "card": data})
        return {"status": "ok", "card": data}

    @app.post("/api/dag/node")
    async def api_add_dag_node(req: CreateDAGNodeRequest) -> Dict[str, Any]:
        try:
            node = center.dag.add_node(
                mission_id=req.mission_id,
                node_id=req.node_id,
                title=req.title,
                prompt=req.prompt,
                assigned_agent=req.assigned_agent,
                depends_on=req.depends_on,
                priority=req.priority,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "status": "ok",
            "node": asdict(node),
            "dag": center.dag.get_dag_snapshot(req.mission_id),
        }

    # --- Module 3: Knowledge Base & Vector Storage UI ---
    @app.post("/api/kb/collections")
    async def api_create_kb_collection(req: CreateKBCollectionRequest) -> Dict[str, Any]:
        info = center.kb.create_collection(
            collection_id=req.collection_id,
            name=req.name,
            description=req.description,
        )
        return {"status": "ok", "collection": asdict(info), "collections": center.kb.list_collections()}

    @app.post("/api/kb/upload")
    async def api_upload_kb_document(req: KBJsonUploadRequest) -> Dict[str, Any]:
        upload_dir = ROOT / "outputs" / "kb_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(req.filename).name
        target_path = upload_dir / safe_name

        if req.content_base64:
            raw_bytes = base64.b64decode(req.content_base64)
            target_path.write_bytes(raw_bytes)
        elif req.content_text is not None:
            target_path.write_text(req.content_text, encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="Either content_text or content_base64 is required")

        progress_trace: List[float] = []

        def _on_prog(pct: float) -> None:
            progress_trace.append(pct)
            center.broadcast_event(
                {
                    "type": "kb_embedding_progress",
                    "collection_id": req.collection_id,
                    "filename": safe_name,
                    "progress_pct": pct,
                }
            )

        try:
            info = center.kb.ingest_document(
                collection_id=req.collection_id,
                file_path=target_path,
                chunk_chars=420,
                progress_callback=_on_prog,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "status": "ok",
            "collection": asdict(info),
            "progress_trace": progress_trace,
            "collections": center.kb.list_collections(),
        }

    @app.post("/api/kb/query")
    async def api_query_kb(req: KBQueryRequest) -> Dict[str, Any]:
        hits = center.kb.query_collection(
            collection_id=req.collection_id,
            query=req.query,
            top_k=req.top_k,
        )
        return {"status": "ok", "results": hits}

    # --- Module 4: Integrations Panel & Developer Workbenches ---
    @app.post("/api/integrations")
    async def api_update_integrations(req: IntegrationsUpdateRequest) -> Dict[str, Any]:
        updates = req.model_dump(exclude_none=True)
        for k, v in updates.items():
            if hasattr(center.integrations, k):
                setattr(center.integrations, k, v)

        # Mirror browser settings into MCPBrowserVisionServer
        center.host.browser.update_integrations(
            searxng_enabled=center.integrations.searxng_enabled,
            searxng_endpoint=center.integrations.searxng_endpoint,
            headless_browser_enabled=center.integrations.headless_browser_enabled,
            vision_screenshots_enabled=center.integrations.vision_screenshots_enabled,
        )
        return {
            "status": "ok",
            "integrations": asdict(center.integrations),
            "hardware_cookbook": get_hardware_cookbook_telemetry(),
        }

    @app.post("/api/workbenches/terminal")
    async def api_run_terminal(req: TerminalCommandRequest) -> Dict[str, Any]:
        out = center.host.dispatch_tool("run_powershell", {"command": req.command})
        return {
            "status": "ok",
            "output": out,
            "terminal_history": [asdict(r) for r in center.host.terminal.session.history[-20:]],
        }

    @app.post("/api/workbenches/vision")
    async def api_capture_vision(req: VisionCaptureRequest) -> Dict[str, Any]:
        out = center.host.dispatch_tool("web_research", {"query": req.query, "url": req.url})
        return {
            "status": "ok",
            "output": out,
            "vision_frames": [asdict(f) for f in center.host.browser.vision_frames[-5:]],
        }

    @app.post("/api/workbenches/rollback-patch")
    async def api_rollback_patch(req: RollbackPatchRequest) -> Dict[str, Any]:
        out = center.host.editor.rollback_patch(req.patch_id)
        return {
            "status": "ok",
            "output": out,
            "diff_ledger": [asdict(d) for d in center.host.editor.diff_history],
        }

    @app.post("/api/workbenches/rollback-git")
    async def api_rollback_git(req: RollbackGitRequest) -> Dict[str, Any]:
        try:
            event = center.time_travel.rollback_to_checkpoint(
                req.checkpoint_id_or_sha,
                reason="ui_manual_time_travel",
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {
            "status": "ok",
            "event": event,
            "git_time_travel": center.time_travel.get_timeline_snapshot(),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        center._ws_clients.add(ws)
        try:
            await ws.send_json({"type": "snapshot", "snapshot": center.get_full_snapshot()})
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                mtype = msg.get("type", "")
                if mtype == "ping":
                    await ws.send_json({"type": "pong", "timestamp": time.time()})
                elif mtype == "chat":
                    prompt = str(msg.get("message", "")).strip()
                    if prompt:
                        loop = asyncio.get_running_loop()
                        res = await loop.run_in_executor(
                            None,
                            lambda: center.execute_chat_mission(
                                user_message=prompt,
                                max_steps=int(msg.get("max_steps", 4)),
                            ),
                        )
                        await ws.send_json({"type": "chat_complete", "result": res})
        except WebSocketDisconnect:
            pass
        finally:
            center._ws_clients.discard(ws)

    return app


# ============================================================================
# 5. Automated End-to-End API + WebSocket Verification Suite
# ============================================================================
def run_self_test() -> None:
    from fastapi.testclient import TestClient

    print("1. Initializing `FeralCommandCenter` on RTX 5070 Ti (`cuda:0`)...")
    if SANDBOX_WORKSPACE.exists():
        shutil.rmtree(SANDBOX_WORKSPACE)
    for db_file in (DAG_DB_PATH, KB_DB_PATH):
        for ext in ("", "-wal", "-shm"):
            p = Path(str(db_file) + ext)
            if p.exists():
                p.unlink()

    center = FeralCommandCenter(workspace_root=SANDBOX_WORKSPACE, load_llm=True)
    app = create_command_center_app(center)

    try:
        with TestClient(app) as client:
            print("\n2. Testing `GET /api/snapshot` & RTX 5070 Ti Hardware Cookbook Telemetry...")
            r_snap = client.get("/api/snapshot")
            assert r_snap.status_code == 200
            snap = r_snap.json()
            hw = snap["integrations_panel"]["hardware_cookbook"]
            print(
                f"   - GPU Detected        : {hw['device_name']} ({hw['architecture']}, {hw['compute_capability']})"
            )
            print(
                f"   - VRAM Telemetry      : {hw['vram_allocated_mib']} MiB allocated / {hw['total_vram_mib']} MiB total ({hw['vram_utilization_pct']}%)"
            )
            print(f"   - Cookbook Recipe     : {hw['cookbook_recommendations']['active_precision_recipe']}")
            assert "RTX 5070 Ti" in hw["device_name"]
            assert hw["vram_allocated_mib"] > 4000.0
            assert len(snap["agents_board"]["cards"]) == 3
            assert len(snap["knowledge_base"]["collections"]) >= 1

            print("\n3. Testing Module 1 (Context Panel Update & Live WebSocket + Chat ReAct Loop)...")
            r_ctx = client.post(
                "/api/context",
                json={"temperature": 0.0, "max_new_tokens": 160},
            )
            assert r_ctx.status_code == 200
            assert r_ctx.json()["context_panel"]["max_new_tokens"] == 160

            with client.websocket_connect("/ws") as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "snapshot"
                ws.send_json({"type": "ping"})
                pong = ws.receive_json()
                assert pong["type"] == "pong"

            r_chat = client.post(
                "/api/chat",
                json={
                    "message": "Read lines 1 to 5 of `src/discounts.py` using `read_file_lines` and tell me what bug is on line 3.",
                    "max_steps": 3,
                },
            )
            assert r_chat.status_code == 200
            chat_res = r_chat.json()
            steps = chat_res["steps"]
            assert len(steps) >= 1
            print(
                f"   - Chat ReAct Steps    : {len(steps)} step(s) | Step 1 speed={steps[0]['tokens_per_sec']} tok/s | VRAM={steps[0]['vram_allocated_mib']} MiB"
            )
            print(f"   - Final Answer Excerpt: {chat_res['final_answer'][:110]!r}")

            print("\n4. Testing Module 2 (Autonomous Agents Board: Create, Move, Pause/Resume & Deploy)...")
            r_ag = client.post(
                "/api/agents/create",
                json={
                    "name": "Unit Test Auditor",
                    "goal": "Read `src/discounts.py` (lines 1-5) with `read_file_lines` and report the formula.",
                    "grid_col": 1,
                    "grid_row": 1,
                },
            )
            assert r_ag.status_code == 200
            agent_id = r_ag.json()["card"]["agent_id"]

            r_mv = client.post(f"/api/agents/{agent_id}/move", json={"grid_col": 2, "grid_row": 1})
            assert r_mv.status_code == 200
            assert r_mv.json()["card"]["grid_col"] == 2

            r_pause = client.post(f"/api/agents/{agent_id}/control", json={"action": "pause"})
            assert r_pause.json()["card"]["status"] == "Paused"
            r_resume = client.post(f"/api/agents/{agent_id}/control", json={"action": "resume"})
            assert r_resume.status_code == 200

            r_dep = client.post(f"/api/agents/{agent_id}/deploy")
            assert r_dep.status_code == 200
            dep_card = r_dep.json()["card"]
            print(
                f"   - Deployed Card `{agent_id}` : status={dep_card['status']}, steps={dep_card['steps_completed']}, console_lines={len(dep_card['console_log'])}"
            )
            assert dep_card["status"] == "Completed"
            assert len(dep_card["console_log"]) >= 4

            print("\n5. Testing Module 3 (Knowledge Base & Vector Storage UI: `.pdf` Upload + Cosine Query)...")
            pdf_tmp = ROOT / "outputs" / "kb_uploads" / "odysseus_ui_spec.pdf"
            pdf_tmp.parent.mkdir(parents=True, exist_ok=True)
            build_minimal_pdf(
                pdf_tmp,
                [
                    "Odysseus Desktop Command Center Specification",
                    "Module 1 provides the Main Chat Hub and Collapsible Context Panel.",
                    "Module 2 provides the Autonomous Agents Board with rolling console logs.",
                    "Module 3 provides ChromaDB SQLite vector collections with progress bars.",
                    "Module 4 provides the Integrations Panel for SearXNG, IMAP SMTP, and Calendar.",
                ],
            )
            pdf_b64 = base64.b64encode(pdf_tmp.read_bytes()).decode("ascii")
            r_up = client.post(
                "/api/kb/upload",
                json={
                    "collection_id": "feral-core-specs",
                    "filename": "odysseus_ui_spec.pdf",
                    "content_base64": pdf_b64,
                },
            )
            assert r_up.status_code == 200
            up_data = r_up.json()
            print(
                f"   - PDF Ingested        : progress_trace={up_data['progress_trace']}% | total_docs={up_data['collection']['document_count']}"
            )
            assert up_data["collection"]["embedding_progress_pct"] == 100.0

            r_q = client.post(
                "/api/kb/query",
                json={
                    "collection_id": "feral-core-specs",
                    "query": "Which module provides the Integrations Panel for SearXNG and IMAP SMTP?",
                    "top_k": 2,
                },
            )
            assert r_q.status_code == 200
            top_hit = r_q.json()["results"][0]
            print(
                f"   - Vector Query Top-1  : doc=`{top_hit['doc_name']}` (similarity={top_hit['similarity']})"
            )
            assert top_hit["doc_name"] == "odysseus_ui_spec.pdf"

            print("\n6. Testing Module 4 (Integrations Panel Toggles: SearXNG, IMAP/SMTP, Calendar)...")
            r_int = client.post(
                "/api/integrations",
                json={
                    "searxng_enabled": True,
                    "searxng_endpoint": "http://127.0.0.1:8080/search",
                    "email_enabled": True,
                    "imap_port": 1143,
                    "calendar_sync_enabled": True,
                },
            )
            assert r_int.status_code == 200
            int_cfg = r_int.json()["integrations"]
            print(
                f"   - Integrations Saved  : searxng={int_cfg['searxng_enabled']}, email={int_cfg['email_enabled']}, calendar={int_cfg['calendar_sync_enabled']}"
            )
            assert int_cfg["searxng_enabled"] is True
            assert int_cfg["email_enabled"] is True
            assert int_cfg["calendar_sync_enabled"] is True

        print(
            "\nPHASE 7 STEP 1 (`server_command_center.py`): PASS -> FastAPI + WebSocket orchestrator verified on RTX 5070 Ti!"
        )
    finally:
        center.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feral Desktop Command Center Server")
    parser.add_argument("--serve", action="store_true", help="Run the live Uvicorn HTTP/WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    if args.serve:
        cc = FeralCommandCenter(workspace_root=SANDBOX_WORKSPACE, load_llm=True)
        fastapi_app = create_command_center_app(cc)
        uvicorn.run(fastapi_app, host=args.host, port=args.port)
    else:
        run_self_test()
