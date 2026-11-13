# Feral — Local Autonomous Software Engineering & Research Agent (`Qwen3.5-9B` + GRPO DoRA)

**Feral** is a local autonomous software engineering, debugging, and research agent built on `unsloth/Qwen3.5-9B-Base` for Windows 11 (`PowerShell`) and NVIDIA Blackwell GPUs (`GeForce RTX 5070 Ti 16 GB VRAM`, `sm_120`). It combines representation engineering, mistake-masked Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO) with verifiable execution rewards, a 4-server Model Context Protocol (MCP) host, self-healing reliability watchdogs, and an **Odysseus-inspired Desktop Command Center UI**.

---

## Quick Start — Launch the Odysseus Command Center UI

```powershell
# 1. Start the live FastAPI + WebSocket Command Center on http://127.0.0.1:8765
python server_command_center.py --serve --host 127.0.0.1 --port 8765

# 2. Or run the full-system autonomous burn-in verification on cuda:0
python test_full_system_burn_in.py
```

---

## Architecture Overview (Phases 0–7)

### Phase 0 & Phase 1 — Representation Engineering & Residual Stream Mechanics
- **`inspect_residual_stream.py`** — Hooks into `model.model.layers` (`[0, 15, 31]`) to measure 4,096-dim residual stream hidden states and L2 norm growth.
- **`collect_refusal_vector.py` & `test_orthogonal_projection.py`** — Extracts normalized contrast vectors at the `<|im_start|>assistant` boundary and applies orthogonal projection across layers `14..19`.
- **`test_agent_brain.py`** — Baseline ReAct loop using `<think>...</think>` reasoning and `<tool_call>` JSON execution.

### Phase 2 — Execution-Verified Trajectory Distillation & Mistake Masking
- **`feral_core.py` & `scenarios/`** — Shared sandbox, `SYSTEM_FERAL` prompt, 7-tool Feral Apex Toolchain (`run_powershell`, `read_file_lines`, `apply_diff_patch`, `write_file`, `ast_repo_map`, `web_research`, `spawn_subagent`), and completion/mistake-masked (`-100`) tokenizer.
- **`distill_verified_traces.py`, `expand_feral_arsenal_dataset.py`, `validate_curriculum_dataset.py`** — Generates and audits `120` execution-verified multi-turn trajectories (`train.jsonl` `100` rows, `val.jsonl` `20` rows).

### Phase 3 & Phase 4 — Rank-32 DoRA SFT, Verifiable-Reward GRPO & GGUF Export
- **`setup_dora_model.py` & `train_sft_agent.py`** — Trains rank-32 DoRA (`use_dora=True`, `lora_alpha=64`) across all `200` linear projections (`val_loss = 0.5101`, `100%` held-out pass rate in `eval_sft_agent.py`).
- **`grpo_reward_sandbox.py` & `train_grpo_agent.py`** — 5-part execution reward sandbox (`max +4.00`) and online GRPO training ($G=4$ rollouts $\times 8$ tasks), improving unit-test pass rate from `87.5%` (`+3.5312`) to **`100.0%` (`+4.0000`)**.
- **`export_gguf.py`** — Fuses DoRA magnitude-direction weights on `cuda:0`, exports `outputs/feral-qwen3.5-9b-grpo-q4_k_m.gguf` (`3,964.0 MiB`, `3.33x` compression), and caches `dora_weight_norms.safetensors` for a **`7.29x` inference speedup** (`1.93 -> 14.10 tok/s`).

### Phase 5 — Feral Autonomous Core Engine & 4-Server MCP Host
- **`feral_llm_client.py`** — Production inference client (`7,642.9 MiB` VRAM, `13.5–14.1 tok/s` streaming via `TextIteratorStreamer`, Context Panel runtime config).
- **`mcp_terminal_server.py`** — Stateful persistent PowerShell session (`cwd` & `$env:` preserved across turns) with watchdog process-tree kill & auto-respawn.
- **`mcp_code_editor.py`** — `ast_repo_map`, `read_file_lines`, `apply_diff_patch`, and `write_file` with pre-commit `ast.parse` syntax guard and patch rollback ledger.
- **`mcp_browser_vision.py`** — `web_research`, `BeautifulSoup` HTML-to-Markdown extractor, SearXNG API integration, and Windows 11 Headless Chromium (`1280x800` PNG) vision capture.
- **`mcp_subagent_worker.py` & `test_feral_core.py`** — Isolated `spawn_subagent` worker (`10.74x` context compression, `0 MiB` extra model VRAM) and unified `FeralMCPHost`.

### Phase 6 — "All-Night Long" Self-Healing Reliability Systems
- **`dag_orchestrator.py`** — Persistent SQLite WAL Hierarchical Task DAG with DFS cycle guard, strike-counted retry, and crash recovery.
- **`context_compactor.py`** — Auto-compacts conversation history (`-50.6%` token reduction), pins working memory to `AGENT_STATE.md`, and powers the 384-dim ChromaDB/SQLite `LocalVectorKnowledgeBase` (`.txt`, `.md`, `.pdf`).
- **`git_time_travel.py`** — Automatic `[GREEN]` Git checkpoints and 3-strike `git reset --hard` rollback.
- **`loop_watchdog.py`** — Canonical SHA-256 tool-call fingerprinting that blocks `A -> A` and `A -> B -> A -> B` loops and escalates sampling temperature (`0.00 -> 0.45`).

### Phase 7 — Odysseus-Inspired Desktop Command Center UI (`server_command_center.py` & `ui/`)
1. **01 // Chat Hub & Collapsible Context Panel** — Left/Center split with Markdown, syntax-highlighted copy-code blocks, collapsible `<think>` stream, and right-hand Context Panel (`SYSTEM_FERAL`, temperature, max tokens, local model selector, and pinned `AGENT_STATE.md`).
2. **02 // Autonomous Agents Board & Task DAG** — Drag-and-drop grid cards with `Idle` / `Running` / `Paused` controls, rolling `[THINK]` / `[TOOL]` / `[OBS]` console logs, and SQLite Task DAG.
3. **03 // Knowledge Base & Vector Storage UI** — Drag-and-drop `.txt`, `.md`, `.pdf` ingestion into ChromaDB/SQLite Collections with live `0%–100%` embedding progress bars and cosine search.
4. **04 // Integrations Panel & Hardware Cookbook** — Toggles for SearXNG, local IMAP/SMTP email, CalDAV calendar sync, Phase 6 watchdogs, and live RTX 5070 Ti telemetry.
5. **05 // Developer Workbenches** — Unified Diff Ledger with 1-click rollback, Git Time-Travel timeline, stateful PowerShell terminal, and Live Headless Chromium Vision Feed.

---

## License

This project is licensed under the [MIT License](LICENSE).
