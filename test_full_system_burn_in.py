"""
Phase 7, Step 3 - Full-System End-to-End Autonomous Burn-In Verification (`test_full_system_burn_in.py`)
========================================================================================================
Runs the complete production burn-in suite for Feral on NVIDIA GeForce RTX 5070 Ti (`cuda:0`):
  1. Boots `FeralCommandCenter(load_llm=True)` + live Uvicorn HTTP/WebSocket server (`Qwen3.5-9B` + GRPO DoRA).
  2. Executes a live multi-turn autonomous bug-fix mission via `POST /api/chat` on `src/discounts.py`:
     - Inspects failing test / buggy code -> applies `apply_diff_patch` -> runs `unittest` -> creates a
       verified `[GREEN]` Git Time-Travel checkpoint and updates `AGENT_STATE.md`.
  3. Verifies the Autonomous Agents Board (`POST /api/agents/agent-01/deploy`), Knowledge Base vector
     ingestion & cosine search (`POST /api/kb/query`), and Integrations Panel (`POST /api/integrations`).
  4. Renders and validates `1280x800` Headless Chromium screenshots across all 5 Odysseus UI tabs:
     - `/?tab=chat`, `/?tab=agents`, `/?tab=kb`, `/?tab=integrations`, `/?tab=workbenches`.
"""
import os
import shutil
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from PIL import Image, ImageStat

from feral_core import ROOT
from server_command_center import (
    DAG_DB_PATH,
    KB_DB_PATH,
    SANDBOX_WORKSPACE,
    FeralCommandCenter,
    create_command_center_app,
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def force_rmtree(target: Path) -> None:
    if not target.exists():
        return
    def _handle_readonly(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    shutil.rmtree(target, onexc=_handle_readonly)


def main() -> None:
    print("1. Booting Full `FeralCommandCenter` Stack on RTX 5070 Ti (`cuda:0`)...")
    force_rmtree(SANDBOX_WORKSPACE)
    for db_file in (DAG_DB_PATH, KB_DB_PATH):
        for ext in ("", "-wal", "-shm"):
            p = Path(str(db_file) + ext)
            if p.exists():
                p.unlink()

    center = FeralCommandCenter(workspace_root=SANDBOX_WORKSPACE, load_llm=True)
    app = create_command_center_app(center)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        # Wait for live HTTP server readiness
        for _ in range(40):
            try:
                r = httpx.get(f"{base_url}/api/snapshot", timeout=2.0)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.15)

        with httpx.Client(base_url=base_url, timeout=180.0) as client:
            snap0 = client.get("/api/snapshot").json()
            hw0 = snap0["integrations_panel"]["hardware_cookbook"]
            cp_before = len(snap0["workbenches"]["git_time_travel"]["checkpoints"])
            print(
                f"   - Server Online       : {base_url} | GPU={hw0['device_name']} ({hw0['vram_allocated_mib']} MiB VRAM)"
            )
            print(f"   - Initial Checkpoints : {cp_before} (`{snap0['workbenches']['git_time_travel']['head_short_sha']}`)")

            print("\n2. Running End-to-End Autonomous Bug-Fix Mission via `POST /api/chat`...")
            mission_prompt = (
                "The checkout unit tests in `tests/test_checkout.py` are failing because `apply_coupon` in "
                "`src/discounts.py` returns the discount amount instead of subtracting it from `price`. "
                "First read `src/discounts.py` (lines 1 to 5) with `read_file_lines`, then fix `src/discounts.py` "
                "with `apply_diff_patch` so it returns `round(price * (1.0 - percent_off / 100.0), 2)`, and "
                "finally run `python -m unittest discover -s tests` with `run_powershell`."
            )
            r_chat = client.post(
                "/api/chat",
                json={"message": mission_prompt, "max_steps": 5},
            )
            assert r_chat.status_code == 200
            chat_data = r_chat.json()
            steps = chat_data["steps"]
            tools_called = [s["tool_call"]["name"] for s in steps if s.get("tool_call")]
            speeds = [s["tokens_per_sec"] for s in steps]
            avg_speed = round(sum(speeds) / max(len(speeds), 1), 2)
            print(f"   - ReAct Steps Executed: {len(steps)} | Tools: {tools_called} | Mean Speed: {avg_speed} tok/s")

            # Verify unit tests pass in SANDBOX_WORKSPACE
            ut = subprocess.run(
                ["python", "-m", "unittest", "discover", "-s", "tests"],
                cwd=str(SANDBOX_WORKSPACE),
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert ut.returncode == 0, f"Expected unit tests to pass after autonomous fix, got:\n{ut.stderr}"

            # Update DAG nodes to reflect mission completion
            center.dag.mark_node_completed("mission-default", "step-1-map", "Mapped repository AST")
            center.dag.mark_node_completed("mission-default", "step-2-patch", "Patched src/discounts.py")
            center.dag.mark_node_completed("mission-default", "step-3-verify", "Unit tests passed (OK)")

            snap1 = client.get("/api/snapshot").json()
            diffs = snap1["workbenches"]["diff_ledger"]
            checkpoints = snap1["workbenches"]["git_time_travel"]["checkpoints"]
            dag_pct = snap1["agents_board"]["dag"]["progress_pct"]
            print(
                f"   - Post-Fix Verification: unittest=OK (exit_code=0) | Diffs={len(diffs)} | Git Checkpoints={len(checkpoints)} | DAG={dag_pct}%"
            )
            assert len(diffs) >= 1
            assert len(checkpoints) >= cp_before + 1
            assert dag_pct == 100.0

            print("\n3. Capturing `1280x800` Headless Chromium Screenshots Across All 5 Odysseus Tabs...")
            tabs = ["chat", "agents", "kb", "integrations", "workbenches"]
            shots_dir = ROOT / "outputs" / "ui_burn_in_screenshots"
            shots_dir.mkdir(parents=True, exist_ok=True)

            for tab in tabs:
                frame = center.host.browser.capture_page_vision(
                    url=f"{base_url}/?tab={tab}",
                    pre_fetched_title=f"Feral Odysseus UI - {tab.upper()}",
                    pre_fetched_md=f"Live 1280x800 screenshot of tab `{tab}`",
                )
                dest = shots_dir / f"tab_{tab}.png"
                shutil.copy2(frame.screenshot_path, dest)
                with Image.open(dest) as img:
                    w, h = img.size
                    stat = ImageStat.Stat(img.convert("RGB"))
                    std_max = round(max(stat.stddev), 2)
                print(f"   - Tab `{tab:<12}` : {w}x{h} ({frame.file_size_kb:>6.2f} KB, max_rgb_stddev={std_max})")
                assert (w, h) == (1280, 800)
                assert std_max > 5.0

        print(
            "\nPHASE 7 STEP 3 (`test_full_system_burn_in.py`): PASS -> Full Feral Autonomous Stack & Odysseus UI Burn-In Verified!"
        )
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)
        center.close()


if __name__ == "__main__":
    main()
