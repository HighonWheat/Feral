"""
Phase 7, Step 2 - Odysseus-Inspired Desktop Command Center UI Verification (`test_command_center_ui.py`)
========================================================================================================
Boots the live FastAPI + Uvicorn server for `FeralCommandCenter`, validates all 4 Odysseus UI hubs
and Developer Workbenches in `ui/index.html`, `ui/styles.css`, and `ui/app.js`, and renders a real
`1280x800` PNG screenshot of the live UI via Windows 11 Headless Chromium (`chrome.exe --headless=new`).
"""
import shutil
import socket
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from bs4 import BeautifulSoup
from PIL import Image, ImageStat

from feral_core import ROOT
from mcp_browser_vision import find_chromium_binary
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


def main() -> None:
    print("1. Auditing `ui/index.html`, `ui/styles.css`, and `ui/app.js` DOM & Module Structure...")
    index_path = ROOT / "ui" / "index.html"
    css_path = ROOT / "ui" / "styles.css"
    js_path = ROOT / "ui" / "app.js"

    assert index_path.exists(), "Missing ui/index.html"
    assert css_path.exists(), "Missing ui/styles.css"
    assert js_path.exists(), "Missing ui/app.js"

    soup = BeautifulSoup(index_path.read_text(encoding="utf-8"), "html.parser")

    required_ids = {
        # Module 1: Chat Hub + Collapsible Right Context Panel
        "view-chat": "Chat Interface Hub",
        "chat-stream": "Chat History Stream",
        "context-sidebar": "Collapsible Right Context Panel",
        "btn-toggle-context": "Context Panel Toggle Button",
        "ctx-model-select": "Available Local Models Dropdown",
        "ctx-system-prompt": "System Prompt Textarea",
        "ctx-temp-slider": "Temperature Slider",
        "ctx-tokens-slider": "Max New Tokens Slider",
        # Module 2: Autonomous Agents Board & Task DAG
        "view-agents": "Autonomous Agents Board View",
        "agents-grid": "Drag-and-Drop / Grid Agent Cards Container",
        "btn-create-agent": "Create Autonomous Agent Button",
        "dag-nodes-list": "Hierarchical SQLite Task DAG List",
        # Module 3: Knowledge Base & Vector Storage UI
        "view-kb": "Knowledge Base & Vector Storage View",
        "kb-dropzone": "Drag-and-Drop (.txt/.md/.pdf) Upload Zone",
        "kb-collections-grid": "ChromaDB / SQLite Collections Grid",
        "kb-query-input": "Semantic Vector Search Input",
        # Module 4: Integrations Panel & Hardware Cookbook
        "view-integrations": "Integrations & Cookbook View",
        "int-searxng-enabled": "SearXNG Web Search Toggle",
        "int-email-enabled": "Local IMAP/SMTP Email Toggle",
        "int-imap-host": "IMAP Host Config",
        "int-smtp-host": "SMTP Host Config",
        "int-calendar-enabled": "CalDAV Calendar Sync Toggle",
        "hw-cookbook-box": "RTX 5070 Ti Hardware Cookbook Panel",
        # Module 5: Developer Workbenches
        "view-workbenches": "Developer Workbenches View",
        "workbench-diff-list": "Unified Diff Ledger",
        "workbench-git-list": "Git Time-Travel Timeline",
        "workbench-term-log": "Persistent PowerShell Console",
        "workbench-vision-box": "Live Headless Chromium Vision Feed",
    }

    for dom_id, label in required_ids.items():
        el = soup.find(id=dom_id)
        assert el is not None, f"Missing required DOM element `#{dom_id}` ({label})"

    print(
        f"   - Verified {len(required_ids)}/{len(required_ids)} required DOM controls across all 5 Odysseus UI hubs"
    )
    print(
        f"   - Asset Sizes: index.html={index_path.stat().st_size:,} B | styles.css={css_path.stat().st_size:,} B | app.js={js_path.stat().st_size:,} B"
    )

    print("\n2. Booting Live Uvicorn Server & Capturing `1280x800` Headless Chromium UI Screenshot...")
    center = FeralCommandCenter(workspace_root=SANDBOX_WORKSPACE, load_llm=False)
    app = create_command_center_app(center)
    port = get_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        # Wait for HTTP readiness
        for _ in range(30):
            try:
                r = httpx.get(f"{base_url}/api/snapshot", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.15)

        r_html = httpx.get(f"{base_url}/", timeout=3.0)
        r_css = httpx.get(f"{base_url}/ui/styles.css", timeout=3.0)
        r_js = httpx.get(f"{base_url}/ui/app.js", timeout=3.0)
        assert r_html.status_code == 200 and "FERAL COMMAND CENTER" in r_html.text
        assert r_css.status_code == 200 and "--bg-void" in r_css.text
        assert r_js.status_code == 200 and "renderMarkdownWithCodeBlocks" in r_js.text

        # Render live UI in Headless Chromium (`1280x800`)
        frame = center.host.browser.capture_page_vision(
            url=f"{base_url}/",
            pre_fetched_title="Feral Odysseus Command Center UI",
            pre_fetched_md="Live 1280x800 render of the Feral Desktop Command Center UI.",
        )
        ui_png = Path(frame.screenshot_path)
        assert ui_png.exists()

        target_copy = ROOT / "outputs" / "feral_command_center_ui.png"
        target_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ui_png, target_copy)

        with Image.open(target_copy) as img:
            w, h = img.size
            stat = ImageStat.Stat(img.convert("RGB"))
            mean_rgb = [round(x, 2) for x in stat.mean]
            std_rgb = [round(x, 2) for x in stat.stddev]

        print(f"   - Live UI URL         : {base_url}/ (HTTP 200 OK)")
        print(f"   - Capture Backend     : {frame.capture_backend} ({find_chromium_binary()})")
        print(f"   - Screenshot Saved    : {target_copy} ({w}x{h}, {frame.file_size_kb} KB)")
        print(f"   - Dark Theme RGB Mean : {mean_rgb} | RGB StdDev: {std_rgb} (non-blank UI verified)")
        assert (w, h) == (1280, 800)
        assert max(std_rgb) > 5.0, "Rendered UI screenshot appears blank!"

        print(
            "\nPHASE 7 STEP 2 (`ui/index.html`, `ui/styles.css`, `ui/app.js`): PASS -> Odysseus Command Center UI verified!"
        )
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)
        center.close()


if __name__ == "__main__":
    main()
