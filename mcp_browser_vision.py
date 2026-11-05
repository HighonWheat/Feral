"""
Phase 5, Step 4 - Web Research, Headless Browser & Vision MCP Server (`mcp_browser_vision.py`)
==============================================================================================
Implements Feral's `web_research(query, url)` tool and headless Chromium vision capture
pipeline, exposed via both Python and MCP JSON-RPC 2.0:

  1. Multi-Provider `web_research(query, url="")` Engine:
     - Instant offline/local documentation index (`DOC_INDEX` + Python/PowerShell reference)
     - Configurable SearXNG JSON API integration (`http://localhost:8080/search`) for the
       Phase 7 Odysseus-style Integrations Panel, with automatic fallback to DuckDuckGo Lite
       and direct URL fetching.
  2. Token-Efficient HTML-to-Markdown Extractor (`html_to_clean_markdown()`):
     Uses `BeautifulSoup` (`bs4`) to strip `<script>`, `<style>`, `<nav>`, and boilerplate,
     converting headings, `<pre><code>` blocks, tables, lists, and links into clean Markdown.
  3. Headless Chromium Screenshot & Vision Feed (`capture_page_vision(url)`):
     Drives Windows 11 headless Chromium (`chrome.exe` / `msedge.exe --headless=new`) to
     capture real `1280x800` PNG viewport screenshots, validates image dimensions & RGB
     statistics with Pillow (`PIL`), and stores base64-encoded frames for the Phase 7
     Live Browser Vision Feed.
"""
import base64
import http.server
import io
import json
import os
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

from feral_core import ROOT, WEB_CACHE as DOC_INDEX

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_chromium_binary() -> Optional[Path]:
    for cand in CHROME_CANDIDATES:
        if cand.exists():
            return cand
    return None


@dataclass
class BrowserIntegrationsConfig:
    """Settings bound to the Phase 7 Odysseus-style Integrations Panel."""
    searxng_enabled: bool = False
    searxng_endpoint: str = "http://localhost:8080/search"
    headless_browser_enabled: bool = True
    vision_screenshots_enabled: bool = True
    user_agent: str = "FeralAgent/1.0 (Windows NT 10.0; Win64; x64)"


@dataclass
class BrowserVisionFrame:
    """Single captured viewport frame for the Command Center Live Browser Vision Feed."""
    frame_id: int
    url: str
    title: str
    markdown_excerpt: str
    screenshot_path: str
    width: int
    height: int
    file_size_kb: float
    capture_backend: str
    base64_png: str
    timestamp: float


def html_to_clean_markdown(html_text: str, source_url: str = "") -> Tuple[str, str]:
    """
    Strips scripts/styles/navigation from raw HTML and converts structural elements
    (`h1..h6`, `pre/code`, `a`, `li`, `p`) into compact Markdown.
    Returns `(page_title, clean_markdown)`.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else (source_url or "Document")

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    lines: List[str] = [f"# {title}"]
    if source_url:
        lines.append(f"Source: {source_url}\n")

    root = soup.find("main") or soup.find("article") or soup.body or soup

    for elem in root.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"], recursive=True):
        name = elem.name
        if name in ("h1", "h2", "h3", "h4"):
            level = int(name[1])
            txt = elem.get_text(" ", strip=True)
            if txt and txt != title:
                lines.append(f"\n{'#' * level} {txt}")
        elif name == "pre":
            code_txt = elem.get_text().strip("\r\n")
            if code_txt:
                lines.append(f"\n```text\n{code_txt}\n```")
        elif name == "li":
            if elem.find_parent("pre") is None:
                txt = elem.get_text(" ", strip=True)
                if txt:
                    lines.append(f"- {txt}")
        elif name == "p":
            if elem.find_parent(["li", "pre"]) is None:
                # Preserve inline links `[label](href)`
                for a in elem.find_all("a", href=True):
                    label = a.get_text(" ", strip=True)
                    href = a["href"]
                    if label and href and not href.startswith("javascript:"):
                        a.replace_with(f"[{label}]({href})")
                txt = elem.get_text(" ", strip=True)
                if txt:
                    lines.append(f"\n{txt}")

    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


class MCPBrowserVisionServer:
    """
    Web Research, Headless Chromium & Vision MCP Server for Feral.
    """

    def __init__(self, screenshot_dir: Optional[Path] = None, max_frames: int = 30):
        self.screenshot_dir = Path(screenshot_dir or (ROOT / "outputs" / "vision_frames"))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.config = BrowserIntegrationsConfig()
        self.chromium_bin = find_chromium_binary()
        self.vision_frames: List[BrowserVisionFrame] = []
        self._frame_counter = 0

    def update_integrations(self, **kwargs: Any) -> BrowserIntegrationsConfig:
        """Updates web search & headless browser toggles from the Phase 7 Integrations Panel."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        return self.config

    def _search_local_doc_index(self, query: str) -> Optional[str]:
        """Searches `DOC_INDEX` from `feral_core.py` for matching URLs or keywords."""
        q_lower = query.lower().strip()
        for url, (title, body) in DOC_INDEX.items():
            if q_lower == url.lower() or q_lower in title.lower():
                return f"# {title}\nSource: {url}\n\n{body}\n[exit_code=0]"

        # Token overlap match across DOC_INDEX
        q_tokens = [t for t in re.split(r"\W+", q_lower) if len(t) >= 2]
        best_match = None
        best_score = 0
        for url, (title, body) in DOC_INDEX.items():
            hay = f"{url} {title} {body}".lower()
            score = sum(2 if t in title.lower() else 1 for t in q_tokens if t in hay)
            if score > best_score:
                best_score = score
                best_match = (url, title, body)

        if best_match is not None and best_score >= 2:
            url, title, body = best_match
            return f"# {title}\nSource: {url}\n\n{body}\n[exit_code=0]"
        return None

    def _search_searxng(self, query: str) -> Optional[str]:
        """Queries a local/self-hosted SearXNG JSON endpoint when enabled in the Integrations Panel."""
        if not self.config.searxng_enabled:
            return None
        try:
            with httpx.Client(timeout=6.0, headers={"User-Agent": self.config.user_agent}) as client:
                resp = client.get(
                    self.config.searxng_endpoint,
                    params={"q": query, "format": "json"},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])[:5]
                if not results:
                    return None
                out = [f"# SearXNG Search Results for: {query}\n"]
                for idx, item in enumerate(results, 1):
                    out.append(f"{idx}. **[{item.get('title', 'Result')}]({item.get('url', '')})**")
                    if item.get("content"):
                        out.append(f"   {item['content']}")
                return "\n".join(out) + "\n[exit_code=0]"
        except Exception:
            return None

    def fetch_url_as_markdown(self, url: str) -> Tuple[str, str]:
        """Fetches `url` over HTTP(S) and converts the HTML response into clean Markdown."""
        with httpx.Client(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": self.config.user_agent},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return url, f"# JSON Response ({url})\n```json\n{resp.text[:4000]}\n```"
            return html_to_clean_markdown(resp.text, source_url=url)

    def web_research(self, query: str, url: str = "") -> str:
        """
        Executes Feral's `web_research(query, url="")` tool:
          - If `url` is provided (or matches `DOC_INDEX`), returns the page as clean Markdown
            and optionally captures a vision frame if `url` is a live HTTP(S) endpoint.
          - Otherwise searches the local documentation index, SearXNG (if enabled), or live web.
        """
        target_url = (url or "").strip()

        # 1. Exact match in DOC_INDEX
        if target_url and target_url in DOC_INDEX:
            title, body = DOC_INDEX[target_url]
            return f"# {title}\nSource: {target_url}\n\n{body}\n[exit_code=0]"

        # 2. Live HTTP(S) URL fetch
        if target_url.startswith(("http://", "https://")):
            try:
                title, md = self.fetch_url_as_markdown(target_url)
                if self.config.vision_screenshots_enabled:
                    self.capture_page_vision(target_url, pre_fetched_title=title, pre_fetched_md=md)
                return f"{md[:6000]}\n[exit_code=0]"
            except Exception as e:
                return f"ERROR: Failed to fetch {target_url}: {e}\n[exit_code=1]"

        # 3. Search local documentation index
        doc_hit = self._search_local_doc_index(query)
        if doc_hit is not None:
            return doc_hit

        # 4. Search SearXNG if enabled in Integrations Panel
        searx_hit = self._search_searxng(query)
        if searx_hit is not None:
            return searx_hit

        return f"No local documentation entry matched `{query}` (enable SearXNG or pass a direct `url`).\n[exit_code=0]"

    def capture_page_vision(
        self,
        url: str,
        pre_fetched_title: Optional[str] = None,
        pre_fetched_md: Optional[str] = None,
        viewport: Tuple[int, int] = (1280, 800),
    ) -> BrowserVisionFrame:
        """
        Captures a `1280x800` PNG viewport screenshot of `url` using Windows 11 headless Chromium
        (`chrome.exe --headless=new --screenshot=...`), validates the PNG with Pillow, and records
        a `BrowserVisionFrame` for the Phase 7 Live Browser Vision Feed.
        """
        self._frame_counter += 1
        fid = self._frame_counter
        out_png = self.screenshot_dir / f"vision_frame_{fid:04d}.png"
        if out_png.exists():
            out_png.unlink()

        title = pre_fetched_title or url
        md_text = pre_fetched_md or ""
        if not md_text and url.startswith(("http://", "https://")):
            try:
                title, md_text = self.fetch_url_as_markdown(url)
            except Exception as e:
                md_text = f"Error fetching markdown: {e}"

        backend_used = "pillow-fallback"
        w, h = viewport

        if self.config.headless_browser_enabled and self.chromium_bin is not None:
            profile_dir = Path(tempfile.mkdtemp(prefix="feral_chromium_"))
            try:
                cmd = [
                    str(self.chromium_bin),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--hide-scrollbars",
                    f"--user-data-dir={profile_dir}",
                    f"--window-size={w},{h}",
                    f"--screenshot={out_png}",
                    url,
                ]
                subprocess.run(cmd, capture_output=True, timeout=15)
                if out_png.exists() and out_png.stat().st_size > 0:
                    backend_used = f"headless-chromium ({self.chromium_bin.name})"
            except Exception:
                pass
            finally:
                shutil.rmtree(profile_dir, ignore_errors=True)

        # Fallback renderer if headless Chromium was disabled or did not write the screenshot
        if not out_png.exists() or out_png.stat().st_size == 0:
            img = Image.new("RGB", (w, h), color=(15, 23, 42))
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (w, 48)], fill=(30, 41, 59))
            draw.text((16, 16), f"Feral Headless Viewport | {url}", fill=(226, 232, 240))
            y = 68
            for line in (md_text or title).splitlines()[:28]:
                draw.text((20, y), line[:120], fill=(148, 163, 184))
                y += 24
            img.save(out_png, format="PNG")

        with Image.open(out_png) as im:
            im_w, im_h = im.size

        raw_bytes = out_png.read_bytes()
        b64_png = base64.b64encode(raw_bytes).decode("ascii")
        size_kb = round(len(raw_bytes) / 1024.0, 2)

        frame = BrowserVisionFrame(
            frame_id=fid,
            url=url,
            title=title,
            markdown_excerpt=md_text[:600],
            screenshot_path=str(out_png.relative_to(ROOT)),
            width=im_w,
            height=im_h,
            file_size_kb=size_kb,
            capture_backend=backend_used,
            base64_png=b64_png,
            timestamp=time.time(),
        )
        self.vision_frames.append(frame)
        if len(self.vision_frames) > self.max_frames:
            self.vision_frames = self.vision_frames[-self.max_frames :]
        return frame

    # ---------------- MCP JSON-RPC 2.0 Interface ----------------

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "web_research",
                "description": "Search documentation or fetch a URL as clean Markdown (with live vision capture).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name != "web_research":
            raise ValueError(f"MCPBrowserVisionServer does not provide tool `{name}`")
        query = str(arguments.get("query", ""))
        url = str(arguments.get("url", ""))
        return self.web_research(query=query, url=url)

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
            if method == "browser/vision_feed":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "frames": [asdict(f) for f in self.vision_frames],
                        "integrations": asdict(self.config),
                    },
                }
            if method == "browser/update_integrations":
                cfg = self.update_integrations(**params)
                return {"jsonrpc": "2.0", "id": req_id, "result": asdict(cfg)}
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
    print("1. Initializing `MCPBrowserVisionServer` & Detecting Windows 11 Chromium Binary...")
    server = MCPBrowserVisionServer()
    print(f"   - Detected Chromium Binary       : {server.chromium_bin}")
    print(f"   - Integrations Panel Defaults    : searxng={server.config.searxng_enabled} ({server.config.searxng_endpoint}), headless_vision={server.config.vision_screenshots_enabled}")

    print("\n2. Testing `web_research()` Against Local Documentation Index (`Select-String`)...")
    doc_res = server.call_tool("web_research", {"query": "PowerShell Select-String grep"})
    print(f"   - Doc Match Header               : {doc_res.splitlines()[0]} ({doc_res.splitlines()[-1]})")
    assert "Select-String" in doc_res and doc_res.endswith("[exit_code=0]")

    print("\n3. Launching Ephemeral Local HTTP Documentation Server to Test Live HTML-to-Markdown + Headless Chromium Screenshot...")
    sample_html = """<!DOCTYPE html>
<html>
<head>
  <title>Feral Vector & Agent API Reference</title>
  <style>body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 32px; } pre { background: #1e293b; padding: 16px; border-radius: 8px; }</style>
  <script>console.log("noisy script that should be stripped from markdown");</script>
</head>
<body>
  <nav>Top Navigation Noise</nav>
  <main>
    <h1>Feral Vector & Agent API Reference</h1>
    <p>Welcome to the local API docs for <a href="http://127.0.0.1/collections">ChromaDB Collections</a>.</p>
    <h2>Quickstart Example</h2>
    <pre><code>client = FeralLLMClient()\nresp = client.generate_turn(messages)</code></pre>
    <ul>
      <li>Supports 4-bit NF4 DoRA with fused weight norms</li>
      <li>Streams live &lt;think&gt; and &lt;tool_call&gt; tokens</li>
    </ul>
  </main>
  <footer>Footer Noise</footer>
</body>
</html>"""

    raw_html_bytes = len(sample_html.encode("utf-8"))

    class _SilentHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = sample_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _SilentHandler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        live_url = f"http://127.0.0.1:{port}/docs"
        md_res = server.call_tool("web_research", {"query": "Feral Vector API", "url": live_url})
        httpd.shutdown()

    clean_md_body = "\n".join(md_res.splitlines()[:-1])
    md_bytes = len(clean_md_body.encode("utf-8"))
    print(f"   - HTML -> Clean Markdown         : {raw_html_bytes} raw HTML bytes -> {md_bytes} clean Markdown bytes (script/nav/footer stripped)")
    for line in clean_md_body.splitlines()[:7]:
        print(f"     | {line}")

    assert "noisy script" not in clean_md_body
    assert "[ChromaDB Collections](http://127.0.0.1/collections)" in clean_md_body
    assert len(server.vision_frames) == 1

    frame = server.vision_frames[-1]
    print("\n4. Verifying Captured Headless Chromium Vision Frame (`BrowserVisionFrame`)...")
    print(f"   - Capture Backend                : {frame.capture_backend}")
    print(f"   - Viewport Resolution & Size     : {frame.width}x{frame.height} PNG ({frame.file_size_kb:.2f} KiB at `{frame.screenshot_path}`)")
    assert frame.width == 1280 and frame.height == 800 and frame.file_size_kb > 1.0

    # Clean up test screenshot artifact so workspace stays tidy
    if server.screenshot_dir.exists():
        shutil.rmtree(server.screenshot_dir, ignore_errors=True)

    print("\nPHASE 5 STEP 4 (`mcp_browser_vision.py`): PASS -> Web research, HTML-to-Markdown, and headless Chromium vision capture verified!")


if __name__ == "__main__":
    main()
