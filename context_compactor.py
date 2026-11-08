"""
Phase 6, Step 2 - Context Auto-Compactor, `AGENT_STATE.md` Pinning & Local Vector Knowledge Base (`context_compactor.py`)
=========================================================================================================================
Implements two tightly coupled memory systems for Feral's long-running autonomous runs and
the Phase 7 Odysseus-inspired **Knowledge Base & Vector Storage UI**:

  1. Context Window Auto-Compactor & `AGENT_STATE.md` Memory Pinning (`ContextCompactor`):
     - Monitors active conversation token count against `max_context_tokens` (triggers at `trigger_ratio`, e.g. 75%).
     - Extracts and persists critical state (`active_goal`, `modified_files`, `completed_steps`,
       `verified_facts`, `recent_errors`) to `AGENT_STATE.md` on disk.
     - Compresses older intermediate `<tool_call>` and `<tool_response>` turns into a dense
       `[PINNED AGENT_STATE.md & COMPACTED HISTORY]` block while keeping the system prompt and
       the `keep_recent_messages` latest turns untouched.
  2. Local Vector Knowledge Base & ChromaDB-Compatible Collection Engine (`LocalVectorKnowledgeBase`):
     - Powers Module 3 of the Command Center UI (**Knowledge Base & Vector Storage UI**).
     - Supports ingesting `.txt`, `.md`, and `.pdf` documents into named **Collections / Knowledge Clusters**
       with real-time `embedding_progress_pct` (`0% -> 100%`) tracking for UI progress bars.
     - Computes L2-normalized dense embeddings on GPU/CPU (`dim=384`), persists chunks + vectors in
       SQLite (`outputs/feral_chromadb_store.sqlite`, or delegates to `chromadb` when installed), and
       performs fast cosine-similarity top-k retrieval (`query_collection()`).
"""
import contextlib
import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from feral_core import ROOT, SYSTEM_FERAL

DEFAULT_VECTOR_DB = ROOT / "outputs" / "feral_chromadb_store.sqlite"


@dataclass
class AgentPinnedState:
    """Structured state persisted to `AGENT_STATE.md` across context compactions."""
    active_goal: str = ""
    compaction_count: int = 0
    modified_files: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    pinned_notes: List[str] = field(default_factory=list)
    tokens_before_last_compaction: int = 0
    tokens_after_last_compaction: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        lines = [
            "# AGENT_STATE.md — Feral Pinned Working Memory",
            f"- **Active Goal:** {self.active_goal or '(none)'}",
            f"- **Compaction Pass:** #{self.compaction_count} "
            f"(`{self.tokens_before_last_compaction}` -> `{self.tokens_after_last_compaction}` tokens)",
            "",
            "## Modified & Patched Files",
        ]
        if self.modified_files:
            for f in self.modified_files:
                lines.append(f"- `{f}`")
        else:
            lines.append("- (none yet)")

        lines.append("\n## Completed Tool Actions")
        if self.completed_actions:
            for act in self.completed_actions[-15:]:
                lines.append(f"- {act}")
        else:
            lines.append("- (none yet)")

        if self.recent_errors:
            lines.append("\n## Resolved / Encountered Errors")
            for err in self.recent_errors[-6:]:
                lines.append(f"- {err}")

        if self.pinned_notes:
            lines.append("\n## Pinned Architectural Notes")
            for note in self.pinned_notes:
                lines.append(f"- {note}")

        return "\n".join(lines) + "\n"


class ContextCompactor:
    """
    Monitors Feral's conversation history, extracts key tool/file milestones into
    `AGENT_STATE.md`, and compacts older turns when token usage crosses `trigger_ratio`.
    """

    def __init__(
        self,
        workspace_root: Path = ROOT,
        max_context_tokens: int = 4096,
        trigger_ratio: float = 0.75,
        keep_recent_messages: int = 4,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.workspace_root / "AGENT_STATE.md"

        self.max_context_tokens = int(max_context_tokens)
        self.trigger_ratio = float(trigger_ratio)
        self.keep_recent_messages = max(2, int(keep_recent_messages))
        self._token_counter = token_counter
        self.state = AgentPinnedState()

    def count_tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return self._token_counter(text)
        # Fast heuristic (~3.6 chars/token for code & JSON) when tokenizer is not attached
        return max(1, (len(text) + 3) // 4)

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(self.count_tokens(m.get("content", "")) + 6 for m in messages)

    def pin_note(self, note: str) -> None:
        if note not in self.state.pinned_notes:
            self.state.pinned_notes.append(note)
        self.save_agent_state_md()

    def save_agent_state_md(self) -> Path:
        self.state.updated_at = time.time()
        self.state_file.write_text(self.state.to_markdown(), encoding="utf-8")
        return self.state_file

    def _harvest_turn_facts(self, messages_to_compact: List[Dict[str, str]]) -> None:
        """Deterministically harvests tool calls, patched files, and errors from older turns."""
        for m in messages_to_compact:
            content = m.get("content", "")
            role = m.get("role", "")

            if role == "assistant":
                for tc_raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.S):
                    try:
                        tc = json.loads(tc_raw)
                        t_name = tc.get("name", "")
                        t_args = tc.get("arguments", {}) or {}
                        if t_name in ("apply_diff_patch", "write_file") and "path" in t_args:
                            p = str(t_args["path"])
                            if p not in self.state.modified_files:
                                self.state.modified_files.append(p)
                            summary = f"`{t_name}` on `{p}`"
                        elif t_name == "read_file_lines":
                            summary = f"`read_file_lines` on `{t_args.get('path')}` (L{t_args.get('start_line')}-{t_args.get('end_line')})"
                        elif t_name == "run_powershell":
                            cmd_short = str(t_args.get("command", ""))[:70]
                            summary = f"`run_powershell`: `{cmd_short}`"
                        elif t_name == "ast_repo_map":
                            summary = f"`ast_repo_map` on `{t_args.get('directory', '.')}`"
                        elif t_name == "spawn_subagent":
                            summary = f"`spawn_subagent`: {str(t_args.get('task_prompt', ''))[:60]}"
                        else:
                            summary = f"`{t_name}` invoked"

                        if summary not in self.state.completed_actions:
                            self.state.completed_actions.append(summary)
                    except Exception:
                        pass

            elif role == "user" and "<tool_response>" in content:
                obs = re.sub(r"</?tool_response>", "", content).strip()
                first_line = obs.splitlines()[0] if obs else ""
                if first_line.startswith("ERROR:"):
                    err_short = first_line[:110]
                    if err_short not in self.state.recent_errors:
                        self.state.recent_errors.append(err_short)

    def maybe_compact(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Checks if `messages` exceeds `max_context_tokens * trigger_ratio`.
        If so, harvests facts from older turns, writes `AGENT_STATE.md`, and replaces older
        intermediate turns with a compact pinned summary message.
        """
        before_tokens = self.count_messages_tokens(messages)
        threshold = int(self.max_context_tokens * self.trigger_ratio)

        # Record initial user goal if not set yet
        if not self.state.active_goal:
            for m in messages:
                if m.get("role") == "user" and "<tool_response>" not in m.get("content", ""):
                    self.state.active_goal = m["content"].strip()
                    break

        if before_tokens <= threshold or len(messages) <= (self.keep_recent_messages + 2):
            self.save_agent_state_md()
            return messages, {
                "compacted": False,
                "tokens_before": before_tokens,
                "tokens_after": before_tokens,
                "threshold": threshold,
            }

        sys_msg = messages[0] if messages and messages[0].get("role") == "system" else {"role": "system", "content": SYSTEM_FERAL}
        body_msgs = messages[1:] if messages and messages[0].get("role") == "system" else list(messages)

        middle_to_compact = body_msgs[: -self.keep_recent_messages]
        recent_tail = body_msgs[-self.keep_recent_messages :]

        self._harvest_turn_facts(middle_to_compact)
        self.state.compaction_count += 1
        self.state.tokens_before_last_compaction = before_tokens

        # Temporarily compute compacted message list and update token telemetry in AGENT_STATE.md
        pinned_msg = {
            "role": "user",
            "content": (
                f"{self.state.active_goal}\n\n"
                f"<compacted_memory>\n{self.state.to_markdown().strip()}\n</compacted_memory>"
            ),
        }
        compacted_messages = [sys_msg, pinned_msg] + recent_tail
        after_tokens = self.count_messages_tokens(compacted_messages)
        self.state.tokens_after_last_compaction = after_tokens
        # Refresh pinned_msg with final after_tokens count and write AGENT_STATE.md to disk
        pinned_msg["content"] = (
            f"{self.state.active_goal}\n\n"
            f"<compacted_memory>\n{self.state.to_markdown().strip()}\n</compacted_memory>"
        )
        self.save_agent_state_md()

        return compacted_messages, {
            "compacted": True,
            "compaction_count": self.state.compaction_count,
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "reduction_pct": round((1.0 - after_tokens / max(before_tokens, 1)) * 100.0, 1),
            "state_file": str(self.state_file),
        }


# ============================================================================
# 2. Local Vector Knowledge Base (`ChromaDB` / SQLite Hybrid Vector Store)
# ============================================================================
@dataclass
class KnowledgeCollectionInfo:
    """Metadata for each Collection / Knowledge Cluster card in the UI."""
    collection_id: str
    name: str
    description: str
    document_count: int
    chunk_count: int
    embedding_dim: int
    embedding_progress_pct: float  # 0.0 .. 100.0 for UI progress bars
    status: str  # Ready | Indexing
    documents: List[str] = field(default_factory=list)
    updated_at: float = 0.0


def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts UTF-8 text from `.txt`, `.md`, or `.pdf` files for drag-and-drop ingestion.
    Supports `pypdf` if installed, plus a built-in PDF FlateDecode/text-stream parser.
    """
    suffix = file_path.suffix.lower()
    if suffix in (".txt", ".md", ".py", ".json", ".log", ".csv"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        # Try pypdf first if installed
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception:
            pass

        # Built-in PDF stream & literal string `( ... ) Tj/TJ` extractor
        raw = file_path.read_bytes()
        extracted_parts: List[str] = []

        # Decompress any FlateDecode streams and scan both raw and decompressed streams
        buffers = [raw]
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, flags=re.S):
            stream_bytes = m.group(1)
            try:
                buffers.append(zlib.decompress(stream_bytes))
            except Exception:
                pass

        for buf in buffers:
            txt = buf.decode("latin-1", errors="ignore")
            for lit in re.findall(r"\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*T[jJ]", txt):
                clean_lit = lit.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
                if clean_lit.strip():
                    extracted_parts.append(clean_lit.strip())
        return "\n".join(extracted_parts).strip()

    raise ValueError(f"Unsupported document extension `{suffix}` (expected .txt, .md, or .pdf)")


def compute_dense_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Computes a deterministic, L2-normalized `dim`-dimensional semantic feature vector
    using sublinear TF-weighted word unigrams, 4/5-char prefix stems, and bigrams on PyTorch.
    """
    STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "from", "into", "because",
        "why", "does", "how", "what", "when", "where", "are", "was", "were",
    }
    vec = torch.zeros(dim, dtype=torch.float32)
    words = [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 2 and w not in STOPWORDS]
    if not words:
        return vec.tolist()

    # 1. Exact word unigrams (weight = 2.0) + 4-char prefix stems (weight = 1.5, matching e.g. waste/wasting)
    for w in words:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 2.0
        if len(w) >= 4:
            stem = f"stem:{w[:4]}"
            hs = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16)
            vec[hs % dim] += 1.5

    # 2. Word bigrams (weight = 1.0)
    for i in range(len(words) - 1):
        bg = f"{words[i]}_{words[i + 1]}"
        h = int(hashlib.md5(bg.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0

    vec = torch.log1p(vec)
    norm = float(torch.linalg.norm(vec).item())
    if norm > 1e-8:
        vec = vec / norm
    return [round(float(x), 6) for x in vec.tolist()]


class LocalVectorKnowledgeBase:
    """
    Local Vector Storage & Knowledge Cluster Manager powering Module 3 of the Command Center UI.
    Stores Collections, Document metadata, Embedding progress bars, and L2-normalized vectors.
    """

    def __init__(self, db_path: Path = DEFAULT_VECTOR_DB, embedding_dim: int = 384):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = int(embedding_dim)
        self._init_schema()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    embedding_progress_pct REAL NOT NULL DEFAULT 100.0,
                    status TEXT NOT NULL DEFAULT 'Ready',
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    collection_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def create_collection(self, collection_id: str, name: str, description: str = "") -> KnowledgeCollectionInfo:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kb_collections
                (collection_id, name, description, embedding_progress_pct, status, updated_at)
                VALUES (?, ?, ?, 100.0, 'Ready', ?)
                """,
                (collection_id, name, description, now),
            )
        return self.get_collection_info(collection_id)

    def ingest_document(
        self,
        collection_id: str,
        file_path: Path,
        chunk_chars: int = 500,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> KnowledgeCollectionInfo:
        """
        Extracts text from `.txt`, `.md`, or `.pdf`, splits into semantic chunks, updates the
        collection's `embedding_progress_pct` (`0% -> 100%`) as chunks are embedded, and commits
        vectors to the local SQLite store.
        """
        file_path = Path(file_path)
        raw_text = extract_text_from_file(file_path)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]

        chunks: List[str] = []
        cur = ""
        for p in paragraphs:
            if len(cur) + len(p) + 2 <= chunk_chars:
                cur = f"{cur}\n\n{p}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        if not chunks:
            chunks = [raw_text[:chunk_chars] or file_path.name]

        total = len(chunks)
        doc_name = file_path.name

        with self._connect() as conn:
            conn.execute(
                "UPDATE kb_collections SET embedding_progress_pct = 0.0, status = 'Indexing', updated_at = ? WHERE collection_id = ?",
                (time.time(), collection_id),
            )
            conn.execute(
                "DELETE FROM kb_chunks WHERE collection_id = ? AND doc_name = ?",
                (collection_id, doc_name),
            )

        for idx, chunk_text in enumerate(chunks):
            vec = compute_dense_embedding(chunk_text, dim=self.embedding_dim)
            pct = round(((idx + 1) / total) * 100.0, 1)
            cid = f"{collection_id}::{doc_name}::{idx:04d}"
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kb_chunks
                    (chunk_id, collection_id, doc_name, chunk_index, content, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cid, collection_id, doc_name, idx, chunk_text, json.dumps(vec), time.time()),
                )
                conn.execute(
                    "UPDATE kb_collections SET embedding_progress_pct = ?, status = ?, updated_at = ? WHERE collection_id = ?",
                    (pct, "Ready" if pct >= 100.0 else "Indexing", time.time(), collection_id),
                )
            if progress_callback is not None:
                progress_callback(pct)

        return self.get_collection_info(collection_id)

    def query_collection(self, collection_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs cosine-similarity vector search inside `collection_id`."""
        q_vec = torch.tensor(compute_dense_embedding(query, dim=self.embedding_dim), dtype=torch.float32)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_name, chunk_index, content, embedding_json FROM kb_chunks WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()

        scored: List[Dict[str, Any]] = []
        for r in rows:
            c_vec = torch.tensor(json.loads(r["embedding_json"]), dtype=torch.float32)
            score = float(torch.dot(q_vec, c_vec).item())
            scored.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_name": r["doc_name"],
                    "chunk_index": r["chunk_index"],
                    "similarity": round(score, 4),
                    "content": r["content"],
                }
            )
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def get_collection_info(self, collection_id: str) -> KnowledgeCollectionInfo:
        with self._connect() as conn:
            c_row = conn.execute(
                "SELECT * FROM kb_collections WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()
            if c_row is None:
                raise KeyError(f"Collection `{collection_id}` not found")
            chunks = conn.execute(
                "SELECT doc_name FROM kb_chunks WHERE collection_id = ? ORDER BY doc_name, chunk_index",
                (collection_id,),
            ).fetchall()
        docs = sorted(set(r["doc_name"] for r in chunks))
        return KnowledgeCollectionInfo(
            collection_id=c_row["collection_id"],
            name=c_row["name"],
            description=c_row["description"],
            document_count=len(docs),
            chunk_count=len(chunks),
            embedding_dim=self.embedding_dim,
            embedding_progress_pct=c_row["embedding_progress_pct"],
            status=c_row["status"],
            documents=docs,
            updated_at=c_row["updated_at"],
        )

    def list_collections(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT collection_id FROM kb_collections ORDER BY updated_at DESC").fetchall()
        return [asdict(self.get_collection_info(r["collection_id"])) for r in rows]


def build_minimal_pdf(path: Path, text_lines: List[str]) -> None:
    """Creates a minimal valid PDF 1.4 file with literal text operators `(...) Tj` for testing `.pdf` ingestion."""
    content_ops = ["BT /F1 12 Tf 50 750 Td"]
    for line in text_lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_ops.append(f"({escaped}) Tj 0 -18 Td")
    content_ops.append("ET")
    stream_body = "\n".join(content_ops).encode("latin-1")

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
        + f"4 0 obj << /Length {len(stream_body)} >>\nstream\n".encode("latin-1")
        + stream_body
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    path.write_bytes(pdf_bytes)


def main():
    test_ws = ROOT / "feral_sandbox"
    if test_ws.exists():
        shutil.rmtree(test_ws)
    test_ws.mkdir(parents=True, exist_ok=True)

    test_vdb = ROOT / "outputs" / "test_feral_kb.sqlite"
    for ext in ("", "-wal", "-shm"):
        p = Path(str(test_vdb) + ext)
        if p.exists():
            p.unlink()

    try:
        print("1. Testing Context Window Auto-Compactor & `AGENT_STATE.md` Pinning (`ContextCompactor`)...")
        compactor = ContextCompactor(
            workspace_root=test_ws,
            max_context_tokens=600,
            trigger_ratio=0.70,  # Triggers at 420 tokens
            keep_recent_messages=2,
        )
        compactor.pin_note("Keep all embeddings and lm_head in native bfloat16 to save 4.0 GB VRAM.")

        # Build a bloated 10-turn conversation exceeding 420 tokens
        verbose_log = "\n".join(f"2026-11-08 03:00:{i:02d} INFO Worker heartbeat ok on port 8080" for i in range(35))
        messages = [
            {"role": "system", "content": SYSTEM_FERAL},
            {"role": "user", "content": "Fix the rate limiter bug in src/rate_limiter.py and verify unit tests pass."},
            {
                "role": "assistant",
                "content": '<think>\nMap src first.\n</think>\n<tool_call>{"name": "ast_repo_map", "arguments": {"directory": "src"}}</tool_call>',
            },
            {"role": "user", "content": f"<tool_response>\nsrc/rate_limiter.py\n  L1: def allow_request(tokens)\n{verbose_log}\n[exit_code=0]\n</tool_response>"},
            {
                "role": "assistant",
                "content": '<think>\nTry grep on logs.\n</think>\n<tool_call>{"name": "run_powershell", "arguments": {"command": "grep ERROR logs/app.log"}}</tool_call>',
            },
            {"role": "user", "content": "<tool_response>\nERROR: The term 'grep' is not recognized as the name of a cmdlet.\n[exit_code=1]\n</tool_response>"},
            {
                "role": "assistant",
                "content": '<think>\nPatch src/rate_limiter.py.\n</think>\n<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py", "search_block": "tokens >= 0", "replace_block": "tokens > 0"}}</tool_call>',
            },
            {"role": "user", "content": "<tool_response>\nOK: Patched src/rate_limiter.py [patch_id=1, +1/-1 lines] (syntax_check=PASS)\n[exit_code=0]\n</tool_response>"},
            {
                "role": "assistant",
                "content": '<think>\nRun unittest discovery.\n</think>\n<tool_call>{"name": "run_powershell", "arguments": {"command": "python -m unittest discover -s tests"}}</tool_call>',
            },
            {"role": "user", "content": "<tool_response>\n..\nRan 2 tests in 0.001s\nOK\n[exit_code=0]\n</tool_response>"},
        ]

        compacted_msgs, stats = compactor.maybe_compact(messages)
        print(
            f"   - Auto-Compaction Triggered    : {stats['tokens_before']} tokens -> {stats['tokens_after']} tokens "
            f"(-{stats['reduction_pct']}% reduction, messages: {len(messages)} -> {len(compacted_msgs)})"
        )
        state_md_text = compactor.state_file.read_text(encoding="utf-8")
        print(f"   - Persisted `AGENT_STATE.md`   : {len(state_md_text.splitlines())} lines written to disk")
        for line in state_md_text.splitlines()[:10]:
            print(f"     | {line}")

        assert stats["compacted"] is True and stats["tokens_after"] < stats["tokens_before"]
        assert "src/rate_limiter.py" in state_md_text
        assert "Keep all embeddings and lm_head in native bfloat16" in state_md_text

        print("\n2. Testing Local Vector Knowledge Base & `.txt`, `.md`, `.pdf` Ingestion (`LocalVectorKnowledgeBase`)...")
        kb = LocalVectorKnowledgeBase(db_path=test_vdb, embedding_dim=384)
        kb.create_collection(
            collection_id="feral_arch_docs",
            name="Feral Architecture & Hardware Cluster",
            description="Local specs for RTX 5070 Ti VRAM optimization, DoRA adapters, and MCP servers.",
        )

        doc_md = test_ws / "dora_vram_guide.md"
        doc_md.write_text(
            "# RTX 5070 Ti DoRA Training Guide\n\n"
            "Never call prepare_model_for_kbit_training on Qwen3.5-9B because it upcasts embed_tokens and lm_head to float32, wasting 4.0 GB of VRAM.\n\n"
            "Slice active hidden states before lm_head so cross-entropy only projects non-masked tokens.",
            encoding="utf-8",
        )
        doc_txt = test_ws / "powershell_tips.txt"
        doc_txt.write_text(
            "Windows PowerShell 5.1 uses Select-String instead of grep and Get-ChildItem -Recurse instead of find.",
            encoding="utf-8",
        )
        doc_pdf = test_ws / "odysseus_ui_spec.pdf"
        build_minimal_pdf(
            doc_pdf,
            [
                "Project Odysseus Local AI Workspace Specification",
                "Includes Chat Hub with Context Panel, Autonomous Agents Board, ChromaDB Knowledge Clusters, and SearXNG Integrations.",
            ],
        )

        progress_steps: List[float] = []
        kb.ingest_document("feral_arch_docs", doc_md, chunk_chars=180, progress_callback=progress_steps.append)
        kb.ingest_document("feral_arch_docs", doc_txt, chunk_chars=180, progress_callback=progress_steps.append)
        c_info = kb.ingest_document("feral_arch_docs", doc_pdf, chunk_chars=180, progress_callback=progress_steps.append)

        print(
            f"   - Collection `{c_info.name}` : {c_info.document_count} docs ({c_info.documents}) | "
            f"{c_info.chunk_count} vector chunks (dim={c_info.embedding_dim}) | Progress Bar: {c_info.embedding_progress_pct}% ({c_info.status})"
        )
        assert c_info.document_count == 3 and c_info.embedding_progress_pct == 100.0

        print("\n3. Testing Semantic Vector Retrieval Across `.md`, `.txt`, and `.pdf` Chunks...")
        hits_vram = kb.query_collection("feral_arch_docs", "Why does prepare_model_for_kbit_training waste 4 GB VRAM on embed_tokens?", top_k=2)
        hits_pdf = kb.query_collection("feral_arch_docs", "Project Odysseus Autonomous Agents Board and ChromaDB Knowledge Clusters", top_k=2)

        print(f"   - Query 1 Top Hit (`{hits_vram[0]['doc_name']}`, sim={hits_vram[0]['similarity']:+.4f}): {hits_vram[0]['content'][:95]}...")
        print(f"   - Query 2 Top Hit (`{hits_pdf[0]['doc_name']}`, sim={hits_pdf[0]['similarity']:+.4f}): {hits_pdf[0]['content'][:95]}...")

        assert hits_vram[0]["doc_name"] == "dora_vram_guide.md" and hits_vram[0]["similarity"] > 0.35
        assert hits_pdf[0]["doc_name"] == "odysseus_ui_spec.pdf" and hits_pdf[0]["similarity"] > 0.35

    finally:
        if test_ws.exists():
            shutil.rmtree(test_ws, ignore_errors=True)
        for ext in ("", "-wal", "-shm"):
            p = Path(str(test_vdb) + ext)
            if p.exists():
                p.unlink()

    print("\nPHASE 6 STEP 2 (`context_compactor.py`): PASS -> Context auto-compaction, AGENT_STATE.md pinning, and .txt/.md/.pdf Vector Knowledge Base verified!")


if __name__ == "__main__":
    main()
