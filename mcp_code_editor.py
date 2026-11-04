"""
Phase 5, Step 3 - Surgical Code Editor MCP Server (`mcp_code_editor.py`)
========================================================================
Implements Feral's 4 surgical repository and code-editing tools (`ast_repo_map`,
`read_file_lines`, `apply_diff_patch`, `write_file`) with `ast.parse` + `py_compile`
syntax verification, automatic rollback on invalid Python edits, and a unified-diff
history ledger for the Phase 7 Desktop Command Center Diff Viewer:

  1. `ast_repo_map(directory)`:
     Walks Python files under `directory` and extracts every `class`, `def`, `async def`,
     argument signature, and 1-based line number using Python's `ast` module.
  2. `read_file_lines(path, start_line, end_line)`:
     Reads only the requested 1-indexed inclusive line slice with formatted line numbers.
  3. `apply_diff_patch(path, search_block, replace_block)`:
     Enforces exact uniqueness (`count == 1`) of `search_block`, verifies `.py` syntax
     via `ast.parse()` and `py_compile.compile(doraise=True)`, auto-rolls back the file
     if syntax fails, and records a `DiffHistoryRecord` (`unified_diff`, `before_content`,
     `after_content`) on success.
  4. `write_file(path, content)`:
     Writes UTF-8 files inside the workspace jail with `.py` syntax verification and
     diff history tracking.
  5. `rollback_patch(patch_id)`:
     Reverts any recorded edit in a single call for the UI Diff Viewer & self-healing loop.
"""
import ast
import difflib
import os
import py_compile
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from feral_core import ROOT

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "lora_model",
    "outputs",
    "checkpoints",
    ".cache",
    ".huggingface",
    "feral_sandbox",
    "feral_ps_sandbox",
    "feral_grpo_sandbox",
}


@dataclass
class DiffHistoryRecord:
    """Audit & rollback entry for the Phase 7 Monaco Diff Viewer."""
    patch_id: int
    path: str
    tool_name: str
    before_content: str
    after_content: str
    unified_diff: str
    lines_added: int
    lines_removed: int
    timestamp: float
    rolled_back: bool = False


class MCPCodeEditorServer:
    """
    Surgical Code Editor MCP Server with workspace path jailing, `py_compile` verification,
    and unified diff history tracking.
    """

    def __init__(self, workspace_root: Path = ROOT, max_diff_history: int = 200):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.max_diff_history = max_diff_history
        self.diff_history: List[DiffHistoryRecord] = []
        self._patch_counter = 0

    def _resolve_safe_path(self, rel_path: str) -> Path:
        p = (self.workspace_root / rel_path).resolve()
        if p != self.workspace_root and self.workspace_root not in p.parents:
            raise ValueError(f"Path escapes workspace root: {rel_path}")
        return p

    def _verify_python_syntax(self, code_text: str, filename: str) -> Optional[str]:
        """
        Validates Python source code using both `ast.parse` and `py_compile.compile`.
        Returns `None` if valid, or a formatted `SyntaxError` diagnostic string if invalid.
        """
        try:
            ast.parse(code_text, filename=filename)
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", encoding="utf-8", delete=False
            ) as tf:
                tf.write(code_text)
                tmp_path = Path(tf.name)
            py_compile.compile(str(tmp_path), cfile=str(tmp_path) + "c", doraise=True)
            return None
        except py_compile.PyCompileError as e:
            return f"PyCompileError: {e.msg}"
        finally:
            if tmp_path is not None:
                for p in (tmp_path, Path(str(tmp_path) + "c")):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

    def _record_diff(
        self,
        rel_path: str,
        tool_name: str,
        before_text: str,
        after_text: str,
    ) -> DiffHistoryRecord:
        self._patch_counter += 1
        patch_id = self._patch_counter

        norm_rel = rel_path.replace("\\", "/")
        diff_lines = list(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"a/{norm_rel}",
                tofile=f"b/{norm_rel}",
                n=3,
            )
        )
        unified = "".join(diff_lines)
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        rec = DiffHistoryRecord(
            patch_id=patch_id,
            path=norm_rel,
            tool_name=tool_name,
            before_content=before_text,
            after_content=after_text,
            unified_diff=unified,
            lines_added=added,
            lines_removed=removed,
            timestamp=time.time(),
            rolled_back=False,
        )
        self.diff_history.append(rec)
        if len(self.diff_history) > self.max_diff_history:
            self.diff_history = self.diff_history[-self.max_diff_history :]
        return rec

    # ---------------- 4 Core Tools ----------------

    def ast_repo_map(self, directory: str = ".") -> str:
        """
        Walks `directory` recursively and returns an AST structural summary of every `.py` file
        (classes, methods, functions, signatures, and 1-based line numbers).
        """
        try:
            target = self._resolve_safe_path(directory)
        except ValueError as e:
            return f"ERROR: {e}\n[exit_code=1]"

        if not target.exists():
            return f"ERROR: Directory not found: {directory}\n[exit_code=1]"

        py_files: List[Path] = []
        if target.is_file() and target.suffix == ".py":
            py_files.append(target)
        else:
            for root_dir, dirs, files in os.walk(target):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
                for fname in sorted(files):
                    if fname.endswith(".py"):
                        py_files.append(Path(root_dir) / fname)

        out_blocks: List[str] = []
        for pf in sorted(py_files):
            rel = pf.relative_to(self.workspace_root).as_posix()
            try:
                src = pf.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src, filename=rel)
            except SyntaxError as e:
                out_blocks.append(f"{rel}\n  L{e.lineno}: <SyntaxError: {e.msg}>")
                continue

            entries: List[str] = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    entries.append(f"  L{node.lineno}: {prefix} {node.name}({ast.unparse(node.args)})")
                elif isinstance(node, ast.ClassDef):
                    entries.append(f"  L{node.lineno}: class {node.name}")
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sprefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                            entries.append(f"    L{sub.lineno}: {sprefix} {sub.name}({ast.unparse(sub.args)})")
            out_blocks.append(f"{rel}\n" + ("\n".join(entries) if entries else "  (no top-level symbols)"))

        body = "\n".join(out_blocks) if out_blocks else "(no Python files found)"
        return f"{body}\n[exit_code=0]"

    def read_file_lines(self, path: str, start_line: int, end_line: int) -> str:
        """Returns 1-indexed inclusive line slice `[start_line, end_line]` with line numbers."""
        try:
            p = self._resolve_safe_path(path)
        except ValueError as e:
            return f"ERROR: {e}\n[exit_code=1]"

        if not p.exists() or not p.is_file():
            return f"ERROR: File not found: {path}\n[exit_code=1]"

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        s = max(1, int(start_line))
        e = min(total, int(end_line))
        norm_rel = p.relative_to(self.workspace_root).as_posix()

        if s > e and total > 0:
            return f"ERROR: Invalid line range {start_line}..{end_line} for {norm_rel} ({total} lines)\n[exit_code=1]"

        header = f"{norm_rel} (lines {s}-{e} of {total})"
        body = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(s, e + 1))
        return f"{header}\n{body}\n[exit_code=0]" if body else f"{header}\n[exit_code=0]"

    def apply_diff_patch(self, path: str, search_block: str, replace_block: str) -> str:
        """
        Replaces an exact, unique `search_block` with `replace_block` in `path`.
        For `.py` files, verifies syntax via `ast.parse` and `py_compile` before saving;
        rolls back untouched if syntax verification fails.
        """
        try:
            p = self._resolve_safe_path(path)
        except ValueError as e:
            return f"ERROR: {e}\n[exit_code=1]"

        if not p.exists() or not p.is_file():
            return f"ERROR: File not found: {path}\n[exit_code=1]"

        raw_before = p.read_text(encoding="utf-8", errors="replace")
        # Normalize CRLF -> LF for matching
        orig_text = raw_before.replace("\r\n", "\n")
        norm_search = search_block.replace("\r\n", "\n")
        norm_replace = replace_block.replace("\r\n", "\n")

        count = orig_text.count(norm_search)
        if count == 0:
            return f"ERROR: SEARCH block not found in {path} (file untouched)\n[exit_code=1]"
        if count > 1:
            return f"ERROR: SEARCH block matched {count} locations in {path}; include more context lines\n[exit_code=1]"

        updated_text = orig_text.replace(norm_search, norm_replace, 1)

        if p.suffix == ".py":
            syntax_err = self._verify_python_syntax(updated_text, filename=path)
            if syntax_err is not None:
                return f"ERROR: {syntax_err} (file untouched, patch rolled back)\n[exit_code=1]"

        p.write_text(updated_text, encoding="utf-8")
        norm_rel = p.relative_to(self.workspace_root).as_posix()
        rec = self._record_diff(norm_rel, "apply_diff_patch", orig_text, updated_text)
        return (
            f"OK: Patched {norm_rel} [patch_id={rec.patch_id}, +{rec.lines_added}/-{rec.lines_removed} lines] "
            f"(syntax_check=PASS)\n[exit_code=0]"
        )

    def write_file(self, path: str, content: str) -> str:
        """Creates or overwrites `path` with UTF-8 `content` after syntax-checking `.py` files."""
        try:
            p = self._resolve_safe_path(path)
        except ValueError as e:
            return f"ERROR: {e}\n[exit_code=1]"

        norm_content = content.replace("\r\n", "\n")
        if p.suffix == ".py":
            syntax_err = self._verify_python_syntax(norm_content, filename=path)
            if syntax_err is not None:
                return f"ERROR: {syntax_err} (write rejected, file untouched)\n[exit_code=1]"

        before_text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n") if p.exists() else ""
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(norm_content, encoding="utf-8")

        norm_rel = p.relative_to(self.workspace_root).as_posix()
        rec = self._record_diff(norm_rel, "write_file", before_text, norm_content)
        return f"OK: Wrote {len(norm_content.splitlines())} lines to {norm_rel} [patch_id={rec.patch_id}]\n[exit_code=0]"

    def rollback_patch(self, patch_id: int) -> str:
        """One-click rollback of a recorded `patch_id` for the UI Diff Viewer or self-healing engine."""
        target_rec: Optional[DiffHistoryRecord] = None
        for rec in reversed(self.diff_history):
            if rec.patch_id == int(patch_id):
                target_rec = rec
                break

        if target_rec is None:
            return f"ERROR: patch_id={patch_id} not found in diff history\n[exit_code=1]"
        if target_rec.rolled_back:
            return f"ERROR: patch_id={patch_id} was already rolled back\n[exit_code=1]"

        p = self._resolve_safe_path(target_rec.path)
        p.write_text(target_rec.before_content, encoding="utf-8")
        target_rec.rolled_back = True
        return f"OK: Rolled back {target_rec.path} from patch_id={patch_id}\n[exit_code=0]"

    # ---------------- MCP JSON-RPC 2.0 Interface ----------------

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "ast_repo_map",
                "description": "List every Python file under `directory` with classes, functions, signatures, and line numbers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                    "required": ["directory"],
                },
            },
            {
                "name": "read_file_lines",
                "description": "Read a 1-indexed inclusive line range `[start_line, end_line]` with line numbers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path", "start_line", "end_line"],
                },
            },
            {
                "name": "apply_diff_patch",
                "description": "Replace a unique `search_block` with `replace_block`; auto-rolls back `.py` syntax errors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "search_block": {"type": "string"},
                        "replace_block": {"type": "string"},
                    },
                    "required": ["path", "search_block", "replace_block"],
                },
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a UTF-8 file with automatic `.py` syntax checking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "ast_repo_map":
            return self.ast_repo_map(str(arguments.get("directory", ".")))
        if name == "read_file_lines":
            return self.read_file_lines(
                str(arguments["path"]),
                int(arguments["start_line"]),
                int(arguments["end_line"]),
            )
        if name == "apply_diff_patch":
            return self.apply_diff_patch(
                str(arguments["path"]),
                str(arguments["search_block"]),
                str(arguments["replace_block"]),
            )
        if name == "write_file":
            return self.write_file(str(arguments["path"]), str(arguments["content"]))
        raise ValueError(f"MCPCodeEditorServer does not provide tool `{name}`")

    def handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        try:
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}
            if method == "tools/call":
                t_name = params.get("name", "")
                t_args = params.get("arguments", {}) or {}
                out = self.call_tool(t_name, t_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": out}],
                        "isError": not out.endswith("[exit_code=0]"),
                    },
                }
            if method == "editor/diff_history":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"diffs": [asdict(d) for d in self.diff_history]},
                }
            if method == "editor/rollback":
                msg = self.rollback_patch(int(params["patch_id"]))
                return {"jsonrpc": "2.0", "id": req_id, "result": {"message": msg}}
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
    test_root = ROOT / "feral_sandbox"
    if test_root.exists():
        shutil.rmtree(test_root)

    editor = MCPCodeEditorServer(workspace_root=test_root)
    try:
        print("1. Creating Sample Multi-File Python Package via `write_file()`...")
        w1 = editor.call_tool(
            "write_file",
            {
                "path": "pkg/billing.py",
                "content": (
                    "class InvoiceEngine:\n"
                    "    def __init__(self, tax_rate: float = 0.10):\n"
                    "        self.tax_rate = tax_rate\n\n"
                    "    def compute_total(self, subtotal: float, discount: float = 0.0) -> float:\n"
                    "        # BUG: adds discount instead of subtracting\n"
                    "        taxed = (subtotal + discount) * (1.0 + self.tax_rate)\n"
                    "        return round(taxed, 2)\n"
                ),
            },
        )
        print(f"   - {w1.splitlines()[0]} ({w1.splitlines()[-1]})")

        print("\n2. Testing `ast_repo_map()` Structural AST Extraction...")
        repo_map = editor.call_tool("ast_repo_map", {"directory": "pkg"})
        for line in repo_map.splitlines()[:-1]:
            print(f"   | {line}")
        assert "class InvoiceEngine" in repo_map and "def compute_total" in repo_map

        print("\n3. Testing `read_file_lines()` Line-Slice Inspection...")
        lines_out = editor.call_tool(
            "read_file_lines",
            {"path": "pkg/billing.py", "start_line": 5, "end_line": 8},
        )
        for line in lines_out.splitlines()[:-1]:
            print(f"   | {line}")
        assert "5 |     def compute_total" in lines_out

        print("\n4. Testing `apply_diff_patch()` Auto-Rollback on Broken Python Syntax (`SyntaxError`)...")
        bad_patch = editor.call_tool(
            "apply_diff_patch",
            {
                "path": "pkg/billing.py",
                "search_block": "        taxed = (subtotal + discount) * (1.0 + self.tax_rate)",
                "replace_block": "        taxed = (subtotal - discount * (1.0 + self.tax_rate",
            },
        )
        print(f"   - Broken Patch Result : {bad_patch.splitlines()[0]} ({bad_patch.splitlines()[-1]})")
        assert "SyntaxError" in bad_patch and bad_patch.endswith("[exit_code=1]")
        # Verify original file on disk was untouched
        disk_after_bad = (test_root / "pkg" / "billing.py").read_text(encoding="utf-8")
        assert "(subtotal + discount)" in disk_after_bad

        print("\n5. Testing Valid `apply_diff_patch()` + Unified Diff History & One-Click Rollback...")
        good_patch = editor.call_tool(
            "apply_diff_patch",
            {
                "path": "pkg/billing.py",
                "search_block": "        taxed = (subtotal + discount) * (1.0 + self.tax_rate)",
                "replace_block": "        taxed = max(0.0, subtotal - discount) * (1.0 + self.tax_rate)",
            },
        )
        print(f"   - Valid Patch Result  : {good_patch.splitlines()[0]} ({good_patch.splitlines()[-1]})")
        assert good_patch.endswith("[exit_code=0]")

        latest_diff = editor.diff_history[-1]
        print(f"   - Unified Diff Recorded (`patch_id={latest_diff.patch_id}`, +{latest_diff.lines_added}/-{latest_diff.lines_removed} lines):")
        for dl in latest_diff.unified_diff.splitlines():
            print(f"     {dl}")

        # Test one-click rollback (`rollback_patch`) and re-apply
        rb_msg = editor.rollback_patch(latest_diff.patch_id)
        print(f"   - One-Click Rollback  : {rb_msg.splitlines()[0]} ({rb_msg.splitlines()[-1]})")
        assert "(subtotal + discount)" in (test_root / "pkg" / "billing.py").read_text(encoding="utf-8")

    finally:
        if test_root.exists():
            shutil.rmtree(test_root, ignore_errors=True)

    print("\nPHASE 5 STEP 3 (`mcp_code_editor.py`): PASS -> AST repo map, line reader, syntax-guarded diff patcher, and rollback verified!")


if __name__ == "__main__":
    main()
