"""
Phase 2, Step 5 - Feral Apex Toolchain
======================================
Builds execution-verified multi-tool trajectories that teach Feral the full Apex Toolchain:

    run_powershell, read_file_lines, apply_diff_patch, ast_repo_map, web_research, spawn_subagent

Every tool below is a REAL implementation. Each trajectory is replayed against a throwaway
mini-repo (feral_sandbox/), so every <tool_response> in the dataset is authentic output, not
hand-written text. Failed tool calls are kept in context but mistake-masked (labels = -100).

Note: web_research reads from a small offline docs cache until the live SearXNG/Playwright
browser MCP is built in Phase 5, Step 4. The call/response format is identical.

Output: feral_arsenal_trajectories.jsonl
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SANDBOX = ROOT / "feral_sandbox"
OUTPUT = ROOT / "feral_arsenal_trajectories.jsonl"

tok = AutoTokenizer.from_pretrained(str(ROOT))

SYSTEM_FERAL = (
    "You are Feral, an autonomous AI software engineering agent running on Windows 11 PowerShell. "
    "Always reason inside <think>...</think> before acting. Prefer surgical tools: map the repo, read only "
    "the lines you need, and patch with exact SEARCH/REPLACE blocks. Verify every change by running it.\n"
    "Available tools:\n"
    "- run_powershell(command: str): Run a PowerShell command in the workspace; returns combined output and exit code.\n"
    "- read_file_lines(path: str, start_line: int, end_line: int): Return the given line range with line numbers.\n"
    "- apply_diff_patch(path: str, search_block: str, replace_block: str): Replace an exact, unique SEARCH block; "
    "Python files are syntax-checked and rolled back on failure.\n"
    "- ast_repo_map(directory: str): List every Python file with its classes, functions, signatures, and line numbers.\n"
    "- web_research(query: str, url: str = \"\"): Search documentation, or open a URL as clean Markdown.\n"
    "- spawn_subagent(task_prompt: str): Delegate an isolated subtask to a fresh-context worker; returns only a concise report.\n"
    "When calling a tool, emit ONLY: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
)

TOOL_SCHEMA = {
    "run_powershell": ["command"],
    "read_file_lines": ["path", "start_line", "end_line"],
    "apply_diff_patch": ["path", "search_block", "replace_block"],
    "ast_repo_map": ["directory"],
    "web_research": ["query"],
    "spawn_subagent": ["task_prompt"],
}

# ============================================================================
# 1. The sandbox mini-repo (2 real bugs: divide-by-zero + empty-list mean)
# ============================================================================
SANDBOX_FILES = {
    "calc/__init__.py": "",
    "calc/ops.py": (
        '"""Basic arithmetic operations for the Feral sandbox."""\n'
        "\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def divide(a, b):\n"
        "    return a / b\n"
        "\n"
        "\n"
        "def percent(part, whole):\n"
        "    return part / whole * 100\n"
    ),
    "calc/stats.py": (
        "from calc.ops import divide\n"
        "\n"
        "\n"
        "class RunningStats:\n"
        "    def __init__(self):\n"
        "        self.values = []\n"
        "\n"
        "    def push(self, x):\n"
        "        self.values.append(x)\n"
        "\n"
        "    def mean(self):\n"
        "        return divide(sum(self.values), len(self.values))\n"
    ),
    "tests/__init__.py": "",
    "tests/test_calc.py": (
        "import unittest\n"
        "from calc.ops import add, divide\n"
        "from calc.stats import RunningStats\n"
        "\n"
        "\n"
        "class TestOps(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
        "\n"
        "    def test_divide_by_zero_returns_none(self):\n"
        "        self.assertIsNone(divide(1, 0))\n"
        "\n"
        "\n"
        "class TestStats(unittest.TestCase):\n"
        "    def test_empty_mean_is_zero(self):\n"
        "        self.assertEqual(RunningStats().mean(), 0.0)\n"
        "\n"
        "    def test_mean(self):\n"
        "        s = RunningStats()\n"
        "        for v in (2, 4, 6):\n"
        "            s.push(v)\n"
        "        self.assertEqual(s.mean(), 4.0)\n"
    ),
}

WEB_CACHE = {
    "https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob": (
        "pathlib - Path.rglob",
        "Path.rglob(pattern)\n\n"
        "Glob the given relative pattern recursively. This is like calling Path.glob() with \"**/\" "
        "added in front of the pattern. Yields all matching files (of any kind).\n\n"
        "Example:\n```python\nfrom pathlib import Path\nfor p in Path('.').rglob('*.py'):\n    print(p)\n```",
    ),
    "https://docs.python.org/3/library/unittest.html#test-discovery": (
        "unittest - Test Discovery",
        "python -m unittest discover [options]\n\n"
        "-s, --start-directory: Directory to start discovery (default: .)\n"
        "-p, --pattern: Pattern to match test files (default: test*.py)\n"
        "-t, --top-level-directory: Top level directory of the project (defaults to start directory)\n\n"
        "All test files must be modules or packages importable from the top-level directory of the project.",
    ),
    "https://learn.microsoft.com/powershell/module/microsoft.powershell.utility/select-string": (
        "Select-String (Microsoft.PowerShell.Utility)",
        "Finds text in strings and files, similar to grep.\n\n"
        "-Pattern: Regular expression to search for\n-Path: Files to search\n"
        "-SimpleMatch: Use a literal match instead of a regex\n\n"
        "Outputs MatchInfo objects with LineNumber, Line, and Path properties.",
    ),
}


def reset_sandbox():
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    for rel, content in SANDBOX_FILES.items():
        p = SANDBOX / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _resolve(path):
    p = (SANDBOX / path).resolve()
    root = SANDBOX.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path escapes workspace: {path}")
    return p


# ============================================================================
# 2. Real tool implementations (the Feral Apex Toolchain)
# ============================================================================
def run_powershell(command):
    # Merge native stderr into stdout as plain strings (avoids PS 5.1 NativeCommandError noise).
    # Blank stderr lines become empty ErrorRecords whose string form is "System.Management.Automation.RemoteException",
    # so ErrorRecords are unwrapped via TargetObject (the original raw line) instead of being stringified.
    wrapped = (
        f"& {{ {command} }} 2>&1 | ForEach-Object {{ "
        "if ($_ -is [System.Management.Automation.ErrorRecord]) { if ($_.TargetObject -ne $null) { \"$($_.TargetObject)\" } "
        "else { $_.Exception.Message } } else { \"$_\" } }; exit $LASTEXITCODE"
    )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", wrapped],
        cwd=SANDBOX, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    out = (res.stdout + res.stderr).rstrip()
    return f"{out}\n[exit_code={res.returncode}]" if out else f"[exit_code={res.returncode}]"


def read_file_lines(path, start_line, end_line):
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: {path} does not exist."
    lines = p.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    s, e = max(1, int(start_line)), min(total, int(end_line))
    body = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(s, e + 1))
    return f"{path} (lines {s}-{e} of {total})\n{body}"


def apply_diff_patch(path, search_block, replace_block):
    p = _resolve(path)
    if not p.exists():
        return f"ERROR: {path} does not exist."
    original = p.read_text(encoding="utf-8")
    hits = original.count(search_block)
    if hits == 0:
        return f"ERROR: SEARCH block not found in {path} (0 matches). File unchanged."
    if hits > 1:
        return f"ERROR: SEARCH block is ambiguous in {path} ({hits} matches). Add more context. File unchanged."
    patched = original.replace(search_block, replace_block, 1)
    p.write_text(patched, encoding="utf-8")
    if p.suffix == ".py":
        try:
            compile(patched, str(p), "exec")
        except SyntaxError as err:
            p.write_text(original, encoding="utf-8")  # rollback
            return f"ERROR: Patch rejected and rolled back. SyntaxError: {err.msg} (line {err.lineno}). {path} is unchanged."
    removed, added = search_block.count("\n") + 1, replace_block.count("\n") + 1
    return f"OK: Patched {path} (1 hunk, -{removed} +{added} lines). Syntax check passed."


def ast_repo_map(directory):
    base = _resolve(directory)
    files = sorted(f for f in base.rglob("*.py") if "__pycache__" not in f.parts)
    out = [f"Repo map of {directory} ({len(files)} files)"]
    for f in files:
        out.append(f.relative_to(SANDBOX).as_posix())
        tree = ast.parse(f.read_text(encoding="utf-8"))
        defs = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defs.append(f"  L{node.lineno:<3} def {node.name}({ast.unparse(node.args)})")
            elif isinstance(node, ast.ClassDef):
                defs.append(f"  L{node.lineno:<3} class {node.name}")
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        defs.append(f"    L{sub.lineno:<3} def {sub.name}({ast.unparse(sub.args)})")
        out.extend(defs or ["  (no definitions)"])
    return "\n".join(out)


def web_research(query, url=""):
    if url:
        if url not in WEB_CACHE:
            return f"ERROR: Could not fetch {url}."
        title, md = WEB_CACHE[url]
        return f"# {title}\nSource: {url}\n\n{md}"
    terms = re.findall(r"[a-z0-9_.]+", query.lower())
    scored = []
    for u, (title, md) in WEB_CACHE.items():
        text = (title + " " + md).lower()
        score = sum(t in text for t in terms)
        if score:
            scored.append((score, u, title, md))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return f'No results for "{query}".'
    lines = [f'Search results for "{query}":']
    for i, (_, u, title, md) in enumerate(scored[:2], 1):
        snippet = md.split("\n\n")[1] if "\n\n" in md else md
        lines.append(f"{i}. {title}\n   {u}\n   {snippet[:160]}")
    return "\n".join(lines)


def spawn_subagent(task_prompt):
    # Deterministic test-runner worker: runs in its own process, returns only a short report.
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        cwd=SANDBOX, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    log = res.stdout + res.stderr
    ran = re.search(r"Ran (\d+) tests?", log)
    n = int(ran.group(1)) if ran else 0
    problems = []
    for block in re.split(r"^=+$", log, flags=re.M)[1:]:
        parts = re.split(r"^-+$", block.strip(), flags=re.M)
        m = re.match(r"(FAIL|ERROR): (\w+)", parts[0].strip())
        if not m:
            continue
        tb = parts[1] if len(parts) > 1 else ""
        frames = re.findall(r'feral_sandbox[\\/](.+?)", line (\d+), in (\w+)', tb)
        where = f"{frames[-1][0].replace(chr(92), '/')}:{frames[-1][1]} in {frames[-1][2]}" if frames else "unknown"
        last = [l.strip() for l in tb.strip().splitlines() if l.strip()]
        problems.append(f"- {m.group(1)} {m.group(2)} at {where} -> {last[-1] if last else 'unknown'}")
    head = f"[subagent report | fresh context | raw log withheld: {len(log)} chars]\nTask: {task_prompt}"
    if res.returncode == 0:
        return f"{head}\nResult: all {n} tests passed."
    return f"{head}\nResult: ran {n} tests, {len(problems)} failing.\n" + "\n".join(problems)


TOOLS = {
    "run_powershell": run_powershell,
    "read_file_lines": read_file_lines,
    "apply_diff_patch": apply_diff_patch,
    "ast_repo_map": ast_repo_map,
    "web_research": web_research,
    "spawn_subagent": spawn_subagent,
}


# ============================================================================
# 3. Trace builder: every tool call is EXECUTED, and its real output is recorded
# ============================================================================
class Trace:
    def __init__(self, tid, stage, prompt):
        self.tid, self.stage = tid, stage
        self.messages = [{"role": "system", "content": SYSTEM_FERAL}, {"role": "user", "content": prompt}]

    def call(self, think, name, args, learn=True, expect=None):
        payload = json.dumps({"name": name, "arguments": args})
        self.messages.append({
            "role": "assistant", "learn": learn,
            "content": f"<think>\n{think}\n</think>\n<tool_call>{payload}</tool_call>",
        })
        obs = TOOLS[name](**args)
        if expect is not None and expect not in obs:
            raise AssertionError(f"[{self.tid}] expected {expect!r} in {name} output, got:\n{obs}")
        self.messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})
        return obs

    def final(self, think, answer):
        self.messages.append({"role": "assistant", "learn": True, "content": f"<think>\n{think}\n</think>\n{answer}"})
        return self


DIVIDE_OLD = "def divide(a, b):\n    return a / b"
DIVIDE_NEW = "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b"
MEAN_OLD = "    def mean(self):\n        return divide(sum(self.values), len(self.values))"
MEAN_NEW = ("    def mean(self):\n        if not self.values:\n            return 0.0\n"
            "        return divide(sum(self.values), len(self.values))")
TEST_CMD = "python -m unittest discover -s tests -t ."


def build_traces():
    traces = []

    # ---------------- Feral_Read ----------------
    reset_sandbox()
    t = Trace("feral_read_001_ops_overview", "Feral_Read", "Show me what functions are defined in calc/ops.py.")
    t.call("I only need calc/ops.py. It is a small module, so reading lines 1-20 covers it without flooding context.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 1, "end_line": 20}, expect="def percent")
    traces.append(t.final(
        "The file has 13 lines and three functions. divide and percent have no zero guard.",
        "`calc/ops.py` defines three functions:\n- `add(a, b)` (line 4)\n- `divide(a, b)` (line 8): returns `a / b` with no zero check\n"
        "- `percent(part, whole)` (line 12): returns `part / whole * 100`, also unguarded\n\n"
        "Both `divide` and `percent` raise `ZeroDivisionError` when the denominator is 0."))

    reset_sandbox()
    t = Trace("feral_read_002_locate_then_read", "Feral_Read",
              "How does RunningStats.mean() compute its result? Only show me the relevant lines.")
    t.call("I don't know which line mean() is on. Select-String gives me the line number cheaply before I read anything.",
           "run_powershell", {"command": "Select-String -Path calc/stats.py -Pattern 'def mean' | ForEach-Object { $_.LineNumber }"},
           expect="11")
    t.call("mean() starts at line 11. A one-line method body means lines 11-12 are all I need.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="def mean")
    traces.append(t.final(
        "mean() delegates to calc.ops.divide with sum and len of self.values. With no values that is divide(0, 0).",
        "`RunningStats.mean()` (`calc/stats.py`, lines 11-12) returns `divide(sum(self.values), len(self.values))`. "
        "It delegates to `calc.ops.divide`, so with no values pushed it calls `divide(0, 0)` and raises `ZeroDivisionError`."))

    # ---------------- Feral_Map ----------------
    reset_sandbox()
    t = Trace("feral_map_001_repo_overview", "Feral_Map", "Give me an overview of this repository's structure.")
    t.call("An overview needs every file's classes and functions, not file contents. ast_repo_map gives that in one call.",
           "ast_repo_map", {"directory": "."}, expect="class RunningStats")
    traces.append(t.final(
        "The map shows a calc package (ops, stats) and a tests package with two TestCase classes.",
        "The repository has two packages:\n- **`calc/`**: `ops.py` (`add`, `divide`, `percent`) and `stats.py` "
        "(`RunningStats` with `__init__`, `push`, `mean`)\n- **`tests/`**: `test_calc.py` with `TestOps` and `TestStats`\n\n"
        "`RunningStats.mean` depends on `calc.ops.divide`."))

    reset_sandbox()
    t = Trace("feral_map_002_find_bug_surgically", "Feral_Map",
              "Where is the empty-input bug in the stats code? Find it without reading every file.")
    t.call("Map the calc package first so I can jump straight to the stats method instead of reading whole files.",
           "ast_repo_map", {"directory": "calc"}, expect="def mean")
    t.call("RunningStats.mean is at L11 in calc/stats.py. Reading just lines 11-12 shows its body.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="len(self.values)")
    traces.append(t.final(
        "len(self.values) is 0 for an empty RunningStats, so mean() divides by zero.",
        "The bug is in `RunningStats.mean()` at `calc/stats.py:12`: `divide(sum(self.values), len(self.values))` "
        "divides by `len(self.values)`, which is `0` when nothing has been pushed. It needs an empty-list guard."))

    # ---------------- Feral_Patch ----------------
    reset_sandbox()
    t = Trace("feral_patch_001_divide_then_test", "Feral_Patch",
              "Fix divide() in calc/ops.py so it returns None when b is 0, then run the test suite.")
    t.call("I need the exact current text of divide() to build a SEARCH block. It is at lines 8-9.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="return a / b")
    t.call("I'll replace the two-line body with a zero guard. The SEARCH block copies the 4-space indentation exactly.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("The patch passed the syntax check. Now run the full test suite to verify behavior.",
           "run_powershell", {"command": TEST_CMD}, expect="failures=1")
    traces.append(t.final(
        "test_divide_by_zero_returns_none now passes, but test_empty_mean_is_zero still fails: mean() returns None instead of 0.0. "
        "That is a separate bug in RunningStats.mean, so I should report it rather than claim everything passes.",
        "`divide()` now returns `None` when `b == 0`, and `test_divide_by_zero_returns_none` passes.\n\n"
        "One failure remains: `test_empty_mean_is_zero` expects `0.0`, but `RunningStats.mean()` now returns `None` for an empty list. "
        "That needs its own guard in `calc/stats.py`."))

    reset_sandbox()
    t = Trace("feral_patch_002_search_miss_recovery", "Feral_Patch", "Make divide() return None for a zero divisor.")
    t.call("Patch divide() directly.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "def divide(a, b):\nreturn a / b",
                                "replace_block": "def divide(a, b):\nif b == 0:\n    return None\nreturn a / b"},
           learn=False, expect="SEARCH block not found")  # MISTAKE MASKED (-100)
    t.call("The SEARCH block matched 0 times, so I guessed the text instead of copying it. "
           "I'll read the exact lines of divide() and copy the whitespace precisely.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="def divide")
    t.call("The body is indented with 4 spaces, which my first SEARCH block dropped. Retry with the exact text.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("Verify both the zero case and a normal case.",
           "run_powershell", {"command": 'python -c "from calc.ops import divide; print(divide(1, 0), divide(6, 3))"'},
           expect="None 2.0")
    traces.append(t.final(
        "divide(1, 0) printed None and divide(6, 3) printed 2.0, so the fix works.",
        "`divide()` now returns `None` for a zero divisor. Verified: `divide(1, 0)` -> `None`, `divide(6, 3)` -> `2.0`."))

    reset_sandbox()
    t = Trace("feral_patch_003_syntax_rollback_recovery", "Feral_Patch",
              "percent() crashes when whole is 0. Make it return 0.0 in that case and verify.")
    t.call("percent() is at lines 12-13. Read the exact text first.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 12, "end_line": 13}, expect="def percent")
    t.call("Add the zero guard.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "    return part / whole * 100",
                                "replace_block": "    if whole == 0\n        return 0.0\n    return part / whole * 100"},
           learn=False, expect="SyntaxError")  # MISTAKE MASKED (-100)
    t.call("The patch was rolled back with a SyntaxError on the new if line: I left off the colon. "
           "The file is unchanged, so the same SEARCH block still matches.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "    return part / whole * 100",
                                "replace_block": "    if whole == 0:\n        return 0.0\n    return part / whole * 100"},
           expect="OK: Patched")
    t.call("Verify the zero case and a normal case.",
           "run_powershell", {"command": 'python -c "from calc.ops import percent; print(percent(1, 0), percent(1, 4))"'},
           expect="0.0 25.0")
    traces.append(t.final(
        "percent(1, 0) printed 0.0 and percent(1, 4) printed 25.0.",
        "`percent()` now returns `0.0` when `whole == 0`. Verified: `percent(1, 0)` -> `0.0`, `percent(1, 4)` -> `25.0`."))

    # ---------------- Feral_Research ----------------
    reset_sandbox()
    t = Trace("feral_research_001_pathlib_rglob", "Feral_Research",
              "How do I recursively find all .py files with pathlib? Check the docs.")
    t.call("Search the docs for the recursive glob API in pathlib.",
           "web_research", {"query": "pathlib recursive glob files", "url": ""}, expect="rglob")
    t.call("The top result is the official Path.rglob page. Open it to confirm the exact semantics.",
           "web_research", {"query": "pathlib Path.rglob",
                            "url": "https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob"}, expect="**/")
    traces.append(t.final(
        "The docs say rglob(pattern) is glob with **/ prepended, which is exactly a recursive search.",
        "Use `Path.rglob`, which is `Path.glob()` with `**/` prepended:\n\n```python\nfrom pathlib import Path\n"
        "for p in Path('.').rglob('*.py'):\n    print(p)\n```"))

    reset_sandbox()
    t = Trace("feral_research_002_unittest_discover", "Feral_Research",
              "Run this project's tests with unittest discovery. Check the docs for the right flags first.")
    t.call("Look up the unittest discover flags before running anything.",
           "web_research", {"query": "unittest discover start directory top level", "url": ""}, expect="Test Discovery")
    t.call("Open the Test Discovery page to read the exact -s and -t semantics.",
           "web_research", {"query": "unittest test discovery",
                            "url": "https://docs.python.org/3/library/unittest.html#test-discovery"}, expect="-t")
    t.call("Tests live in tests/ and import the calc package from the project root, so use -s tests -t .",
           "run_powershell", {"command": TEST_CMD}, expect="errors=2")
    traces.append(t.final(
        "Discovery ran 4 tests: 2 pass and 2 error with ZeroDivisionError (divide by zero, and empty mean which calls divide(0, 0)).",
        "Ran `python -m unittest discover -s tests -t .` (start in `tests/`, import from the project root).\n\n"
        "Result: 4 tests, **2 errors**, both `ZeroDivisionError`:\n- `test_divide_by_zero_returns_none`\n"
        "- `test_empty_mean_is_zero` (because `mean()` calls `divide(0, 0)`)"))

    # ---------------- Feral_Delegate ----------------
    reset_sandbox()
    t = Trace("feral_delegate_001_test_triage", "Feral_Delegate",
              "Run the full test suite and tell me what's broken, but keep the noisy logs out of our main context.")
    t.call("Raw test logs would bloat my context. A subagent can run the suite in a fresh context and return only the failures.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and report each failing test with its root-cause frame."},
           expect="2 failing")
    traces.append(t.final(
        "The subagent found 2 failures, and both trace back to calc/ops.py:9 in divide. One root cause, two symptoms.",
        "2 of 4 tests are failing, and both share one root cause: `divide()` in `calc/ops.py` line 9 has no zero check.\n"
        "- `test_divide_by_zero_returns_none` calls `divide(1, 0)` directly\n"
        "- `test_empty_mean_is_zero` reaches it through `RunningStats.mean()` -> `divide(0, 0)`"))

    # ---------------- Feral_Combo ----------------
    reset_sandbox()
    t = Trace("feral_combo_001_full_fix", "Feral_Combo",
              "The test suite is failing. Find the root cause, fix it across files, and make all tests pass.")
    t.call("Start with triage in a fresh context so the raw logs stay out of mine.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and report each failing test with its root-cause frame."},
           expect="2 failing")
    t.call("Both failures end in divide. Map calc to see how ops and stats connect before editing.",
           "ast_repo_map", {"directory": "calc"}, expect="def divide")
    t.call("divide is at L8. Read its exact body for the SEARCH block.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="return a / b")
    t.call("Guard divide against a zero divisor.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("With divide returning None, an empty mean() would return None, but the test expects 0.0. Read mean() at L11.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="def mean")
    t.call("Add an empty-list guard that returns 0.0 before calling divide.",
           "apply_diff_patch", {"path": "calc/stats.py", "search_block": MEAN_OLD, "replace_block": MEAN_NEW}, expect="OK: Patched")
    t.call("Both patches passed the syntax check. Run the full suite to verify.",
           "run_powershell", {"command": TEST_CMD}, expect="OK")
    traces.append(t.final(
        "All 4 tests pass. Two surgical edits across two files fixed both failures.",
        "All 4 tests pass. Root cause and fixes:\n1. `calc/ops.py`: `divide()` now returns `None` when `b == 0`.\n"
        "2. `calc/stats.py`: `RunningStats.mean()` returns `0.0` for an empty list before calling `divide`."))

    reset_sandbox()
    t = Trace("feral_combo_002_add_method", "Feral_Combo",
              "Add a count() method to RunningStats that returns how many values were pushed, and verify it works.")
    t.call("Map calc to find RunningStats and its existing methods.",
           "ast_repo_map", {"directory": "calc"}, expect="def push")
    t.call("push is at L8 and mean at L11. Read 8-12 so I can insert count() right after push.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 8, "end_line": 12}, expect="self.values.append(x)")
    t.call("Anchor on the full push() method so the SEARCH block is unique, then append count() after it.",
           "apply_diff_patch", {"path": "calc/stats.py",
                                "search_block": "    def push(self, x):\n        self.values.append(x)\n",
                                "replace_block": "    def push(self, x):\n        self.values.append(x)\n\n"
                                                 "    def count(self):\n        return len(self.values)\n"},
           expect="OK: Patched")
    t.call("Verify by pushing 3 values and printing count().",
           "run_powershell", {"command": 'python -c "from calc.stats import RunningStats; s = RunningStats(); '
                                         's.push(1); s.push(2); s.push(3); print(s.count())"'}, expect="3")
    traces.append(t.final(
        "count() printed 3 after three pushes.",
        "Added `RunningStats.count()` in `calc/stats.py` (returns `len(self.values)`). Verified: after pushing 3 values, `count()` -> `3`."))

    # ---------------- Anchor_General (25%, no tools) ----------------
    anchors = [
        ("anchor_feral_001_read_ranges", "When should you read a line range instead of a whole file?",
         "This is a workflow question; no tools are needed.",
         "Read a line range whenever you already know roughly where the code is (from `ast_repo_map`, `Select-String`, or a traceback). "
         "A 2,000-line file costs ~20K tokens of context; the 10 lines you actually need cost ~100. Smaller context means faster "
         "inference, less KV-cache VRAM, and less irrelevant text distracting the next decision. Read whole files only when they are small "
         "or you genuinely need the full structure."),
        ("anchor_feral_002_search_replace", "Why patch code with SEARCH/REPLACE blocks instead of rewriting the whole file?",
         "A design question about editing strategy. Answer directly.",
         "Rewriting a whole file forces the model to reproduce every unchanged line perfectly; one dropped line or changed indent silently "
         "breaks code that had nothing to do with the task. A SEARCH/REPLACE block only touches the exact text being changed, costs far fewer "
         "output tokens, and fails loudly (0 matches) instead of corrupting the file when the model's view of the code is stale."),
        ("anchor_feral_003_subagents", "When is it worth delegating to a subagent instead of doing the work yourself?",
         "A judgment question about delegation. No tools required.",
         "Delegate when a subtask is self-contained and produces a lot of noise you don't need to keep: running a test suite, scanning many "
         "files, or researching one library. The subagent burns its own fresh context and returns a short report, so the main context keeps "
         "only the conclusion. Don't delegate tiny tasks or work that needs the main conversation's full history; the handoff overhead "
         "outweighs the benefit."),
        ("anchor_feral_004_unique_match", "Why must a SEARCH block match exactly once in the file?",
         "This is about patch safety. Answer directly.",
         "If a SEARCH block matches 0 times, the agent's picture of the file is wrong and it must re-read before editing. If it matches 2+ "
         "times (for example a repeated `return None`), the tool can't know which occurrence was meant, and guessing could edit the wrong "
         "function. Requiring exactly one match turns both cases into an explicit error, so the fix is always to add surrounding context "
         "lines until the block is unique."),
    ]
    for tid, prompt, think, ans in anchors:
        traces.append(Trace(tid, "Anchor_General", prompt).final(think, ans))

    return traces


# ============================================================================
# 4. Tokenize with completion-only + mistake masking
# ============================================================================
def tokenize(trace):
    input_ids, labels, masked = [], [], 0
    for msg in trace.messages:
        header = tok.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False)
        body = tok.encode(f"{msg['content']}<|im_end|>\n", add_special_tokens=False)
        input_ids += header + body
        labels += [-100] * len(header)
        if msg["role"] == "assistant" and msg.get("learn", True):
            labels += body
        else:
            labels += [-100] * len(body)
            masked += msg["role"] == "assistant"
    return input_ids, labels, masked


# ============================================================================
# 5. Built-in audit (runs before anything is committed)
# ============================================================================
def audit(records):
    errors = []
    for r in records:
        tid = r["id"]
        if len(r["input_ids"]) != len(r["labels"]):
            errors.append(f"{tid}: input_ids/labels length mismatch")
        for msg in r["messages"]:
            if msg["role"] == "user" and "System.Management.Automation" in msg["content"]:
                errors.append(f"{tid}: PowerShell wrapper noise leaked into a tool_response")
            if msg["role"] != "assistant":
                continue
            c = msg["content"]
            if c.count("<think>") != 1 or c.count("</think>") != 1 or c.index("<think>") > c.index("</think>"):
                errors.append(f"{tid}: unbalanced <think> tags")
            for raw in re.findall(r"<tool_call>(.*?)</tool_call>", c, flags=re.S):
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError as e:
                    errors.append(f"{tid}: invalid tool_call JSON ({e})")
                    continue
                name = call.get("name")
                if name not in TOOL_SCHEMA:
                    errors.append(f"{tid}: unknown tool {name}")
                    continue
                missing = [k for k in TOOL_SCHEMA[name] if k not in call.get("arguments", {})]
                if missing:
                    errors.append(f"{tid}: {name} missing args {missing}")
        # Leak check: decode every contiguous trained span; none may contain tool output or other roles.
        span = []
        for t, l in zip(r["input_ids"] + [0], r["labels"] + [-100]):
            if l != -100:
                span.append(l)
            elif span:
                text = tok.decode(span)
                if "<tool_response>" in text or "<|im_start|>" in text:
                    errors.append(f"{tid}: tool_response/role header leaked into labels")
                span = []
    return errors


def main():
    print("1. Replaying Feral Apex Toolchain trajectories against the live sandbox repo...")
    traces = build_traces()
    shutil.rmtree(SANDBOX, ignore_errors=True)
    print(f"   -> {len(traces)} trajectories executed, every tool call ran for real.\n")

    print("2. Tokenizing with completion-only + mistake masking...")
    records, stage_n, stage_tot, stage_act, tool_use = [], Counter(), Counter(), Counter(), Counter()
    total_masked = 0
    for t in traces:
        ids, labels, masked = tokenize(t)
        total_masked += masked
        records.append({"id": t.tid, "stage": t.stage, "messages": t.messages, "input_ids": ids, "labels": labels})
        stage_n[t.stage] += 1
        stage_tot[t.stage] += len(ids)
        stage_act[t.stage] += sum(l != -100 for l in labels)
        for m in t.messages:
            if m["role"] != "assistant":
                continue  # system prompt holds a non-JSON placeholder example
            for raw in re.findall(r"<tool_call>(.*?)</tool_call>", m["content"], flags=re.S):
                tool_use[json.loads(raw)["name"]] += 1

    print("\n=== FERAL APEX TOOLCHAIN - STAGE SUMMARY ===")
    for st in ["Feral_Read", "Feral_Map", "Feral_Patch", "Feral_Research", "Feral_Delegate", "Feral_Combo", "Anchor_General"]:
        tot, act = stage_tot[st], stage_act[st]
        print(f"  {st:16s} | Traces: {stage_n[st]:2d} | Total Tokens: {tot:5d} | Active: {act:5d} ({act / tot * 100:4.1f}%)")
    gt, ga = sum(stage_tot.values()), sum(stage_act.values())
    print("-" * 80)
    print(f"  {'TOTAL':16s} | Traces: {len(records):2d} | Total Tokens: {gt:5d} | Active: {ga:5d} ({ga / gt * 100:4.1f}%)")
    print(f"  Anchor ratio: {stage_n['Anchor_General'] / len(records) * 100:.1f}% | Mistake-masked assistant turns: {total_masked}")
    print(f"  Max sequence length: {max(len(r['input_ids']) for r in records)} tokens")
    print("\n=== TOOL CALL COVERAGE ===")
    for name in TOOL_SCHEMA:
        print(f"  {name:18s}: {tool_use[name]:2d} calls")

    print("\n3. Auditing dataset integrity...")
    errors = audit(records)
    uncovered = [n for n in TOOL_SCHEMA if tool_use[n] == 0]
    if uncovered:
        errors.append(f"tools never used: {uncovered}")
    if total_masked != 2:
        errors.append(f"expected 2 mistake-masked turns, found {total_masked}")
    if errors:
        print("AUDIT: FAIL")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("  [PASS] <think> tags balanced in every assistant turn")
    print("  [PASS] every <tool_call> is valid JSON with a known tool and required args")
    print("  [PASS] zero <tool_response> / role-header tokens leaked into trained labels")
    print("  [PASS] all 6 Apex tools covered")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nAUDIT: PASS -> wrote {len(records)} trajectories to {OUTPUT.name}")


if __name__ == "__main__":
    main()
