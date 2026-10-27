"""
Phase 2, Step 5 - Feral Apex Toolchain (`expand_feral_arsenal_dataset.py`)
==========================================================================
Generates `feral_arsenal_trajectories.jsonl` using the shared `feral_core.py` engine
so 100% of trajectories share `SYSTEM_FERAL` and execute against a real throwaway repo.

Every stage has >= 4 trajectories so every stage is represented in both train.jsonl and val.jsonl.
"""
import json
import os
import re
from collections import Counter

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer

from feral_core import (
    ROOT,
    TOOL_SCHEMA,
    Sandbox,
    Trace,
    anchor_trace,
    to_record,
    write_and_summarize,
)
from scenarios.code_fix_tasks import ANCHOR_TASKS

OUTPUT = ROOT / "feral_arsenal_trajectories.jsonl"
STAGE_ORDER = [
    "Feral_Read",
    "Feral_Map",
    "Feral_Patch",
    "Feral_Research",
    "Feral_Delegate",
    "Feral_Combo",
    "Anchor_General",
]

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
    "calc/formatter.py": (
        "def format_metric(name, val, decimals=2):\n"
        "    if val is None:\n"
        "        return f'{name}: N/A'\n"
        "    return f'{name}: {val:.{decimals}f}'\n"
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

DIVIDE_OLD = "def divide(a, b):\n    return a / b"
DIVIDE_NEW = "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b"
MEAN_OLD = "    def mean(self):\n        return divide(sum(self.values), len(self.values))"
MEAN_NEW = (
    "    def mean(self):\n"
    "        if not self.values:\n"
    "            return 0.0\n"
    "        return divide(sum(self.values), len(self.values))"
)
TEST_CMD = "python -m unittest discover -s tests -t ."


def build_traces(sb: Sandbox):
    traces = []

    # ---------------- Feral_Read (4 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_read_001_ops_overview", "Feral_Read", "Show me what functions are defined in calc/ops.py.")
    t.call("Read lines 1-20 of calc/ops.py to inspect its functions without flooding context.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 1, "end_line": 20}, expect="def percent")
    traces.append(t.final(
        "The file has 13 lines and three functions (`add`, `divide`, `percent`). `divide` and `percent` have no zero guard.",
        "`calc/ops.py` defines three functions:\n- `add(a, b)` (line 4)\n- `divide(a, b)` (line 8): unguarded `a / b`\n"
        "- `percent(part, whole)` (line 12): unguarded `part / whole * 100`"))

    sb.reset()
    t = Trace(sb, "feral_read_002_locate_then_read", "Feral_Read",
              "How does RunningStats.mean() compute its result? Only show me the relevant lines.")
    t.call("Locate `def mean` in calc/stats.py with Select-String before reading lines.",
           "run_powershell", {"command": "Select-String -Path calc/stats.py -Pattern 'def mean' | ForEach-Object { $_.LineNumber }"},
           expect="11")
    t.call("mean() starts at line 11. Read lines 11-12.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="def mean")
    traces.append(t.final(
        "mean() calls divide(sum(self.values), len(self.values)), which raises ZeroDivisionError when empty.",
        "`RunningStats.mean()` (`calc/stats.py`, lines 11-12) returns `divide(sum(self.values), len(self.values))`, "
        "which calls `divide(0, 0)` on an empty instance."))

    sb.reset()
    t = Trace(sb, "feral_read_003_formatter_check", "Feral_Read",
              "How does calc/formatter.py handle None values in format_metric?")
    t.call("Read lines 1-10 of calc/formatter.py.",
           "read_file_lines", {"path": "calc/formatter.py", "start_line": 1, "end_line": 10}, expect="def format_metric")
    traces.append(t.final(
        "Lines 2-3 check `if val is None:` and return `f'{name}: N/A'`.",
        "`format_metric(name, val, decimals=2)` in `calc/formatter.py` checks `if val is None:` on line 2 and returns `f'{name}: N/A'`."))

    sb.reset()
    t = Trace(sb, "feral_read_004_test_assertions", "Feral_Read",
              "What exact assertion does TestStats.test_empty_mean_is_zero make in tests/test_calc.py?")
    t.call("Find the line number of test_empty_mean_is_zero first.",
           "run_powershell", {"command": "Select-String -Path tests/test_calc.py -Pattern 'test_empty_mean_is_zero' | ForEach-Object { $_.LineNumber }"},
           expect="15")
    t.call("Read lines 15-17 of tests/test_calc.py.",
           "read_file_lines", {"path": "tests/test_calc.py", "start_line": 15, "end_line": 17}, expect="assertEqual")
    traces.append(t.final(
        "Line 16 asserts `self.assertEqual(RunningStats().mean(), 0.0)`.",
        "`TestStats.test_empty_mean_is_zero` (`tests/test_calc.py`, lines 15-16) asserts `self.assertEqual(RunningStats().mean(), 0.0)`."))

    # ---------------- Feral_Map (4 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_map_001_repo_overview", "Feral_Map", "Give me an overview of this repository's structure.")
    t.call("Use ast_repo_map on `.` to list every Python file, class, and function signature.",
           "ast_repo_map", {"directory": "."}, expect="class RunningStats")
    traces.append(t.final(
        "The map shows calc/ (ops, stats, formatter) and tests/ (test_calc).",
        "Repository structure:\n- **`calc/ops.py`**: `add`, `divide`, `percent`\n"
        "- **`calc/stats.py`**: `RunningStats` (`__init__`, `push`, `mean`)\n"
        "- **`calc/formatter.py`**: `format_metric`\n"
        "- **`tests/test_calc.py`**: `TestOps` and `TestStats`"))

    sb.reset()
    t = Trace(sb, "feral_map_002_find_bug_surgically", "Feral_Map",
              "Where is the empty-input bug in the stats code? Find it without reading every file.")
    t.call("Map the calc directory first to find the stats class and method line numbers.",
           "ast_repo_map", {"directory": "calc"}, expect="def mean")
    t.call("RunningStats.mean is at L11 in calc/stats.py. Read lines 11-12.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="len(self.values)")
    traces.append(t.final(
        "len(self.values) is 0 when empty, causing divide(0, 0).",
        "The bug is in `RunningStats.mean()` at `calc/stats.py:12`: `divide(sum(self.values), len(self.values))` "
        "passes `0` as the denominator when `self.values` is empty."))

    sb.reset()
    t = Trace(sb, "feral_map_003_tests_inventory", "Feral_Map",
              "List every test class and test method in the tests/ directory with their line numbers.")
    t.call("Run ast_repo_map on the `tests` directory.",
           "ast_repo_map", {"directory": "tests"}, expect="class TestOps")
    traces.append(t.final(
        "tests/test_calc.py contains TestOps (2 tests) and TestStats (2 tests).",
        "In `tests/test_calc.py`:\n- **`TestOps`** (L6): `test_add` (L7), `test_divide_by_zero_returns_none` (L10)\n"
        "- **`TestStats`** (L14): `test_empty_mean_is_zero` (L15), `test_mean` (L18)"))

    sb.reset()
    t = Trace(sb, "feral_map_004_signature_audit", "Feral_Map",
              "Check the function signatures in calc/ and verify which ones have default parameter values.")
    t.call("Map calc/ with ast_repo_map to inspect all function signatures.",
           "ast_repo_map", {"directory": "calc"}, expect="format_metric")
    traces.append(t.final(
        "Only `format_metric(name, val, decimals=2)` in calc/formatter.py has a default parameter.",
        "Across `calc/`, only `format_metric(name, val, decimals=2)` in `calc/formatter.py` (L1) defines a default parameter (`decimals=2`)."))

    # ---------------- Feral_Patch (5 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_patch_001_divide_then_test", "Feral_Patch",
              "Fix divide() in calc/ops.py so it returns None when b is 0, then run the test suite.")
    t.call("Read lines 8-9 of calc/ops.py to get the exact SEARCH block.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="return a / b")
    t.call("Patch divide() with a zero check.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("Run the unittest suite to verify.",
           "run_powershell", {"command": TEST_CMD}, expect="failures=1")
    traces.append(t.final(
        "test_divide_by_zero_returns_none passes, while test_empty_mean_is_zero now fails with None != 0.0.",
        "`divide()` now returns `None` when `b == 0` (`test_divide_by_zero_returns_none` passes). "
        "`test_empty_mean_is_zero` still fails because `RunningStats.mean()` needs its own empty-list guard."))

    sb.reset()
    t = Trace(sb, "feral_patch_002_search_miss_recovery", "Feral_Patch", "Make divide() return None for a zero divisor.")
    t.call("Patch divide() directly.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "def divide(a, b):\nreturn a / b",
                                "replace_block": "def divide(a, b):\nif b == 0:\n    return None\nreturn a / b"},
           learn=False, expect="SEARCH block not found")
    t.call("The SEARCH block missed because I guessed indentation. Read lines 8-9 of calc/ops.py first.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="def divide")
    t.call("Apply the patch with the exact 4-space indentation.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("Verify both zero and non-zero divisors.",
           "run_powershell", {"command": 'python -c "from calc.ops import divide; print(divide(1, 0), divide(6, 3))"'},
           expect="None 2.0")
    traces.append(t.final(
        "divide(1, 0) returned None and divide(6, 3) returned 2.0.",
        "`divide()` in `calc/ops.py` now returns `None` when `b == 0`. Verified: `None 2.0`."))

    sb.reset()
    t = Trace(sb, "feral_patch_003_syntax_rollback_recovery", "Feral_Patch",
              "percent() crashes when whole is 0. Make it return 0.0 in that case and verify.")
    t.call("Read lines 12-13 of calc/ops.py first.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 12, "end_line": 13}, expect="def percent")
    t.call("Patch percent() with a zero check.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "    return part / whole * 100",
                                "replace_block": "    if whole == 0\n        return 0.0\n    return part / whole * 100"},
           learn=False, expect="SyntaxError")
    t.call("apply_diff_patch rolled back the edit due to a missing colon on `if whole == 0:`. Re-apply with valid syntax.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": "    return part / whole * 100",
                                "replace_block": "    if whole == 0:\n        return 0.0\n    return part / whole * 100"},
           expect="OK: Patched")
    t.call("Run a quick verification command.",
           "run_powershell", {"command": 'python -c "from calc.ops import percent; print(percent(1, 0), percent(1, 4))"'},
           expect="0.0 25.0")
    traces.append(t.final(
        "percent(1, 0) returned 0.0 and percent(1, 4) returned 25.0.",
        "Patched `percent()` in `calc/ops.py` to return `0.0` when `whole == 0`. Verified: `0.0 25.0`."))

    sb.reset()
    t = Trace(sb, "feral_patch_004_formatter_custom_na", "Feral_Patch",
              "Update format_metric in calc/formatter.py so None values render as '<name>: null' instead of '<name>: N/A', and verify.")
    t.call("Read lines 1-5 of calc/formatter.py.",
           "read_file_lines", {"path": "calc/formatter.py", "start_line": 1, "end_line": 5}, expect="N/A")
    t.call("Replace `'N/A'` with `'null'` using apply_diff_patch.",
           "apply_diff_patch", {"path": "calc/formatter.py",
                                "search_block": "    if val is None:\n        return f'{name}: N/A'",
                                "replace_block": "    if val is None:\n        return f'{name}: null'"},
           expect="OK: Patched")
    t.call("Verify format_metric with None and a float.",
           "run_powershell", {"command": 'python -c "from calc.formatter import format_metric; print(format_metric(\'loss\', None), \'|\', format_metric(\'loss\', 1.234))"'},
           expect="loss: null | loss: 1.23")
    traces.append(t.final(
        "format_metric now prints `loss: null | loss: 1.23`.",
        "Updated `format_metric` in `calc/formatter.py` to format `None` as `<name>: null`. Verified: `loss: null | loss: 1.23`."))

    sb.reset()
    t = Trace(sb, "feral_patch_005_stats_empty_guard", "Feral_Patch",
              "Patch RunningStats.mean() in calc/stats.py so it returns 0.0 when self.values is empty, and verify with python -c.")
    t.call("Read lines 10-12 of calc/stats.py.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 10, "end_line": 12}, expect="def mean")
    t.call("Patch mean() with an `if not self.values: return 0.0` guard.",
           "apply_diff_patch", {"path": "calc/stats.py", "search_block": MEAN_OLD, "replace_block": MEAN_NEW},
           expect="OK: Patched")
    t.call("Verify empty and non-empty RunningStats.mean().",
           "run_powershell", {"command": 'python -c "from calc.stats import RunningStats; s = RunningStats(); print(s.mean()); s.push(4); s.push(6); print(s.mean())"'},
           expect="0.0\n5.0")
    traces.append(t.final(
        "Empty mean returned 0.0 and [4, 6] returned 5.0.",
        "Patched `RunningStats.mean()` in `calc/stats.py` to return `0.0` when empty. Verified output: `0.0` and `5.0`."))

    # ---------------- Feral_Research (4 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_research_001_pathlib_rglob", "Feral_Research",
              "How do I recursively find all .py files with pathlib? Check the docs.")
    t.call("Search the docs for pathlib recursive glob.",
           "web_research", {"query": "pathlib recursive glob files", "url": ""}, expect="rglob")
    t.call("Open the Path.rglob entry to confirm the exact method signature and example.",
           "web_research", {"query": "pathlib Path.rglob",
                            "url": "https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob"}, expect="**/")
    traces.append(t.final(
        "Path.rglob(pattern) is equivalent to glob with **/ in front.",
        "Use `Path.rglob('*.py')`:\n```python\nfrom pathlib import Path\nfor p in Path('.').rglob('*.py'):\n    print(p)\n```"))

    sb.reset()
    t = Trace(sb, "feral_research_002_unittest_discover", "Feral_Research",
              "Run this project's tests with unittest discovery. Check the docs for the right flags first.")
    t.call("Search the unittest docs for test discovery flags.",
           "web_research", {"query": "unittest discover start directory top level", "url": ""}, expect="Test Discovery")
    t.call("Open the unittest test-discovery page to confirm `-s` and `-t`.",
           "web_research", {"query": "unittest test discovery",
                            "url": "https://docs.python.org/3/library/unittest.html#test-discovery"}, expect="-t")
    t.call("Run unittest discovery with `-s tests -t .`.",
           "run_powershell", {"command": TEST_CMD}, expect="errors=2")
    traces.append(t.final(
        "Ran 4 tests with 2 ZeroDivisionError failures.",
        "Ran `python -m unittest discover -s tests -t .`: 4 tests executed, **2 errors** (`test_divide_by_zero_returns_none` and `test_empty_mean_is_zero`)."))

    sb.reset()
    t = Trace(sb, "feral_research_003_select_string_literal", "Feral_Research",
              "Check the PowerShell Select-String docs for how to do a literal (non-regex) match, then use it to find 'a / b' in calc/ops.py.")
    t.call("Search the docs for Select-String literal matching.",
           "web_research", {"query": "Select-String literal match regex", "url": ""}, expect="Select-String")
    t.call("Open the Select-String doc page to verify the flag name.",
           "web_research", {"query": "Select-String SimpleMatch",
                            "url": "https://learn.microsoft.com/powershell/module/microsoft.powershell.utility/select-string"},
           expect="-SimpleMatch")
    t.call("Run Select-String with -SimpleMatch on calc/ops.py.",
           "run_powershell", {"command": "Select-String -Path calc/ops.py -Pattern 'a / b' -SimpleMatch | ForEach-Object { \"$($_.LineNumber): $($_.Line.Trim())\" }"},
           expect="9: return a / b")
    traces.append(t.final(
        "`-SimpleMatch` disables regex parsing and found `9: return a / b`.",
        "The `-SimpleMatch` switch runs a literal search in `Select-String`. In `calc/ops.py`, it matched line 9: `return a / b`."))

    sb.reset()
    t = Trace(sb, "feral_research_004_ast_parse_docs", "Feral_Research",
              "Check the Python ast docs for how ast.parse inspects function definitions and line numbers.")
    t.call("Search the docs for ast.parse FunctionDef.",
           "web_research", {"query": "ast.parse FunctionDef lineno", "url": ""}, expect="ast.parse")
    t.call("Open the ast.parse documentation entry.",
           "web_research", {"query": "ast.parse",
                            "url": "https://docs.python.org/3/library/ast.html#ast.parse"}, expect="ast.FunctionDef")
    traces.append(t.final(
        "ast.parse returns a module tree whose `.body` contains `ast.FunctionDef` and `ast.ClassDef` nodes with `.name`, `.lineno`, and `ast.unparse(node.args)`.",
        "`ast.parse(source)` builds an AST whose `tree.body` exposes `ast.FunctionDef` and `ast.ClassDef` nodes. "
        "Each node provides `node.name`, `node.lineno`, and `ast.unparse(node.args)` for the signature."))

    # ---------------- Feral_Delegate (4 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_delegate_001_test_triage", "Feral_Delegate",
              "Run the full test suite and tell me what's broken, but keep the noisy logs out of our main context.")
    t.call("Delegate the test run to spawn_subagent so raw tracebacks stay out of the main context.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and report each failing test with its root-cause frame."},
           expect="2 failing")
    traces.append(t.final(
        "Both failing tests (`test_divide_by_zero_returns_none` and `test_empty_mean_is_zero`) fail at `calc/ops.py:9` in `divide`.",
        "2 of 4 tests are failing, both raising `ZeroDivisionError` at `calc/ops.py:9` in `divide`:\n"
        "- `test_divide_by_zero_returns_none`\n- `test_empty_mean_is_zero`"))

    sb.reset()
    t = Trace(sb, "feral_delegate_002_verify_after_divide_patch", "Feral_Delegate",
              "Patch divide() to return None on zero, then use a subagent to check if any tests still fail.")
    t.call("Patch divide() in calc/ops.py.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW},
           expect="OK: Patched")
    t.call("Delegate test suite verification to spawn_subagent.",
           "spawn_subagent", {"task_prompt": "Run unittest discovery and report any remaining failures."},
           expect="1 failing")
    traces.append(t.final(
        "The subagent reports 1 remaining failure: `test_empty_mean_is_zero` (`AssertionError: None != 0.0`).",
        "After patching `divide()`, 3 of 4 tests pass. Only `test_empty_mean_is_zero` still fails (`AssertionError: None != 0.0`)."))

    sb.reset()
    t = Trace(sb, "feral_delegate_003_verify_after_stats_patch", "Feral_Delegate",
              "Patch RunningStats.mean() to return 0.0 when empty, then have a subagent run the test suite and report what's left.")
    t.call("Patch RunningStats.mean() in calc/stats.py.",
           "apply_diff_patch", {"path": "calc/stats.py", "search_block": MEAN_OLD, "replace_block": MEAN_NEW},
           expect="OK: Patched")
    t.call("Ask a subagent to run the test suite and summarize remaining failures.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and list any failing tests."},
           expect="1 failing")
    traces.append(t.final(
        "The subagent confirmed `test_empty_mean_is_zero` now passes and only `test_divide_by_zero_returns_none` still fails.",
        "`test_empty_mean_is_zero` now passes. The subagent reports 1 remaining error: `test_divide_by_zero_returns_none` at `calc/ops.py:9`."))

    sb.reset()
    t = Trace(sb, "feral_delegate_004_triage_then_read_root_cause", "Feral_Delegate",
              "Use a subagent to find which file and line the test suite crashes on, then read just that function.")
    t.call("Spawn a subagent to triage the test suite.",
           "spawn_subagent", {"task_prompt": "Run the tests and report the exact file and line of any crash."},
           expect="calc/ops.py:9")
    t.call("The crash is at calc/ops.py:9. Read lines 8-9.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="def divide")
    traces.append(t.final(
        "Lines 8-9 show `def divide(a, b): return a / b` with no zero-divisor check.",
        "The subagent isolated the crash to `calc/ops.py:9`. Reading lines 8–9 confirms `divide(a, b)` performs `return a / b` with no zero check."))

    # ---------------- Feral_Combo (4 traces) ----------------
    sb.reset()
    t = Trace(sb, "feral_combo_001_full_fix", "Feral_Combo",
              "The test suite is failing. Find the root cause, fix it across files, and make all tests pass.")
    t.call("Triage the test suite in a subagent first.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and report each failing test with its root-cause frame."},
           expect="2 failing")
    t.call("Map calc/ to see the module layout.",
           "ast_repo_map", {"directory": "calc"}, expect="def divide")
    t.call("Read divide() at lines 8-9 of calc/ops.py.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="return a / b")
    t.call("Patch divide() to return None when b == 0.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("Read RunningStats.mean() at lines 11-12 of calc/stats.py.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="def mean")
    t.call("Guard mean() against an empty list so it returns 0.0.",
           "apply_diff_patch", {"path": "calc/stats.py", "search_block": MEAN_OLD, "replace_block": MEAN_NEW}, expect="OK: Patched")
    t.call("Run the full test suite to verify all 4 tests pass.",
           "run_powershell", {"command": TEST_CMD}, expect="OK")
    traces.append(t.final(
        "All 4 unit tests pass.",
        "Fixed both bugs (`divide()` zero check in `calc/ops.py` and empty-list check in `RunningStats.mean()` in `calc/stats.py`). All 4 tests pass."))

    sb.reset()
    t = Trace(sb, "feral_combo_002_add_method", "Feral_Combo",
              "Add a count() method to RunningStats that returns how many values were pushed, and verify it works.")
    t.call("Map calc/ to locate RunningStats.",
           "ast_repo_map", {"directory": "calc"}, expect="def push")
    t.call("Read lines 8-12 of calc/stats.py.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 8, "end_line": 12}, expect="self.values.append(x)")
    t.call("Patch calc/stats.py to add count() right after push().",
           "apply_diff_patch", {"path": "calc/stats.py",
                                "search_block": "    def push(self, x):\n        self.values.append(x)\n",
                                "replace_block": "    def push(self, x):\n        self.values.append(x)\n\n"
                                                 "    def count(self):\n        return len(self.values)\n"},
           expect="OK: Patched")
    t.call("Verify RunningStats.count() via python -c.",
           "run_powershell", {"command": 'python -c "from calc.stats import RunningStats; s = RunningStats(); s.push(1); s.push(2); s.push(3); print(s.count())"'},
           expect="3")
    traces.append(t.final(
        "RunningStats.count() returned 3.",
        "Added `RunningStats.count()` in `calc/stats.py` and verified it returns `3` after 3 pushes."))

    sb.reset()
    t = Trace(sb, "feral_combo_003_add_summary_method", "Feral_Combo",
              "Add a summary(name) method to RunningStats in calc/stats.py that formats the mean using format_metric from calc.formatter, fix empty mean(), and verify.")
    t.call("Map calc/ to inspect both formatter.py and stats.py.",
           "ast_repo_map", {"directory": "calc"}, expect="format_metric")
    t.call("Read calc/stats.py lines 1-12.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 1, "end_line": 12}, expect="class RunningStats")
    t.call("Patch calc/stats.py to import format_metric, guard empty mean(), and add summary(self, name).",
           "apply_diff_patch",
           {
               "path": "calc/stats.py",
               "search_block": "from calc.ops import divide\n\n\nclass RunningStats:",
               "replace_block": "from calc.formatter import format_metric\nfrom calc.ops import divide\n\n\nclass RunningStats:",
           },
           expect="OK: Patched")
    t.call("Add the empty-list guard in mean() and the new summary(self, name) method.",
           "apply_diff_patch",
           {
               "path": "calc/stats.py",
               "search_block": MEAN_OLD,
               "replace_block": MEAN_NEW + "\n\n    def summary(self, name):\n        return format_metric(name, self.mean())",
           },
           expect="OK: Patched")
    t.call("Verify summary() on both empty and populated RunningStats instances.",
           "run_powershell",
           {"command": 'python -c "from calc.stats import RunningStats; s = RunningStats(); print(s.summary(\'ms\')); s.push(10); s.push(20); print(s.summary(\'ms\'))"'},
           expect="ms: 0.00\nms: 15.00")
    traces.append(t.final(
        "summary('ms') printed `ms: 0.00` when empty and `ms: 15.00` after pushing 10 and 20.",
        "Added `RunningStats.summary(name)` using `format_metric` and guarded `mean()` on empty input. Verified: `ms: 0.00` and `ms: 15.00`."))

    sb.reset()
    t = Trace(sb, "feral_combo_004_fix_all_and_subagent_verify", "Feral_Combo",
              "Fix both divide() and RunningStats.mean() using surgical patches, then confirm the full test suite passes via a subagent.")
    t.call("Read divide() in calc/ops.py.",
           "read_file_lines", {"path": "calc/ops.py", "start_line": 8, "end_line": 9}, expect="def divide")
    t.call("Patch divide() to return None on zero.",
           "apply_diff_patch", {"path": "calc/ops.py", "search_block": DIVIDE_OLD, "replace_block": DIVIDE_NEW}, expect="OK: Patched")
    t.call("Read mean() in calc/stats.py.",
           "read_file_lines", {"path": "calc/stats.py", "start_line": 11, "end_line": 12}, expect="def mean")
    t.call("Patch mean() to return 0.0 on empty values.",
           "apply_diff_patch", {"path": "calc/stats.py", "search_block": MEAN_OLD, "replace_block": MEAN_NEW}, expect="OK: Patched")
    t.call("Delegate the final test verification to spawn_subagent.",
           "spawn_subagent", {"task_prompt": "Run the unittest suite and confirm all tests pass."},
           expect="all 4 tests passed")
    traces.append(t.final(
        "The subagent confirmed all 4 tests passed.",
        "Patched `calc/ops.py` and `calc/stats.py`, and verified via subagent that **all 4 tests passed**."))

    # ---------------- Anchor_General (remaining 4 anchors from ANCHOR_TASKS) ----------------
    for tid, prompt, think, ans in ANCHOR_TASKS[14:]:
        traces.append(anchor_trace(tid, prompt, think, ans))

    return traces


def main():
    tok = AutoTokenizer.from_pretrained(str(ROOT))
    sb = Sandbox("feral_sandbox", SANDBOX_FILES)
    print("1. Replaying Feral Apex Toolchain trajectories against feral_sandbox/...")
    try:
        traces = build_traces(sb)
    finally:
        sb.cleanup()

    print(f"2. Tokenizing {len(traces)} trajectories with completion-only + mistake masking...")
    records = []
    masked_total = 0
    tool_use = Counter()
    for tr in traces:
        rec, masked = to_record(tr, tok)
        records.append(rec)
        masked_total += masked
        for m in tr.messages:
            if m["role"] == "assistant":
                for raw in re.findall(r"<tool_call>(.*?)</tool_call>", m["content"], flags=re.S):
                    tool_use[json.loads(raw)["name"]] += 1

    write_and_summarize(records, masked_total, OUTPUT, STAGE_ORDER)
    print("\n=== TOOL CALL COVERAGE (STEP 5) ===")
    for name in TOOL_SCHEMA:
        if tool_use[name]:
            print(f"  {name:18s}: {tool_use[name]:2d} calls")


if __name__ == "__main__":
    main()
