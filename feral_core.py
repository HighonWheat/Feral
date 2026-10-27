"""
Feral shared core
=================
Single source of truth for every dataset generator:

  - SYSTEM_FERAL: the one system prompt Feral is trained (and later run) with
  - TOOL_SCHEMA:  tool names + required arguments
  - Sandbox:      REAL implementations of every tool, run inside a throwaway folder
  - Trace:        records a trajectory while executing each tool call for real
  - tokenize():   completion-only + mistake-masked tokenization

Generators import from here so the prompt and tools can never drift apart again.
"""
import ast
import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SYSTEM_FERAL = (
    "You are Feral, an autonomous AI software engineering agent running on Windows 11 PowerShell. "
    "Always reason inside <think>...</think> before acting. Prefer surgical tools: map the repo, read only "
    "the lines you need, and patch with exact SEARCH/REPLACE blocks. Verify every change by running it.\n"
    "Available tools:\n"
    "- run_powershell(command: str): Run a PowerShell command in the workspace; returns combined output and exit code.\n"
    "- read_file_lines(path: str, start_line: int, end_line: int): Return the given line range with line numbers.\n"
    "- apply_diff_patch(path: str, search_block: str, replace_block: str): Replace an exact, unique SEARCH block; "
    "Python files are syntax-checked and rolled back on failure.\n"
    "- write_file(path: str, content: str): Create a new file (or fully overwrite a small one) with UTF-8 content.\n"
    "- ast_repo_map(directory: str): List every Python file with its classes, functions, signatures, and line numbers.\n"
    "- web_research(query: str, url: str = \"\"): Search documentation, or open a URL as clean Markdown.\n"
    "- spawn_subagent(task_prompt: str): Delegate an isolated subtask to a fresh-context worker; returns only a concise report.\n"
    "When calling a tool, emit ONLY: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
)

TOOL_SCHEMA = {
    "run_powershell": ["command"],
    "read_file_lines": ["path", "start_line", "end_line"],
    "apply_diff_patch": ["path", "search_block", "replace_block"],
    "write_file": ["path", "content"],
    "ast_repo_map": ["directory"],
    "web_research": ["query"],
    "spawn_subagent": ["task_prompt"],
}

# Offline documentation cache for web_research (replaced by the live browser MCP in Phase 5, Step 4).
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
    "https://docs.python.org/3/library/ast.html#ast.parse": (
        "ast - Abstract Syntax Trees (ast.parse)",
        "ast.parse(source, filename='<unknown>', mode='exec')\n\n"
        "Parse the source into an AST node. Walk top-level definitions via `tree.body` and inspect "
        "`ast.FunctionDef` (`node.name`, `node.lineno`, `ast.unparse(node.args)`) and `ast.ClassDef`.",
    ),
    "https://docs.python.org/3/library/subprocess.html#subprocess.run": (
        "subprocess - subprocess.run",
        "subprocess.run(args, *, capture_output=False, text=None, timeout=None, cwd=None, check=False)\n\n"
        "Run the command described by `args`. Wait for command to complete, then return a `CompletedProcess` "
        "instance with `.returncode`, `.stdout`, and `.stderr`.",
    ),
}


class SkipTrace(Exception):
    """Raised when a scenario doesn't behave as designed (e.g. a 'mistake' command actually succeeds)."""


def _ps_wrap(command):
    # - Native stderr lines come back as ErrorRecords: unwrap them to the raw line (blank lines stay blank).
    # - Cmdlet errors (e.g. 'wc' not recognized) become "ERROR: <message>" and force a non-zero exit code.
    # - Out-String -Stream formats objects exactly like an interactive console would.
    return (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        "$global:__feral_err = $false\n"
        "& { " + command + " } 2>&1 | ForEach-Object {\n"
        "  if ($_ -is [System.Management.Automation.ErrorRecord]) {\n"
        "    if ($_.FullyQualifiedErrorId -like 'NativeCommandError*') {\n"
        "      if ($null -ne $_.TargetObject) { \"$($_.TargetObject)\" } else { '' }\n"
        "    } else { $global:__feral_err = $true; \"ERROR: $($_.Exception.Message)\" }\n"
        "  } else { $_ }\n"
        "} | Out-String -Stream -Width 160\n"
        "if ($global:__feral_err) { exit 1 } elseif ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 0 }\n"
    )


def _decode_clixml(stderr):
    """Windows PowerShell serializes stderr as CLIXML when redirected. Keep only error text; drop progress records."""
    if "#< CLIXML" not in stderr:
        return stderr
    import html
    pre, _, xml = stderr.partition("#< CLIXML")
    errors = re.findall(r'<S S="Error">(.*?)</S>', xml, flags=re.S)
    text = "".join(html.unescape(e).replace("_x000D__x000A_", "\n").replace("_x000A_", "\n") for e in errors)
    # Drop the wrapper's own source line + caret marker from parser errors; keep the human-readable message.
    lines = [l for l in text.splitlines() if not l.startswith(("At line:", "+ ")) and l.strip()]
    lines = [l.replace("    + CategoryInfo", "CategoryInfo").replace("    + FullyQualifiedErrorId", "FullyQualifiedErrorId")
             for l in lines if "CategoryInfo" not in l and "FullyQualifiedErrorId" not in l]
    return pre + ("ERROR: " + " ".join(l.strip() for l in lines) if lines else "")


class Sandbox:
    """A throwaway workspace folder with real tool implementations bound to it."""

    def __init__(self, name, files):
        self.root = ROOT / name
        self.files = files

    def reset(self, extra_files=None):
        self.cleanup()
        for rel, content in {**self.files, **(extra_files or {})}.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def cleanup(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _resolve(self, path):
        p = (self.root / path).resolve()
        root = self.root.resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return p

    # ---------------- tools ----------------
    def run_powershell(self, command):
        encoded = base64.b64encode(_ps_wrap(command).encode("utf-16-le")).decode()
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out after 90s.\n[exit_code=124]"
        out = "\n".join(line.rstrip() for line in (res.stdout + _decode_clixml(res.stderr)).splitlines()).strip("\n")
        return f"{out}\n[exit_code={res.returncode}]" if out.strip() else f"[exit_code={res.returncode}]"

    def read_file_lines(self, path, start_line, end_line):
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: {path} does not exist."
        lines = p.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        s, e = max(1, int(start_line)), min(total, int(end_line))
        body = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(s, e + 1))
        return f"{path} (lines {s}-{e} of {total})\n{body}"

    def apply_diff_patch(self, path, search_block, replace_block):
        p = self._resolve(path)
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
                return (f"ERROR: Patch rejected and rolled back. SyntaxError: {err.msg} (line {err.lineno}). "
                        f"{path} is unchanged.")
        removed, added = search_block.count("\n") + 1, replace_block.count("\n") + 1
        return f"OK: Patched {path} (1 hunk, -{removed} +{added} lines). Syntax check passed."

    def write_file(self, path, content):
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: Wrote {len(content.encode('utf-8'))} bytes to {path}."

    def ast_repo_map(self, directory):
        base = self._resolve(directory)
        files = sorted(f for f in base.rglob("*.py") if "__pycache__" not in f.parts)
        out = [f"Repo map of {directory} ({len(files)} files)"]
        for f in files:
            out.append(f.relative_to(self.root).as_posix())
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

    def web_research(self, query, url=""):
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

    def spawn_subagent(self, task_prompt):
        # Deterministic test-runner worker: own process, returns only a short report.
        res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        log = res.stdout + res.stderr
        ran = re.search(r"Ran (\d+) tests?", log)
        n = int(ran.group(1)) if ran else 0
        problems = []
        sandbox_dir = re.escape(self.root.name)
        for block in re.split(r"^=+$", log, flags=re.M)[1:]:
            parts = re.split(r"^-+$", block.strip(), flags=re.M)
            m = re.match(r"(FAIL|ERROR): (\w+)", parts[0].strip())
            if not m:
                continue
            tb = parts[1] if len(parts) > 1 else ""
            frames = re.findall(sandbox_dir + r'[\\/](.+?)", line (\d+), in (\w+)', tb)
            where = (f"{frames[-1][0].replace(chr(92), '/')}:{frames[-1][1]} in {frames[-1][2]}"
                     if frames else "unknown")
            last = [l.strip() for l in tb.strip().splitlines() if l.strip()]
            problems.append(f"- {m.group(1)} {m.group(2)} at {where} -> {last[-1] if last else 'unknown'}")
        head = f"[subagent report | fresh context | raw log withheld: {len(log)} chars]\nTask: {task_prompt}"
        if res.returncode == 0:
            return f"{head}\nResult: all {n} tests passed."
        return f"{head}\nResult: ran {n} tests, {len(problems)} failing.\n" + "\n".join(problems)

    def tools(self):
        return {name: getattr(self, name) for name in TOOL_SCHEMA}


class Trace:
    """Builds one trajectory. Every call() EXECUTES the tool and records its real output."""

    def __init__(self, sandbox, tid, stage, prompt, group=None):
        self.sandbox, self.tid, self.stage = sandbox, tid, stage
        self.group = group or tid
        self.messages = [{"role": "system", "content": SYSTEM_FERAL}, {"role": "user", "content": prompt}]

    def call(self, think, name, args, learn=True, expect=None, must_fail=False):
        payload = json.dumps({"name": name, "arguments": args})
        self.messages.append({
            "role": "assistant", "learn": learn,
            "content": f"<think>\n{think}\n</think>\n<tool_call>{payload}</tool_call>",
        })
        obs = self.sandbox.tools()[name](**args)
        if must_fail and (obs.endswith("[exit_code=0]") or obs.startswith("OK")):
            raise SkipTrace(f"[{self.tid}] expected {name} to fail, but it succeeded:\n{obs}")
        if expect is not None and expect not in obs:
            raise AssertionError(f"[{self.tid}] expected {expect!r} in {name} output, got:\n{obs}")
        self.messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})
        return obs

    def final(self, think, answer):
        self.messages.append({"role": "assistant", "learn": True, "content": f"<think>\n{think}\n</think>\n{answer}"})
        return self


def anchor_trace(tid, prompt, think, answer):
    """A no-tool reasoning trace (prevents catastrophic forgetting of general knowledge)."""
    return Trace(None, tid, "Anchor_General", prompt).final(think, answer)


def tool_output(obs):
    """Strip the trailing [exit_code=N] line from a run_powershell observation."""
    return re.sub(r"\n?\[exit_code=-?\d+\]$", "", obs).strip()


def tokenize(messages, tok):
    """Completion-only + mistake masking: train only on assistant turns with learn=True."""
    input_ids, labels, masked = [], [], 0
    for msg in messages:
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


def to_record(trace, tok):
    ids, labels, masked = tokenize(trace.messages, tok)
    return {"id": trace.tid, "stage": trace.stage, "group": trace.group, "messages": trace.messages,
            "input_ids": ids, "labels": labels}, masked


def write_and_summarize(records, masked_total, out_path, stage_order):
    from collections import Counter
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n, tot, act = Counter(), Counter(), Counter()
    for r in records:
        n[r["stage"]] += 1
        tot[r["stage"]] += len(r["input_ids"])
        act[r["stage"]] += sum(l != -100 for l in r["labels"])
    print(f"\n=== {Path(out_path).name} ===")
    for st in stage_order:
        if n[st]:
            print(f"  {st:20s} | Traces: {n[st]:3d} | Tokens: {tot[st]:6d} | Active: {act[st]:6d} ({act[st] / tot[st] * 100:4.1f}%)")
    gt, ga = sum(tot.values()), sum(act.values())
    print("-" * 82)
    print(f"  {'TOTAL':20s} | Traces: {len(records):3d} | Tokens: {gt:6d} | Active: {ga:6d} ({ga / gt * 100:4.1f}%)")
    print(f"  Mistake-masked assistant turns: {masked_total}")
