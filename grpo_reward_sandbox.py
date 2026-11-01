"""
Phase 4, Step 1 - Deterministic Code & Terminal Verification Sandbox (`grpo_reward_sandbox.py`)
===============================================================================================
Implements the deterministic Reinforcement Learning with Verifiable Rewards (RLVR / GRPO)
grading environment for Feral:

    R_total = R_format + R_json_schema + R_execution_pass + R_stderr_recovery + R_conciseness

Every candidate rollout generated during GRPO is parsed and executed inside an isolated
`feral_core.Sandbox` directory against real PowerShell commands and Python unit assertions.
Also exports `GRPO_TASKS`, `score_rollout()`, and `compute_grpo_advantages()` for Step 2.
"""
import json
import math
import re
import statistics
from typing import Dict, List

from feral_core import SYSTEM_FERAL, TOOL_SCHEMA, Sandbox, tool_output

SANDBOX_FILES = {
    "src/rate_limiter.py": (
        "def allow_request(tokens):\n"
        "    if tokens >= 0:\n"
        "        return True\n"
        "    return False\n"
    ),
    "src/paginator.py": (
        "def page_slice(items, page, per_page):\n"
        "    start = page * per_page\n"
        "    return items[start:start + per_page]\n"
    ),
    "src/env_parser.py": (
        "def parse_port(raw, default=8080):\n"
        "    return int(raw)\n"
    ),
    "src/percentage.py": (
        "def safe_ratio(num, den):\n"
        "    return round(num / den, 2)\n"
    ),
    "src/slugify.py": (
        "def make_slug(title):\n"
        "    return title.strip().replace(' ', '-')\n"
    ),
    "src/clamp_score.py": (
        "def clamp_score(x):\n"
        "    return max(0.0, min(x, 0.0))\n"
    ),
    "logs/app.log": (
        "2026-10-30 10:00:01 INFO  Booting worker\n"
        "2026-10-30 10:01:12 ERROR Connection refused on 127.0.0.1:5432\n"
        "2026-10-30 10:02:00 WARN  Retrying in 5s\n"
        "2026-10-30 10:02:05 ERROR Max retries exceeded\n"
        "2026-10-30 10:03:00 INFO  Fallback cache active\n"
    ),
    "config/service.json": '{\n  "service": "feral-core",\n  "max_workers": 8,\n  "retry": true\n}\n',
}


# ============================================================================
# 1. Verifiable RLVR / GRPO Task Suite (Code Patching + PowerShell Recovery)
# ============================================================================
GRPO_TASKS: List[Dict] = [
    {
        "id": "grpo_01_rate_limiter_patch",
        "kind": "code_patch",
        "expect_tool": "apply_diff_patch",
        "target_path": "src/rate_limiter.py",
        "verify_cmd": 'python -c "from src.rate_limiter import allow_request; print(allow_request(0), allow_request(1), allow_request(-1))"',
        "expect_output": "False True False",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "src/rate_limiter.py allows requests when tokens == 0. Patch allow_request so it only returns True when tokens > 0.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRead src/rate_limiter.py first.\n</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "src/rate_limiter.py", "start_line": 1, "end_line": 10}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/rate_limiter.py (lines 1-4 of 4)\n   1 | def allow_request(tokens):\n   2 |     if tokens >= 0:\n   3 |         return True\n   4 |     return False\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_02_one_based_paginator",
        "kind": "code_patch",
        "expect_tool": "apply_diff_patch",
        "target_path": "src/paginator.py",
        "verify_cmd": 'python -c "from src.paginator import page_slice; print(page_slice([10, 20, 30, 40], 1, 2), page_slice([10, 20, 30, 40], 2, 2))"',
        "expect_output": "[10, 20] [30, 40]",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "page_slice in src/paginator.py treats `page` as 0-indexed instead of 1-indexed (page=1 should start at index 0). Patch it.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRead src/paginator.py lines 1-10.\n</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "src/paginator.py", "start_line": 1, "end_line": 10}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/paginator.py (lines 1-3 of 3)\n   1 | def page_slice(items, page, per_page):\n   2 |     start = page * per_page\n   3 |     return items[start:start + per_page]\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_03_safe_ratio_zero_den",
        "kind": "code_patch",
        "expect_tool": "apply_diff_patch",
        "target_path": "src/percentage.py",
        "verify_cmd": 'python -c "from src.percentage import safe_ratio; print(safe_ratio(5, 0), safe_ratio(1, 4))"',
        "expect_output": "0.0 0.25",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "safe_ratio in src/percentage.py crashes when den == 0. Patch it to return 0.0 when den == 0.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRead src/percentage.py lines 1-10.\n</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "src/percentage.py", "start_line": 1, "end_line": 10}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/percentage.py (lines 1-2 of 2)\n   1 | def safe_ratio(num, den):\n   2 |     return round(num / den, 2)\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_04_clamp_score_upper_bound",
        "kind": "code_patch",
        "expect_tool": "apply_diff_patch",
        "target_path": "src/clamp_score.py",
        "verify_cmd": 'python -c "from src.clamp_score import clamp_score; print(clamp_score(-0.5), clamp_score(0.4), clamp_score(1.5))"',
        "expect_output": "0.0 0.4 1.0",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "clamp_score in src/clamp_score.py always returns 0.0 because the upper bound in min(x, 0.0) should be 1.0. Patch it.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRead src/clamp_score.py lines 1-10.\n</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "src/clamp_score.py", "start_line": 1, "end_line": 10}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/clamp_score.py (lines 1-2 of 2)\n   1 | def clamp_score(x):\n   2 |     return max(0.0, min(x, 0.0))\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_05_slugify_lowercase",
        "kind": "code_patch",
        "expect_tool": "apply_diff_patch",
        "target_path": "src/slugify.py",
        "verify_cmd": 'python -c "from src.slugify import make_slug; print(make_slug(\'  Hello World \'))"',
        "expect_output": "hello-world",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "make_slug in src/slugify.py does not lowercase the string. Patch it so make_slug('  Hello World ') returns 'hello-world'.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRead src/slugify.py lines 1-10.\n</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "src/slugify.py", "start_line": 1, "end_line": 10}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/slugify.py (lines 1-2 of 2)\n   1 | def make_slug(title):\n   2 |     return title.strip().replace(' ', '-')\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_06_stderr_grep_count_recovery",
        "kind": "stderr_recovery",
        "expect_tool": "run_powershell",
        "forbidden_cmd_tokens": ["grep"],
        "expect_output": "2",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "Count how many ERROR lines appear in logs/app.log.",
            },
            {
                "role": "assistant",
                "content": '<think>\nRun `grep -c ERROR logs/app.log`.\n</think>\n<tool_call>{"name": "run_powershell", "arguments": {"command": "grep -c ERROR logs/app.log"}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nERROR: The term 'grep' is not recognized as the name of a cmdlet, function, script file, or operable program.\n[exit_code=1]\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_07_stderr_jq_json_recovery",
        "kind": "stderr_recovery",
        "expect_tool": "run_powershell",
        "forbidden_cmd_tokens": ["jq"],
        "expect_output": "8",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "Read the max_workers integer field from config/service.json.",
            },
            {
                "role": "assistant",
                "content": '<think>\nUse jq to read max_workers.\n</think>\n<tool_call>{"name": "run_powershell", "arguments": {"command": "jq .max_workers config/service.json"}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nERROR: The term 'jq' is not recognized as the name of a cmdlet, function, script file, or operable program.\n[exit_code=1]\n</tool_response>",
            },
        ],
    },
    {
        "id": "grpo_08_stderr_wc_lines_recovery",
        "kind": "stderr_recovery",
        "expect_tool": "run_powershell",
        "forbidden_cmd_tokens": ["wc"],
        "expect_output": "5",
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "How many total lines are in logs/app.log?",
            },
            {
                "role": "assistant",
                "content": '<think>\nRun `wc -l logs/app.log`.\n</think>\n<tool_call>{"name": "run_powershell", "arguments": {"command": "wc -l logs/app.log"}}</tool_call>',
            },
            {
                "role": "user",
                "content": "<tool_response>\nERROR: The term 'wc' is not recognized as the name of a cmdlet, function, script file, or operable program.\n[exit_code=1]\n</tool_response>",
            },
        ],
    },
]


# ============================================================================
# 2. Deterministic 5-Part Reward Function Suite
# ============================================================================
def score_rollout(task: Dict, completion: str, sb: Sandbox) -> Dict:
    """
    Grades a candidate assistant completion on `task` using 5 deterministic rewards:
        R_total = R_format + R_json_schema + R_execution_pass + R_stderr_recovery + R_conciseness
    """
    text = completion.strip()

    # --- 1. R_format (-1.0 to +0.5) ---
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    tool_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S)
    strict_layout = bool(re.fullmatch(r"<think>.*?</think>\s*<tool_call>.*?</tool_call>", text, flags=re.S))

    if (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and len(tool_matches) == 1
        and strict_layout
    ):
        r_format = 0.50
    elif think_match is not None and len(tool_matches) == 1:
        r_format = 0.20
    else:
        r_format = -1.00

    # --- 2. R_conciseness (-0.25 to +0.25) ---
    if think_match is not None:
        think_words = len(think_match.group(1).strip().split())
        if 6 <= think_words <= 65:
            r_conciseness = 0.25
        elif 66 <= think_words <= 120:
            r_conciseness = 0.10
        else:
            r_conciseness = -0.25
    else:
        r_conciseness = -0.25

    # --- 3. R_json_schema (-0.5 to +0.5) ---
    parsed_call = None
    r_json_schema = -0.50
    if len(tool_matches) == 1:
        try:
            call = json.loads(tool_matches[0])
            name = call.get("name")
            args = call.get("arguments")
            if isinstance(args, dict) and name in TOOL_SCHEMA:
                req = TOOL_SCHEMA[name]
                if all(k in args and str(args[k]).strip() != "" for k in req):
                    if name == task["expect_tool"]:
                        r_json_schema = 0.50
                        parsed_call = (name, args)
                    else:
                        r_json_schema = 0.10
        except Exception:
            r_json_schema = -0.50

    # --- 4. R_execution_pass (-0.5 to +2.0) & 5. R_stderr_recovery (-0.5 to +0.75) ---
    r_execution_pass = -0.50
    r_stderr_recovery = 0.00
    exec_detail = "not_executed"

    if parsed_call is not None:
        tool_name, tool_args = parsed_call
        sb.reset()
        try:
            obs = sb.tools()[tool_name](**tool_args)
        except Exception as exc:
            obs = f"ERROR: {exc}"

        if task["kind"] == "code_patch":
            if tool_Args_path := tool_args.get("path"):
                path_ok = tool_Args_path.replace("\\", "/") == task["target_path"]
            else:
                path_ok = False

            if obs.startswith("OK: Patched") and path_ok:
                # Patch applied and passed py_compile! Now run the unit test verification:
                verify_obs = sb.run_powershell(task["verify_cmd"])
                verify_out = tool_output(verify_obs)
                if verify_obs.endswith("[exit_code=0]") and task["expect_output"] in verify_out:
                    r_execution_pass = 2.00
                    r_stderr_recovery = 0.75
                    exec_detail = f"patch_and_test_pass ({verify_out})"
                else:
                    # Syntactically valid patch, but failed unit test assertion
                    r_execution_pass = 0.50
                    r_stderr_recovery = 0.00
                    exec_detail = f"patch_ok_but_test_failed ({verify_out[:60]})"
            else:
                r_execution_pass = -0.50
                r_stderr_recovery = -0.25
                exec_detail = obs.splitlines()[0][:80]

        elif task["kind"] == "stderr_recovery":
            cmd_str = str(tool_args.get("command", ""))
            used_forbidden = any(
                re.search(rf"\b{re.escape(tok)}\b", cmd_str)
                for tok in task.get("forbidden_cmd_tokens", [])
            )
            clean_out = tool_output(obs)

            if used_forbidden:
                r_stderr_recovery = -0.50
                r_execution_pass = -0.50
                exec_detail = "repeated_forbidden_linux_cmd"
            elif obs.endswith("[exit_code=0]") and clean_out == task["expect_output"]:
                r_execution_pass = 2.00
                r_stderr_recovery = 0.75
                exec_detail = f"recovery_pass ({clean_out})"
            elif obs.endswith("[exit_code=0]"):
                r_execution_pass = 0.25
                r_stderr_recovery = 0.25
                exec_detail = f"exit_0_wrong_output ({clean_out[:40]})"
            else:
                r_execution_pass = -0.50
                r_stderr_recovery = 0.00
                exec_detail = obs.splitlines()[0][:80]

    r_total = round(
        r_format + r_json_schema + r_execution_pass + r_stderr_recovery + r_conciseness,
        4,
    )
    return {
        "R_total": r_total,
        "R_format": r_format,
        "R_json_schema": r_json_schema,
        "R_execution_pass": r_execution_pass,
        "R_stderr_recovery": r_stderr_recovery,
        "R_conciseness": r_conciseness,
        "detail": exec_detail,
    }


def compute_grpo_advantages(rewards: List[float], eps: float = 1e-6) -> List[float]:
    """
    Computes group-relative normalized advantages for a group of G candidate rollouts:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards)
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [round((r - mean_r) / (std_r + eps), 4) for r in rewards]


# ============================================================================
# 3. Sandbox Self-Test on 2 Tasks x G=4 Candidate Rollouts
# ============================================================================
def main():
    sb = Sandbox("feral_grpo_sandbox", SANDBOX_FILES)
    print("1. Initializing Deterministic GRPO Reward Sandbox (`feral_grpo_sandbox/`)...")
    print(f"   - Loaded {len(GRPO_TASKS)} verifiable RLVR tasks (Code Patching + Windows STDERR Recovery)\n")

    try:
        # Test Suite A: Code Patching Task (`grpo_01_rate_limiter_patch`) with G=4 candidate rollouts
        task_patch = GRPO_TASKS[0]
        candidates_patch = [
            (
                "Candidate 1 (Gold Surgical Patch -> Passes Unit Test)",
                '<think>\nallow_request currently returns True when tokens >= 0. I will patch `if tokens >= 0:` to `if tokens > 0:` with apply_diff_patch.\n</think>\n'
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py", "search_block": "def allow_request(tokens):\\n    if tokens >= 0:", "replace_block": "def allow_request(tokens):\\n    if tokens > 0:"}}</tool_call>',
            ),
            (
                "Candidate 2 (Valid Syntax Patch, Wrong Logic `tokens > 1` -> Fails Unit Test)",
                '<think>\nI will patch allow_request in src/rate_limiter.py so it checks tokens > 1.\n</think>\n'
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py", "search_block": "def allow_request(tokens):\\n    if tokens >= 0:", "replace_block": "def allow_request(tokens):\\n    if tokens > 1:"}}</tool_call>',
            ),
            (
                "Candidate 3 (SEARCH Block Indentation Miss -> Rejected by Sandbox)",
                '<think>\nPatch src/rate_limiter.py to check tokens > 0.\n</think>\n'
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py", "search_block": "def allow_request(tokens):\\nif tokens >= 0:", "replace_block": "def allow_request(tokens):\\nif tokens > 0:"}}</tool_call>',
            ),
            (
                "Candidate 4 (Missing <think> Tag + Malformed JSON -> Format & Schema Penalty)",
                '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py"}}</tool_call>',
            ),
        ]

        # Test Suite B: Windows STDERR Recovery Task (`grpo_06_stderr_grep_count_recovery`) with G=4 rollouts
        task_rec = GRPO_TASKS[5]
        candidates_rec = [
            (
                "Candidate 1 (Gold Native PowerShell Recovery -> Select-String .Count == 2)",
                '<think>\n`grep` is not available in Windows PowerShell. I will use `(Select-String -Path logs/app.log -Pattern \'ERROR\').Count`.\n</think>\n'
                '<tool_call>{"name": "run_powershell", "arguments": {"command": "(Select-String -Path logs/app.log -Pattern \'ERROR\').Count"}}</tool_call>',
            ),
            (
                "Candidate 2 (Runs Select-String Without .Count -> Exit 0, Wrong Output Format)",
                '<think>\nSwitch from grep to PowerShell Select-String on logs/app.log.\n</think>\n'
                '<tool_call>{"name": "run_powershell", "arguments": {"command": "Select-String -Path logs/app.log -Pattern \'ERROR\'"}}</tool_call>',
            ),
            (
                "Candidate 3 (Repeats Forbidden Linux `grep` Command -> Penalty)",
                '<think>\nTry running grep again with different flags.\n</think>\n'
                '<tool_call>{"name": "run_powershell", "arguments": {"command": "grep ERROR logs/app.log | wc -l"}}</tool_call>',
            ),
            (
                "Candidate 4 (No Tool Call Emitted -> Format & Schema Penalty)",
                "<think>\nThere are probably 2 ERROR lines in the log.\n</think>\nThere are 2 ERROR lines.",
            ),
        ]

        for suite_name, task, candidates in [
            ("SUITE A: Code Patching Verification (`grpo_01_rate_limiter_patch`)", task_patch, candidates_patch),
            ("SUITE B: Windows STDERR Recovery (`grpo_06_stderr_grep_count_recovery`)", task_rec, candidates_rec),
        ]:
            print(f"=== {suite_name} ===")
            scores = [score_rollout(task, comp, sb) for _, comp in candidates]
            rewards = [s["R_total"] for s in scores]
            advantages = compute_grpo_advantages(rewards)

            print(f"{'Candidate':48s} | {'R_fmt':>6s} | {'R_json':>6s} | {'R_exec':>6s} | {'R_rec':>6s} | {'R_conc':>6s} | {'R_total':>7s} | {'Adv A_i':>8s}")
            print("-" * 112)
            for (label, _), sc, adv in zip(candidates, scores, advantages):
                print(
                    f"{label[:48]:48s} | {sc['R_format']:+6.2f} | {sc['R_json_schema']:+6.2f} | "
                    f"{sc['R_execution_pass']:+6.2f} | {sc['R_stderr_recovery']:+6.2f} | "
                    f"{sc['R_conciseness']:+6.2f} | {sc['R_total']:+7.2f} | {adv:+8.4f}"
                )
            print(f"  Group Mean Reward (mu_r): {statistics.mean(rewards):+.4f} | Group Std (sigma_r): {statistics.pstdev(rewards):.4f}\n")

            # Verify strict ordering: Candidate 1 > Candidate 2 > Candidate 3 > Candidate 4
            if not (rewards[0] == 4.0 and rewards[0] > rewards[1] > rewards[2] > rewards[3]):
                raise AssertionError(f"Unexpected reward ordering in {suite_name}: {rewards}")
            if not (advantages[0] > 1.0 and advantages[-1] < -1.0):
                raise AssertionError(f"Unexpected GRPO advantages in {suite_name}: {advantages}")

    finally:
        sb.cleanup()

    print("SANDBOX VERIFICATION: PASS -> All 5 reward components and GRPO group advantages verified!")


if __name__ == "__main__":
    main()
