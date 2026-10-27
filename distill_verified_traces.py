"""
Phase 2, Step 4 - Automated Execution-Verified Distillation
===========================================================
Builds `agent_curriculum_train.jsonl` using the unified Feral system prompt (`SYSTEM_FERAL`)
and real execution inside throwaway sandboxes (`feral_core.Sandbox`).

Scenario files you can edit to add your own tasks:
  - scenarios/powershell_tasks.py  (PS_TASKS + CREATE_TASKS)
  - scenarios/code_fix_tasks.py    (CODE_FIX_TASKS + ANCHOR_TASKS)
"""
import os
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer

from feral_core import (
    ROOT,
    Sandbox,
    SkipTrace,
    Trace,
    anchor_trace,
    to_record,
    tool_output,
    write_and_summarize,
)
from scenarios.code_fix_tasks import ANCHOR_TASKS, CODE_FIX_TASKS
from scenarios.powershell_tasks import CREATE_TASKS, PS_TASKS, WORKSPACE_FILES

OUTPUT = ROOT / "agent_curriculum_train.jsonl"
STAGE_ORDER = ["Stage_A_Format", "Stage_B_Recovery", "Stage_C_Code_Verify", "Anchor_General"]


def build_all_traces(sb: Sandbox):
    traces = []

    # 1. Stage A: Clean first-try native PowerShell execution
    for spec in PS_TASKS:
        sb.reset()
        t = Trace(sb, f"stage_a_{spec['id']}", "Stage_A_Format", spec["prompt"], group=f"ps_{spec['id']}")
        obs = t.call(spec["think"], "run_powershell", {"command": spec["command"]}, expect="[exit_code=0]")
        out = tool_output(obs)
        traces.append(t.final(
            "The command succeeded with exit_code=0. I can report the exact output now.",
            spec["answer"].format(out=out),
        ))

    # 2. Stage B: Mistake-masked Linux-in-PowerShell failure -> native PowerShell recovery
    for spec in PS_TASKS:
        if "linux" not in spec:
            continue
        sb.reset()
        t = Trace(sb, f"stage_b_{spec['id']}", "Stage_B_Recovery", spec["prompt_b"], group=f"ps_{spec['id']}")
        try:
            t.call(
                f"Run `{spec['linux']}` to answer the request.",
                "run_powershell",
                {"command": spec["linux"]},
                learn=False,       # MISTAKE MASKED (-100)
                must_fail=True,
            )
        except SkipTrace as exc:
            print(f"  [skip] {exc}")
            continue
        obs = t.call(spec["fix_think"], "run_powershell", {"command": spec["command"]}, learn=True, expect="[exit_code=0]")
        out = tool_output(obs)
        traces.append(t.final(
            "The native PowerShell command succeeded cleanly.",
            spec["answer"].format(out=out),
        ))

    # 3. Stage C (Part 1): Inspect -> Surgical Diff Patch -> Execute Verification
    for spec in CODE_FIX_TASKS:
        sb.reset(extra_files={spec["file"]: spec["buggy"]})
        t = Trace(sb, f"stage_c_fix_{spec['id']}", "Stage_C_Code_Verify", spec["prompt"], group=f"fix_{spec['id']}")
        t.call(spec["think_read"], "read_file_lines", {"path": spec["file"], "start_line": 1, "end_line": 20})
        t.call(
            spec["think_patch"],
            "apply_diff_patch",
            {"path": spec["file"], "search_block": spec["search"], "replace_block": spec["replace"]},
            expect="OK: Patched",
        )
        obs = t.call(
            f"Run `{spec['verify_cmd']}` to verify the fix works.",
            "run_powershell",
            {"command": spec["verify_cmd"]},
            expect=spec["expect"],
        )
        traces.append(t.final(
            "The patched script executed with exit_code=0 and produced the expected output.",
            spec["answer"].format(out=tool_output(obs)),
        ))

    # 4. Stage C (Part 2): Create script with write_file -> Execute -> (Optional mistake-masked fix)
    for spec in CREATE_TASKS:
        sb.reset()
        t = Trace(sb, f"stage_c_create_{spec['id']}", "Stage_C_Code_Verify", spec["prompt"], group=f"create_{spec['id']}")
        if "buggy_content" in spec:
            # First write contains a realistic bug (masked so we don't train the bug)
            t.call(
                f"Write `{spec['path']}` using `write_file`.",
                "write_file",
                {"path": spec["path"], "content": spec["buggy_content"]},
                learn=False,
                expect="OK: Wrote",
            )
            t.call(
                f"Run `{spec['run']}` to test the new script.",
                "run_powershell",
                {"command": spec["run"]},
                learn=False,
                must_fail=True,
            )
            t.call(
                spec["fix_think"],
                "apply_diff_patch",
                {"path": spec["path"], "search_block": spec["search"], "replace_block": spec["replace"]},
                learn=True,
                expect="OK: Patched",
            )
            obs = t.call(
                f"Re-run `{spec['run']}` to confirm the fix.",
                "run_powershell",
                {"command": spec["run"]},
                learn=True,
                expect="[exit_code=0]",
            )
        else:
            t.call(
                f"Create `{spec['path']}` with `write_file`.",
                "write_file",
                {"path": spec["path"], "content": spec["content"]},
                expect="OK: Wrote",
            )
            obs = t.call(
                f"Run `{spec['run']}` to verify the script output.",
                "run_powershell",
                {"command": spec["run"]},
                expect="[exit_code=0]",
            )
        traces.append(t.final(
            "The script ran cleanly and printed the verified result.",
            spec["answer"].format(out=tool_output(obs)),
        ))

    # 5. Anchor_General: Pure conceptual reasoning without tool calls (first 14 anchors here; rest in Step 5)
    for tid, prompt, think, ans in ANCHOR_TASKS[:14]:
        traces.append(anchor_trace(tid, prompt, think, ans))

    return traces


def main():
    tok = AutoTokenizer.from_pretrained(str(ROOT))
    sb = Sandbox("feral_ps_sandbox", WORKSPACE_FILES)
    print("1. Executing live PowerShell & code-verification trajectories inside feral_ps_sandbox/...")
    try:
        traces = build_all_traces(sb)
    finally:
        sb.cleanup()

    print(f"2. Tokenizing {len(traces)} trajectories with completion-only + mistake masking...")
    records = []
    masked_total = 0
    for tr in traces:
        rec, masked = to_record(tr, tok)
        records.append(rec)
        masked_total += masked

    write_and_summarize(records, masked_total, OUTPUT, STAGE_ORDER)


if __name__ == "__main__":
    main()
