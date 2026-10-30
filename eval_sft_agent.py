"""
Phase 3, Step 4 - Adapter Benchmark: Base vs. Fine-Tuned Feral Agent (`eval_sft_agent.py`)
==========================================================================================
Loads `Qwen3.5-9B-Base` (4-bit NF4) alongside the trained Epoch-2 DoRA checkpoint from
`lora_model/feral_sft_dora` and runs a deterministic side-by-side benchmark across 4 unseen
engineering challenges:

  1. Unseen Windows PowerShell STDERR Recovery (`du -b` -> native `Get-ChildItem | Measure-Object`)
  2. Unseen Surgical Code Patching (`read_file_lines` context -> `apply_diff_patch` with exact SEARCH/REPLACE)
  3. Unseen AST Repository Mapping (`ast_repo_map` tool selection)
  4. Unseen Conceptual Anchor Check (must reason in `<think>` and answer directly with ZERO `<tool_call>`)

Uses `with model.disable_adapter():` to evaluate the untouched Base model on the exact same
weights and prompts in a single GPU load.
"""
import json
import math
import os
import re
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from feral_core import ROOT, SYSTEM_FERAL, TOOL_SCHEMA

hf_logging.set_verbosity_error()

CHECKPOINT_DIR = ROOT / "lora_model" / "feral_sft_dora"

BENCHMARK_SUITES = [
    {
        "id": "bench_01_stderr_recovery",
        "title": "Unseen Windows STDERR Recovery (`du` -> Native PowerShell)",
        "expect_tool": "run_powershell",
        "expect_substrings": ["Get-ChildItem", "Measure-Object"],
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {"role": "user", "content": "Calculate the total size in bytes of all .log files inside the logs/ folder."},
            {
                "role": "assistant",
                "content": "<think>\nRun `du -b logs/*.log` to get the byte sizes.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"du -b logs/*.log\"}}</tool_call>",
            },
            {
                "role": "user",
                "content": "<tool_response>\nERROR: The term 'du' is not recognized as the name of a cmdlet, function, script file, or operable program.\n[exit_code=1]\n</tool_response>",
            },
        ],
    },
    {
        "id": "bench_02_surgical_diff_patch",
        "title": "Unseen Surgical Code Patch (`apply_diff_patch` after `read_file_lines`)",
        "expect_tool": "apply_diff_patch",
        "expect_substrings": ["src/rate_limiter.py", "tokens > 0"],
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "src/rate_limiter.py allows requests even when tokens == 0. Read it and patch allow_request so it requires tokens > 0.",
            },
            {
                "role": "assistant",
                "content": "<think>\nRead lines 1-10 of src/rate_limiter.py first to inspect allow_request.\n</think>\n<tool_call>{\"name\": \"read_file_lines\", \"arguments\": {\"path\": \"src/rate_limiter.py\", \"start_line\": 1, \"end_line\": 10}}</tool_call>",
            },
            {
                "role": "user",
                "content": "<tool_response>\nsrc/rate_limiter.py (lines 1-5 of 5)\n   1 | def allow_request(tokens):\n   2 |     if tokens >= 0:\n   3 |         return True\n   4 |     return False\n   5 | \n</tool_response>",
            },
        ],
    },
    {
        "id": "bench_03_ast_repo_map",
        "title": "Unseen Repository Structural Mapping (`ast_repo_map`)",
        "expect_tool": "ast_repo_map",
        "expect_substrings": ["auth"],
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "Give me a structural map of all Python classes and functions inside the auth/ directory without reading full files.",
            },
        ],
    },
    {
        "id": "bench_04_anchor_no_tool_tourettes",
        "title": "Unseen Conceptual Reasoning Anchor (Zero Tool-Call Tourette's Check)",
        "expect_tool": None,  # Must NOT emit a <tool_call>!
        "expect_substrings": ["activation", "recomput"],
        "messages": [
            {"role": "system", "content": SYSTEM_FERAL},
            {
                "role": "user",
                "content": "Explain why activation/gradient checkpointing saves VRAM during backpropagation and what the compute tradeoff is.",
            },
        ],
    },
]


def format_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def grade_response(text: str, expect_tool: str | None, expect_substrings: list[str]) -> dict:
    has_think = (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.index("<think>") < text.index("</think>")
    )
    tool_matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S)

    if expect_tool is None:
        # Anchor task: must NOT call any tool
        tool_ok = len(tool_matches) == 0 and "<tool_call>" not in text
        parsed_tool = "NONE" if tool_ok else f"UNEXPECTED({len(tool_matches)})"
    else:
        tool_ok = False
        parsed_tool = "NONE"
        if len(tool_matches) == 1:
            try:
                payload = json.loads(tool_matches[0])
                name = payload.get("name")
                args = payload.get("arguments", {})
                parsed_tool = str(name)
                req_keys = TOOL_SCHEMA.get(name, [])
                if name == expect_tool and all(k in args for k in req_keys):
                    tool_ok = True
            except Exception:
                parsed_tool = "INVALID_JSON"

    lower_text = text.lower()
    subs_ok = all(s.lower() in lower_text for s in expect_substrings)
    passed = has_think and tool_ok and subs_ok
    return {
        "passed": passed,
        "has_think": has_think,
        "tool_ok": tool_ok,
        "parsed_tool": parsed_tool,
        "subs_ok": subs_ok,
    }


@torch.no_grad()
def generate_turn(model, tok, prompt: str, max_new_tokens: int = 220) -> str:
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list(dict.fromkeys([tok.eos_token_id, im_end_id]))
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0, input_len:]
    # Strip trailing <|im_end|> or eos token when decoding
    if len(gen_ids) > 0 and int(gen_ids[-1].item()) in eos_ids:
        gen_ids = gen_ids[:-1]
    return tok.decode(gen_ids, skip_special_tokens=False).strip()


def preview_one_line(text: str, width: int = 110) -> str:
    clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return clean if len(clean) <= width else clean[:width - 3] + "..."


def main():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Trained DoRA checkpoint not found at {CHECKPOINT_DIR}")

    print("1. Loading 4-bit NF4 Qwen3.5-9B-Base + Trained Epoch-2 Feral DoRA Adapter...")
    tok = AutoTokenizer.from_pretrained(str(ROOT))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(ROOT),
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    model.eval()

    print(f"   - Loaded DoRA checkpoint from: {CHECKPOINT_DIR.relative_to(ROOT)}\n")
    print("2. Running Side-by-Side Unseen Engineering Benchmark (Untouched Base vs. Feral DoRA)...")
    print("=" * 100)

    base_score = 0
    feral_score = 0
    results = []

    for item in BENCHMARK_SUITES:
        prompt = format_prompt(item["messages"])

        # A) Untouched Base Model (DoRA adapters disabled)
        with model.disable_adapter():
            base_out = generate_turn(model, tok, prompt)
        base_grade = grade_response(base_out, item["expect_tool"], item["expect_substrings"])
        base_score += int(base_grade["passed"])

        # B) Fine-Tuned Feral DoRA Agent (DoRA adapters active)
        feral_out = generate_turn(model, tok, prompt)
        feral_grade = grade_response(feral_out, item["expect_tool"], item["expect_substrings"])
        feral_score += int(feral_grade["passed"])

        print(f"\n[{item['id']}] {item['title']}")
        print("-" * 100)
        print(
            f"  BASE MODEL  -> [{'PASS' if base_grade['passed'] else 'FAIL'}] "
            f"<think>={base_grade['has_think']} | tool={base_grade['parsed_tool']} (ok={base_grade['tool_ok']}) | content_ok={base_grade['subs_ok']}"
        )
        print(f"    Output: {preview_one_line(base_out)}")
        print(
            f"  FERAL DoRA  -> [{'PASS' if feral_grade['passed'] else 'FAIL'}] "
            f"<think>={feral_grade['has_think']} | tool={feral_grade['parsed_tool']} (ok={feral_grade['tool_ok']}) | content_ok={feral_grade['subs_ok']}"
        )
        print(f"    Output: {preview_one_line(feral_out)}")

        results.append({
            "id": item["id"],
            "base": {"grade": base_grade, "output": base_out},
            "feral": {"grade": feral_grade, "output": feral_out},
        })

    n_total = len(BENCHMARK_SUITES)
    print("\n" + "=" * 100)
    print("=== PHASE 3 FINAL BENCHMARK SCORECARD ===")
    print(f"  Untouched Base Model (`Qwen3.5-9B-Base`) : {base_score}/{n_total} ({base_score / n_total * 100:.0f}%)")
    print(f"  Fine-Tuned Feral DoRA (`feral_sft_dora`) : {feral_score}/{n_total} ({feral_score / n_total * 100:.0f}%)")
    print("=" * 100)

    if feral_score != n_total:
        raise RuntimeError(f"Expected Feral DoRA to pass all {n_total} benchmark suites, got {feral_score}/{n_total}!")
    print("\nBENCHMARK: PASS -> 100% <think> + <tool_call> compliance, surgical patching, and STDERR recovery verified!")


if __name__ == "__main__":
    main()
