"""
Phase 2, Step 6 - Dataset Integrity & Token-Masking Audit
=========================================================
Merges every curriculum source into one dataset, audits each row, and writes a stratified
train/validation split for Phase 3 SFT.

Sources:
    agent_curriculum_train.jsonl      (Phase 2, Step 4: execution-verified PowerShell curriculum)
    feral_arsenal_trajectories.jsonl  (Phase 2, Step 5: Feral Apex Toolchain)

Outputs:
    train.jsonl, val.jsonl

Exits with code 1 if any check fails, so nothing broken can reach training.
"""
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
SOURCES = ["agent_curriculum_train.jsonl", "feral_arsenal_trajectories.jsonl"]
MAX_SEQ_LEN = 4096          # planned SFT context window
VAL_FRACTION = 0.15
SEED = 42

KNOWN_TOOLS = {
    "run_powershell": ["command"],
    "write_file": ["path", "content"],
    "read_file_lines": ["path", "start_line", "end_line"],
    "apply_diff_patch": ["path", "search_block", "replace_block"],
    "ast_repo_map": ["directory"],
    "web_research": ["query"],
    "spawn_subagent": ["task_prompt"],
}

tok = AutoTokenizer.from_pretrained(str(ROOT))
IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")


def retokenize(messages):
    """Rebuild input_ids/labels from messages exactly as the generators do."""
    ids, labels = [], []
    for msg in messages:
        header = tok.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False)
        body = tok.encode(f"{msg['content']}<|im_end|>\n", add_special_tokens=False)
        ids += header + body
        labels += [-100] * len(header)
        trained = msg["role"] == "assistant" and msg.get("learn", True)
        labels += body if trained else [-100] * len(body)
    return ids, labels


def check_record(r):
    errs = []
    tid = r["id"]
    msgs = r["messages"]

    # 1. Structure: system -> user -> ... -> assistant
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[-1]["role"] != "assistant":
        errs.append("bad turn structure (must be system, user, ..., assistant)")
    if not msgs[-1].get("learn", True):
        errs.append("final assistant turn is masked (model would never learn to finish)")

    for i, m in enumerate(msgs):
        c = m["content"]
        if m["role"] == "assistant":
            # 2. Balanced <think>
            if c.count("<think>") != 1 or c.count("</think>") != 1 or c.index("<think>") > c.index("</think>"):
                errs.append(f"turn {i}: unbalanced <think> tags")
            # 3. Valid tool calls
            for raw in re.findall(r"<tool_call>(.*?)</tool_call>", c, flags=re.S):
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError as e:
                    errs.append(f"turn {i}: invalid tool_call JSON ({e})")
                    continue
                name = call.get("name")
                if name not in KNOWN_TOOLS:
                    errs.append(f"turn {i}: unknown tool {name!r}")
                    continue
                missing = [k for k in KNOWN_TOOLS[name] if k not in call.get("arguments", {})]
                if missing:
                    errs.append(f"turn {i}: {name} missing args {missing}")
            if c.count("<tool_call>") != c.count("</tool_call>"):
                errs.append(f"turn {i}: unbalanced <tool_call> tags")
            # 4. Every tool call must be answered by a tool_response in the next turn
            if "<tool_call>" in c and (i + 1 >= len(msgs) or "<tool_response>" not in msgs[i + 1]["content"]):
                errs.append(f"turn {i}: tool_call without a following tool_response")
        elif "<tool_response>" in c and c.count("<tool_response>") != c.count("</tool_response>"):
            errs.append(f"turn {i}: unbalanced <tool_response> tags")
        if "System.Management.Automation" in c:
            errs.append(f"turn {i}: PowerShell wrapper noise in content")

    # 5. Stored tokens must exactly match a fresh re-tokenization (catches stale or hand-edited rows)
    ids, labels = retokenize(msgs)
    if ids != r["input_ids"] or labels != r["labels"]:
        errs.append("stored input_ids/labels do not match fresh tokenization")
    if len(r["input_ids"]) != len(r["labels"]):
        errs.append("input_ids/labels length mismatch")
    if len(r["input_ids"]) > MAX_SEQ_LEN:
        errs.append(f"sequence {len(r['input_ids'])} > MAX_SEQ_LEN {MAX_SEQ_LEN}")

    # 6. Positional leak check: map every token position to the turn that owns it. Trained positions
    #    must lie ONLY inside learn=True assistant bodies, and those bodies must be fully trained.
    #    (A text search would false-positive on answers that merely *mention* "<tool_response>".)
    pos = 0
    for i, m in enumerate(msgs):
        h = len(tok.encode(f"<|im_start|>{m['role']}\n", add_special_tokens=False))
        b = len(tok.encode(f"{m['content']}<|im_end|>\n", add_special_tokens=False))
        head_labels = r["labels"][pos:pos + h]
        body_labels = r["labels"][pos + h:pos + h + b]
        if any(l != -100 for l in head_labels):
            errs.append(f"turn {i}: role header tokens are trained")
        should_train = m["role"] == "assistant" and m.get("learn", True)
        if should_train and any(l == -100 for l in body_labels):
            errs.append(f"turn {i}: learn=True assistant body is partially masked")
        if not should_train and any(l != -100 for l in body_labels):
            errs.append(f"turn {i}: {m['role']} (learn={m.get('learn', True)}) tokens leaked into trained labels")
        pos += h + b
    if pos != len(r["labels"]):
        errs.append("token positions do not cover the full sequence")

    # 7. Stop token: the model must be trained to emit <|im_end|> once per learned assistant turn
    learned_turns = sum(1 for m in msgs if m["role"] == "assistant" and m.get("learn", True))
    trained_im_end = sum(1 for lab in r["labels"] if lab == IM_END_ID)
    if trained_im_end != learned_turns:
        errs.append(f"trained <|im_end|> count {trained_im_end} != learned assistant turns {learned_turns}")

    return [f"{tid}: {e}" for e in errs]


def main():
    print("1. Loading and merging sources...")
    records = []
    for src in SOURCES:
        path = ROOT / src
        if not path.exists():
            print(f"   MISSING: {src}")
            sys.exit(1)
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        for row in rows:
            row["source"] = src
        records += rows
        print(f"   {src:34s}: {len(rows):2d} rows")
    print(f"   -> {len(records)} merged rows\n")

    print("2. Auditing every row...")
    errors = []
    ids_seen = Counter(r["id"] for r in records)
    errors += [f"duplicate id: {k} (x{v})" for k, v in ids_seen.items() if v > 1]
    hashes = Counter(hashlib.sha1(json.dumps(r["input_ids"]).encode()).hexdigest() for r in records)
    errors += [f"{v} rows share identical token content" for v in hashes.values() if v > 1]
    for r in records:
        errors += check_record(r)

    checks = [
        "turn structure + final assistant turn is trained",
        "balanced <think> / <tool_call> / <tool_response> tags",
        "every <tool_call> is valid JSON with a known tool and required args",
        "every <tool_call> is answered by a <tool_response>",
        "stored tokens == fresh re-tokenization",
        "positional mask: only learn=True assistant bodies trained (system/user/tool/mistake turns = -100)",
        "<|im_end|> stop token trained once per learned assistant turn",
        f"no sequence longer than {MAX_SEQ_LEN} tokens",
        "no duplicate ids or duplicate token content",
    ]
    if errors:
        print("AUDIT: FAIL")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    for c in checks:
        print(f"  [PASS] {c}")

    print("\n3. Dataset statistics")
    lengths = sorted(len(r["input_ids"]) for r in records)
    active = [sum(l != -100 for l in r["labels"]) for r in records]
    p90 = lengths[int(0.9 * (len(lengths) - 1))]
    print(f"  Sequence length  min {lengths[0]} | median {int(statistics.median(lengths))} | p90 {p90} | max {lengths[-1]}")
    print(f"  Total tokens {sum(lengths)} | trained tokens {sum(active)} ({sum(active) / sum(lengths) * 100:.1f}%)")
    masked = sum(1 for r in records for m in r["messages"] if m["role"] == "assistant" and not m.get("learn", True))
    print(f"  Mistake-masked assistant turns: {masked}")
    sys_prompts = Counter(r["messages"][0]["content"][:60] for r in records)
    print(f"  Distinct system prompts: {len(sys_prompts)}")
    for p, n in sys_prompts.items():
        print(f"    {n:2d} rows | {p}...")

    print("\n4. Stratified train/val split (seed={}, val={:.0%} per stage, stages with <3 rows stay in train)".format(SEED, VAL_FRACTION))
    by_stage = defaultdict(list)
    for r in records:
        by_stage[r["stage"]].append(r)
    rng = random.Random(SEED)
    train, val = [], []
    for stage in sorted(by_stage):
        rows = sorted(by_stage[stage], key=lambda x: x["id"])
        rng.shuffle(rows)
        k = max(1, round(len(rows) * VAL_FRACTION)) if len(rows) >= 3 else 0
        val += rows[:k]
        train += rows[k:]
        print(f"  {stage:20s} | total {len(rows):2d} | train {len(rows) - k:2d} | val {k}")

    # Leakage check: no validation prompt may also appear as a training prompt
    train_prompts = {r["messages"][1]["content"] for r in train}
    leaked = [r["id"] for r in val if r["messages"][1]["content"] in train_prompts]
    if leaked:
        print(f"SPLIT: FAIL - val prompts also in train: {leaked}")
        sys.exit(1)
    print(f"  [PASS] no val prompt appears in train")
    print(f"  Val ids: {', '.join(sorted(r['id'] for r in val))}")

    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(ROOT / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"\nAUDIT: PASS -> train.jsonl ({len(train)} rows) + val.jsonl ({len(val)} rows)")


if __name__ == "__main__":
    main()
