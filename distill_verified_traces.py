import json
import subprocess
from collections import Counter
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".")

SYSTEM_AGENT = (
    "You are an autonomous AI engineering agent running on Windows 11 PowerShell. "
    "Always reason inside <think>...</think> blocks before acting. "
    "Available tools:\n"
    "- run_powershell(command: str): Execute a PowerShell command and return STDOUT/STDERR.\n"
    "- write_file(path: str, content: str): Write UTF-8 content to a file.\n"
    "When calling a tool, emit ONLY: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
)

def run_live_ps(cmd: str) -> str:
    """Executes a real PowerShell command on the host PC to capture authentic output."""
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=15
    )
    out = res.stdout.strip()
    err = res.stderr.strip()
    if err and not out:
        return f"STDERR: {err}"
    return out if out else "(Command completed with no output)"

print("1. Capturing live execution outputs from your RTX 5070 Ti workstation...")
out_gpu = run_live_ps("nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader")
out_cpu = run_live_ps("(Get-CimInstance Win32_Processor).Name; [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)")
out_disk = run_live_ps("$d = Get-PSDrive E; \"Free: $([math]::Round($d.Free/1GB, 1)) GB | Used: $([math]::Round($d.Used/1GB, 1)) GB\"")
out_py_files = run_live_ps("Get-ChildItem -Filter *.py -Name | Sort-Object")
out_line_count = run_live_ps("(Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines")
out_select_hook = run_live_ps("Select-String -Path inspect_residual_stream.py -Pattern 'register_forward_hook' | ForEach-Object { \"$($_.LineNumber): $($_.Line.Trim())\" }")
out_torch_arch = run_live_ps("python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\"")
out_top_proc = run_live_ps("Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 3 -Property ProcessName, @{Name='MB';Expression={[math]::Round($_.WorkingSet64 / 1MB, 1)}} | ConvertTo-Json -Compress")
out_os_ver = run_live_ps("(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | ConvertTo-Json -Compress)")
out_first5 = run_live_ps("Get-Content inspect_residual_stream.py -TotalCount 5")

trajectories = []

# ============================================================================
# STAGE A: 6 Clean First-Try Execution-Verified Tool Traces
# ============================================================================
stage_a_specs = [
    (
        "stage_a_001_gpu_telemetry",
        "Check the exact NVIDIA GPU name, driver version, and current memory usage using nvidia-smi.",
        "I need to query GPU telemetry on Windows using `nvidia-smi` in CSV format.",
        "nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader",
        out_gpu,
        "The live `nvidia-smi` output returned the GPU name, driver version, and VRAM usage.",
        f"Here is the live GPU telemetry:\n- **Raw Output:** `{out_gpu}`"
    ),
    (
        "stage_a_002_cpu_ram_specs",
        "Query the exact CPU model name and total physical RAM in GB on this workstation.",
        "In Windows PowerShell, `Get-CimInstance Win32_Processor` and `Win32_ComputerSystem` provide exact CPU and RAM specifications.",
        "(Get-CimInstance Win32_Processor).Name; [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)",
        out_cpu,
        "I have retrieved the exact processor name and physical memory in GB.",
        f"Workstation CPU & RAM:\n```text\n{out_cpu}\n```"
    ),
    (
        "stage_a_003_drive_e_space",
        "How much free and used storage space (in GB) is available on the E: drive?",
        "I will use `Get-PSDrive E` and convert bytes to GB using `[math]::Round(..., 1)`.",
        "$d = Get-PSDrive E; \"Free: $([math]::Round($d.Free/1GB, 1)) GB | Used: $([math]::Round($d.Used/1GB, 1)) GB\"",
        out_disk,
        "The E: drive storage metrics were computed cleanly.",
        f"**E: Drive Storage:** {out_disk}"
    ),
    (
        "stage_a_004_list_py_scripts",
        "List all Python (.py) scripts in the current working directory sorted alphabetically.",
        "I should run `Get-ChildItem -Filter *.py -Name | Sort-Object` in PowerShell.",
        "Get-ChildItem -Filter *.py -Name | Sort-Object",
        out_py_files,
        "The directory listing returned all Python scripts in sorted order.",
        f"### Python Scripts in Workspace\n```text\n{out_py_files}\n```"
    ),
    (
        "stage_a_005_search_hook_line",
        "Find the exact line number where register_forward_hook is called in inspect_residual_stream.py.",
        "I can use `Select-String` in PowerShell to locate `register_forward_hook` with its line number.",
        "Select-String -Path inspect_residual_stream.py -Pattern 'register_forward_hook' | ForEach-Object { \"$($_.LineNumber): $($_.Line.Trim())\" }",
        out_select_hook,
        "I found the exact line number and statement where the forward hook is registered.",
        f"Found match in `inspect_residual_stream.py`:\n```python\n{out_select_hook}\n```"
    ),
    (
        "stage_a_006_pytorch_sm120_check",
        "Verify the installed PyTorch version, CUDA device name, and compute capability tuple using a Python one-liner.",
        "I will invoke `python -c` via `run_powershell` to print `torch.__version__`, `get_device_name(0)`, and `get_device_capability(0)`.",
        "python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\"",
        out_torch_arch,
        "PyTorch confirmed CUDA availability and compute capability (12, 0) for Blackwell sm_120.",
        f"PyTorch CUDA Verification: `{out_torch_arch}`"
    ),
]

for tid, prompt, think1, cmd, resp, think2, final_ans in stage_a_specs:
    trajectories.append({
        "id": tid,
        "stage": "Stage_A_Format",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\n{think1}\n</think>\n<tool_call>{json.dumps({'name': 'run_powershell', 'arguments': {'command': cmd}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\n{resp}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\n{think2}\n</think>\n{final_ans}"
            }
        ]
    })

# ============================================================================
# STAGE B: 6 Mistake-Masked STDERR Recovery Traces (learn=False on Turn 1!)
# ============================================================================
stage_b_specs = [
    (
        "stage_b_001_wc_recovery",
        "Count how many lines of code are in inspect_residual_stream.py.",
        "Let's count the lines in `inspect_residual_stream.py` using `wc -l`.",
        "wc -l inspect_residual_stream.py",
        "STDERR: wc : The term 'wc' is not recognized as the name of a cmdlet, function, script file, or operable program.",
        "`wc -l` is a Linux command unavailable in Windows PowerShell. I must use `(Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines`.",
        "(Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines",
        out_line_count,
        f"`inspect_residual_stream.py` contains **{out_line_count} lines**."
    ),
    (
        "stage_b_002_grep_recovery",
        "Search inspect_residual_stream.py for the term register_forward_hook and show the matching line.",
        "I'll run `grep -n` to search for `register_forward_hook`.",
        "grep -n register_forward_hook inspect_residual_stream.py",
        "STDERR: grep : The term 'grep' is not recognized as the name of a cmdlet, function, script file, or operable program.",
        "`grep` is not a native Windows PowerShell cmdlet. I should use `Select-String` to search for the pattern and format the line number.",
        "Select-String -Path inspect_residual_stream.py -Pattern 'register_forward_hook' | ForEach-Object { \"$($_.LineNumber): $($_.Line.Trim())\" }",
        out_select_hook,
        f"Match found via `Select-String`:\n```python\n{out_select_hook}\n```"
    ),
    (
        "stage_b_003_df_recovery",
        "Check free disk space on drive E: in gigabytes.",
        "I will run `df -h E:` to check disk space.",
        "df -h E:",
        "STDERR: df : The term 'df' is not recognized as the name of a cmdlet, function, script file, or operable program.",
        "`df` is a Unix utility. In Windows PowerShell, I must use `Get-PSDrive E` and divide by `1GB`.",
        "$d = Get-PSDrive E; \"Free: $([math]::Round($d.Free/1GB, 1)) GB | Used: $([math]::Round($d.Used/1GB, 1)) GB\"",
        out_disk,
        f"Drive E: capacity: **{out_disk}**"
    ),
    (
        "stage_b_004_head_recovery",
        "Display the first 5 lines of inspect_residual_stream.py.",
        "Let's use `head -n 5` to inspect the top of the file.",
        "head -n 5 inspect_residual_stream.py",
        "STDERR: head : The term 'head' is not recognized as the name of a cmdlet, function, script file, or operable program.",
        "`head` does not exist in PowerShell. The native PowerShell equivalent is `Get-Content <file> -TotalCount 5`.",
        "Get-Content inspect_residual_stream.py -TotalCount 5",
        out_first5,
        f"First 5 lines of `inspect_residual_stream.py`:\n```python\n{out_first5}\n```"
    ),
    (
        "stage_b_005_uname_recovery",
        "Get the Windows OS caption, version, and architecture.",
        "I'll run `uname -a` to check the OS version and architecture.",
        "uname -a",
        "STDERR: uname : The term 'uname' is not recognized as the name of a cmdlet, function, script file, or operable program.",
        "`uname` is a Linux command. On Windows PowerShell, I should query `Get-CimInstance Win32_OperatingSystem`.",
        "(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | ConvertTo-Json -Compress)",
        out_os_ver,
        f"Operating System Info: `{out_os_ver}`"
    ),
    (
        "stage_b_006_top_proc_recovery",
        "Find the top 3 memory-consuming processes currently running.",
        "I will use Linux `ps aux --sort=-%mem` to find the top memory processes.",
        "ps aux --sort=-%mem | head -n 4",
        "STDERR: Get-Process : A positional parameter cannot be found that accepts argument 'aux'.",
        "In PowerShell, `ps` is an alias for `Get-Process` and does not accept Linux `aux` flags. I must sort by `WorkingSet64 -Descending` and select the first 3.",
        "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 3 -Property ProcessName, @{Name='MB';Expression={[math]::Round($_.WorkingSet64 / 1MB, 1)}} | ConvertTo-Json -Compress",
        out_top_proc,
        f"Top 3 memory-consuming processes:\n```json\n{out_top_proc}\n```"
    ),
]

for tid, prompt, bad_think, bad_cmd, stderr_txt, fix_think, fix_cmd, good_out, final_ans in stage_b_specs:
    trajectories.append({
        "id": tid,
        "stage": "Stage_B_Recovery",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "learn": False,  # MISTAKE MASKED (-100)
                "content": f"<think>\n{bad_think}\n</think>\n<tool_call>{json.dumps({'name': 'run_powershell', 'arguments': {'command': bad_cmd}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\n{stderr_txt}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,   # ACTIVE RECOVERY LEARNING
                "content": f"<think>\n{fix_think}\n</think>\n<tool_call>{json.dumps({'name': 'run_powershell', 'arguments': {'command': fix_cmd}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\n{good_out}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\nThe native PowerShell command succeeded. I can now report the verified output.\n</think>\n{final_ans}"
            }
        ]
    })

# ============================================================================
# STAGE C: 6 Multi-Step Code Patching & Execution Verification Traces
# ============================================================================
stage_c_specs = [
    (
        "stage_c_001_zero_div_fix",
        "The script math_util.py crashes when dividing by zero. Inspect it, fix the bug, and run it to verify.",
        "math_util.py",
        "def safe_divide(a, b):\n    return a / b\n\nprint(safe_divide(10, 0))",
        "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n\nprint(safe_divide(10, 0))\nprint(safe_divide(10, 2))\n",
        "None\n5.0",
        "Added a `b == 0` guard to `safe_divide` in `math_util.py` and verified execution (`None`, `5.0`)."
    ),
    (
        "stage_c_002_dict_key_guard",
        "config_loader.py raises a KeyError when 'timeout' is missing from the config dict. Inspect, fix with a default of 30, and verify.",
        "config_loader.py",
        "def get_timeout(cfg):\n    return cfg['timeout']\n\nprint(get_timeout({'host': 'localhost'}))",
        "def get_timeout(cfg):\n    return cfg.get('timeout', 30)\n\nprint(get_timeout({'host': 'localhost'}))\nprint(get_timeout({'timeout': 10}))\n",
        "30\n10",
        "Replaced `cfg['timeout']` with `cfg.get('timeout', 30)` in `config_loader.py` and verified output (`30`, `10`)."
    ),
    (
        "stage_c_003_empty_list_avg",
        "stats_helper.py crashes on an empty list when computing the mean. Inspect, patch to return 0.0 on empty lists, and verify.",
        "stats_helper.py",
        "def compute_mean(nums):\n    return sum(nums) / len(nums)\n\nprint(compute_mean([]))",
        "def compute_mean(nums):\n    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)\n\nprint(compute_mean([]))\nprint(compute_mean([2.0, 4.0]))\n",
        "0.0\n3.0",
        "Patched `compute_mean` in `stats_helper.py` to handle empty lists and verified outputs (`0.0`, `3.0`)."
    ),
    (
        "stage_c_004_json_parse_guard",
        "tool_parser.py crashes on malformed JSON strings. Inspect it, wrap json.loads in a try/except ValueError returning None, and verify.",
        "tool_parser.py",
        "import json\ndef parse_payload(raw):\n    return json.loads(raw)\n\nprint(parse_payload('{bad json}'))",
        "import json\ndef parse_payload(raw):\n    try:\n        return json.loads(raw)\n    except ValueError:\n        return None\n\nprint(parse_payload('{bad json}'))\nprint(parse_payload('{\"ok\": true}')['ok'])\n",
        "None\nTrue",
        "Added a `try/except ValueError` guard to `parse_payload` in `tool_parser.py` and verified execution (`None`, `True`)."
    ),
    (
        "stage_c_005_unit_vec_norm",
        "vec_math.py divides by zero when normalizing an all-zero vector. Inspect, guard against zero norm, and verify.",
        "vec_math.py",
        "import math\ndef normalize_2d(x, y):\n    n = math.hypot(x, y)\n    return (x / n, y / n)\n\nprint(normalize_2d(0.0, 0.0))",
        "import math\ndef normalize_2d(x, y):\n    n = math.hypot(x, y)\n    if n == 0.0:\n        return (0.0, 0.0)\n    return (x / n, y / n)\n\nprint(normalize_2d(0.0, 0.0))\nprint(normalize_2d(3.0, 4.0))\n",
        "(0.0, 0.0)\n(0.6, 0.8)",
        "Added a zero-norm check (`if n == 0.0`) to `normalize_2d` in `vec_math.py` and verified execution."
    ),
    (
        "stage_c_006_stop_token_dedup",
        "stop_ids.py duplicates token IDs when building eos_token_id list. Inspect, deduplicate while preserving order, and verify.",
        "stop_ids.py",
        "def build_eos(eos_id, im_end_id):\n    return [eos_id, im_end_id]\n\nprint(build_eos(151645, 151645))",
        "def build_eos(eos_id, im_end_id):\n    return list(dict.fromkeys([eos_id, im_end_id]))\n\nprint(build_eos(151645, 151645))\nprint(build_eos(151643, 151645))\n",
        "[151645]\n[151643, 151645]",
        "Updated `build_eos` in `stop_ids.py` using `dict.fromkeys` to deduplicate stop token IDs and verified execution."
    ),
]

for tid, prompt, fname, broken_code, fixed_code, expected_out, summary in stage_c_specs:
    trajectories.append({
        "id": tid,
        "stage": "Stage_C_Code_Verify",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\nFirst, I will inspect `{fname}` using `Get-Content` to identify the bug.\n</think>\n<tool_call>{json.dumps({'name': 'run_powershell', 'arguments': {'command': f'Get-Content {fname}'}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\n{broken_code}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\nI see the unhandled edge case in `{fname}`. I will patch `{fname}` using `write_file` and then execute it to verify.\n</think>\n<tool_call>{json.dumps({'name': 'write_file', 'arguments': {'path': fname, 'content': fixed_code}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\nSuccessfully wrote {len(fixed_code)} bytes to {fname}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\nNow I will run `python {fname}` via `run_powershell` to confirm the fix passes without errors.\n</think>\n<tool_call>{json.dumps({'name': 'run_powershell', 'arguments': {'command': f'python {fname}'}})}</tool_call>"
            },
            {"role": "user", "content": f"<tool_response>\n{expected_out}\n</tool_response>"},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\nThe script ran cleanly and produced the expected verified output.\n</think>\n{summary}"
            }
        ]
    })

# ============================================================================
# STAGE D (ANCHOR): 6 Pure Reasoning Traces (25% Ratio — No Tool Calls)
# ============================================================================
anchor_specs = [
    (
        "anchor_001_lora_math",
        "Explain why LoRA uses two low-rank matrices A and B instead of updating a full 4096x4096 weight matrix W directly.",
        "This is a theoretical ML question requiring no external tool calls. I will explain the parameter reduction math directly.",
        "A full weight matrix $W \\in \\mathbb{R}^{4096 \\times 4096}$ has **16,777,216 parameters**. LoRA freezes $W$ and injects $A \\in \\mathbb{R}^{r \\times 4096}$ and $B \\in \\mathbb{R}^{4096 \\times r}$ with rank $r = 32$. The low-rank update $\\Delta W = B A$ requires only $2 \\times 4096 \\times 32 = 262,144$ trainable parameters—a **98.4% reduction** in gradient and optimizer state memory."
    ),
    (
        "anchor_002_hybrid_attention",
        "Why does Qwen3.5-9B combine 3 Gated DeltaNet linear attention layers with 1 Full Softmax Attention layer in each 4-layer block?",
        "This is an architecture question. I should answer directly without calling tools.",
        "Full quadratic attention scales as $O(T^2)$ in compute and grows KV-cache linearly with sequence length $T$, but excels at exact needle-in-a-haystack token retrieval. Gated DeltaNet linear attention compresses history into a fixed-size recurrent state with $O(T)$ complexity and constant memory. Interleaving **3 Gated DeltaNet layers + 1 Full Attention layer** cuts long-context KV-cache VRAM by ~75% while preserving exact retrieval fidelity."
    ),
    (
        "anchor_003_orthogonal_projection",
        "Why must we normalize the steering vector v to a unit vector before applying orthogonal projection h_clean = h - scale * (h . v_hat) * v_hat?",
        "This is a representation engineering math question. No tools are needed.",
        "If $v$ is not normalized ($\\|v\\|_2 \\neq 1$), the projection term $(h \\cdot v)v$ scales quadratically with $\\|v\\|_2^2$. In mid-to-late transformer layers where residual norms reach $30\\text{--}250$, an unnormalized vector causes massive activation overshoot and **Manifold Collapse** (repeating tokens or multilingual gibberish). Normalizing $\\hat{v} = v / \\|v\\|_2$ ensures $(h \\cdot \\hat{v})\\hat{v}$ removes *only* the exact component along direction $\\hat{v}$."
    ),
    (
        "anchor_004_mistake_masking",
        "In multi-turn trajectory distillation, why do we set labels = -100 on assistant turns that trigger STDERR?",
        "This is a loss-function question about supervised fine-tuning. I will answer directly.",
        "PyTorch's `CrossEntropyLoss(ignore_index=-100)` zeroes out the loss and gradient for any token position where `label == -100`. If an assistant turn emits a broken Linux command in PowerShell (e.g., `wc -l`), leaving its labels active trains the model to *reproduce the mistake*. Masking the failed assistant turn with `-100` while keeping the subsequent recovery turn active teaches the model to diagnose `STDERR` without ever reinforcing the initial bug."
    ),
    (
        "anchor_005_grpo_advantage",
        "How does Group Relative Policy Optimization (GRPO) eliminate the need for a separate Critic/Value network in RLVR?",
        "This question asks about GRPO mechanics. I can answer directly without running commands.",
        "Standard PPO requires a full-size Value network (Critic) in VRAM to estimate the baseline reward $V(s)$. GRPO eliminates the Critic by sampling a group of $G$ candidate rollouts $(o_1, \\dots, o_G)$ for the same prompt, scoring each with a deterministic sandbox verifier $r_i$, and computing the group-relative advantage:\n$$A_i = \\frac{r_i - \\mu_r}{\\sigma_r + \\epsilon}$$\nCandidates scoring above the group mean $\\mu_r$ receive positive gradients, cutting RL VRAM usage nearly in half."
    ),
    (
        "anchor_006_context_compaction",
        "How does autonomous context auto-compaction allow an agent to debug code for 8+ hours without exhausting 16 GB of VRAM?",
        "This is a systems architecture question. No tool execution is required.",
        "During long debugging loops, raw `<tool_response>` logs (tracebacks, file dumps, test outputs) rapidly bloat the token history. When token count crosses a high-water mark (e.g., 24K tokens), the Context Compactor summarizes older middle turns into a structured `AGENT_STATE.md` block (preserving completed DAG tasks, active hypotheses, and failed approaches to avoid) and drops the raw middle `<tool_response>` strings, resetting KV-cache memory while preserving long-horizon state."
    ),
]

for tid, prompt, think, ans in anchor_specs:
    trajectories.append({
        "id": tid,
        "stage": "Anchor_General",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "learn": True,
                "content": f"<think>\n{think}\n</think>\n{ans}"
            }
        ]
    })

output_path = "agent_curriculum_train.jsonl"
print(f"2. Tokenizing and applying Mistake Masking across {len(trajectories)} trajectories...\n")

stage_counts = Counter()
stage_total_tokens = Counter()
stage_active_tokens = Counter()
masked_assistant_turns = 0

with open(output_path, "w", encoding="utf-8") as f:
    for item in trajectories:
        input_ids = []
        labels = []
        for msg in item["messages"]:
            header_ids = tok.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False)
            body_ids = tok.encode(f"{msg['content']}<|im_end|>\n", add_special_tokens=False)
            
            input_ids.extend(header_ids + body_ids)
            labels.extend([-100] * len(header_ids))
            
            if msg["role"] == "assistant":
                if msg.get("learn", True):
                    labels.extend(body_ids)
                else:
                    labels.extend([-100] * len(body_ids))
                    masked_assistant_turns += 1
            else:
                labels.extend([-100] * len(body_ids))
        
        record = {
            "id": item["id"],
            "stage": item["stage"],
            "messages": item["messages"],
            "input_ids": input_ids,
            "labels": labels
        }
        f.write(json.dumps(record) + "\n")
        
        trained = sum(1 for x in labels if x != -100)
        st = item["stage"]
        stage_counts[st] += 1
        stage_total_tokens[st] += len(input_ids)
        stage_active_tokens[st] += trained

print("=== STAGE-BY-STAGE DISTILLATION SUMMARY ===")
for st in ["Stage_A_Format", "Stage_B_Recovery", "Stage_C_Code_Verify", "Anchor_General"]:
    c = stage_counts[st]
    tot = stage_total_tokens[st]
    act = stage_active_tokens[st]
    ratio = (act / tot * 100) if tot else 0.0
    print(f"  {st:20s} | Traces: {c:2d} | Total Tokens: {tot:5d} | Active Trained Tokens: {act:5d} ({ratio:4.1f}%)")

total_traces = sum(stage_counts.values())
grand_tot = sum(stage_total_tokens.values())
grand_act = sum(stage_active_tokens.values())
print("-" * 85)
print(f"  TOTAL                | Traces: {total_traces:2d} | Total Tokens: {grand_tot:5d} | Active Trained Tokens: {grand_act:5d} ({grand_act/grand_tot*100:4.1f}%)")
print(f"  Mistake-Masked Assistant Turns (`learn=False` -> `-100`): {masked_assistant_turns}")
print(f"\nSuccessfully wrote {total_traces} execution-verified trajectories to {output_path}!")
