import json
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

curriculum_data = [
    # --- STAGE A: Clean First-Try Tool Execution ---
    {
        "id": "stage_a_001_gpu_check",
        "stage": "Stage_A_Format",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": "Check the exact NVIDIA GPU name, driver version, and current memory usage using nvidia-smi."},
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nI need to query the GPU name, driver version, and memory usage on Windows.\nI can run `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader` via `run_powershell`.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nNVIDIA GeForce RTX 5070 Ti, 572.16, 6420 MiB, 16303 MiB\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nThe command returned the exact telemetry for the RTX 5070 Ti. I can now present the results clearly.\n</think>\n- **GPU:** NVIDIA GeForce RTX 5070 Ti\n- **Driver Version:** 572.16\n- **VRAM Usage:** 6,420 MiB / 16,303 MiB"
            }
        ]
    },

    # --- STAGE B: STDERR Recovery (WITH MISTAKE MASKING: learn=False on the failed attempt!) ---
    {
        "id": "stage_b_001_stderr_recovery",
        "stage": "Stage_B_Recovery",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": "Find all Python files in the current workspace and count how many lines of code are in inspect_residual_stream.py."},
            {
                "role": "assistant",
                "learn": False,  # MASKED! Do not train the model to emit the broken Linux command!
                "content": "<think>\nLet's list the Python files and count the lines in `inspect_residual_stream.py`.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"ls *.py; wc -l inspect_residual_stream.py\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nSTDERR: wc : The term 'wc' is not recognized as the name of a cmdlet, function, script file, or operable program.\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,   # ACTIVE! Train heavily on diagnosing STDERR and switching to native PowerShell!
                "content": "<think>\nThe previous command failed because `wc -l` is a Linux utility and does not exist in Windows PowerShell.\nI must replace `wc -l` with native PowerShell: `(Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines` and use `Get-ChildItem -Filter *.py -Name`.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"Get-ChildItem -Filter *.py -Name; (Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\ncollect_refusal_vector.py\ninspect_residual_stream.py\ntest_agent_brain.py\ntest_orthogonal_projection.py\n69\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nThe native PowerShell command succeeded. There are 4 Python scripts and `inspect_residual_stream.py` has 69 lines.\n</think>\n### Workspace Python Files\n- `collect_refusal_vector.py`\n- `inspect_residual_stream.py` (**69 lines**)\n- `test_agent_brain.py`\n- `test_orthogonal_projection.py`"
            }
        ]
    },

    # --- STAGE C: Multi-Step Code Patching & Verification ---
    {
        "id": "stage_c_001_code_patch_verify",
        "stage": "Stage_C_Code_Verify",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": "The script math_util.py is crashing when dividing by zero. Inspect it, fix the bug, and run it to verify the fix."},
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nFirst, I should read the contents of `math_util.py` to see how division is currently implemented.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"Get-Content math_util.py\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\ndef safe_divide(a, b):\n    return a / b\n\nprint(safe_divide(10, 0))\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\n`safe_divide(a, b)` directly computes `a / b` without checking if `b == 0`, causing a `ZeroDivisionError` when `safe_divide(10, 0)` runs.\nI will use `write_file` to patch `math_util.py` so it returns `None` when `b == 0`, and then run `python math_util.py` to verify.\n</think>\n<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"math_util.py\", \"content\": \"def safe_divide(a, b):\\n    if b == 0:\\n        return None\\n    return a / b\\n\\nprint(safe_divide(10, 0))\\nprint(safe_divide(10, 2))\\n\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nSuccessfully wrote 118 bytes to math_util.py\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nNow I must execute `python math_util.py` via `run_powershell` to verify the script runs cleanly with zero exceptions.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"python math_util.py\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nNone\n5.0\n</tool_response>"
            },
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nThe patched script executed cleanly and printed `None` for `10 / 0` and `5.0` for `10 / 2`.\n</think>\n`math_util.py` has been patched with a zero-divisor check (`if b == 0: return None`) and verified via execution (`None`, `5.0`)."
            }
        ]
    },

    # --- ANCHOR (25% Ratio): Direct Reasoning Without Tool Calls (Prevents Catastrophic Forgetting) ---
    {
        "id": "anchor_001_pure_reasoning",
        "stage": "Anchor_General",
        "messages": [
            {"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": "Explain why LoRA uses two low-rank matrices A and B instead of updating the full weight matrix W directly."},
            {
                "role": "assistant",
                "learn": True,
                "content": "<think>\nThis is a conceptual ML question that does not require running any external PowerShell commands or tools. I should answer directly and concisely.\n</think>\nA full weight matrix $W \\in \\mathbb{R}^{d \\times d}$ (such as $4096 \\times 4096$) contains $16.77\\text{ million}$ parameters per layer. Updating it directly requires storing full-size gradients and optimizer states in VRAM.\n\nLoRA freezes $W$ and injects two low-rank matrices $A \\in \\mathbb{R}^{r \\times d}$ and $B \\in \\mathbb{R}^{d \\times r}$ where rank $r \\ll d$ (e.g., $r = 32$). The update $\\Delta W = B A$ only trains $2 \\times 4096 \\times 32 = 262,144$ parameters—a **98.4% reduction** in trainable parameters while preserving full expressivity."
            }
        ]
    }
]

output_path = "agent_curriculum_train.jsonl"
print(f"Building {output_path} with Mistake-Masking support...\n")

with open(output_path, "w", encoding="utf-8") as f:
    for item in curriculum_data:
        input_ids = []
        labels = []
        for msg in item["messages"]:
            header_ids = tok.encode(f"<|im_start|>{msg['role']}\n", add_special_tokens=False)
            body_ids = tok.encode(f"{msg['content']}<|im_end|>\n", add_special_tokens=False)
            
            input_ids.extend(header_ids + body_ids)
            labels.extend([-100] * len(header_ids))
            
            # Only train on assistant turns where learn == True!
            if msg["role"] == "assistant" and msg.get("learn", True):
                labels.extend(body_ids)
            else:
                labels.extend([-100] * len(body_ids))
        
        record = {
            "id": item["id"],
            "stage": item["stage"],
            "input_ids": input_ids,
            "labels": labels
        }
        f.write(json.dumps(record) + "\n")
        
        trained = sum(1 for x in labels if x != -100)
        print(f"[{item['stage']:19s}] ID: {item['id']:28s} | Total Tokens: {len(input_ids):4d} | Active Trained Tokens: {trained:4d}")

print(f"\nSuccessfully wrote {len(curriculum_data)} pre-tokenized masked trajectories to {output_path}!")
