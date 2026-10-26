import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".")

# 1. Regenerate clean trajectory (using @' '@ so PowerShell never eats backticks!)
trajectory = {
    "id": "trace_001_stage_b_error_recovery",
    "category": "stage_b_stderr_recovery",
    "messages": [
        {
            "role": "system",
            "content": "You are an autonomous AI engineering agent running on Windows 11 PowerShell. Always reason inside <think>...</think> blocks before acting. When you need to execute a command, emit <tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"...\"}}</tool_call>."
        },
        {
            "role": "user",
            "content": "Find all Python files in the current workspace and count how many lines of code are in inspect_residual_stream.py."
        },
        {
            "role": "assistant",
            "content": "<think>\nThe user wants a list of all Python files in the current directory and the line count of `inspect_residual_stream.py`.\nLet's try running `ls *.py` and `wc -l inspect_residual_stream.py` first.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"ls *.py; wc -l inspect_residual_stream.py\"}}</tool_call>"
        },
        {
            "role": "user",
            "content": "<tool_response>\nSTDERR: wc : The term 'wc' is not recognized as the name of a cmdlet, function, script file, or operable program.\nAt line:1 char:10\n+ ls *.py; wc -l inspect_residual_stream.py\n+          ~~\n</tool_response>"
        },
        {
            "role": "assistant",
            "content": "<think>\nThe command failed because `wc -l` is a Linux bash utility, not a native Windows PowerShell cmdlet.\nIn PowerShell, I should use `Get-ChildItem -Filter *.py` to list the files and `(Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines` to count the lines.\nLet's execute the corrected PowerShell command now.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"Get-ChildItem -Filter *.py | Select-Object Name; (Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines\"}}</tool_call>"
        },
        {
            "role": "user",
            "content": "<tool_response>\nName\n----\ncollect_refusal_vector.py\ninspect_residual_stream.py\ntest_agent_brain.py\ntest_orthogonal_projection.py\n69\n</tool_response>"
        },
        {
            "role": "assistant",
            "content": "<think>\nThe corrected PowerShell command succeeded. There are 4 Python files in the workspace (`collect_refusal_vector.py`, `inspect_residual_stream.py`, `test_agent_brain.py`, `test_orthogonal_projection.py`), and `inspect_residual_stream.py` contains exactly 69 lines of code. I can now report the final result.\n</think>\n### Workspace Python Files\n- `collect_refusal_vector.py`\n- `inspect_residual_stream.py` (**69 lines**)\n- `test_agent_brain.py`\n- `test_orthogonal_projection.py`"
        }
    ]
}

# 2. Build input_ids and labels token-by-token per message turn
input_ids = []
labels = []

for msg in trajectory["messages"]:
    role_header = f"<|im_start|>{msg['role']}\n"
    body_and_end = f"{msg['content']}<|im_end|>\n"
    
    header_ids = tok.encode(role_header, add_special_tokens=False)
    body_ids = tok.encode(body_and_end, add_special_tokens=False)
    
    input_ids.extend(header_ids)
    input_ids.extend(body_ids)
    
    # We NEVER train the model to predict the role header (<|im_start|>assistant\n)
    labels.extend([-100] * len(header_ids))
    
    if msg["role"] == "assistant":
        # Train on 100% of the assistant's <think>, <tool_call>, and final answer + <|im_end|>!
        labels.extend(body_ids)
    else:
        # Mask out system prompts, user prompts, and <tool_response> outputs with -100!
        labels.extend([-100] * len(body_ids))

total_tokens = len(input_ids)
trained_tokens = sum(1 for x in labels if x != -100)
masked_tokens = total_tokens - trained_tokens

print(f"Total Trajectory Tokens : {total_tokens}")
print(f"Masked Tokens (-100)    : {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%) -> Ignored by Loss")
print(f"Active Training Tokens  : {trained_tokens} ({trained_tokens/total_tokens*100:.1f}%) -> Updated by Gradients")

# 3. Verify that ONLY assistant thoughts/actions are active in labels
active_decoded = tok.decode([t for t in labels if t != -100])
print("\n--- DECODED ACTIVE TRAINING TARGETS (WHAT THE MODEL ACTUALLY LEARNS) ---")
print(active_decoded)
