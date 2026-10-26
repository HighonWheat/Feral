import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".")

# Gold-Standard Multi-Turn Agentic Trajectories (ChatML + <think> + <tool_call> + Error Recovery)
trajectories = [
    {
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
                "content": "<think>\nThe user wants a list of all Python files in the current directory and the line count of inspect_residual_stream.py.\nLet's try running ls *.py and wc -l inspect_residual_stream.py first.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"ls *.py; wc -l inspect_residual_stream.py\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nSTDERR: wc : The term 'wc' is not recognized as the name of a cmdlet, function, script file, or operable program.\nAt line:1 char:10\n+ ls *.py; wc -l inspect_residual_stream.py\n+          ~~\n</tool_response>"
            },
            {
                "role": "assistant",
                "content": "<think>\nThe command failed because wc -l is a Linux bash utility, not a native Windows PowerShell cmdlet.\nIn PowerShell, I should use Get-ChildItem -Filter *.py to list the files and (Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines to count the lines.\nLet's execute the corrected PowerShell command now.\n</think>\n<tool_call>{\"name\": \"run_powershell\", \"arguments\": {\"command\": \"Get-ChildItem -Filter *.py | Select-Object Name; (Get-Content inspect_residual_stream.py | Measure-Object -Line).Lines\"}}</tool_call>"
            },
            {
                "role": "user",
                "content": "<tool_response>\nName\n----\ncollect_refusal_vector.py\ninspect_residual_stream.py\ntest_agent_brain.py\ntest_orthogonal_projection.py\n69\n</tool_response>"
            },
            {
                "role": "assistant",
                "content": "<think>\nThe corrected PowerShell command succeeded. There are 4 Python files in the workspace (collect_refusal_vector.py, inspect_residual_stream.py, 	est_agent_brain.py, 	est_orthogonal_projection.py), and inspect_residual_stream.py contains exactly 69 lines of code. I can now report the final result.\n</think>\n### Workspace Python Files\n- collect_refusal_vector.py\n- inspect_residual_stream.py (**69 lines**)\n- 	est_agent_brain.py\n- 	est_orthogonal_projection.py"
            }
        ]
    }
]

# Format into raw ChatML training text and save to JSONL
output_file = "agent_traces_prototype.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in trajectories:
        raw_text = ""
        for msg in item["messages"]:
            raw_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        record = {
            "id": item["id"],
            "category": item["category"],
            "text": raw_text
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(trajectories)} multi-turn trajectory to {output_file}!")

# Verify tokenization and token count
tokens = tok.encode(raw_text)
print(f"Total tokens in trajectory: {len(tokens)}")
print("\n--- EXACT RAW TRAINING SEQUENCE PREVIEW ---")
print(raw_text)
