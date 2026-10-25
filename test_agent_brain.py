import json
import re
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Loading Qwen3.5-9B onto RTX 5070 Ti...")
tok = AutoTokenizer.from_pretrained(".")
model = AutoModelForCausalLM.from_pretrained(
    ".",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map="auto"
)

im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
stop_ids = [tok.eos_token_id, im_end_id]

# The ongoing conversation memory
history = """<|im_start|>system
You are an autonomous system agent on Windows. You have a tool to run PowerShell commands.
Available tool:
{"name": "run_powershell", "description": "Execute a PowerShell command", "parameters": {"command": "string"}}
To run a command, think inside <think></think> and output <tool_call>{"name": "run_powershell", "arguments": {"command": "..."}}</tool_call>.
After the tool runs, you will receive the output inside <tool_response></tool_response>. Then analyze the real numbers and give the user a clear final answer.<|im_end|>
<|im_start|>user
Check my computer's top 3 memory-hungry processes AND check how much free disk space I have on my E: drive. Run commands to find out, then summarize what you found.<|im_end|>
"""

# Allow the agent to loop up to 4 steps autonomously
for step in range(1, 5):
    print(f"\n================ AGENT STEP {step} ================")
    history += "<|im_start|>assistant\n<think>\n"
    
    inputs = tok(history, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        eos_token_id=stop_ids
    )
    
    response = tok.decode(out[0][inputs["input_ids"].shape[-1]:])
    print(response)
    history += response + "\n"
    
    # Check if the AI wants to run a tool
    match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    if match:
        tool_data = json.loads(match.group(1).strip())
        cmd = tool_data["arguments"]["command"]
        print(f"\n[RUNNING POWERSHELL]: {cmd}")
        
        result = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
        output_text = (result.stdout or result.stderr).strip()
        print(f"[SYSTEM OUTPUT]:\n{output_text}\n")
        
        # Feed the real output BACK into the AI's memory!
        history += f"<|im_start|>user\n<tool_response>\n{output_text}\n</tool_response><|im_end|>\n"
    else:
        print("\n[AGENT FINISHED TASK!]")
        break
