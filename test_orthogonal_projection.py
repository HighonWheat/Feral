import os
import sys
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

# 1. Ensure UTF-8 output on Windows to prevent UnicodeEncodeError crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model_path = "."
print("Loading tokenizer and 4-bit Qwen3.5-9B...")

tok = AutoTokenizer.from_pretrained(model_path)
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
stop_ids = [tok.eos_token_id, im_end_id, im_start_id]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="cuda:0"
)
model.eval()

# 2. Balanced Contrast Pairs across Shell, Coding, and Factual Q&A
chatty_prompts = [
    "<|im_start|>system\nAlways begin your response with a long, conversational explanation before giving any code.<|im_end|>\n<|im_start|>user\nCheck the current directory.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nAlways begin your response with a long, conversational explanation before giving any code.<|im_end|>\n<|im_start|>user\nList running processes.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nAlways begin your response with friendly greetings and extensive polite introductory filler.<|im_end|>\n<|im_start|>user\nWrite a Python function to reverse a string.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nAlways begin your response with friendly greetings and extensive polite introductory filler.<|im_end|>\n<|im_start|>user\nHow do I sort a list in Python?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nAlways include conversational pleasantries and long introductory background.<|im_end|>\n<|im_start|>user\nWhat is the speed of light in meters per second?<|im_end|>\n<|im_start|>assistant\n",
]

direct_prompts = [
    "<|im_start|>system\nOutput ONLY a powershell codeblock with zero introductory text.<|im_end|>\n<|im_start|>user\nCheck the current directory.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nOutput ONLY a powershell codeblock with zero introductory text.<|im_end|>\n<|im_start|>user\nList running processes.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nOutput ONLY the raw code solution with zero conversational filler or greetings.<|im_end|>\n<|im_start|>user\nWrite a Python function to reverse a string.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nOutput ONLY the raw code solution with zero conversational filler or greetings.<|im_end|>\n<|im_start|>user\nHow do I sort a list in Python?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nProvide only the direct factual answer with zero pleasantries or filler.<|im_end|>\n<|im_start|>user\nWhat is the speed of light in meters per second?<|im_end|>\n<|im_start|>assistant\n",
]

layers = model.model.layers
# Target middle "semantic style" sweet-spot layers (14 to 19)
sweet_spot_layers = list(range(14, 20))

captured_chatty = {i: [] for i in sweet_spot_layers}
captured_direct = {i: [] for i in sweet_spot_layers}

def make_collector(storage_dict, layer_idx):
    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        storage_dict[layer_idx].append(hidden[0, -1, :].detach().float().cpu())
    return hook_fn

print(f"\nExtracting unit direction vectors across Sweet-Spot Layers {sweet_spot_layers}...")

for p in chatty_prompts:
    hooks = [layers[i].register_forward_hook(make_collector(captured_chatty, i)) for i in sweet_spot_layers]
    with torch.no_grad():
        model(**tok(p, return_tensors="pt").to("cuda:0"))
    for h in hooks:
        h.remove()

for p in direct_prompts:
    hooks = [layers[i].register_forward_hook(make_collector(captured_direct, i)) for i in sweet_spot_layers]
    with torch.no_grad():
        model(**tok(p, return_tensors="pt").to("cuda:0"))
    for h in hooks:
        h.remove()

# Compute UNIT vectors pointing in the "Chatty" direction (chatty - direct)
unit_chatty_vectors = {}
for i in sweet_spot_layers:
    mean_chatty = torch.stack(captured_chatty[i]).mean(dim=0)
    mean_direct = torch.stack(captured_direct[i]).mean(dim=0)
    v_chatty = mean_chatty - mean_direct
    v_unit = (v_chatty / torch.norm(v_chatty)).to(device="cuda:0", dtype=torch.bfloat16)
    unit_chatty_vectors[i] = v_unit

# Pure Orthogonal Projection Hook: h_clean = h - scale * (h . v_unit) * v_unit
def make_clean_projection_hook(layer_idx, scale=1.0):
    v_unit = unit_chatty_vectors[layer_idx]
    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        proj_magnitude = torch.matmul(hidden, v_unit).unsqueeze(-1)
        hidden_clean = hidden - (scale * proj_magnitude * v_unit)
        if isinstance(output, tuple):
            return (hidden_clean,) + output[1:]
        return hidden_clean
    return hook_fn

test_queries = [
    "Give me the PowerShell command to check my IP address.",
    "Write a Python function to check if a word is a palindrome.",
]

for query in test_queries:
    test_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tok(test_prompt, return_tensors="pt").to("cuda:0")

    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    print("\n--- [1] NORMAL GENERATION (UNTOUCHED) ---")
    with torch.no_grad():
        out_normal = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=stop_ids
        )
    print(tok.decode(out_normal[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    # Safe scales: 0.8, 1.2, 1.5 (avoiding scale > 2.0 to prevent representation collapse)
    for scale_test in [0.8, 1.2, 1.5]:
        surgery_hooks = [
            layers[i].register_forward_hook(make_clean_projection_hook(i, scale=scale_test))
            for i in sweet_spot_layers
        ]
        print(f"\n--- [2] ORTHOGONAL PROJECTION (Layers 14-19, scale={scale_test}) ---")
        with torch.no_grad():
            out_steered = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=stop_ids
            )
        print(tok.decode(out_steered[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        for h in surgery_hooks:
            h.remove()
