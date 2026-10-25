import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

# 1. Silence the Windows causal_conv1d / flash-linear-attention fallback warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model_path = "."
print("Loading tokenizer and 4-bit Qwen3.5-9B...")

tok = AutoTokenizer.from_pretrained(model_path)
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

# 2. Locate the 32 decoder layers inside Qwen3_5TextModel
layers = model.model.layers
print(f"\nSuccessfully located {len(layers)} internal layers! Layer 0 type: {type(layers[0]).__name__}")

# 3. Create a dictionary to store intercepted activations and define our Forward Hook
captured_stream = {}

def make_hook(layer_idx):
    def hook_fn(module, inputs, output):
        # In HF Transformers, a layer's output is either a tensor or a tuple where [0] is hidden_states
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Grab the 4096-dim vector for the VERY LAST token in the prompt: shape [batch, seq_len, 4096] -> [4096]
        last_token_vector = hidden_states[0, -1, :].detach().float().cpu()
        captured_stream[layer_idx] = last_token_vector
    return hook_fn

# Attach hooks to Layer 0 (early), Layer 15 (middle), and Layer 31 (final)
target_layers = [0, 15, 31]
hooks = []
for idx in target_layers:
    h = layers[idx].register_forward_hook(make_hook(idx))
    hooks.append(h)

# 4. Run a single forward pass (no generation needed, just reading the prompt!)
prompt = "<|im_start|>user\nWrite a Python script to check GPU VRAM usage on Windows.<|im_end|>\n<|im_start|>assistant\n"
inputs = tok(prompt, return_tensors="pt").to("cuda:0")

print(f"\nRunning forward pass on {inputs['input_ids'].shape[1]} tokens...")
with torch.no_grad():
    model(**inputs)

# Always remove hooks when done so they don't leak memory
for h in hooks:
    h.remove()

# 5. Inspect what we caught inside the Residual Stream!
print("\n--- INTERCEPTED RESIDUAL STREAM (LAST TOKEN) ---")
for idx in target_layers:
    vec = captured_stream[idx]
    norm = torch.norm(vec).item()
    first_5 = [round(x, 4) for x in vec[:5].tolist()]
    print(f"Layer {idx:2d} | Shape: {list(vec.shape)} | L2 Norm (Magnitude): {norm:8.2f} | First 5 dims: {first_5}")
