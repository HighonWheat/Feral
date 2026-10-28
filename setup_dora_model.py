"""
Phase 3, Step 1 - QLoRA / DoRA Adapter Injection & Trainable Parameter Inspection
=================================================================================
Loads `unsloth/Qwen3.5-9B-Base` in 4-bit NF4 (`bfloat16` compute, double quantization)
on the RTX 5070 Ti (`sm_120`), inspects the hybrid Gated DeltaNet + Full Attention + MLP
linear projections, and injects Weight-Decomposed Low-Rank Adapters (DoRA, r=32, alpha=64).

Also exports `load_feral_dora_model()` so Phase 3 Steps 2 and 3 can import the exact same setup.
"""
import os
import warnings
from collections import Counter

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

MODEL_PATH = "."
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# High-dimensional feature projections across all 32 hybrid layers:
#   - 8 Full Softmax Attention layers (`self_attn`): q_proj, k_proj, v_proj, o_proj
#   - 24 Gated DeltaNet layers (`linear_attn`):      in_proj_qkv, in_proj_z, out_proj
#   - 32 SwiGLU MLP blocks (`mlp`):                  gate_proj, up_proj, down_proj
# Note: `in_proj_a` and `in_proj_b` in `linear_attn` have shape (32, 4096) (32 scalar head decay gates),
# where out_features == r (32), so low-rank decomposition is neither needed nor bottlenecked there.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_feral_dora_model(
    model_path: str = MODEL_PATH,
    r: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    use_dora: bool = True,
    gradient_checkpointing: bool = True,
):
    """Loads 4-bit NF4 Qwen3.5-9B-Base, prepares k-bit training, and attaches DoRA/LoRA adapters."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    # Freeze all base weights and keep embeddings/lm_head/norms in native bfloat16.
    # (Calling PEFT's `prepare_model_for_kbit_training` blindly upcasts 2.03B embedding + lm_head
    # parameters from bfloat16 to float32, wasting ~4.5 GB of VRAM on a 16 GB GPU!)
    for param in model.parameters():
        param.requires_grad = False

    if gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_dora=use_dora,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    torch.cuda.empty_cache()
    return model, tok


def inspect_architecture_and_adapters(model, tok):
    layers = model.base_model.model.model.layers
    print(f"1. Hybrid Architecture Verified: {len(layers)} Decoder Layers")
    print("   - Layer 0 (Gated DeltaNet `linear_attn`) projections:")
    for name, mod in layers[0].linear_attn.named_children():
        if hasattr(mod, "in_features"):
            tag = " [TARGETED]" if name in TARGET_MODULES else " [SKIPPED: scalar head gate, out_features == r]"
            print(f"       {name:14s}: ({mod.out_features:5d}, {mod.in_features:5d}){tag}")

    print("   - Layer 3 (Full Softmax `self_attn`) projections:")
    for name, mod in layers[3].self_attn.named_children():
        if hasattr(mod, "in_features"):
            print(f"       {name:14s}: ({mod.out_features:5d}, {mod.in_features:5d}) [TARGETED]")

    print("   - Layer 0 (SwiGLU `mlp`) projections:")
    for name, mod in layers[0].mlp.named_children():
        if hasattr(mod, "in_features"):
            print(f"       {name:14s}: ({mod.out_features:5d}, {mod.in_features:5d}) [TARGETED]")

    # Count injected adapter modules by projection type
    injected_counts = Counter()
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in mod.lora_A:
            leaf = name.split(".")[-1]
            injected_counts[leaf] += 1

    print(f"\n2. Injected DoRA Modules Across 32 Layers (r={LORA_RANK}, alpha={LORA_ALPHA}, scaling={LORA_ALPHA / LORA_RANK:.1f}):")
    for mod_name in TARGET_MODULES:
        print(f"   - {mod_name:14s}: {injected_counts[mod_name]:2d} layers wrapped")
    print(f"   -> Total Linear4bit modules wrapped with DoRA: {sum(injected_counts.values())}")

    # Single-matrix concrete math check on Layer 0 linear_attn.out_proj (4096 x 4096)
    out_proj = layers[0].linear_attn.out_proj
    a_shape = tuple(out_proj.lora_A["default"].weight.shape)
    b_shape = tuple(out_proj.lora_B["default"].weight.shape)
    m_shape = tuple(out_proj.lora_magnitude_vector["default"].weight.shape)
    full_params = out_proj.in_features * out_proj.out_features
    dora_params = (a_shape[0] * a_shape[1]) + (b_shape[0] * b_shape[1]) + m_shape[0]
    reduction = (1.0 - dora_params / full_params) * 100.0

    print("\n3. Single-Matrix Decomposition Check (`layers[0].linear_attn.out_proj`, 4096 x 4096):")
    print(f"   - Frozen NF4 base weight W_0 : (4096, 4096) -> {full_params:10,d} logical params (packed into {out_proj.base_layer.weight.numel():,d} uint8 bytes)")
    print(f"   - Trainable lora_A (bf16)    : {str(a_shape):12s} -> {a_shape[0] * a_shape[1]:10,d} params")
    print(f"   - Trainable lora_B (bf16)    : {str(b_shape):12s} -> {b_shape[0] * b_shape[1]:10,d} params")
    print(f"   - Trainable DoRA magnitude m : {str(m_shape):12s} -> {m_shape[0]:10,d} params")
    print(f"   - Total trainable for matrix : {dora_params:10,d} params ({reduction:.2f}% reduction vs full 16.78M matrix)")

    # Global parameter census (count logical parameters by doubling packed 4-bit NF4 uint8 storage)
    trainable_params = 0
    lora_ab_params = 0
    dora_mag_params = 0
    frozen_logical_params = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            trainable_params += n
            if "lora_magnitude_vector" in name:
                dora_mag_params += n
            else:
                lora_ab_params += n
        else:
            # 4-bit NF4 packs 2 weights per uint8 byte
            frozen_logical_params += p.numel() * (2 if p.dtype == torch.uint8 else 1)

    total_logical = frozen_logical_params + trainable_params
    pct = trainable_params / total_logical * 100.0

    print("\n4. Global Parameter Census (Logical Weights vs Trainable DoRA Adapters):")
    print(f"   - Frozen Base Parameters (4-bit NF4 + bf16 norms/embeds) : {frozen_logical_params:14,d}")
    print(f"   - Trainable LoRA Directional Matrices (A + B)            : {lora_ab_params:14,d}")
    print(f"   - Trainable DoRA Magnitude Vectors (m)                   : {dora_mag_params:14,d}")
    print(f"   - Total Trainable Parameters (bfloat16)                  : {trainable_params:14,d} ({pct:.2f}% of {total_logical:,d})")

    # Forward pass & RTX 5070 Ti VRAM telemetry check
    ids = tok("<|im_start|>user\nPing Feral DoRA check.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model(**ids)
    logits = out.logits
    has_nan = bool(torch.isnan(logits).any() or torch.isinf(logits).any())

    alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserv_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print("\n5. RTX 5070 Ti VRAM & Forward-Pass Sanity Check:")
    print(f"   - Output logits shape : {list(logits.shape)} | dtype: {logits.dtype} | NaN/Inf: {has_nan}")
    print(f"   - GPU VRAM Allocated  : {alloc_mb:7.1f} MiB / {total_mb:.0f} MiB ({alloc_mb / total_mb * 100:.1f}%)")
    print(f"   - GPU VRAM Reserved   : {reserv_mb:7.1f} MiB / {total_mb:.0f} MiB")

    if has_nan or sum(injected_counts.values()) != 200:
        raise RuntimeError("DoRA verification failed!")
    print("\nVERIFICATION: PASS -> 200 hybrid projections wrapped with DoRA (r=32, alpha=64) and verified on cuda:0!")


if __name__ == "__main__":
    print("Loading 4-bit NF4 Qwen3.5-9B-Base and injecting DoRA adapters (r=32, alpha=64)...\n")
    feral_model, feral_tok = load_feral_dora_model()
    inspect_architecture_and_adapters(feral_model, feral_tok)
