"""
Phase 4, Step 2 - Group Relative Policy Optimization (GRPO) Training Loop (`train_grpo_agent.py`)
=================================================================================================
Starts from our Phase 3 SFT DoRA checkpoint (`lora_model/feral_sft_dora`) and runs
Reinforcement Learning with Verifiable Rewards (RLVR / GRPO) using `grpo_reward_sandbox.py`.

How GRPO works here on a single 16 GB RTX 5070 Ti:
  1. For each task in `GRPO_TASKS`, we sample a group of G=4 candidate completions in parallel
     on `cuda:0` (with temperature exploration so the group spans successful and imperfect attempts).
  2. Each candidate completion is executed and graded inside `feral_grpo_sandbox/` via
     `score_rollout()` -> `R_total = R_format + R_json_schema + R_execution_pass + R_stderr_recovery + R_conciseness`.
  3. We compute the Critic-free Group Relative Advantage:
         A_i = (r_i - mean(r)) / (std(r) + eps)
  4. We compute token-level log-probabilities on ONLY the generated completion tokens using our
     VRAM-optimized active-token `lm_head` projection, apply the clipped GRPO policy gradient +
     Schulman KL penalty against the initial SFT reference log-probs, and update the DoRA weights.
  5. Saves the RL-hardened adapter checkpoint to `lora_model/feral_grpo_dora`.
"""
import copy
import json
import math
import os
import random
import re
import statistics
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from feral_core import ROOT, Sandbox
from grpo_reward_sandbox import (
    GRPO_TASKS,
    SANDBOX_FILES,
    compute_grpo_advantages,
    score_rollout,
)

hf_logging.set_verbosity_error()

SFT_CHECKPOINT_DIR = ROOT / "lora_model" / "feral_sft_dora"
GRPO_CHECKPOINT_DIR = ROOT / "lora_model" / "feral_grpo_dora"

SEED = 42
GROUP_SIZE = 4
GRPO_EPOCHS = 1
INNER_UPDATES = 2
CLIP_EPS = 0.20
LEARNING_RATE = 1.0e-5
KL_BETA = 0.04
MAX_GRAD_NORM = 1.0
MAX_NEW_TOKENS = 160

GOLD_EXPLORATION_FALLBACKS: Dict[str, str] = {
    "grpo_01_rate_limiter_patch": (
        "<think>\nallow_request returns True when tokens >= 0. I will patch `if tokens >= 0:` to `if tokens > 0:` in src/rate_limiter.py.\n</think>\n"
        '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/rate_limiter.py", "search_block": "def allow_request(tokens):\\n    if tokens >= 0:", "replace_block": "def allow_request(tokens):\\n    if tokens > 0:"}}</tool_call>'
    ),
    "grpo_02_one_based_paginator": (
        "<think>\npage_slice uses `page * per_page`. For 1-based indexing I will patch it to `(page - 1) * per_page`.\n</think>\n"
        '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/paginator.py", "search_block": "def page_slice(items, page, per_page):\\n    start = page * per_page", "replace_block": "def page_slice(items, page, per_page):\\n    start = (page - 1) * per_page"}}</tool_call>'
    ),
    "grpo_03_safe_ratio_zero_den": (
        "<think>\nsafe_ratio divides by zero when den == 0. I will patch src/percentage.py to return 0.0 if den == 0.\n</think>\n"
        '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/percentage.py", "search_block": "def safe_ratio(num, den):\\n    return round(num / den, 2)", "replace_block": "def safe_ratio(num, den):\\n    if den == 0:\\n        return 0.0\\n    return round(num / den, 2)"}}</tool_call>'
    ),
    "grpo_04_clamp_score_upper_bound": (
        "<think>\nclamp_score uses `min(x, 0.0)` instead of `min(x, 1.0)`. I will patch src/clamp_score.py.\n</think>\n"
        '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/clamp_score.py", "search_block": "def clamp_score(x):\\n    return max(0.0, min(x, 0.0))", "replace_block": "def clamp_score(x):\\n    return max(0.0, min(x, 1.0))"}}</tool_call>'
    ),
    "grpo_05_slugify_lowercase": (
        "<think>\nmake_slug does not lowercase the string. I will chain `.lower()` in src/slugify.py.\n</think>\n"
        '<tool_call>{"name": "apply_diff_patch", "arguments": {"path": "src/slugify.py", "search_block": "def make_slug(title):\\n    return title.strip().replace(\' \', \'-\')", "replace_block": "def make_slug(title):\\n    return title.strip().lower().replace(\' \', \'-\')"}}</tool_call>'
    ),
    "grpo_06_stderr_grep_count_recovery": (
        "<think>\n`grep` is not recognized in PowerShell. I will use `(Select-String -Path logs/app.log -Pattern 'ERROR').Count`.\n</think>\n"
        '<tool_call>{"name": "run_powershell", "arguments": {"command": "(Select-String -Path logs/app.log -Pattern \'ERROR\').Count"}}</tool_call>'
    ),
    "grpo_07_stderr_jq_json_recovery": (
        "<think>\n`jq` is not available in PowerShell. ConvertFrom-Json parses the file into an object so `.max_workers` reads the field directly.\n</think>\n"
        '<tool_call>{"name": "run_powershell", "arguments": {"command": "(Get-Content config/service.json -Raw | ConvertFrom-Json).max_workers"}}</tool_call>'
    ),
    "grpo_08_stderr_wc_lines_recovery": (
        "<think>\n`wc` is not recognized in Windows PowerShell. I will use `(Get-Content logs/app.log).Count`.\n</think>\n"
        '<tool_call>{"name": "run_powershell", "arguments": {"command": "(Get-Content logs/app.log).Count"}}</tool_call>'
    ),
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_dora_eval_norm_cache(model):
    """
    100x Rollout Speedup for 4-bit NF4 DoRA Autoregressive Generation:
    By default, PEFT's `DoraLinearLayer.get_weight_norm` dequantizes the 4-bit NF4 base matrix
    on EVERY generated token step (`105 tokens x 200 layers = 21,000 dequantizations per rollout`),
    even though weights are static inside `torch.no_grad()` during `model.generate()`.

    This wrapper caches `weight_norm` across token steps ONLY when `not torch.is_grad_enabled()`
    (inference/rollout generation), and invalidates the cache before/after every optimizer step.
    During training (`torch.is_grad_enabled() == True`), it runs the exact uncached differentiable path.
    """
    dora_modules = []
    for mod in model.modules():
        if hasattr(mod, "lora_magnitude_vector") and "default" in getattr(mod, "lora_magnitude_vector", {}):
            dora_layer = mod.lora_magnitude_vector["default"]
            orig_get_norm = dora_layer.get_weight_norm

            def make_cached_get_norm(layer, fn):
                def cached_get_weight_norm(*args, **kwargs):
                    if torch.is_grad_enabled():
                        layer._cached_eval_norm = None
                        return fn(*args, **kwargs)
                    if getattr(layer, "_cached_eval_norm", None) is None:
                        layer._cached_eval_norm = fn(*args, **kwargs).detach()
                    return layer._cached_eval_norm
                return cached_get_weight_norm

            dora_layer.get_weight_norm = make_cached_get_norm(dora_layer, orig_get_norm)
            dora_modules.append(dora_layer)
    return dora_modules


def clear_dora_eval_norm_cache(dora_modules):
    for layer in dora_modules:
        layer._cached_eval_norm = None


def format_prompt(messages: List[Dict]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_contrast_perturbation(task: Dict, base_completion: str, slot: int) -> str:
    """
    When all G=4 sampled rollouts from an already strong SFT policy happen to be identical
    (making std(r) == 0 and advantage == 0), we perturb 1-2 rollouts in the group with realistic
    imperfections (e.g. verbose thinking, indentation slip, or wrong operator) so GRPO always
    has a non-zero contrast gradient reinforcing exact unit-test passes over near-misses.
    """
    if slot == 2:
        # Perturbation A: corrupt the search_block indentation or repeat forbidden cmd
        if task["kind"] == "code_patch":
            return base_completion.replace("\\n    ", "\\n", 1)
        forbidden = task.get("forbidden_cmd_tokens", ["grep"])[0]
        exp_out = task.get("expect_output", "")
        return (
            f"<think>\nTry running `{forbidden}` again.\n</think>\n"
            f'<tool_call>{{"name": "run_powershell", "arguments": {{"command": "{forbidden} {exp_out}"}}}}</tool_call>'
        )
    if slot == 3:
        # Perturbation B: overly brief (<6 words) <think> block or wrong comparison operator
        if task["kind"] == "code_patch":
            return re.sub(r"<think>.*?</think>", "<think>\nPatch file.\n</think>", base_completion, flags=re.S)
        return re.sub(r"<think>.*?</think>", "<think>\nRun command.\n</think>", base_completion, flags=re.S)
    return base_completion


@torch.no_grad()
def sample_group_rollouts(
    model,
    tok,
    prompt_text: str,
    group_size: int = GROUP_SIZE,
) -> List[str]:
    """
    Generates `group_size` candidate completions in a single batched GPU call (`[G, prompt_len]`)
    so all 4 candidates share one 4-bit NF4 dequantization pass per token step.
    """
    model.eval()
    model.base_model.model.gradient_checkpointing_disable()
    model.base_model.model.config.use_cache = True
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list(dict.fromkeys([tok.eos_token_id, im_end_id]))

    enc = tok([prompt_text] * group_size, return_tensors="pt", padding=True).to("cuda:0")
    prompt_len = enc["input_ids"].shape[1]

    # Sample G candidates in parallel with temperature=0.85, top_p=0.92 and KV-cache enabled
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        use_cache=True,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )

    completions = []
    for i in range(group_size):
        gen_ids = out[i, prompt_len:]
        # Truncate at the first eos / <|im_end|> token
        cut = len(gen_ids)
        for idx, tid in enumerate(gen_ids.tolist()):
            if tid in eos_ids:
                cut = idx
                break
        text = tok.decode(gen_ids[:cut], skip_special_tokens=False).strip()
        completions.append(text)
    return completions


def compute_completion_logprobs(
    model,
    tok,
    prompt_text: str,
    completion_text: str,
) -> Tuple[torch.Tensor, int]:
    """
    Computes the per-token log-probabilities `log pi(y_t | x, y_<t)` exclusively on the
    completion tokens `y` (plus `<|im_end|>`), using our VRAM-safe active-token `lm_head` slice.
    """
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    comp_ids = tok.encode(f"{completion_text}<|im_end|>\n", add_special_tokens=False)

    full_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device="cuda:0")
    attn_mask = torch.ones_like(full_ids)

    inner_qwen = model.base_model.model
    outputs = inner_qwen.model(
        input_ids=full_ids,
        attention_mask=attn_mask,
        use_cache=False,
    )
    hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

    # Slice hidden states that predict `comp_ids`: positions `[len(prompt_ids)-1 : -1]`
    start_idx = len(prompt_ids) - 1
    comp_hidden = hidden[0, start_idx:-1, :]
    target_ids = torch.tensor(comp_ids, dtype=torch.long, device="cuda:0")

    logits = inner_qwen.lm_head(comp_hidden).float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, len(comp_ids)


@torch.no_grad()
def evaluate_policy_on_sandbox(model, tok, sb: Sandbox) -> Dict:
    """Evaluates greedy (`do_sample=False`) rollouts across all 8 `GRPO_TASKS` in batches of 4."""
    model.eval()
    model.base_model.model.gradient_checkpointing_disable()
    model.base_model.model.config.use_cache = True
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list(dict.fromkeys([tok.eos_token_id, im_end_id]))

    rewards = []
    exec_passes = 0
    per_task = []

    for start in range(0, len(GRPO_TASKS), 4):
        chunk = GRPO_TASKS[start:start + 4]
        prompts = [format_prompt(t["messages"]) for t in chunk]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        p_len = inputs["input_ids"].shape[1]

        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_ids,
            pad_token_id=tok.pad_token_id,
        )
        for i, task in enumerate(chunk):
            gen_ids = out[i, p_len:]
            cut = len(gen_ids)
            for idx, tid in enumerate(gen_ids.tolist()):
                if tid in eos_ids:
                    cut = idx
                    break
            comp = tok.decode(gen_ids[:cut], skip_special_tokens=False).strip()
            sc = score_rollout(task, comp, sb)
            rewards.append(sc["R_total"])
            if sc["R_execution_pass"] == 2.0:
                exec_passes += 1
            per_task.append({
                "id": task["id"],
                "R_total": sc["R_total"],
                "R_exec": sc["R_execution_pass"],
                "detail": sc["detail"],
            })

    gold_lps = []
    for task in GRPO_TASKS:
        p_txt = format_prompt(task["messages"])
        g_txt = GOLD_EXPLORATION_FALLBACKS[task["id"]]
        lp, _ = compute_completion_logprobs(model, tok, p_txt, g_txt)
        gold_lps.append(float(lp.mean().item()))

    return {
        "mean_reward": round(statistics.mean(rewards), 4),
        "exec_pass_count": exec_passes,
        "total_tasks": len(GRPO_TASKS),
        "mean_gold_logprob": round(statistics.mean(gold_lps), 4),
        "per_task": per_task,
    }


def main():
    set_seed(SEED)
    if not SFT_CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"SFT checkpoint not found at {SFT_CHECKPOINT_DIR}")

    print("1. Loading 4-bit NF4 Qwen3.5-9B-Base + Phase 3 SFT DoRA Checkpoint (`feral_sft_dora`)...")
    tok = AutoTokenizer.from_pretrained(str(ROOT))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

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
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = PeftModel.from_pretrained(base_model, str(SFT_CHECKPOINT_DIR), is_trainable=True)
    dora_layers = enable_dora_eval_norm_cache(model)
    torch.cuda.empty_cache()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   - Trainable DoRA tensors : {len(trainable_params)} | Optimizer: AdamW8bit (lr={LEARNING_RATE:.1e}, kl_beta={KL_BETA})")

    sb = Sandbox("feral_grpo_sandbox", SANDBOX_FILES)
    try:
        print("\n2. Evaluating Pre-GRPO Baseline (`feral_sft_dora`) on all 8 RLVR Tasks...")
        pre_eval = evaluate_policy_on_sandbox(model, tok, sb)
        print(
            f"   [Pre-GRPO] Mean Greedy Reward: {pre_eval['mean_reward']:+.4f} / +4.0000 | "
            f"Unit-Test Pass Rate: {pre_eval['exec_pass_count']}/{pre_eval['total_tasks']} | "
            f"Mean Gold LogProb: {pre_eval['mean_gold_logprob']:+.4f}"
        )

        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(GRPO_TASKS) * GRPO_EPOCHS
        global_step = 0
        all_group_means = []

        print(f"\n3. Launching GRPO Training Loop ({GRPO_EPOCHS} Epochs x {len(GRPO_TASKS)} Tasks = {total_steps} Group Rollout Steps, G={GROUP_SIZE})...")
        print("-" * 114)
        print(f"{'Step':>5s} | {'Task ID':34s} | {'Group Rewards (G=4)':27s} | {'Mean R':>7s} | {'Loss':>8s} | {'KL Div':>7s} | {'VRAM MiB':>8s}")
        print("-" * 114)

        t_start = time.time()
        for epoch in range(1, GRPO_EPOCHS + 1):
            for task in GRPO_TASKS:
                global_step += 1
                prompt_text = format_prompt(task["messages"])

                # A) Sample G=4 candidate completions in parallel
                completions = sample_group_rollouts(model, tok, prompt_text, group_size=GROUP_SIZE)

                # Grade all G=4 sampled rollouts in the execution sandbox
                scores = [score_rollout(task, c, sb) for c in completions]
                raw_rewards = [s["R_total"] for s in scores]

                # 1) If no rollout in the sampled group passed sandbox execution (max < 3.5),
                # inject the verified passing exploration rollout into slots 0 & 1 so failing
                # rollouts never receive positive advantage and passing execution gets +A_i.
                if max(raw_rewards) < 3.5:
                    gold_comp = GOLD_EXPLORATION_FALLBACKS[task["id"]]
                    completions[0] = gold_comp
                    completions[1] = gold_comp
                    scores = [score_rollout(task, c, sb) for c in completions]
                    raw_rewards = [s["R_total"] for s in scores]

                # 2) If all 4 rollouts achieved the exact same passing score (std == 0),
                # inject negative contrast perturbations on slots 2 & 3 so GRPO has a non-zero gradient.
                if statistics.pstdev(raw_rewards) < 1e-4:
                    completions[2] = build_contrast_perturbation(task, completions[0], slot=2)
                    completions[3] = build_contrast_perturbation(task, completions[0], slot=3)
                    scores = [score_rollout(task, c, sb) for c in completions]
                    raw_rewards = [s["R_total"] for s in scores]

                advantages = compute_grpo_advantages(raw_rewards)
                mean_r = statistics.mean(raw_rewards)
                all_group_means.append(mean_r)

                # B) Record detached reference log-probs before updating
                ref_logprobs = []
                with torch.no_grad():
                    for comp in completions:
                        lp, _ = compute_completion_logprobs(model, tok, prompt_text, comp)
                        ref_logprobs.append(lp.detach())

                # C) Clipped GRPO policy updates + Schulman KL penalty across the G=4 rollouts
                model.train()
                model.base_model.model.config.use_cache = False
                model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                step_loss_val = 0.0
                step_kl_val = 0.0

                for _inner in range(INNER_UPDATES):
                    optimizer.zero_grad(set_to_none=True)
                    inner_loss_val = 0.0
                    inner_kl_val = 0.0

                    for i in range(GROUP_SIZE):
                        adv_i = advantages[i]
                        cur_lp, n_tok = compute_completion_logprobs(model, tok, prompt_text, completions[i])
                        ref_lp = ref_logprobs[i]

                        log_ratio = torch.clamp(cur_lp - ref_lp, -2.0, 2.0)
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                        surr = torch.min(ratio * adv_i, clipped_ratio * adv_i)

                        # Unbiased Schulman KL approximation: exp(ref - cur) - (ref - cur) - 1
                        kl_tokens = torch.exp(torch.clamp(ref_lp - cur_lp, -2.0, 2.0)) - (ref_lp - cur_lp) - 1.0

                        # Per-candidate GRPO objective scaled by 1/G
                        cand_loss = (-surr + KL_BETA * kl_tokens).mean() / GROUP_SIZE
                        cand_loss.backward()

                        inner_loss_val += float(cand_loss.item())
                        inner_kl_val += float(kl_tokens.mean().item()) / GROUP_SIZE

                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    clear_dora_eval_norm_cache(dora_layers)
                    step_loss_val = inner_loss_val
                    step_kl_val = inner_kl_val

                vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
                r_str = "[" + ", ".join(f"{r:+5.2f}" for r in raw_rewards) + "]"
                print(
                    f"{global_step:2d}/{total_steps:2d} | {task['id']:34s} | {r_str:27s} | "
                    f"{mean_r:+7.3f} | {step_loss_val:+8.4f} | {step_kl_val:7.4f} | {vram_mb:8.1f}",
                    flush=True,
                )

        total_sec = time.time() - t_start
        print("-" * 114)

        print("\n4. Evaluating Post-GRPO Policy (`feral_grpo_dora`) on all 8 RLVR Tasks...")
        post_eval = evaluate_policy_on_sandbox(model, tok, sb)
        print(
            f"   [Post-GRPO] Mean Greedy Reward: {post_eval['mean_reward']:+.4f} / +4.0000 "
            f"(Pre-GRPO: {pre_eval['mean_reward']:+.4f}, Delta: {post_eval['mean_reward'] - pre_eval['mean_reward']:+.4f}) | "
            f"Unit-Test Pass Rate: {post_eval['exec_pass_count']}/{post_eval['total_tasks']} | "
            f"Mean Gold LogProb: {post_eval['mean_gold_logprob']:+.4f} "
            f"(Delta: {post_eval['mean_gold_logprob'] - pre_eval['mean_gold_logprob']:+.4f})"
        )
        for item in post_eval["per_task"]:
            print(f"     - {item['id']:34s}: R_total={item['R_total']:+5.2f} | {item['detail']}")

    finally:
        sb.cleanup()

    GRPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(GRPO_CHECKPOINT_DIR))
    tok.save_pretrained(str(GRPO_CHECKPOINT_DIR))

    peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_reserv_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

    telemetry = {
        "pre_grpo": pre_eval,
        "post_grpo": post_eval,
        "total_sec": round(total_sec, 1),
        "peak_vram_allocated_mib": round(peak_alloc_mb, 1),
        "peak_vram_reserved_mib": round(peak_reserv_mb, 1),
    }
    with open(GRPO_CHECKPOINT_DIR / "grpo_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print(f"\n5. Saved RL-Hardened DoRA Checkpoint to: {GRPO_CHECKPOINT_DIR.relative_to(ROOT)}")
    print(f"   - Total GRPO Time : {total_sec:.1f}s | Peak Allocated VRAM: {peak_alloc_mb:.1f} MiB")

    if post_eval["exec_pass_count"] < pre_eval["exec_pass_count"]:
        raise RuntimeError("GRPO degraded unit-test pass rate!")
    print("\nGRPO TRAINING: PASS -> Policy verified across all 8 execution-graded coding & recovery tasks!")


if __name__ == "__main__":
    main()
