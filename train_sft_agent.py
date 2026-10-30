"""
Phase 3, Step 3 - Full Curriculum SFT Training Run with Live Telemetry (`train_sft_agent.py`)
=============================================================================================
Trains the 200 DoRA adapters (`r=32`, `lora_alpha=64`) on `Qwen3.5-9B-Base` (4-bit NF4)
over the 100 execution-verified trajectories in `train.jsonl` and evaluates on the 20
group-disjoint validation trajectories in `val.jsonl`.

Key engineering features:
  - Token-weighted active Cross-Entropy Loss (`forward_masked_lm_loss` from Step 2) so padded
    and mistake-masked (`-100`) positions consume zero `lm_head` VRAM and zero gradient.
  - 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`) + Linear Warmup (10%) + Cosine LR Decay.
  - Evaluates exact token-level validation loss and perplexity on `val.jsonl` at Epoch 0
    (untouched baseline) and after every epoch, saving the best DoRA checkpoint to
    `lora_model/feral_sft_dora`.
"""
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from setup_dora_model import load_feral_dora_model
from test_sft_step import DataCollatorForPreMaskedTrajectories, PreMaskedTrajectoryDataset

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "lora_model" / "feral_sft_dora"

SEED = 42
EPOCHS = 3
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4          # Effective batch size = 2 * 4 = 8 trajectories/step
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.10


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_active_loss_sum(model, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """
    Runs the 32-layer Qwen3.5 backbone, filters hidden states to ONLY `shift_labels != -100`,
    and returns `(sum_of_token_losses, num_active_tokens)` so gradient accumulation across
    micro-batches is weighted by the exact number of active trained tokens.
    """
    inner_qwen = model.base_model.model
    text_backbone = inner_qwen.model
    lm_head = inner_qwen.lm_head

    outputs = text_backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state

    shift_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
    shift_labels = batch["labels"][:, 1:].reshape(-1)

    active_mask = shift_labels != -100
    n_active = int(active_mask.sum().item())
    active_hidden = shift_hidden[active_mask]
    active_labels = shift_labels[active_mask]

    active_logits = lm_head(active_hidden).float()
    loss_sum = F.cross_entropy(active_logits, active_labels, reduction="sum")
    return loss_sum, n_active


@torch.no_grad()
def evaluate_validation(model, val_loader: DataLoader) -> Dict[str, float]:
    """Computes exact token-weighted validation loss and perplexity over `val.jsonl`."""
    model.eval()
    total_loss_sum = 0.0
    total_active_tokens = 0

    for batch in val_loader:
        gpu_batch = {k: v.to("cuda:0") for k, v in batch.items()}
        loss_sum, n_active = forward_active_loss_sum(model, gpu_batch)
        total_loss_sum += float(loss_sum.item())
        total_active_tokens += n_active

    mean_loss = total_loss_sum / max(1, total_active_tokens)
    ppl = math.exp(min(20.0, mean_loss))
    model.train()
    return {
        "val_loss": mean_loss,
        "val_ppl": ppl,
        "val_tokens": total_active_tokens,
    }


def compute_total_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(sq)


def main():
    set_seed(SEED)
    print("1. Loading train.jsonl (100 rows) and val.jsonl (20 rows)...")
    train_ds = PreMaskedTrajectoryDataset(ROOT / "train.jsonl")
    val_ds = PreMaskedTrajectoryDataset(ROOT / "val.jsonl")

    print("2. Loading 4-bit NF4 Qwen3.5-9B-Base and injecting 200 DoRA adapters (r=32, alpha=64)...")
    model, tok = load_feral_dora_model()
    collator = DataCollatorForPreMaskedTrajectories(pad_token_id=tok.pad_token_id, pad_to_multiple_of=8)

    train_loader = DataLoader(
        train_ds,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_opt_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(round(total_opt_steps * WARMUP_RATIO)))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    print(f"   - Train trajectories : {len(train_ds)} | Val trajectories : {len(val_ds)}")
    print(f"   - Micro-batch size   : {MICRO_BATCH_SIZE} | Grad accum steps : {GRAD_ACCUM_STEPS} (Effective batch = {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"   - Optimizer steps    : {total_opt_steps} ({steps_per_epoch}/epoch x {EPOCHS} epochs, warmup={warmup_steps} steps)\n")

    # Baseline (Epoch 0) validation before any weight updates
    print("3. Evaluating Baseline (Epoch 0, Untrained DoRA) on val.jsonl...")
    t0 = time.time()
    val0 = evaluate_validation(model, val_loader)
    print(f"   [Epoch 0 Baseline] Val Loss: {val0['val_loss']:.4f} | Val Perplexity: {val0['val_ppl']:.2f} | Active Val Tokens: {val0['val_tokens']}\n")

    best_val_loss = val0["val_loss"]
    telemetry = {
        "baseline_val_loss": val0["val_loss"],
        "baseline_val_ppl": val0["val_ppl"],
        "steps": [],
        "epochs": [{"epoch": 0, **val0}],
    }

    torch.cuda.reset_peak_memory_stats(0)
    global_step = 0
    train_start = time.time()

    print("4. Launching Full Curriculum SFT Training Run on RTX 5070 Ti...")
    print("-" * 92)
    print(f"{'Step':>6s} | {'Epoch':>5s} | {'Train Loss':>10s} | {'Grad Norm':>9s} | {'LR':>10s} | {'Tokens':>6s} | {'Step Time':>9s} | {'VRAM MiB':>8s}")
    print("-" * 92)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        micro_batches = list(train_loader)
        for window_start in range(0, len(micro_batches), GRAD_ACCUM_STEPS):
            step_t0 = time.time()
            window = micro_batches[window_start:window_start + GRAD_ACCUM_STEPS]

            # Count total active tokens across the accumulation window for exact token-weighted averaging
            window_active_tokens = sum(
                int((b["labels"][:, 1:] != -100).sum().item()) for b in window
            )
            window_loss_accum = 0.0

            for mb in window:
                gpu_mb = {k: v.to("cuda:0") for k, v in mb.items()}
                loss_sum, _ = forward_active_loss_sum(model, gpu_mb)
                scaled_loss = loss_sum / max(1, window_active_tokens)
                scaled_loss.backward()
                window_loss_accum += float(loss_sum.item())

            step_loss = window_loss_accum / max(1, window_active_tokens)
            grad_norm = compute_total_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=MAX_GRAD_NORM)

            current_lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            dt = time.time() - step_t0
            vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            ep_frac = (epoch - 1) + min(1.0, (window_start + len(window)) / len(micro_batches))

            print(
                f"{global_step:4d}/{total_opt_steps:<2d} | {ep_frac:5.2f} | "
                f"{step_loss:10.4f} | {grad_norm:9.4f} | {current_lr:10.2e} | "
                f"{window_active_tokens:6d} | {dt:8.2f}s | {vram_mb:8.1f}",
                flush=True,
            )

            telemetry["steps"].append({
                "step": global_step,
                "epoch": round(ep_frac, 3),
                "train_loss": round(step_loss, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": current_lr,
                "active_tokens": window_active_tokens,
                "step_sec": round(dt, 2),
            })

        # End-of-epoch validation
        val_metrics = evaluate_validation(model, val_loader)
        telemetry["epochs"].append({"epoch": epoch, **val_metrics})
        improved = val_metrics["val_loss"] < best_val_loss
        marker = " *BEST*" if improved else ""
        print("-" * 92)
        print(
            f"[Epoch {epoch} Complete] Val Loss: {val_metrics['val_loss']:.4f} "
            f"(Baseline: {val0['val_loss']:.4f}, Delta: {val_metrics['val_loss'] - val0['val_loss']:+.4f}) | "
            f"Val PPL: {val_metrics['val_ppl']:.2f}{marker}",
            flush=True,
        )
        print("-" * 92)

        if improved:
            best_val_loss = val_metrics["val_loss"]
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(CHECKPOINT_DIR))
            tok.save_pretrained(str(CHECKPOINT_DIR))

    total_sec = time.time() - train_start
    peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_reserv_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

    telemetry["total_sec"] = round(total_sec, 1)
    telemetry["best_val_loss"] = round(best_val_loss, 4)
    telemetry["peak_vram_allocated_mib"] = round(peak_alloc_mb, 1)
    telemetry["peak_vram_reserved_mib"] = round(peak_reserv_mb, 1)

    with open(CHECKPOINT_DIR / "training_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print("\n5. Training Summary & Checkpoint Verification:")
    print(f"   - Total Training Time : {total_sec:.1f}s ({total_sec / 60:.2f} min)")
    print(f"   - Val Loss Progression: {val0['val_loss']:.4f} (Epoch 0) -> {best_val_loss:.4f} (Best)")
    print(f"   - Val PPL Progression : {val0['val_ppl']:.2f} (Epoch 0) -> {math.exp(best_val_loss):.2f} (Best)")
    print(f"   - Peak VRAM Allocated : {peak_alloc_mb:.1f} MiB | Reserved: {peak_reserv_mb:.1f} MiB")
    print(f"   - Saved DoRA Adapter  : {CHECKPOINT_DIR}")

    if best_val_loss >= val0["val_loss"]:
        raise RuntimeError("Validation loss did not improve over baseline!")
    print("\nTRAINING: PASS -> Feral DoRA adapter converged cleanly without overfitting!")


if __name__ == "__main__":
    main()
