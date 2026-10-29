"""
Phase 3, Step 2 - Custom Pre-Masked Data Collator & Single-Batch Forward/Backward Sanity Test
=============================================================================================
Builds and verifies:
  1. `PreMaskedTrajectoryDataset`: Loads pre-tokenized `input_ids` and mistake-masked `labels`
     from `train.jsonl` / `val.jsonl`.
  2. `DataCollatorForPreMaskedTrajectories`: Pads variable-length sequences in a batch using
     `pad_token_id` for `input_ids`, `0` for `attention_mask`, and `-100` for `labels`—preserving
     every internal `-100` mask (system, user, tool_response, and `learn=False` mistake turns).
  3. Runs a real 2-step Forward + Backward + 8-bit AdamW update on the RTX 5070 Ti (`sm_120`)
     to verify initial cross-entropy loss, gradient flow across `lora_B`, `lora_A`, and
     `lora_magnitude_vector`, loss descent on Step 1, and peak GPU VRAM.
"""
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from setup_dora_model import load_feral_dora_model

ROOT = Path(__file__).resolve().parent


def forward_masked_lm_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    VRAM-Optimized Exact Cross-Entropy Loss for 248K-Vocab Models (Qwen3.5):
    Standard HF `ForCausalLMLoss` projects ALL `B * T` tokens through `lm_head` (248,320 vocab)
    and upcasts the full `[B, T, 248320]` tensor to `float32` + `.contiguous()`, burning ~4.5 GB
    of VRAM even when >70% of tokens are masked with `-100`.

    Instead, we run the 32-layer backbone to get `[B, T, 4096]` hidden states, filter down to
    ONLY the active (`shift_labels != -100`) positions, and run `lm_head` + `float32` cross-entropy
    exclusively on those trained tokens. Mathematically 100% identical loss & gradients!
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
    active_hidden = shift_hidden[active_mask]
    active_labels = shift_labels[active_mask]

    active_logits = lm_head(active_hidden).float()
    return F.cross_entropy(active_logits, active_labels)


class PreMaskedTrajectoryDataset(Dataset):
    """Dataset wrapper for pre-tokenized, mistake-masked JSONL trajectories."""

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 4096):
        self.path = Path(jsonl_path)
        self.rows = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ids = r["input_ids"][:max_seq_len]
                labs = r["labels"][:max_seq_len]
                self.rows.append({
                    "id": r["id"],
                    "stage": r["stage"],
                    "input_ids": ids,
                    "labels": labs,
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


class DataCollatorForPreMaskedTrajectories:
    """
    Pads variable-length trajectories to the longest sequence in the batch:
      - `input_ids`      padded on the right with `pad_token_id`
      - `attention_mask` `1` for real tokens, `0` for right-padding
      - `labels`         padded on the right with `-100` (never touches internal `-100` masks!)
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_ids = []
        batch_mask = []
        batch_labels = []

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)

            batch_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_mask.append([1] * len(ids) + [0] * pad_len)
            batch_labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def compute_grad_norms(model):
    """Computes L2 gradient norms separated by DoRA component (`lora_A`, `lora_B`, `magnitude`)."""
    sq_a, sq_b, sq_m = 0.0, 0.0, 0.0
    n_with_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g_norm_sq = p.grad.detach().float().norm(2).item() ** 2
            if g_norm_sq > 0:
                n_with_grad += 1
            if "lora_A" in name:
                sq_a += g_norm_sq
            elif "lora_B" in name:
                sq_b += g_norm_sq
            elif "lora_magnitude_vector" in name:
                sq_m += g_norm_sq
    total = math.sqrt(sq_a + sq_b + sq_m)
    return math.sqrt(sq_a), math.sqrt(sq_b), math.sqrt(sq_m), total, n_with_grad


def main():
    print("1. Loading train.jsonl and testing DataCollatorForPreMaskedTrajectories...")
    ds = PreMaskedTrajectoryDataset(ROOT / "train.jsonl")
    # Pick 2 trajectories of different lengths: one Stage_B_Recovery (with mistake masking) + one Stage_C_Code_Verify
    sample_b = next(r for r in ds.rows if r["stage"] == "Stage_B_Recovery")
    sample_c = next(r for r in ds.rows if r["stage"] == "Stage_C_Code_Verify")

    model, tok = load_feral_dora_model()
    collator = DataCollatorForPreMaskedTrajectories(pad_token_id=tok.pad_token_id, pad_to_multiple_of=8)
    batch = collator([sample_b, sample_c])

    len_b, act_b = len(sample_b["input_ids"]), sum(x != -100 for x in sample_b["labels"])
    len_c, act_c = len(sample_c["input_ids"]), sum(x != -100 for x in sample_c["labels"])
    batch_shape = list(batch["input_ids"].shape)
    batch_active = int((batch["labels"] != -100).sum().item())
    batch_masked = int((batch["labels"] == -100).sum().item())

    print(f"   - Row 0 ({sample_b['id']:28s}) : len={len_b:4d} | active trained={act_b:4d} | masked (-100)={len_b - act_b:4d}")
    print(f"   - Row 1 ({sample_c['id']:28s}) : len={len_c:4d} | active trained={act_c:4d} | masked (-100)={len_c - act_c:4d}")
    print(f"   - Collated Tensor Shape (padded to x8) : {batch_shape}")
    print(f"   - Total Active Labels in Batch         : {batch_active} (expected {act_b + act_c})")
    print(f"   - Total Masked/Padded Labels (-100)    : {batch_masked}")

    if batch_active != (act_b + act_c):
        raise AssertionError("Collator corrupted active label count!")
    # Verify padding positions in Row 0 have attention_mask == 0 and labels == -100
    if not torch.all(batch["labels"][0, len_b:] == -100) or not torch.all(batch["attention_mask"][0, len_b:] == 0):
        raise AssertionError("Right-padding mask verification failed!")
    print("   [PASS] Collator preserved exact Mistake Masking and padded right-hand positions with -100.\n")

    print("2. Running Step 0 Forward + Backward Pass on RTX 5070 Ti (Batch Size = 2)...")
    model.train()
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
        weight_decay=0.01,
    )

    gpu_batch = {k: v.to("cuda:0") for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats(0)

    optimizer.zero_grad(set_to_none=True)
    loss0 = forward_masked_lm_loss(model, gpu_batch)
    loss0.backward()

    norm_a0, norm_b0, norm_m0, norm_tot0, n_grad0 = compute_grad_norms(model)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step()

    print(f"   - Step 0 Cross-Entropy Loss : {loss0.item():.4f} (Perplexity: {math.exp(loss0.item()):.2f})")
    print(f"   - Step 0 Grad Norm (lora_B) : {norm_b0:.4f}  (non-zero immediately because A != 0)")
    print(f"   - Step 0 Grad Norm (mag m)  : {norm_m0:.4f}  (non-zero immediately on output scales)")
    print(f"   - Step 0 Grad Norm (lora_A) : {norm_a0:.4f}  (0.0 at Step 0 because lora_B starts at 0!)")
    print(f"   - Step 0 Total Grad Norm    : {norm_tot0:.4f} across {n_grad0} active adapter tensors\n")

    print("3. Running Step 1 Forward + Backward Pass (Verifying lora_A Wakes Up & Loss Drops)...")
    optimizer.zero_grad(set_to_none=True)
    loss1 = forward_masked_lm_loss(model, gpu_batch)
    loss1.backward()
    norm_a1, norm_b1, norm_m1, norm_tot1, n_grad1 = compute_grad_norms(model)

    peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_reserv_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    print(f"   - Step 1 Cross-Entropy Loss : {loss1.item():.4f} (Delta: {loss1.item() - loss0.item():+.4f})")
    print(f"   - Step 1 Grad Norm (lora_B) : {norm_b1:.4f}")
    print(f"   - Step 1 Grad Norm (mag m)  : {norm_m1:.4f}")
    print(f"   - Step 1 Grad Norm (lora_A) : {norm_a1:.4f}  (Woke up after Step 0 updated lora_B!)")
    print(f"   - Step 1 Total Grad Norm    : {norm_tot1:.4f} across {n_grad1} active adapter tensors\n")

    print("4. RTX 5070 Ti Training VRAM Telemetry:")
    print(f"   - Peak VRAM Allocated : {peak_alloc_mb:7.1f} MiB / {total_mb:.0f} MiB ({peak_alloc_mb / total_mb * 100:.1f}%)")
    print(f"   - Peak VRAM Reserved  : {peak_reserv_mb:7.1f} MiB / {total_mb:.0f} MiB")

    if math.isnan(loss0.item()) or math.isnan(loss1.item()):
        raise RuntimeError("NaN loss detected!")
    if loss1.item() >= loss0.item():
        raise RuntimeError(f"Expected Step 1 loss ({loss1.item():.4f}) < Step 0 loss ({loss0.item():.4f})!")
    if norm_a1 <= 0.0:
        raise RuntimeError("lora_A gradients failed to wake up on Step 1!")

    print("\nSANITY TEST: PASS -> Pre-masked collator, DoRA backward pass, and 8-bit AdamW verified!")


if __name__ == "__main__":
    main()
