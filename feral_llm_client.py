"""
Phase 5, Step 1 - Feral Streaming LLM Client & Tool-Call Engine (`feral_llm_client.py`)
=======================================================================================
Core local inference engine for Feral, powering both the autonomous multi-server MCP host
(Phase 5) and the Odysseus-inspired Desktop Command Center UI (Phase 7):

  1. Fast-Fused 4-Bit DoRA Loader (`FeralLLMClient`):
     Loads `Qwen3.5-9B-Base` (4-bit NF4) + our Phase 4 RL-hardened adapter (`lora_model/feral_grpo_dora`)
     and binds the precomputed `dora_weight_norms.safetensors` vector directly into `dora_layer.forward`,
     delivering the 7.29x generation speedup (`~14+ tok/s`) on RTX 5070 Ti (`cuda:0`).
  2. Context Panel & Local Model Registry (`list_available_local_models()`):
     Discovers all local checkpoints in `E:\\Qwen3.5 9b` (`Feral-Qwen3.5-9B-GRPO-DoRA`,
     `Feral-Qwen3.5-9B-SFT-DoRA`, `Qwen3.5-9B-Base`, and `feral-qwen3.5-9b-grpo-q4_k_m.gguf`)
     and exposes live controls for `system_prompt`, `temperature`, `top_p`, and `max_new_tokens`.
  3. Real-Time `<think>` & `<tool_call>` Token Streamer (`stream_generate()`):
     Uses `TextIteratorStreamer` to yield structured real-time events (`think_delta`, `answer_delta`,
     `tool_call`, `metrics`) for both the Chat Interface and the Autonomous Agents Board console log.
  4. Deterministic Tool-Call Parser & JSON Schema Validator (`parse_feral_output()`):
     Extracts `<think>...</think>`, parses `<tool_call>...</tool_call>`, validates against Feral's
     7-tool `TOOL_SCHEMA`, and formats `<tool_response>` observations for multi-turn ReAct loops.
"""
import json
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from transformers.utils import logging as hf_logging

from feral_core import ROOT, SYSTEM_FERAL, TOOL_SCHEMA

hf_logging.set_verbosity_error()

GRPO_CHECKPOINT_DIR = ROOT / "lora_model" / "feral_grpo_dora"
SFT_CHECKPOINT_DIR = ROOT / "lora_model" / "feral_sft_dora"
NORM_CACHE_PATH = GRPO_CHECKPOINT_DIR / "dora_weight_norms.safetensors"
GGUF_PATH = ROOT / "outputs" / "feral-qwen3.5-9b-grpo-q4_k_m.gguf"

VALID_TOOLS: Dict[str, List[str]] = {k: list(v) for k, v in TOOL_SCHEMA.items()}


@dataclass
class GenerationConfigState:
    """Runtime generation controls bound to the UI Right Sidebar ('Context Panel')."""
    model_id: str = "Feral-Qwen3.5-9B-GRPO-DoRA"
    system_prompt: str = SYSTEM_FERAL
    temperature: float = 0.0
    top_p: float = 0.92
    max_new_tokens: int = 256


@dataclass
class ParsedFeralTurn:
    """Structured representation of a single assistant turn."""
    raw_text: str
    think: str
    tool_call: Optional[Dict[str, Any]]
    final_answer: str
    is_valid_tool_call: bool
    validation_detail: str
    tokens_generated: int = 0
    elapsed_sec: float = 0.0
    tokens_per_sec: float = 0.0
    vram_allocated_mib: float = 0.0


def list_available_local_models() -> List[Dict[str, Any]]:
    """
    Scans `E:\\Qwen3.5 9b` and returns metadata for the Right Sidebar ('Context Panel')
    local model selector dropdown and Odysseus-style Hardware Cookbook.
    """
    models = [
        {
            "id": "Feral-Qwen3.5-9B-GRPO-DoRA",
            "label": "Feral Qwen3.5-9B (GRPO RLVR + DoRA r=32)",
            "backend": "transformers+peft-4bit",
            "path": str(GRPO_CHECKPOINT_DIR.relative_to(ROOT)),
            "available": GRPO_CHECKPOINT_DIR.exists(),
            "vram_estimate_gib": 7.6,
            "recommended": True,
        },
        {
            "id": "Feral-Qwen3.5-9B-SFT-DoRA",
            "label": "Feral Qwen3.5-9B (Phase 3 SFT DoRA r=32)",
            "backend": "transformers+peft-4bit",
            "path": str(SFT_CHECKPOINT_DIR.relative_to(ROOT)),
            "available": SFT_CHECKPOINT_DIR.exists(),
            "vram_estimate_gib": 7.6,
            "recommended": False,
        },
        {
            "id": "Feral-Qwen3.5-9B-GGUF-Q4_K_M",
            "label": "Feral Qwen3.5-9B GGUF v3 (Q4_K_M / Q8_0)",
            "backend": "gguf-v3",
            "path": str(GGUF_PATH.relative_to(ROOT)) if GGUF_PATH.exists() else "outputs/feral-qwen3.5-9b-grpo-q4_k_m.gguf",
            "available": GGUF_PATH.exists(),
            "size_mib": round(GGUF_PATH.stat().st_size / (1024 ** 2), 1) if GGUF_PATH.exists() else 0.0,
            "vram_estimate_gib": 5.2,
            "recommended": False,
        },
        {
            "id": "Qwen3.5-9B-Base-ZeroShot",
            "label": "Unsloth Qwen3.5-9B-Base (Untouched Base)",
            "backend": "transformers-4bit",
            "path": ".",
            "available": (ROOT / "model.safetensors.index.json").exists(),
            "vram_estimate_gib": 7.2,
            "recommended": False,
        },
    ]
    return models


def _repair_json_string(raw_json: str) -> str:
    """Repairs minor JSON formatting artifacts (e.g. trailing commas before `}` or `]`)."""
    s = raw_json.strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_feral_output(text: str) -> ParsedFeralTurn:
    """
    Parses a raw assistant completion into:
      - `think`: inner monologue inside `<think>...</think>`
      - `tool_call`: parsed dict `{"name": ..., "arguments": {...}}` inside `<tool_call>...</tool_call>`
      - `final_answer`: user-facing markdown response when no tool call is emitted
    """
    cleaned = text.strip()
    for stop_tok in ("<|im_end|>", "<|endoftext|>"):
        if cleaned.endswith(stop_tok):
            cleaned = cleaned[: -len(stop_tok)].strip()

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", cleaned, flags=re.S)
    think_text = think_match.group(1).strip() if think_match else ""

    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, flags=re.S)
    tool_call_obj: Optional[Dict[str, Any]] = None
    is_valid_tc = False
    detail = "final_answer"

    if tc_match:
        raw_json = _repair_json_string(tc_match.group(1))
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                t_name = parsed["name"]
                t_args = parsed["arguments"]
                if t_name in VALID_TOOLS and isinstance(t_args, dict):
                    missing = [k for k in VALID_TOOLS[t_name] if k not in t_args]
                    if not missing:
                        tool_call_obj = {"name": t_name, "arguments": t_args}
                        is_valid_tc = True
                        detail = f"valid_tool_call:{t_name}"
                    else:
                        detail = f"missing_required_args:{missing}"
                else:
                    detail = f"unknown_tool:{t_name}"
            else:
                detail = "invalid_tool_call_keys"
        except Exception as e:
            detail = f"json_decode_error:{e}"

    # Extract final answer by stripping `<think>...</think>` and `<tool_call>...</tool_call>`
    answer_text = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S)
    answer_text = re.sub(r"<tool_call>.*?</tool_call>", "", answer_text, flags=re.S).strip()

    return ParsedFeralTurn(
        raw_text=cleaned,
        think=think_text,
        tool_call=tool_call_obj,
        final_answer=answer_text,
        is_valid_tool_call=is_valid_tc,
        validation_detail=detail,
    )


class FeralLLMClient:
    """
    Production local inference engine for Feral (`Qwen3.5-9B` + GRPO DoRA + Fast Norm Fusion).
    Provides synchronous generation, real-time token streaming, and multi-turn ReAct execution.
    """

    def __init__(
        self,
        checkpoint_dir: Path = GRPO_CHECKPOINT_DIR,
        norm_cache_path: Path = NORM_CACHE_PATH,
        device: str = "cuda:0",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.norm_cache_path = Path(norm_cache_path)
        self.device = device
        self.config_state = GenerationConfigState()

        self.tokenizer = AutoTokenizer.from_pretrained(str(ROOT))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_ids = list(dict.fromkeys([self.tokenizer.eos_token_id, im_end_id]))

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(ROOT),
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.checkpoint_dir),
            is_trainable=False,
        )
        self.model.eval()
        self.model.base_model.model.config.use_cache = True

        self.fused_dora_count = self._bind_fast_dora_inference()
        self._warmup_cuda_kernels()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _warmup_cuda_kernels(self):
        """Runs a 2-token dummy decode pass so CUDA/bitsandbytes kernels are warm before the first user turn."""
        dummy = self.tokenizer("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(self.device)
        _ = self.model.generate(**dummy, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        torch.cuda.synchronize()

    def _bind_fast_dora_inference(self) -> int:
        """
        Binds precomputed `mag_norm_scale = (magnitude / weight_norm).view(1, -1)` into all 200
        `DoraLinear4bitLayer.forward` hooks so inference never dequantizes 4-bit weights to
        recompute column norms during token generation (7.29x speedup).
        """
        norm_tensors: Dict[str, torch.Tensor] = {}
        if self.norm_cache_path.exists():
            with safe_open(str(self.norm_cache_path), framework="pt", device=self.device) as f:
                for k in f.keys():
                    norm_tensors[k] = f.get_tensor(k)

        bound = 0
        from peft.utils.integrations import dequantize_module_weight

        for name, mod in self.model.named_modules():
            if hasattr(mod, "lora_magnitude_vector") and "default" in getattr(mod, "lora_magnitude_vector", {}):
                dora_layer = mod.lora_magnitude_vector["default"]
                with torch.no_grad():
                    if name in norm_tensors:
                        w_norm = norm_tensors[name].to(device=self.device, dtype=torch.bfloat16)
                    else:
                        lora_a_mod = mod.lora_A["default"]
                        lora_b_mod = mod.lora_B["default"]
                        scaling_val = mod.scaling["default"]
                        w_dq = dequantize_module_weight(mod.base_layer).to(torch.bfloat16)
                        lw = (lora_b_mod.weight @ lora_a_mod.weight).to(torch.bfloat16)
                        w_norm = dora_layer.get_weight_norm(
                            weight=w_dq, lora_weight=lw, scaling=scaling_val, adapter_name="default"
                        ).detach()
                    mag_norm_scale = (dora_layer.weight.to(torch.bfloat16) / w_norm.clamp(min=1e-8)).view(1, -1).detach()

                def make_fast_dora_forward(scale_vec):
                    def fast_dora_forward(x, *, lora_A, lora_B, scaling, base_layer, base_result=None, adapter_name="default"):
                        lora_result = lora_B(lora_A(x))
                        if base_result is not None and base_layer.bias is not None:
                            base_result = base_result - base_layer.bias
                        return (scale_vec - 1) * base_result + scale_vec * lora_result * scaling
                    return fast_dora_forward

                dora_layer.forward = make_fast_dora_forward(mag_norm_scale)
                bound += 1
        return bound

    def update_context_panel(
        self,
        *,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationConfigState:
        """Updates runtime parameters from the UI Right Sidebar ('Context Panel')."""
        if model_id is not None:
            self.config_state.model_id = model_id
        if system_prompt is not None:
            self.config_state.system_prompt = system_prompt
        if temperature is not None:
            self.config_state.temperature = max(0.0, min(float(temperature), 2.0))
        if top_p is not None:
            self.config_state.top_p = max(0.05, min(float(top_p), 1.0))
        if max_new_tokens is not None:
            self.config_state.max_new_tokens = max(16, min(int(max_new_tokens), 4096))
        return self.config_state

    def format_chatml(self, messages: List[Dict[str, str]]) -> str:
        """Formats a conversation history into Qwen3.5 ChatML with `system_prompt` prepended if missing."""
        msgs = list(messages)
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": self.config_state.system_prompt})
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @torch.no_grad()
    def generate_turn(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> ParsedFeralTurn:
        """Runs a single assistant turn and returns a validated `ParsedFeralTurn` with GPU telemetry."""
        prompt = self.format_chatml(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        temp = self.config_state.temperature if temperature is None else float(temperature)
        max_toks = self.config_state.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        do_sample = temp > 1e-4

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_toks,
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": self.eos_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = self.config_state.top_p

        use_base_only = self.config_state.model_id == "Qwen3.5-9B-Base-ZeroShot"
        t0 = time.perf_counter()
        if use_base_only:
            with self.model.disable_adapter():
                out = self.model.generate(**gen_kwargs)
        else:
            out = self.model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen_ids = out[0, prompt_len:]
        cut = len(gen_ids)
        for idx, tid in enumerate(gen_ids.tolist()):
            if tid in self.eos_token_ids:
                cut = idx
                break
        n_tok = max(cut, 1)
        raw_text = self.tokenizer.decode(gen_ids[:cut], skip_special_tokens=False).strip()

        parsed = parse_feral_output(raw_text)
        parsed.tokens_generated = n_tok
        parsed.elapsed_sec = round(elapsed, 3)
        parsed.tokens_per_sec = round(n_tok / max(elapsed, 1e-6), 2)
        parsed.vram_allocated_mib = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
        return parsed

    def stream_turn(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, ParsedFeralTurn]:
        """
        Streams live tokens via `TextIteratorStreamer`, yielding real-time UI events:
          - `{"type": "token", "text": chunk, "section": "think" | "tool_call" | "answer"}`
          - `{"type": "complete", "parsed": ParsedFeralTurn}`
        Ideal for the Main Chat Hub and the Autonomous Agents Board rolling console log.
        """
        prompt = self.format_chatml(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        temp = self.config_state.temperature if temperature is None else float(temperature)
        max_toks = self.config_state.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        do_sample = temp > 1e-4

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )
        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_toks,
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": self.eos_token_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = self.config_state.top_p

        t0 = time.perf_counter()
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        accum = []
        current_section = "answer"
        chunk_count = 0

        for piece in streamer:
            if not piece:
                continue
            clean_piece = piece.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            accum.append(clean_piece)
            joined = "".join(accum)

            if "<think>" in joined and "</think>" not in joined:
                current_section = "think"
            elif "<tool_call>" in joined and "</tool_call>" not in joined:
                current_section = "tool_call"
            elif "</think>" in joined and "<tool_call>" not in joined:
                current_section = "answer"

            if clean_piece:
                chunk_count += 1
                yield {
                    "type": "token",
                    "text": clean_piece,
                    "section": current_section,
                }

        thread.join()
        elapsed = time.perf_counter() - t0
        full_text = "".join(accum).strip()
        n_tok = len(self.tokenizer.encode(full_text, add_special_tokens=False))
        parsed = parse_feral_output(full_text)
        parsed.tokens_generated = n_tok
        parsed.elapsed_sec = round(elapsed, 3)
        parsed.tokens_per_sec = round(n_tok / max(elapsed, 1e-6), 2)
        parsed.vram_allocated_mib = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)

        yield {"type": "complete", "parsed": parsed}
        return parsed

    def run_react_loop(
        self,
        user_prompt: str,
        tool_executor: Callable[[str, Dict[str, Any]], str],
        max_steps: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes a multi-turn autonomous ReAct loop (`<think>` -> `<tool_call>` -> `<tool_response>` -> Answer),
        returning the full conversation trace and per-step telemetry for the Autonomous Agents Board.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.config_state.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        steps_log = []

        for step_idx in range(1, max_steps + 1):
            turn = self.generate_turn(messages)
            messages.append({"role": "assistant", "content": turn.raw_text})

            step_entry: Dict[str, Any] = {
                "step": step_idx,
                "think": turn.think,
                "tool_call": turn.tool_call,
                "final_answer": turn.final_answer,
                "tokens_per_sec": turn.tokens_per_sec,
                "elapsed_sec": turn.elapsed_sec,
            }

            if not turn.is_valid_tool_call or turn.tool_call is None:
                steps_log.append(step_entry)
                return {
                    "status": "completed",
                    "steps": steps_log,
                    "final_answer": turn.final_answer,
                    "messages": messages,
                }

            obs = tool_executor(turn.tool_call["name"], turn.tool_call["arguments"])
            step_entry["tool_observation"] = obs
            steps_log.append(step_entry)
            messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})

        return {
            "status": "max_steps_reached",
            "steps": steps_log,
            "final_answer": steps_log[-1]["final_answer"] if steps_log else "",
            "messages": messages,
        }


def main():
    print("1. Auditing Local Model Registry for UI Context Panel (`list_available_local_models()`)...")
    local_models = list_available_local_models()
    for m in local_models:
        status = "READY" if m["available"] else "MISSING"
        print(f"   - [{status}] {m['id']:30s} | Backend: {m['backend']:22s} | Est. VRAM: {m['vram_estimate_gib']} GiB")

    print("\n2. Initializing `FeralLLMClient` (`Qwen3.5-9B-Base` 4-Bit NF4 + `feral_grpo_dora` + Fast DoRA Norm Fusion)...")
    client = FeralLLMClient()
    print(f"   - Bound Fast DoRA Forward Hooks : {client.fused_dora_count} / 200 projections")
    print(f"   - Active Context Panel Config   : model={client.config_state.model_id}, temp={client.config_state.temperature}, max_tokens={client.config_state.max_new_tokens}")

    print("\n3. Testing Real-Time `<think>` + `<tool_call>` Token Streaming (`stream_turn()`)...")
    stream_msgs = [
        {"role": "user", "content": "Inspect lines 1 to 20 of src/rate_limiter.py before making any edits."}
    ]
    streamed_chunks = []
    sections_seen = set()
    final_parsed: Optional[ParsedFeralTurn] = None

    for ev in client.stream_turn(stream_msgs):
        if ev["type"] == "token":
            streamed_chunks.append(ev["text"])
            sections_seen.add(ev["section"])
        elif ev["type"] == "complete":
            final_parsed = ev["parsed"]

    assert final_parsed is not None, "Streamer did not emit completion event"
    print(f"   - Streamed Chunks Received      : {len(streamed_chunks)} chunks across sections {sorted(sections_seen)}")
    print(f"   - Streamed `<think>` Monologue  : {final_parsed.think}")
    print(f"   - Parsed `<tool_call>` Payload  : {json.dumps(final_parsed.tool_call)}")
    print(f"   - Schema Validation             : {final_parsed.validation_detail} (is_valid={final_parsed.is_valid_tool_call})")
    print(f"   - Streaming Speed & VRAM        : {final_parsed.tokens_per_sec:.2f} tok/s ({final_parsed.tokens_generated} tokens in {final_parsed.elapsed_sec:.3f}s) | Allocated VRAM: {final_parsed.vram_allocated_mib:.1f} MiB")

    print("\n4. Testing Multi-Turn Autonomous ReAct Loop (`run_react_loop()`)...")
    from feral_core import Sandbox
    from grpo_reward_sandbox import SANDBOX_FILES

    sb = Sandbox("feral_client_test_sb", SANDBOX_FILES)
    sb.reset()
    try:
        def execute_in_sandbox(name: str, args: Dict[str, Any]) -> str:
            return getattr(sb, name)(**args)

        react_res = client.run_react_loop(
            user_prompt="Count how many ERROR lines appear in logs/app.log.",
            tool_executor=execute_in_sandbox,
            max_steps=3,
        )
        print(f"   - ReAct Loop Status             : {react_res['status']} in {len(react_res['steps'])} turns")
        for s in react_res["steps"]:
            if s["tool_call"]:
                print(f"     * Turn {s['step']} Tool Call ({s['tokens_per_sec']:5.2f} tok/s): {json.dumps(s['tool_call'])} -> Obs: {s.get('tool_observation', '').splitlines()[0]}")
            else:
                print(f"     * Turn {s['step']} Final Answer ({s['tokens_per_sec']:5.2f} tok/s): {s['final_answer']}")
    finally:
        sb.cleanup()

    if not final_parsed.is_valid_tool_call or react_res["status"] != "completed":
        raise RuntimeError("FeralLLMClient verification failed!")

    print("\nPHASE 5 STEP 1 (`feral_llm_client.py`): PASS -> Streaming `<think>`, `<tool_call>` validation, and ReAct loop verified!")


if __name__ == "__main__":
    main()
