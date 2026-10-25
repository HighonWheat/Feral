# Feral (Uncensored Qwen3.5-9B & Autonomous Agent)

**Feral** is an experimental project built on top of `Qwen/Qwen3.5-9B` focused on mechanistic interpretability, activation steering / abliteration (uncensoring via residual stream refusal-direction projection), and local autonomous tool-use execution on consumer GPUs (4-bit NF4 via `bitsandbytes`).

## Project Structure

- **`inspect_residual_stream.py`** — Loads `Qwen3.5-9B` in 4-bit NF4 (`bfloat16` compute) and attaches PyTorch forward hooks across the 32 internal decoder layers (`Layer 0`, `Layer 15`, `Layer 31`) to intercept and inspect 4096-dimensional residual stream hidden state vectors on the final prompt token.
- **`test_agent_brain.py`** — Autonomous multi-step ReAct agent loop using `<think>...</think>` reasoning chains and structured `<tool_call>` JSON execution to run live Windows PowerShell commands and process `<tool_response>` feedback.
- **`LICENSE`** — MIT License.

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5070 Ti with 4-bit quantization)
- `torch`, `transformers`, `bitsandbytes`, `accelerate`

## Usage

Place the `Qwen3.5-9B` model weights and tokenizer files in the project root directory (automatically ignored by `.gitignore`), then run:

```bash
# Inspect internal layer activations / residual stream vectors
python inspect_residual_stream.py

# Run the autonomous PowerShell system agent
python test_agent_brain.py
```

## License

This project is licensed under the [MIT License](LICENSE).
