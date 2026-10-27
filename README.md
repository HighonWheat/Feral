# Feral (Autonomous Qwen3.5-9B Software Engineering Agent)

**Feral** is a local autonomous software engineering and research agent built on `Qwen/Qwen3.5-9B` (4-bit NF4 / `bfloat16` compute on RTX 5070 Ti `sm_120`), combining representation engineering (orthogonal residual stream projection) with execution-verified multi-tool trajectory distillation and mistake masking.

## Project Structure

- **`inspect_residual_stream.py`** — Hooks into `model.model.layers` (`[0, 15, 31]`) to inspect 4,096-dim residual stream hidden states and L2 norm growth.
- **`collect_refusal_vector.py` & `test_orthogonal_projection.py`** — Extracts contrast vectors at the `<|im_start|>assistant` decision boundary and applies unit-vector orthogonal projection across sweet-spot layers `14..19`.
- **`test_agent_brain.py`** — Autonomous ReAct loop using `<think>...</think>` reasoning and `<tool_call>` JSON execution.
- **`feral_core.py`** — Shared execution sandbox, unified `SYSTEM_FERAL` prompt, Feral Apex Toolchain implementations (`run_powershell`, `read_file_lines`, `apply_diff_patch`, `write_file`, `ast_repo_map`, `web_research`, `spawn_subagent`), and completion/mistake-masked tokenizer.
- **`scenarios/`** — Modular task libraries (`powershell_tasks.py` and `code_fix_tasks.py`) used to generate verified training trajectories.
- **`distill_verified_traces.py`** — Executes real PowerShell and Python bug-fix tasks in a throwaway sandbox to build `agent_curriculum_train.jsonl`.
- **`expand_feral_arsenal_dataset.py`** — Replays multi-tool surgical repository workflows against a sandbox repo to build `feral_arsenal_trajectories.jsonl`.
- **`validate_curriculum_dataset.py`** — Audits all trajectories (unified prompt, balanced tags, tool JSON schema, positional `-100` label masking, stop tokens) and writes a group-disjoint stratified `train.jsonl` / `val.jsonl` split.

## How to Add Your Own Training Tasks

1. **Add a PowerShell or CLI task:** Append a dict to `PS_TASKS` in [`scenarios/powershell_tasks.py`](scenarios/powershell_tasks.py) (include an optional `linux` field to auto-generate a mistake-masked `Stage_B_Recovery` trace).
2. **Add a Python bug-fix task:** Append a dict to `CODE_FIX_TASKS` in [`scenarios/code_fix_tasks.py`](scenarios/code_fix_tasks.py) with `buggy`, `search`, `replace`, and `verify_cmd`.
3. **Regenerate and audit the dataset:**
   ```powershell
   python distill_verified_traces.py
   python expand_feral_arsenal_dataset.py
   python validate_curriculum_dataset.py
   ```

## License

This project is licensed under the [MIT License](LICENSE).
