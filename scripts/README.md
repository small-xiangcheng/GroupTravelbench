# GroupTravelbench Scripts

This directory contains every script for data restore, model deployment, inference, and evaluation. All configuration lives in [`base_config.sh`](./base_config.sh).

## Quick Start

```bash
# 1. Restore evaluation data (sandbox cache + POI metadata)
bash scripts/restore_data.sh

# 2. Edit the config (model name, API key, etc.; values are placeholders)
vim scripts/base_config.sh

# 3. Optional: serve a local model (skip this when using a cloud API)
source scripts/base_config.sh && source scripts/vllm_server.sh

# 4. Run the full pipeline in the background (logs are written automatically)
bash scripts/run_pipeline.sh --bg
```

### Evaluating a cloud model

Change two lines in `base_config.sh`; no local deployment is needed:

```bash
export USE_CUSTOM_ENDPOINT=false      # do not use local vLLM
export MODEL_NAME="your-model-name"   # cloud model name
```

Then run:
```bash
source scripts/base_config.sh && bash scripts/run_pipeline.sh --bg
```

---

## Configuration

### Model and inference

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_NAME` | Model under evaluation (also names the output directory) | `your-model-name` |
| `MODEL_PATH` | Local weight path (only for local serving) | `/path/to/your/model` |
| `MODEL_GPU_LIST` | GPU list | `"0,1,2,3"` |
| `MODEL_TP_SIZE` | Tensor-parallel size | `1` |
| `MODEL_DP_SIZE` | Data-parallel size (DP × TP must equal the GPU count) | `4` |
| `MODEL_INFERENCE_MODE` | Inference mode | `think` / `no-think` |
| `USE_CUSTOM_ENDPOINT` | `true` = local vLLM, `false` = cloud API | `false` |
| `ENABLE_THINKING` | Agent (model under test) thinking | `true` |
| `USER_ENABLE_THINKING` | User-simulator thinking (intended ON) | `true` |
| `TOOL_ENABLE_THINKING` | Tool-simulator thinking (intended OFF) | `false` |
| `USER_LLM` | User-simulator model | `your-user-simulator-model` |
| `TOOL_LLM` | Tool-simulator model (cache-miss fallback) | `your-tool-simulator-model` |
| `INFER_MAX_CONCURRENCY` | Max inference concurrency | `100` |


### Embedding model

Needed only for **similarity retrieval**. If unset, the simulator falls back to random retrieval.

| Variable | Description | Example |
|----------|-------------|---------|
| `EMBEDDING_MODEL_PATH` | Embedding-model path | `/path/to/your/embedding-model` |
| `EMBEDDING_MODEL_NAME` | Embedding-model name | `your-embedding-model-name` |
| `EMBEDDING_SERVICE_URL` | Embedding-service URL | `http://localhost:40001/v1` |
| `EMBEDDING_API_KEY` | Embedding-service key (**optional**; see note below) | `your-embedding-api-key` |
| `EMBEDDING_PORT` | Service port | `40001` |
| `EMBEDDING_GPU_LIST` | GPU list | `"4,5,6,7"` |
| `EMBEDDING_TP_SIZE` / `EMBEDDING_DP_SIZE` | TP / DP size | `2` / `2` |

`EMBEDDING_API_KEY` is only needed when the embedding service sits behind an authenticated gateway (e.g. an internal proxy).

### API

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Global default API key | `your-api-key` |
| `OPENAI_API_BASE` | Global default API base URL | `https://your-api-endpoint/v1` |

Any OpenAI-compatible endpoint works.

### Per-role API overrides (optional)

Use these when the Agent, User simulator, Tool simulator, or LLM-Judge need a different endpoint or key. **Leave empty to inherit the global defaults.**

| Variable | Description |
|----------|-------------|
| `AGENT_API_BASE` / `AGENT_API_KEY` | Agent-specific endpoint / key |
| `USER_API_BASE` / `USER_API_KEY` | User-simulator endpoint / key |
| `TOOL_API_BASE` / `TOOL_API_KEY` | Tool-simulator endpoint / key |
| `LLM_JUDGE_API_BASE` / `LLM_JUDGE_API_KEY` | LLM-Judge endpoint / key |

Typical setup: Agent on local vLLM, other roles on a cloud API with their own keys:
```bash
export OPENAI_API_KEY="your-default-key"
export OPENAI_API_BASE="https://your-api-endpoint/v1"

export AGENT_API_BASE=""                  # controlled by USE_CUSTOM_ENDPOINT + MODEL_SERVICE_URL
export USER_API_KEY="your-user-sim-key"   # User simulator uses a different key
export TOOL_API_BASE="https://your-other-endpoint/v1"
export TOOL_API_KEY="your-tool-sim-key"
```

### Evaluation

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_JUDGE_MODEL` | LLM-Judge model | `your-judge-model` |
| `LLM_JUDGE_CONCURRENCY` | LLM-Judge concurrency | `50` |

---

## Scripts

### `restore_data.sh` — restore evaluation data

```bash
bash scripts/restore_data.sh
```

Reassembles the compressed shards under `data_archive/`, unpacks `sandbox_cache/` (read during inference) and `poi2category/` (read during evaluation), and verifies every file with sha256. **This is the first step after cloning.**

### `run_pipeline.sh` — full pipeline

```bash
bash scripts/run_pipeline.sh [OPTIONS]
```

Runs **inference → evaluation → report**.

| Option | Description |
|--------|-------------|
| `--bg` | Run in the background (nohup; log to `Logs/Agent-{MODEL_NAME}.log`) |
| `--skip-infer` | Skip inference (evaluate existing output) |
| `--skip-llm-judge` | Skip the LLM-Judge (rule-based metrics only; faster, no extra API cost) |
| `--no-resume` | Fresh run (truncate the log and do not skip finished tasks). Resume is the default. |

**Resume**: logs are appended by default, and inference resumes from the existing output file. Use `--no-resume` to start over.

**Concurrency safety**: each launch snapshots the current config into process-local variables. Editing `base_config.sh` afterwards does not affect a running instance.

### `infer.sh` — inference

```bash
source scripts/base_config.sh && bash scripts/infer.sh
```

- Input: `datas/test.jsonl`
- Output: `infer_output/Agent-{MODEL_NAME}/test.jsonl`
- Default: 3 trials, `--max-turn 15`; concurrency is `INFER_MAX_CONCURRENCY`
- Extra arguments are forwarded to the CLI, e.g. `bash scripts/infer.sh --no-resume`

### `eval.sh` — evaluation

```bash
source scripts/base_config.sh && bash scripts/eval.sh [OPTIONS]
```

Reads `infer_output/Agent-{MODEL_NAME}/test.jsonl` by default.

| Option | Description |
|--------|-------------|
| `--skip-llm-judge` | Skip the LLM-Judge |
| `--input=<path>` | Custom input path |

### `vllm_server.sh` — local Agent model

```bash
source scripts/base_config.sh && source scripts/vllm_server.sh
```

Serves the main model with vLLM and exports `MODEL_SERVICE_URL` when ready.

- Checks that DP × TP equals the GPU count
- Enables the reasoning parser when `MODEL_INFERENCE_MODE=think`
- Startup timeout is 600s; logs go to `logs/`

> Must be executed with `source`, otherwise `MODEL_SERVICE_URL` will not enter the current shell.

### `vllm_embedding_server.sh` — local embedding model

```bash
source scripts/base_config.sh && source scripts/vllm_embedding_server.sh
```

Serves the embedding model used for sandbox similarity retrieval and exports `EMBEDDING_SERVICE_URL`.

- Starts with `--runner pooling` and prefix caching enabled
- Uses a separate GPU group so it does not contend with the Agent model

---

## Output layout

```
GroupTravelbench/
├── infer_output/
│   └── Agent-your-model-name/
│       └── test.jsonl                    # inference results
├── eval_output/
│   └── Agent-your-model-name/
│       ├── report.md                     # 📊 final report
│       ├── multi_eval_result.json        # detailed rule-based results
│       └── llm_judge_result.json         # detailed LLM-Judge results
├── Logs/
│   └── Agent-your-model-name.log         # full-pipeline log
└── logs/                                 # vLLM deployment logs
```

---

## Report contents

`eval_output/Agent-xxx/report.md` has four sections:

1. **Score Overview** — Group Utility plus the mean of the other four dimensions
2. **Scores by Dimension** — Preference Completeness / Group Utility / Group Fairness / Plan Validity / LLM-Judge, split by easy / medium / hard
3. **LLM-Judge Sub-dimensions** — per-subdimension scores, including meta-judge agreement
4. **Execution Statistics** — plan-generation rate, average duration, average tool calls, **cache hit rate**, tool error rate, and the distribution of completion reasons

---

## Common workflows

| Scenario | Command |
|----------|---------|
| First-time data setup after clone | `bash scripts/restore_data.sh` |
| Fresh evaluation including inference | `bash scripts/run_pipeline.sh --bg --no-resume` |
| Resume (default) | `bash scripts/run_pipeline.sh --bg` |
| Evaluation only (inference already done) | `bash scripts/run_pipeline.sh --bg --skip-infer` |
| Fast evaluation (no LLM-Judge) | `bash scripts/run_pipeline.sh --bg --skip-llm-judge` |
| Foreground debugging | `bash scripts/run_pipeline.sh --skip-llm-judge` |

---

## Sandbox tool simulator

### Overview

Evaluation always runs in `isolated` (offline) mode and **never calls a third-party tool API**:

- **Cache hit** → return the pre-collected real tool response (zero cost, zero latency, fully reproducible)
- **Cache miss** → `TOOL_LLM` invents a response from historical examples and records it in `*_missed.json`

```
Agent calls a tool → SandboxBaseTool._cached_execute()
    ├── look up sandbox_cache/*_cache.json
    │     ├── HIT  → return the cached result
    │     └── MISS → LLMToolSimulator.simulate_tool_response()
    │                  ├── retrieve similar examples (random or FAISS)
    │                  ├── build a prompt and call TOOL_LLM
    │                  └── record the result in *_missed.json
    └── return the result to the Agent
```

### Retrieval modes

| Mode | Description | Embedding precompute |
|------|-------------|----------------------|
| **Random retrieval** (default fallback) | Sample `max_examples` entries from the cache | No |
| **Similarity retrieval** (recommended) | Find the K most similar examples via embeddings | Yes |

### Precomputing embeddings (similarity retrieval)

```bash
# 1. Prepare an embedding service (local or an existing URL)
source scripts/base_config.sh && source scripts/vllm_embedding_server.sh

# 2. Precompute (run once)
python -m group_travelbench.simulators.precompute_embeddings --cache-dir ./sandbox_cache

# Or pass arguments explicitly
python -m group_travelbench.simulators.precompute_embeddings \
    --cache-dir ./sandbox_cache \
    --service-url http://localhost:40001/v1 \
    --model-name your-embedding-model-name \
    --batch-size 1024
```

**Output**: one `*_embeddings.npz` next to each `*_cache.json`:
```
sandbox_cache/
├── travel_search_flights_cache.json       # original cache (shipped with the repo)
├── travel_search_flights_embeddings.npz   # precomputed embeddings (generated locally)
├── search_poi_cache.json
├── search_poi_embeddings.npz
└── ...
```

### Notes

1. **Cache hit rate is the main lever**: `sandbox_cache/` already holds a full cache for all 10 tools. Hits call no LLM.
2. **Similarity retrieval needs** the matching `*_embeddings.npz` files and a runtime `EMBEDDING_SERVICE_URL`.
3. **Automatic fallback**: missing embedding config or `*_embeddings.npz` files fall back to random retrieval; the run does not abort.
4. **Keep Tool-LLM temperature low**: scripts set `TOOL_LLM_ARGS` to `temperature=0.0` so simulated outputs stay stable.
5. **FAISS is optional**: without `faiss`, the simulator uses numpy dot-product search. Behavior is the same, just slower.
