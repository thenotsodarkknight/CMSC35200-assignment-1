# CMSC 35200: Deep Learning Systems 
## Assignment 1

Five workflows built around the Argonne ALCF Sophia inference service and local Ollama/nanoGPT environments.

## Reports & Documentation

* [`outputs/summary.md`](./outputs/summary.md) – English + Spanish one-page summaries.
* [`outputs/summary_validation.txt`](./outputs/summary_validation.txt) – numeric parity check (English vs. Spanish).
* [`outputs/part_notes.md`](./outputs/part_notes.md) – half-page notes for each part with metrics, challenges, and observations.

## Quick Map

| Path | Purpose |
| --- | --- |
| [`task1.py`](./task1.py) | Runs the 4-model “telephone” chain on Sophia |
| [`task2.py`](./task2.py) | Installs Open WebUI, writes [`.env.openwebui`](./.env.openwebui), optional `--serve` and `--healthcheck` |
| [`task3.py`](./task3.py) | 50-gene disease analysis on Sophia |
| [`task4.py`](./task4.py) | Replays the gene analysis locally via Ollama |
| [`task4_eval.py`](./task4_eval.py) | Compares local runs vs. Sophia baseline ([`outputs/gene_analysis_local/comparison.md`](./outputs/gene_analysis_local/comparison.md)) |
| [`task5.py`](./task5.py) | nanoGPT Shakespeare fine-tune helper (plot + sample) |
| [`outputs/`](./outputs) | All artifacts: telephone logs, gene summaries, local comparison, nanoGPT curve, summaries |
| [`openwebui_state/`](./openwebui_state) | Open WebUI persistent config directory |
| [`nanoGPT/`](./nanoGPT) | Forked nanoGPT repo (includes checkpoints, logs, sample script) |
| [`model_servers.yaml`](./model_servers.yaml) | Sophia model base URL + list of models for Open WebUI |
| [`.env.openwebui`](./.env.openwebui) | Environment exports used by Open WebUI (`OPEN_WEBUI_CONFIG_DIR`, tokens, etc.) |

Artifacts:
* `outputs/nanogpt_learning_curve.png`
* `outputs/nanogpt_sample.txt`
* `nanoGPT/out/ckpt.pt` checkpoints

## Prerequisites

1. Python 3.11 (the script installs `python@3.11` via Homebrew if missing).
2. Globus login enabling Argonne ALCF Sophia access (tokens stored under `~/.globus`).
3. Ollama ↔ local model server service started.

## Workflow at a Glance

### 1. Telephone Chain ([`task1.py`](./task1.py))

```bash
python3 task1.py
```

Outputs:
* `outputs/telephone/telephone_runs.json` – raw data (prompt, stages, timings).
* `outputs/telephone/telephone_runs.md` – readable summary.

### 2. Open WebUI ([`task2.py`](./task2.py))

Build venv + config:
```bash
python3 task2.py               
python3 task2.py --healthcheck 
```

Manual launch:
```bash
source .venv_openwebui/bin/activate
set -a && source .env.openwebui && set +a
open-webui serve --host 127.0.0.1 --port 8080
```

Open WebUI stores state in [`openwebui_state/`](./openwebui_state). Inside the UI verify the Sophia provider under “Settings → Connections”.

### 3. Gene Analysis ([`task3.py`](./task3.py))

```bash
python3 task3.py --seed 42 --model meta-llama/Meta-Llama-3.1-70B-Instruct
```

Outputs:
* `outputs/gene_analysis/selected_genes.json`
* `outputs/gene_analysis/raw_model_responses.json` (batched raw responses)
* `outputs/gene_analysis/results.json` (per-gene records)
* `outputs/gene_analysis/summary.{json,md}`
* `outputs/gene_analysis/spot_audit.md` (checks TP53/BRCA1 if present)

### 4. Local Models ([`task4.py`](./task4.py) + [`task4_eval.py`](./task4_eval.py))

Ensure Ollama is running and the models are pulled:
```bash
ollama pull llama3.2:3b
ollama pull phi3:3.8b
```

Run the analysis against Ollama:
```bash
python3 task4.py --model llama3.2:3b --batch-size 5
python3 task4.py --model phi3:3.8b --batch-size 5
```

Compare local vs. Sophia:
```bash
python3 task4_eval.py
```

Reports:
* `outputs/gene_analysis_local/<model>/summary.json`
* [`outputs/gene_analysis_local/comparison.md`](./outputs/gene_analysis_local/comparison.md) – agreement/disagreement counts.

### 5. nanoGPT ([`task5.py`](./task5.py))

This repo includes the upstream [`nanoGPT/`](./nanoGPT) clone. Training ran once already; to re-run or just regenerate plots/samples:

```bash
python3 task5.py          # re-trains, logs to nanoGPT/out/training_log.txt
python3 task5.py --skip-train
```

## Suggested Verification Sequence

1. `python3 task1.py` (check `outputs/telephone/*`).
2. `python3 task2.py --healthcheck` (inspect `outputs/openwebui_healthcheck.json`).
3. `python3 task3.py` (review gene outputs + spot audit).
4. `python3 task4.py ...` for each Ollama model, then `python3 task4_eval.py`.
5. `python3 task5.py --skip-train` (confirm plot + sample).

If you plan to commit this repo, remove or rotate any sensitive tokens (the current [`.env.openwebui`](./.env.openwebui) includes a live Sophia access token placeholder).

## Notes

* The healthcheck currently reports “unexpected” for `openai/gpt-oss-20b` if the model returns non-`ok` text; this is saved in the JSON for troubleshooting.
* Ollama comparisons are relative to the Sophia baseline
* For reproducibility, all scripts accept `--seed` or `--model` flags where applicable; see `python3 <task>.py --help` for details.
