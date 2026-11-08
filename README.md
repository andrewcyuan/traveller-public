# Traveller

Minimal inference and evaluation code for Traveller's Delta retrieval method on LongMemEval V2. No training code, adapters, saved runs, or datasets are included.

This repository contains only the winning state-search strategy, with high reasoning effort and a fixed Qwen reader.

## Install

Use `uv` with Python 3.12 on Linux or macOS, with `curl` and SQLite FTS5 available.

```sh
uv sync --locked
```

## Get the data

```sh
uv run --locked python scripts/prepare_data.py --tier small --output data/small
```

This downloads the pinned public dataset, verifies official checksums, and builds the native trajectory store. Allow at least 15 GiB of free space; screenshot archives include more trajectories than Small needs. Keep `data/small-downloads` because the normalized store can refer to its images. Downloads resume after interruption.

For Medium, use `--tier medium --output data/medium`. Use separate directories for each tier. Medium retrieval uses each question's exact official history subset.

## Start an evaluation

Set an Openrouter key in the `.env`. The default protocol uses `openai/gpt-6-luna` for retrieval, `typesafe/jev-1.13` for ranking, `qwen/qwen3.5-9b` for the reader, and `openai/gpt-5.2` for model-graded questions.

The highest score I achieved was using Gemini 3.8 flash high. I wrote a separate adapter to the ai studio gemini CLI for this, but you should be able to use openrouter too. (openrouter was being unreliable for gemini while I was developing)

```sh
export OPENROUTER_API_KEY='your-key'

# Validate inputs and build search indexes without model calls.
uv run --locked python -m delta_lme prepare \
  --data data/small --output runs/state-search \
  --tier small --workers 2 --budget-usd 15

# Run retrieval, then answering, then grading.
uv run --locked python -m delta_lme run \
  --data data/small --output runs/state-search \
  --tier small --workers 2 --budget-usd 15
```

The budget API spending cap is derived from monitoring Openrouter spend. Using luna, my runs generally cost around $12, using gemini, around $76, and using opus, around $300.

For Medium, change both `--data` and `--tier`, and use a fresh output directory.

Optional direct-provider routes require `uv sync --locked --extra providers`; use `uv run --locked --extra providers python` for these runs. Set `DELTA_CONTROLLER_MODEL=gemini-3.8-flash`, `DELTA_CONTROLLER_PROVIDER=google`, and `GEMINI_API_KEY` for Gemini retrieval. Set `DELTA_READER_PROVIDER=together` and `TOGETHER_API_KEY` for Together Qwen. Set these before both preparation and execution. OpenRouter is still required for ranking and grading. These settings change the protocol; record them with any score. Together reader requests retain the first 10 images and all text.

## Outputs

Open `runs/state-search/run-viewer.html` when the run completes. The directory also contains `summary.json`, `research-journal.md`, the state-search audit, and the viewer integrity manifest. Rebuild the viewer offline with:

```sh
uv run --locked python -m evals.run_viewer runs/state-search/viewer-runs.jsonl.gz \
  --output runs/state-search/run-viewer.html --expected-runs 451
```

Viewer generation makes no model calls.

## Source

`delta_lme/` contains the runner, state-search retrieval, API clients, and budget handling. `strategies/delta/` contains prompts and source inspection tools. `evals/run_viewer/` contains offline reporting. `vendor/LongMemEval-V2/` contains only the native modules required at runtime. See [third-party notices](attribution.md) for source pins and the vendor modification.
