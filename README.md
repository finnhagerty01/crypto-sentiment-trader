# Crypto Sentiment Trader

Deterministic research project for testing a small BTCUSDT market-only trading baseline. This repository is experimental software and is not financial advice.

The active rebuild lives under `rebuild/`. The repository-root application has
been quarantined under `legacy/` as historical reference material.

## Supported Baseline

The current supported workflow is deliberately narrow:

```text
saved BTCUSDT 1h OHLCV
  -> causal market features and noise flags
  -> Logistic Regression
  -> chronological validation
  -> long-or-cash backtest after fees and slippage
  -> reproducible artifacts and report
```

Reddit sentiment, derivatives data, shorting, ensembles, paper trading, and live trading are unavailable in the rebuild baseline.

## Installation

Install the standalone rebuild environment from `rebuild/`:

```bash
cd rebuild
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

## Tests

Run the replacement suite from `rebuild/`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
```

The quarantined legacy root suite is retained under `legacy/runtime/tests/`
for reference and has known failures documented in
`docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`.

## Offline Demo

The integration test demonstrates the complete offline workflow from the local BTC fixture:

```bash
cd rebuild
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/integration/test_offline_baseline_workflow.py
```

For manual runs, first collect or create a saved market Parquet dataset with its `.metadata.json` sidecar, then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader run-baseline \
  --config configs/baseline.yaml \
  --market-data artifacts/datasets/btcusdt_1h_raw_example.parquet
```

## Real Data Collection

`collect-market` is the only baseline command that may access the network. It requires an explicit UTC date range:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader collect-market \
  --config configs/baseline.yaml \
  --start 2025-01-01T00:00:00Z \
  --end 2025-04-01T00:00:00Z \
  --output-dir artifacts/datasets
```

## Separate Commands

Build a feature dataset from a saved raw dataset:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader build-dataset \
  --config configs/baseline.yaml \
  --input artifacts/datasets/btcusdt_1h_raw_example.parquet
```

Train and save the baseline model:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader train \
  --config configs/baseline.yaml \
  --dataset artifacts/datasets/btcusdt_1h_features_example.parquet
```

Backtest a saved model on the final holdout:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader backtest \
  --config configs/baseline.yaml \
  --dataset artifacts/datasets/btcusdt_1h_features_example.parquet \
  --model artifacts/models/btcusdt_1h_logistic_example.joblib
```

## Artifact Layout

Generated files stay under `rebuild/artifacts/`:

```text
artifacts/
  datasets/   raw and feature Parquet files plus metadata/config sidecars
  models/     joblib model artifacts plus metadata sidecars
  backtests/  report bundles with predictions, trades, equity, metrics, benchmarks, and summary
```

Commands refuse to overwrite existing artifact paths.

## Documentation

See:

- `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`
- `legacy/README.md`
- `rebuild/codex_rebuild/00_MASTER_PLAN.md`
- `rebuild/codex_rebuild/01_TARGET_ARCHITECTURE.md`
- `rebuild/codex_rebuild/HANDOFF.md`
