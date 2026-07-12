# Rebuild User Testing Guide

This guide is for manually checking that each completed rebuild phase works on
your machine. It should be updated as new phases are implemented.

Run all rebuild commands from the standalone project:

```bash
cd rebuild
```

Use the rebuild environment, not the legacy root project:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
```

Expected result after Phase 08:

```text
78 passed
```

The current suite also reports 24 `joblib`/NumPy deprecation warnings from
model artifact save/load tests and the offline workflow test. Those warnings
are known and do not indicate a Phase 08 failure.

## What Codex Tests Versus What You Should Test

Codex runs the required automated tests before finishing each phase and records
the commands in `rebuild/codex_rebuild/HANDOFF.md`.

You should still run smoke tests after each phase to verify:

- your local dependencies resolve;
- the CLI entry point works in your shell;
- generated artifacts appear where expected;
- commands are using the standalone `rebuild/` package, not legacy root code.

## Current Phase Status

| Phase | Status | User-facing check |
|---|---:|---|
| 02 Baseline and safety | Complete | Rebuild tests run independently from legacy tests. |
| 03 Package scaffold and config | Complete | `crypto-trader --help` works and config loads. |
| 04 Market data pipeline | Complete | `collect-market` can save BTCUSDT 1h raw data. |
| 05 Features, noise, and target | Complete | Feature, clipping, and target unit tests pass. |
| 06 Model and validation | Complete | Logistic model, validation, and artifact tests pass. |
| 07 Backtest and reporting | Complete | Backtest, benchmark, metric, and report tests pass. |
| 08 CLI and reproducible run | Complete | End-to-end offline baseline CLI test passes. |
| 09 Legacy quarantine and docs | Not started | No manual check yet. |

Phases 10 and 11 are gated future work. Do not test them until the handoff says
their entry criteria are satisfied.

## Baseline Commands

Run the full standalone rebuild suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
```

Run the explicit unit plus integration suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit tests/integration
```

Check imports and bytecode compilation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q src/trader
```

Check the CLI entry point:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help
```

After Phase 08, the help output should list:

```text
collect-market
build-dataset
train
backtest
run-baseline
```

All five commands are operational after Phase 08. `collect-market` is the only
command that may fetch external data; `build-dataset`, `train`, `backtest`, and
`run-baseline` operate from saved local artifacts.

## Phase 02 Checks: Standalone Test Boundary

Run from `rebuild/`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/test_import.py
```

Expected result:

```text
1 passed
```

This confirms the standalone package imports without running legacy root tests.

## Phase 03 Checks: Package and Config

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/test_config.py tests/unit/test_cli.py
```

Run a direct config smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "from trader.config import load_config; c = load_config('configs/baseline.yaml'); print(c.data.symbol, c.data.interval)"
```

Expected output:

```text
BTCUSDT 1h
```

Check the train command parser:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader train --help
```

Expected result after Phase 08:

- exit status is zero;
- help text includes `--dataset` and `--output-dir`.

## Phase 04 Checks: Market Data Pipeline

Run the offline market-data tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/data tests/integration/test_market_fixture_storage.py
```

The automated tests cover schema validation, fake-client collection, fixture
loading, Parquet storage, metadata sidecars, and deterministic hashes.

To manually collect a small live sample from Binance US:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader collect-market \
  --start 2026-01-01T00:00:00Z \
  --end 2026-01-02T00:00:00Z
```

Expected result:

- the command prints the number of saved rows;
- it prints a `metadata content_hash`;
- a Parquet file is created in `artifacts/datasets/`;
- a matching `.metadata.json` sidecar is created next to it.

Generated files under `rebuild/artifacts/` are intentionally ignored by Git.

To read a saved dataset, replace `<dataset>` with the Parquet path printed by
the command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "from trader.data.storage import read_market_dataset; data = read_market_dataset('<dataset>'); print(data.head()); print(len(data))"
```

Expected result:

- the dataset loads without network access;
- columns are `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`;
- timestamps are UTC;
- symbol is `BTCUSDT`.

## Phase 05 Checks: Features, Noise, and Target

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/features
```

Run a fixture smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "import pandas as pd; from trader.config import load_config; from trader.features.market import build_feature_dataset, MODEL_FEATURE_COLUMNS; data = pd.read_csv('tests/fixtures/btcusdt_1h.csv'); features = build_feature_dataset(data, load_config('configs/baseline.yaml')); print(len(features), len(MODEL_FEATURE_COLUMNS)); print(features[['timestamp', 'target']].tail(2).to_string(index=False))"
```

Expected result:

- the command prints the fixture row count;
- model feature columns are created;
- early rows may contain warm-up missing values;
- the final target can be blank because no future close exists.

## Phase 06 Checks: Model and Validation

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/modeling
```

Expected result after Phase 07:

```text
14 passed
```

This covers deterministic Logistic Regression training, chronological
walk-forward validation, final-holdout isolation, model artifact save/load, and
feature-order compatibility checks.

## Phase 07 Checks: Backtest and Reporting

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/backtest tests/unit/reporting
```

Expected result:

```text
11 passed
```

These tests verify:

- a signal at bar `t` fills at bar `t+1` open;
- buy and sell fees and slippage are hand-verifiable;
- no short position is possible;
- cash never becomes negative;
- final open positions are force-closed at the final close;
- drawdown and zero-trade metrics are valid;
- benchmarks use the same period as the strategy;
- report output is deterministic.

To run a small manual backtest smoke test without network access:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "import pandas as pd; from trader.config import BacktestConfig, CostsConfig; from trader.backtest.engine import run_long_cash_backtest; from trader.backtest.metrics import calculate_backtest_metrics; market = pd.read_csv('tests/fixtures/btcusdt_1h.csv'); predictions = pd.DataFrame({'timestamp': market['timestamp'], 'signal': [0, 1, 1, 0, 0, 1]}); result = run_long_cash_backtest(market, predictions, backtest_config=BacktestConfig(initial_capital=10000.0), costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005)); print(result.trades[['side', 'signal_timestamp', 'fill_timestamp', 'reason']].to_string(index=False)); print(calculate_backtest_metrics(result.equity, result.trades)['total_return'])"
```

Expected result:

- trade fill timestamps are after their signal timestamps, except `final_close`
  which uses the final bar's close by design;
- only `buy` and `sell` sides appear;
- a numeric total return is printed.

## Phase 08 Checks: CLI and Reproducible Run

Run the focused Phase 08 automated checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/test_cli.py tests/integration/test_offline_baseline_workflow.py
```

Expected result:

```text
4 passed
```

The integration test creates a deterministic saved market dataset from the
local BTC fixture in a temporary directory, runs `crypto-trader run-baseline`
twice, verifies dataset/model/report artifacts, reads predictions, metrics,
and benchmark files, compares predictions and metrics across runs, and asserts
that no market collection path is used.

Manual end-to-end baseline smoke test with a saved raw market dataset:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader run-baseline \
  --config configs/baseline.yaml \
  --market-data artifacts/datasets/<raw-market-dataset>.parquet
```

Expected result:

- a feature dataset is created under `artifacts/datasets/`;
- a model artifact is created under `artifacts/models/`;
- a report bundle is created under `artifacts/backtests/`;
- stdout prints strategy, cash, buy-and-hold, and momentum metric comparisons;
- rerunning into the same explicit artifact paths fails instead of silently
  overwriting existing output.

To run the commands separately:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader build-dataset \
  --config configs/baseline.yaml \
  --input artifacts/datasets/<raw-market-dataset>.parquet
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader train \
  --config configs/baseline.yaml \
  --dataset artifacts/datasets/<feature-dataset>.parquet
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader backtest \
  --config configs/baseline.yaml \
  --dataset artifacts/datasets/<feature-dataset>.parquet \
  --model artifacts/models/<model-artifact>.joblib
```

## What Not To Use

Do not use the root-level legacy commands as rebuild validation. The root
project still has known legacy test failures and is not the active rebuild
implementation.

Do not use these root server artifacts as test fixtures:

```text
master_reddit_server.csv
reddit_archive_server.csv
trade_logs_server.log
```

They are user-owned reference artifacts and should remain untouched unless you
explicitly decide otherwise.

## After Each Future Phase

After Codex completes a phase, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help
```

Then run the one user-facing command that the phase made operational. Check
`HANDOFF.md` for the exact command and expected result.

If the command produces artifacts, confirm that:

- the files appear under `rebuild/artifacts/`;
- metadata exists where documented;
- rerunning from saved artifacts does not require network access unless the
  command is explicitly a collection command.
