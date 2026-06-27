# Phase 08: CLI and Reproducible Baseline Run

## Objective

Connect the completed components through explicit CLI commands and prove one deterministic end-to-end baseline run.

## Required commands

Implement:

```text
crypto-trader collect-market
crypto-trader build-dataset
crypto-trader train
crypto-trader backtest
crypto-trader run-baseline
```

### `collect-market`

- May access the network.
- Saves validated raw data and metadata.
- Requires explicit date range or documented defaults.

### `build-dataset`

- Reads a saved raw dataset.
- Writes a feature dataset and metadata.
- Never fetches data.

### `train`

- Reads a saved feature dataset.
- Runs development walk-forward validation.
- Saves a fitted model artifact.
- Does not run a trading loop.

### `backtest`

- Reads saved data and a model artifact.
- Evaluates the final holdout or an explicit period.
- Writes the complete report bundle.

### `run-baseline`

- Orchestrates build, train, and backtest from an already saved market dataset.
- Must not fetch data unless an explicit flag requests collection.
- Prints output artifact locations and a concise metric comparison.

## Reproducibility requirements

- Input dataset is identified by content hash.
- Configuration is copied into artifacts.
- Random seeds are fixed.
- Re-running with the same data and configuration yields the same predictions and metrics within numerical tolerance.
- Output directories must not overwrite prior runs silently.
- Failures leave a clear error and no artifact falsely marked complete.

## Documentation

Rewrite the root `README.md` around the new core workflow:

1. Project purpose and non-financial-advice notice.
2. Supported baseline scope.
3. Installation.
4. Test commands.
5. Offline fixture/demo command.
6. Real data collection command.
7. Dataset build, train, and backtest commands.
8. Artifact layout.
9. Explicit statement that live trading is unavailable.
10. Link to the repository review and rebuild documentation.

## Required integration test

Run the entire offline workflow against `rebuild/tests/fixtures/btcusdt_1h.csv` in a temporary directory and verify:

- dataset artifact exists;
- model artifact exists;
- report bundle exists;
- predictions are deterministic;
- metrics and benchmark files are readable;
- no network or credentials are used.

## Acceptance criteria

- A new developer can reproduce the offline baseline from the README.
- The end-to-end integration test passes.
- The core workflow contains no imports from repository-root legacy code.
- `run-baseline` works from saved data with one command.
