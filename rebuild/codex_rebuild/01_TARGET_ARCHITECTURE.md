# Target Architecture

## Design rules

The replacement system must separate:

1. external data collection;
2. deterministic dataset construction;
3. model training;
4. chronological evaluation;
5. backtest simulation;
6. artifact reporting;
7. future paper execution.

Training must not occur inside an execution loop. External data must not be fetched implicitly by a backtest. A saved dataset must be sufficient to reproduce a result.

## Target directory structure

Build the new implementation as a standalone project alongside the current code:

```text
rebuild/
  pyproject.toml
  README.md
  src/
    trader/
      __init__.py
      cli.py
      config.py
      data/
        __init__.py
        market.py
        schemas.py
        storage.py
      features/
        __init__.py
        market.py
        noise.py
        target.py
      modeling/
        __init__.py
        baseline.py
        validation.py
        artifacts.py
      backtest/
        __init__.py
        engine.py
        benchmarks.py
        metrics.py
      reporting/
        __init__.py
        writer.py
      sentiment/
        __init__.py
        # Added only after the market-only baseline is accepted.
      paper/
        __init__.py
        # Added only after the paper-trading gate is accepted.
  configs/
    baseline.yaml
  tests/
    unit/
      data/
      features/
      modeling/
      backtest/
    integration/
    fixtures/
      btcusdt_1h.csv
  artifacts/
    datasets/
    models/
    backtests/
    paper_state/

docs/
  codex_rebuild/
```

Generated artifact directories should be ignored by Git except for small intentional fixtures and documentation examples.

## Core contracts

### Market dataset

Canonical columns:

```text
timestamp
symbol
open
high
low
close
volume
```

Requirements:

- `timestamp` is UTC.
- One row exists per closed hourly candle.
- Rows are unique by `(timestamp, symbol)`.
- Prices are positive.
- Volume is non-negative.
- Rows are sorted by timestamp and symbol.
- Incomplete candles are rejected before storage.

### Feature dataset

Initial feature columns:

```text
return_1h
return_6h
return_24h
realized_volatility_24h
volume_ratio_24h
rsi_14
```

Noise-related columns may include:

```text
<feature>_clipped
<feature>_missing
noise_band
```

Raw OHLCV may remain in the dataset for backtest execution but must not automatically become model features.

### Target

The initial binary target is:

```text
1 if next_return > round_trip_cost + volatility_multiplier * current_volatility
0 otherwise
```

The exact return horizon, volatility unit, and cost calculation must be documented and tested.

### Model

The baseline model is a scikit-learn pipeline:

```text
imputation if required
    -> StandardScaler
    -> LogisticRegression
```

Fitted preprocessing belongs inside the model artifact. Scaling must never use validation or holdout observations.

### Backtest

The core strategy is long or cash:

- Generate a signal after bar `t` closes.
- Execute at bar `t+1` open.
- Never short.
- Use one position at a time for BTC.
- Apply explicit entry and exit fees and slippage.
- Force-close at the final available bar for complete accounting.

## Configuration

`rebuild/configs/baseline.yaml` should contain only values used by the core pipeline:

```yaml
data:
  symbol: BTCUSDT
  interval: 1h

features:
  volatility_window: 24
  volume_window: 24
  rsi_window: 14
  clipping_window: 168
  clipping_mad_multiplier: 8.0

target:
  horizon_bars: 1
  volatility_multiplier: 0.10

model:
  probability_threshold: 0.55
  regularization_c: 1.0

validation:
  minimum_train_bars: 1000
  test_bars: 168
  step_bars: 168
  final_holdout_fraction: 0.20

costs:
  fee_per_side: 0.001
  slippage_per_side: 0.0005

backtest:
  initial_capital: 10000.0
```

Defaults can change through evidence, but unused configuration is not permitted.

## Dependency target

Core runtime dependencies should be approximately:

- `numpy`
- `pandas`
- `pyarrow`
- `PyYAML`
- `scikit-learn`
- `joblib`

Data collection may add a lightweight HTTP dependency. Reddit, tree-model, visualization, and transformer dependencies must not be required for the market-only baseline.

Development dependencies should include:

- `pytest`
- optional linting/type-checking tools only if configured and used

## Legacy transition

Do not move all existing code immediately.

1. Build and test `rebuild/src/trader/`.
2. Switch documented commands to the new package.
3. Prove the deterministic end-to-end run.
4. Quarantine the old system under `legacy/` or remove it using Git history.
5. Do not mix old and new imports.

The new package must not import from the repository-root legacy packages:

- `src.analysis`
- `src.models`
- `src.features`
- `src.risk`
- `src.execution`

Small algorithms may be reimplemented cleanly with tests, but legacy modules must not become hidden dependencies.
