# Phase 07: Backtest and Reporting

## Objective

Implement a transparent long-or-cash backtest using next-bar execution, explicit costs, benchmarks, and reproducible report artifacts.

## Backtest behavior

Implement in `rebuild/src/trader/backtest/engine.py`:

- one BTC position or cash;
- no leverage;
- no shorting;
- signal calculated from information through bar `t`;
- entry or exit at bar `t+1` open;
- buy fill increased by slippage;
- sell fill reduced by slippage;
- fee charged on each side;
- no action when the next bar is unavailable;
- final open position closed using a documented final-price rule.

Do not add stops, take-profit, confidence sizing, cooldowns, or portfolio correlation controls in this phase.

## Metrics

Implement in `rebuild/src/trader/backtest/metrics.py`:

- total return;
- annualized return only when statistically meaningful;
- maximum drawdown;
- Sharpe ratio with a documented hourly annualization assumption;
- trade count;
- win rate;
- average trade return;
- profit factor;
- turnover;
- total fees;
- total slippage;
- exposure percentage.

Handle zero-trade and zero-variance cases without misleading infinities.

## Benchmarks

Implement in `rebuild/src/trader/backtest/benchmarks.py`:

1. Always cash.
2. BTC buy and hold with entry/exit costs.
3. A simple causal momentum rule defined before viewing holdout results.

All strategies must use the same date range and cost assumptions.

## Reporting

Implement `rebuild/src/trader/reporting/writer.py` to create a versioned backtest directory containing:

```text
config.yaml
dataset_metadata.json
model_metadata.json
fold_metrics.csv
predictions.csv
trades.csv
equity.csv
metrics.json
benchmark_metrics.json
summary.md
```

Plots are optional and must not add a heavy default dependency.

## Required tests

- A signal at `t` cannot fill at `t`.
- Buy and sell cost calculations are hand-verifiable.
- No short position is possible.
- Cash never becomes negative.
- Final position accounting is correct.
- Drawdown calculation matches a known equity sequence.
- Zero-trade metrics are valid.
- Benchmarks use identical periods.
- Report output is deterministic except explicitly allowed timestamps/run IDs.

## Boundaries

- Do not tune the probability threshold using holdout results.
- Do not add multi-asset accounting.
- Do not reuse the legacy backtest engine.
- Do not fetch data.

## Acceptance criteria

- A local fixture can produce a complete backtest report.
- All trades are explainable from prediction and fill rows.
- Strategy and benchmark costs are explicit and comparable.
- Report artifacts contain enough information to reproduce the run.
