# Codex Rebuild Master Plan

## Purpose

This directory is an implementation handoff for separate Codex sessions. The objective is to replace the current coupled system with a small, deterministic research pipeline described in:

- `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`
- `docs/codex_rebuild/01_TARGET_ARCHITECTURE.md`

The first complete version is deliberately limited to:

```text
saved BTCUSDT 1h OHLCV
    -> causal feature generation and noise handling
    -> Logistic Regression
    -> chronological validation
    -> long-or-cash backtest after costs
    -> reproducible artifacts and report
```

Reddit, derivatives, tuning, ensembles, shorting, paper trading, and live execution are excluded from the first usable version.

## Project boundary

All rebuild implementation work belongs under the standalone `rebuild/` project:

```text
rebuild/
  pyproject.toml
  configs/
  src/trader/
  tests/
  artifacts/
```

The repository-root application is legacy reference material. Do not add new implementation modules to the root `src/`, `tests/`, `configs/`, or `scripts/` directories.

## Required execution order

Each numbered file is intended to be completed by a fresh Codex instance.

1. `02_BASELINE_AND_SAFETY.md`
2. `03_PACKAGE_SCAFFOLD_AND_CONFIG.md`
3. `04_MARKET_DATA_PIPELINE.md`
4. `05_FEATURES_NOISE_AND_TARGET.md`
5. `06_MODEL_AND_VALIDATION.md`
6. `07_BACKTEST_AND_REPORTING.md`
7. `08_CLI_AND_REPRODUCIBLE_RUN.md`
8. `09_LEGACY_QUARANTINE_AND_DOCS.md`
9. `10_SENTIMENT_EXPERIMENT.md`
10. `11_PAPER_TRADING_GATE.md`

Files 10 and 11 are gated future work. Do not start them merely because earlier files exist; their entry criteria must be satisfied.

## Instructions for every Codex instance

Before editing:

1. Read this file.
2. Read `01_TARGET_ARCHITECTURE.md`.
3. Read the assigned phase file completely.
4. Read `HANDOFF.md`.
5. Inspect the current Git status and preserve unrelated user changes.
6. Inspect code produced by prior phases rather than assuming the target tree is complete.
7. Run rebuild commands from `rebuild/` unless the assigned phase explicitly says otherwise.

While working:

- Stay within the assigned phase.
- Prefer small modules with explicit inputs and outputs.
- Add or update tests with every behavior change.
- Unit tests must not require network access, credentials, or mutable external data.
- Use UTC timestamps and make timezone behavior explicit.
- All rolling, scaling, clipping, and filtering operations must be causal.
- Never initialize Binance, Reddit, or another external client during imports.
- Never enable real trading.
- Do not silently preserve a legacy abstraction merely because tests depend on it.
- Do not delete untracked server artifacts or user data.

Before finishing:

1. Run the phase-specific tests.
2. Run the full clean-package test suite.
3. Run `git diff --check`.
4. Update `HANDOFF.md` with files changed, commands run, results, decisions, and remaining blockers.
5. Report failures honestly. Do not weaken tests simply to obtain a green result.

## Definition of complete for the core rebuild

Phases 02 through 09 are complete when:

- `rebuild/src/trader/` contains the active implementation.
- One documented command can run a deterministic BTC market-only backtest from a saved dataset.
- The command requires no Reddit or Binance credentials.
- Training and holdout data are chronologically separated.
- Feature transformations are fitted on training history only.
- Predictions execute on the next bar.
- Fees and slippage are included.
- Cash, buy-and-hold, and momentum benchmarks are reported.
- Output includes configuration, dataset identity, predictions, trades, equity, metrics, and model metadata.
- Core tests pass without network access.
- The old implementation is clearly quarantined or documented as legacy.
- Heavy unused dependencies are removed from the default installation.

## Non-goals for the core rebuild

- Maximizing profitability.
- Preserving every legacy API.
- Reproducing historical saved-model results.
- AWS deployment.
- Live exchange execution.
- Hyperparameter optimization.
- A generalized multi-asset framework.

The primary success criterion is a trustworthy experimental system, not a sophisticated one.
