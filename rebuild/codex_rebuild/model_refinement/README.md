# Model Refinement Module

## Purpose

This module is a research track between Phase 10 and Phase 11. It exists
because the first market-only baseline produced no long entries with the
default `0.55` probability threshold, and paper trading should not begin until
there is evidence of a stable model candidate.

This module is not a continuation of the numbered rebuild phases. Phase 11
remains the paper-trading gate and must not start until refinement produces a
candidate worth testing in paper mode.

## Scope

Allowed:

- Threshold and probability diagnostics.
- Target strictness and horizon experiments.
- Controlled market feature group experiments.
- Reproducible experiment orchestration.
- Re-evaluation of Phase 10 sentiment only after a better market-only setup is
  selected.

Excluded:

- Shorting.
- Paper trading.
- Live trading.
- Tree-model ensembles, hyperparameter optimization, or transformer models.
- Direct selection from final holdout performance.

## Required execution order

Each file is intended to be completed by a separate Codex session:

1. `01_THRESHOLD_SWEEP_AND_DIAGNOSTICS.md`
2. `02_TARGET_AND_HORIZON_EXPERIMENTS.md`
3. `03_MARKET_FEATURE_GROUPS.md`
4. `04_EXPERIMENT_ORCHESTRATION.md`
5. `05_SENTIMENT_REEVALUATION_GATE.md`
6. `06_MODEL_CLASS_COMPARISON.md`

Do not skip ahead. Each step should produce artifacts or tests that make the
next step more trustworthy.

## Instructions for every refinement instance

Before editing:

1. Read `rebuild/codex_rebuild/00_MASTER_PLAN.md`.
2. Read `rebuild/codex_rebuild/01_TARGET_ARCHITECTURE.md`.
3. Read `rebuild/codex_rebuild/HANDOFF.md`.
4. Read this `README.md`.
5. Read the assigned model-refinement file completely.
6. Inspect current Git status and preserve unrelated user changes.
7. Run commands from `rebuild/` unless the assigned file says otherwise.

While working:

- Keep implementation under `rebuild/`.
- Do not change root legacy code.
- Do not start Phase 11.
- Keep the long-or-cash backtest as the reference execution model.
- Keep Logistic Regression as the model class unless a later refinement doc
  explicitly changes that.
- Preserve chronological train/validation/holdout boundaries.
- Select thresholds, targets, horizons, and feature groups from development
  folds only.
- Treat final holdout as confirmation, not selection.

Before finishing:

1. Run the phase-specific tests.
2. Run `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
   `rebuild/`.
3. Run `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
   src/trader` from `rebuild/`.
4. Run `git diff --check` from the repository root.
5. Update `rebuild/codex_rebuild/HANDOFF.md` with files changed, commands run,
   decisions, results, and remaining blockers.

## Success criteria

The module succeeds only if it produces a reproducible candidate that:

- beats cash, buy-and-hold, and momentum on development folds after costs;
- has acceptable turnover and drawdown;
- is confirmed on final holdout without being selected from holdout;
- has exact config, feature list, threshold, target, fold metrics, backtest
  metrics, and benchmark metrics saved.

If no candidate passes those criteria, Phase 11 remains blocked. That is a
valid outcome.
