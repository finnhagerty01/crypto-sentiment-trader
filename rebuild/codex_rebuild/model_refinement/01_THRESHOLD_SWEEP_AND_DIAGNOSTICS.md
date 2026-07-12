# Model Refinement 01: Threshold Sweep and Diagnostics

## Entry criteria

Start only when:

- Phase 08 baseline artifacts can be reproduced.
- Phase 10 is complete or explicitly deferred.
- The current market-only run shows low probabilities or no trades at the
  default threshold.
- The user wants model refinement before paper trading.

## Objective

Determine whether the existing Logistic Regression model has any usable signal
below the current `0.55` probability threshold, without changing target,
features, model class, costs, or backtest behavior.

## Required structure

Add threshold analysis code under:

```text
rebuild/src/trader/modeling/
```

Suggested module:

```text
thresholds.py
```

Add tests under:

```text
rebuild/tests/unit/modeling/test_thresholds.py
```

Do not modify the long-or-cash backtest engine except to add reusable helper
interfaces if absolutely required and covered by tests.

## Threshold candidates

Evaluate exactly:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
```

## Required behavior

- Refit the existing Logistic Regression model once per chronological
  development fold.
- Generate validation probabilities once per fold.
- Evaluate every threshold against those same probabilities.
- Convert probabilities to long/cash signals using `probability >= threshold`.
- Backtest each threshold with the existing long-or-cash engine.
- Report classification metrics and trading metrics per fold and threshold.
- Aggregate development-fold performance into a threshold summary.
- Select a threshold from development folds only.
- Evaluate final holdout only after a threshold has been selected.

## Selection policy

Use a deterministic conservative policy:

1. Exclude thresholds with zero trades across all development folds.
2. Exclude thresholds with negative median development total return.
3. Exclude thresholds that underperform cash on median development total
   return.
4. Among remaining thresholds, choose the one with the highest median
   development total return.
5. Break ties by lower median max drawdown.
6. Break remaining ties by lower turnover.
7. If no threshold survives, report no selected threshold and keep Phase 11
   blocked.

Do not tune this policy on holdout.

## Required artifacts

When wired into workflow or CLI, write:

```text
threshold_fold_metrics.csv
threshold_summary.csv
selected_threshold.json
holdout_threshold_metrics.json
```

It is acceptable in this step to expose the core function first and add CLI
wiring in a later orchestration step.

## Tests

Unit tests must cover:

- thresholds are evaluated from the same probability vector;
- threshold comparison is deterministic;
- zero-trade thresholds are excluded from selection;
- final holdout is not used during threshold selection;
- selected threshold is `None` when all thresholds fail the policy;
- backtest signals use `>= threshold`.

## Acceptance criteria

- Focused tests pass.
- Full rebuild tests pass.
- The output can distinguish "no signal" from "threshold too strict."
- `HANDOFF.md` records whether a lower threshold produced development-fold
  evidence worth carrying forward.
