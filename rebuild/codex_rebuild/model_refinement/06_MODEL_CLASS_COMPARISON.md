# Model Refinement 06: Model Class Comparison

## Entry criteria

Start only after Model Refinement 05 determines whether Phase 10 sentiment adds
stable incremental signal. Do not use this phase to rescue a weak candidate by
changing many controls at once.

## Objective

Test whether the existing Logistic Regression model is underfitting nonlinear
crypto dynamics under the already selected data, target, feature, validation,
threshold, cost, and long-or-cash backtest controls.

## Fixed controls

Every model class must use identical:

- saved market dataset;
- sentiment decision from Refinement 05;
- feature and target data;
- chronological development folds;
- final holdout split;
- threshold policy;
- costs;
- long-or-cash backtest behavior.

Holdout is confirmation only. It must not be used for ranking or model-class
selection.

## Candidate models

1. Baseline: existing scikit-learn Logistic Regression.
2. Scikit-learn Random Forest with a tiny predefined grid.
3. Scikit-learn HistGradientBoosting or GradientBoosting with a tiny predefined
   grid.
4. Optional XGBoost only if the dependency is explicitly approved before adding
   it.

## Constraints

- No broad hyperparameter optimization.
- No Optuna or automated search.
- No paper trading or live trading.
- No shorting.
- No selecting from holdout performance.
- No model accepted unless it improves development folds and confirms on
  holdout without materially worse drawdown or turnover.
- Probability calibration must be explicit before probability thresholds are
  compared across model classes.
- Feature importance and diagnostics should be reported, but not used as
  selection criteria.

## Required artifacts

Write a reproducible run directory containing:

- `model_class_summary.csv`
- `threshold_summary.csv`
- `fold_metrics.csv`
- `holdout_metrics.csv`
- `benchmark_metrics.csv`
- `model_diagnostics.csv`
- `model_class_decision.json`
- per-candidate resolved config and metadata

## Required tests

Tests must verify:

- identical chronological splits across model classes;
- identical feature and target data across model classes;
- threshold selection from development folds only;
- holdout is not used for ranking;
- deterministic results for fixed seeds;
- XGBoost is skipped unless the dependency is explicitly enabled.

## Acceptance criteria

Refinement 06 is complete only when the comparison artifacts are reproducible
and the decision file clearly states whether a non-logistic model has earned
future consideration. Phase 11 remains blocked unless the selected candidate
also satisfies the paper-trading gate and the user explicitly approves
proceeding.
