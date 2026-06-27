# Phase 06: Model and Chronological Validation

## Objective

Implement a regularized Logistic Regression baseline, expanding walk-forward validation, and a final untouched holdout.

## Required work

1. Implement the baseline estimator in `rebuild/src/trader/modeling/baseline.py`.
2. Use a scikit-learn `Pipeline` so fitted preprocessing is stored with the classifier.
3. Include:
   - imputation only if the Phase 05 contract permits missing model values;
   - `StandardScaler`;
   - `LogisticRegression`;
   - fixed random state where applicable.
4. Expose:
   - `fit`;
   - positive-class probability prediction;
   - binary signal generation from the configured probability threshold;
   - feature names;
   - model metadata.
5. Implement chronological split utilities in `rebuild/src/trader/modeling/validation.py`.
6. Reserve the final configured fraction as a holdout before walk-forward development validation.
7. Within the development period, use expanding windows with configured train, test, and step sizes.
8. Refit a fresh pipeline for every fold.
9. Report fold-level classification metrics and probability diagnostics, but do not optimize parameters in this phase.
10. Implement model save/load in `rebuild/src/trader/modeling/artifacts.py` using `joblib` plus JSON metadata.

## Artifact metadata

Include:

- schema version;
- training date range;
- dataset content hash;
- feature names and order;
- complete relevant configuration;
- library versions;
- target definition;
- probability threshold;
- validation fold metrics;
- creation timestamp.

Loading must validate feature compatibility and reject incompatible artifacts.

## Required tests

- Training data always precedes validation data.
- Holdout rows never appear in development folds.
- Scaler statistics are fitted per training fold.
- Saved and loaded models produce identical probabilities.
- Feature order mismatch is rejected.
- Probability outputs are bounded.
- Single-class training windows fail clearly or are skipped explicitly.
- Repeated training with identical inputs is deterministic within expected numerical tolerance.

## Boundaries

- No Optuna.
- No feature selection.
- No Random Forest, XGBoost, LightGBM, or ensemble.
- Do not inspect holdout performance to choose model parameters.
- No trading simulation yet.

## Acceptance criteria

- A model can be trained solely from the Phase 05 dataset.
- Walk-forward metrics are available per fold.
- A final model artifact can be saved and loaded.
- Holdout isolation is proven by tests.
