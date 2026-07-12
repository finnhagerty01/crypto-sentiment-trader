# Model Refinement 04: Experiment Orchestration

## Entry criteria

Start only when:

- Threshold sweep support exists.
- Target and horizon experiments are configurable.
- Feature groups are configurable.
- Phase 11 remains blocked.

## Objective

Add one reproducible experiment runner that compares named model-refinement
variants from a saved market dataset and writes deterministic comparison
artifacts.

## Required structure

Update:

```text
rebuild/src/trader/cli.py
rebuild/src/trader/workflow.py
```

Suggested new module:

```text
rebuild/src/trader/modeling/experiments.py
```

Add tests:

```text
rebuild/tests/integration/test_experiment_grid.py
rebuild/tests/unit/test_cli.py
```

## CLI

Add:

```bash
crypto-trader run-experiment-grid \
  --market-data artifacts/datasets/raw.parquet \
  --experiment-config configs/experiments/market_signal_grid.yaml \
  --output-dir artifacts/experiments \
  --run-id market-refinement-001
```

The command must never fetch external data.

## Experiment config

Use named variants:

```yaml
variants:
  - name: baseline_h1_roundtrip_v010
    config: configs/baseline.yaml

  - name: h6_oneway_v005_trend
    config: configs/baseline.yaml
    overrides:
      target.horizon_bars: 6
      target.cost_buffer: one_way
      target.volatility_multiplier: 0.05
      features.enabled_groups:
        - baseline
        - trend
```

Rules:

- Variant names must be unique path-safe identifiers.
- Overrides must target known config fields.
- Each variant starts from a full config file, then applies overrides.
- The resolved config for each variant must be saved.

## Required output

Write:

```text
artifacts/experiments/<run_id>/
  experiment_config.yaml
  variant_summary.csv
  target_distribution.csv
  threshold_summary.csv
  fold_metrics.csv
  holdout_metrics.csv
  benchmark_metrics.csv
  variants/<variant_name>/resolved_config.yaml
```

## Ranking policy

Rank variants using development folds only:

1. Exclude variants with no selected threshold.
2. Exclude variants with negative median development total return.
3. Exclude variants that do not beat cash on median development total return.
4. Rank by median development total return.
5. Break ties by lower median max drawdown.
6. Break remaining ties by lower median turnover.

Final holdout metrics are reported but not used for rank selection.

## Tests

Integration tests must cover:

- command runs from a local saved market fixture;
- no network or credentials are required;
- variant configs are resolved and saved;
- unknown overrides fail clearly;
- duplicate variant names fail clearly;
- final holdout metrics are present but separate from development ranking;
- output is deterministic for a fixed `run_id`.

## Acceptance criteria

- Focused tests pass.
- Full rebuild tests pass.
- One command can compare threshold, target, horizon, and feature-group
  variants from a saved market dataset.
- The experiment report identifies whether a candidate is good enough to
  consider for Phase 11.
