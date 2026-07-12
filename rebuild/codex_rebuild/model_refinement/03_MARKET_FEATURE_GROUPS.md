# Model Refinement 03: Market Feature Groups

## Entry criteria

Start only when:

- Target and horizon configuration is explicit and tested.
- The current baseline feature set is still available as `baseline`.
- Phase 11 remains blocked.

## Objective

Add controlled causal market feature groups that can be tested independently
instead of adding all possible features at once.

## Required structure

Update:

```text
rebuild/src/trader/config.py
rebuild/src/trader/features/market.py
rebuild/configs/baseline.yaml
```

Add or update tests:

```text
rebuild/tests/unit/features/test_market_features.py
rebuild/tests/unit/test_config.py
```

## Configuration change

Extend feature config:

```yaml
features:
  enabled_groups:
    - baseline
```

The default must preserve current behavior by enabling only `baseline`.

Allowed groups:

```text
baseline
trend
volatility
volume
calendar
momentum_reversal
```

Reject unknown groups and duplicate groups.

## Feature groups

Keep existing baseline features:

```text
return_1h
return_6h
return_24h
realized_volatility_24h
volume_ratio_24h
rsi_14
```

Add these groups:

```text
trend:
- sma_24_distance
- sma_168_distance
- ema_12_26_distance

volatility:
- high_low_range_1h
- realized_volatility_change_24h
- volatility_percentile_168h

volume:
- volume_zscore_24h
- volume_zscore_168h
- dollar_volume_zscore_168h

calendar:
- hour_sin
- hour_cos
- day_of_week_sin
- day_of_week_cos
- weekend_flag

momentum_reversal:
- return_3h
- return_12h
- return_72h
- drawdown_from_24h_high
- distance_from_24h_low
```

## Feature rules

- All features must be causal at bar `t`.
- Rolling statistics must use only rows through bar `t`, never future rows.
- Raw OHLCV must not become model features.
- Non-calendar raw features receive clipped columns and missingness flags.
- Calendar features are bounded deterministic encodings and do not need MAD
  clipping.
- `MODEL_FEATURE_COLUMNS` must reflect the enabled groups for the current
  config, so replace the fixed global model-feature tuple with a config-aware
  helper while preserving a baseline default for existing callers.

## Tests

Unit tests must cover:

- baseline-only config preserves the current feature columns;
- each group can be enabled with baseline;
- unknown and duplicate groups are rejected;
- future row mutation does not change earlier feature rows;
- warm-up missingness flags are set before imputation;
- calendar encodings are bounded and deterministic;
- model metadata records exact feature names for enabled groups.

## Acceptance criteria

- Focused tests pass.
- Full rebuild tests pass.
- Existing baseline artifacts can still be produced with default config.
- New feature groups are available for controlled experiments but not enabled
  by default.
