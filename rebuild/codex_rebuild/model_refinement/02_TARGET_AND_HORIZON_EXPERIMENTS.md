# Model Refinement 02: Target and Horizon Experiments

## Entry criteria

Start only when:

- Threshold sweep diagnostics are implemented and recorded in `HANDOFF.md`.
- The current target distribution and threshold behavior are understood.
- Phase 11 remains blocked.

## Objective

Make target strictness explicit and test whether the one-hour round-trip-cost
target is too sparse for the baseline model.

## Required structure

Update:

```text
rebuild/src/trader/config.py
rebuild/src/trader/features/target.py
rebuild/configs/baseline.yaml
```

Add or update tests:

```text
rebuild/tests/unit/features/test_target.py
rebuild/tests/unit/test_config.py
```

## Configuration change

Extend target config to include:

```yaml
target:
  horizon_bars: 1
  cost_buffer: round_trip
  volatility_multiplier: 0.10
```

Allowed `cost_buffer` values:

```text
none
one_way
round_trip
```

Backwards compatibility is not required for generated artifacts, but the
default `baseline.yaml` must preserve the current target behavior by setting
`cost_buffer: round_trip`.

## Target definitions

Use:

```text
next_return = close[t + horizon_bars] / close[t] - 1
```

Cost buffer:

```text
none       = 0
one_way    = fee_per_side + slippage_per_side
round_trip = 2 * (fee_per_side + slippage_per_side)
```

Noise band:

```text
noise_band = cost_buffer + volatility_multiplier * realized_volatility_24h
```

Target:

```text
target = 1 when next_return > noise_band
target = 0 when next_return <= noise_band
target = NA when future close or volatility is unavailable
```

## Experiment grid

The later orchestration step should evaluate:

```text
horizon_bars: 1, 3, 6, 12, 24
cost_buffer: none, one_way, round_trip
volatility_multiplier: 0.00, 0.05, 0.10
```

This step only needs the implementation, validation, and target-distribution
helper needed to support that grid.

## Required diagnostics

Add a helper that reports, for a feature dataset:

- row count;
- labeled row count;
- positive count;
- negative count;
- positive rate;
- unlabeled count;
- first and last labeled timestamp.

## Tests

Unit tests must cover:

- hand-calculated labels for each `cost_buffer`;
- `horizon_bars` uses the correct future close;
- rows near the end become unlabeled when no future close exists;
- `baseline.yaml` keeps the previous round-trip behavior;
- invalid `cost_buffer` values are rejected;
- positive-rate diagnostics are correct.

## Acceptance criteria

- Focused tests pass.
- Full rebuild tests pass.
- Existing baseline behavior is preserved by default.
- Target distribution diagnostics are available for experiment comparison.
