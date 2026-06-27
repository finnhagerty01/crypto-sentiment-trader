# Phase 05: Features, Noise Handling, and Target

## Objective

Build the causal six-feature dataset, robust noise handling, and the volatility-aware target. This phase must make leakage prevention explicit and testable.

## Required features

Implement in `rebuild/src/trader/features/market.py`:

- `return_1h`;
- `return_6h`;
- `return_24h`;
- `realized_volatility_24h`;
- `volume_ratio_24h`;
- `rsi_14`.

Keep OHLCV columns for later execution accounting, but expose one explicit model-feature list.

## Noise handling

Implement in `rebuild/src/trader/features/noise.py`:

1. Causal rolling median/MAD clipping for selected heavy-tailed features.
2. Missingness flags before imputation.
3. Clear warm-up behavior.
4. No centered rolling windows.
5. No global statistics fitted on the full dataset.

The clipping limits for row `t` must use observations strictly before `t`. Shift rolling statistics by one bar and test that changing a future row cannot alter an earlier transformed row.

If MAD is zero or unavailable, use a documented fallback that does not create infinities.

## Target

Implement in `rebuild/src/trader/features/target.py`:

```text
next_return = close[t + horizon] / close[t] - 1
round_trip_cost = 2 * (fee_per_side + slippage_per_side)
noise_band[t] = round_trip_cost
                + volatility_multiplier * realized_volatility_24h[t]
target[t] = 1 if next_return > noise_band[t] else 0
```

Document whether `realized_volatility_24h` is hourly standard deviation or a horizon-scaled value. Keep units consistent.

The final rows without a future label must be excluded from training but may remain valid inference rows.

## Dataset builder

Create one pure function that accepts validated OHLCV plus configuration and returns:

- execution columns;
- raw feature columns if retained;
- clipped model features;
- missingness flags;
- `noise_band`;
- `next_return`;
- `target`.

It must not fetch data or write artifacts.

## Required tests

- Known returns from a hand-calculated sequence.
- RSI bounds and warm-up behavior.
- Volume ratio behavior with constant volume.
- Realized volatility units.
- Future mutation does not change past features.
- Current extreme observation does not set its own clipping threshold.
- Target responds correctly to costs and volatility.
- Feature list contains no target, future return, or raw future value.
- Output is deterministic.

## Boundaries

- No sentiment or Kalman filtering.
- No model fitting.
- No automatic feature selection.
- Do not add more technical indicators.

## Acceptance criteria

- Feature generation is pure and deterministic.
- Every transformation is causal.
- Model features are explicitly enumerated.
- The target includes costs and a volatility buffer.
- Tests demonstrate no future leakage.
