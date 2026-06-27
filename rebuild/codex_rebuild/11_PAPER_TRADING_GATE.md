# Phase 11: Gated Paper Trading

## Entry criteria

Do not start until:

- The research pipeline is stable and reproducible.
- A model artifact is versioned and loadable.
- The user explicitly requests paper execution.
- Live trading remains disabled.

## Objective

Implement persistent, restart-safe paper execution as a separate consumer of a trained model artifact.

## Required structure

Use:

```text
rebuild/src/trader/paper/
  broker.py
  state.py
  orders.py
  runner.py
  reconciliation.py
```

Paper execution must not train models.

## Required behavior

- Load an explicit model artifact.
- Load only closed market bars.
- Generate at most one decision per bar.
- Persist cash, position, orders, fills, fees, and model/dataset versions.
- Use idempotent decision and order identifiers.
- Restore state after restart.
- Reconcile stored state before processing a new bar.
- Refuse to run if model features or schema are incompatible.
- Support `--run-once`.
- Require no Binance trading credentials.

## Safety boundaries

- Do not implement a `--live` flag.
- Do not import the legacy Binance executor.
- Do not submit exchange orders.
- Do not assume a process remains alive.
- Do not retrain automatically.
- Do not add multi-asset support.

## Required tests

- Restart preserves state.
- Reprocessing the same bar creates no duplicate order.
- A failed write does not partially update account state.
- Missing model artifact fails closed.
- Incompatible feature schema fails closed.
- Paper fills use documented prices and costs.
- No network is required for state-machine tests.

## Acceptance criteria

- Paper trading is a separate command and process from training.
- State is durable and auditable.
- Repeated scheduler invocations are idempotent.
- No path to real order submission exists.
