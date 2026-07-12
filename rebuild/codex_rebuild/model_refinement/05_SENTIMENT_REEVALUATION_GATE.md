# Model Refinement 05: Sentiment Reevaluation Gate

## Entry criteria

Start only when:

- Experiment orchestration is complete.
- A best market-only candidate has been selected from development folds.
- Final holdout confirms the market-only candidate is not obviously spurious.
- Phase 10 sentiment experiment code exists.
- Phase 11 remains blocked.

## Objective

Re-evaluate Phase 10 sentiment features against the best fixed market-only
setup. Sentiment must prove incremental value; it is not assumed useful.

## Required structure

Use existing sentiment code under:

```text
rebuild/src/trader/sentiment/
```

Add only the minimum glue needed to compare sentiment variants through the
experiment orchestration output.

Suggested tests:

```text
rebuild/tests/unit/sentiment/test_experiment_gate.py
rebuild/tests/integration/test_experiment_grid.py
```

## Fixed controls

Hold these constant:

- market dataset;
- selected market-only target;
- selected horizon;
- selected threshold policy;
- selected market feature groups;
- Logistic Regression model class;
- chronological folds;
- final holdout split;
- costs;
- long-or-cash backtest behavior.

Only sentiment features may change.

## Sentiment variant order

Evaluate in the Phase 10 order:

1. Hourly VADER mean sentiment.
2. Post count and explicit missingness.
3. Reliability-shrunk sentiment.
4. Causal 6-hour EWMA.
5. Causal 24-hour EWMA.
6. Fast-minus-slow EWMA.
7. One sentiment lag.
8. Kalman-filtered latent sentiment.

Do not add all sentiment features in one uncontrolled run.

## Keep/reject policy

Keep sentiment only if it:

- improves multiple development folds;
- improves final holdout net return versus the fixed market-only candidate;
- does not materially worsen drawdown;
- does not create unacceptable turnover;
- preserves missingness and reliability diagnostics.

If sentiment fails, leave it as research-only and keep the market-only
candidate as the only possible Phase 11 input.

## Tests

Tests must cover:

- sentiment variants reuse identical market controls;
- only sentiment feature columns change between sentiment variants;
- future sentiment observations do not alter earlier feature rows;
- missing sentiment hours remain explicit;
- output includes a valid "sentiment does not help" conclusion path.

## Acceptance criteria

- Focused tests pass.
- Full rebuild tests pass.
- Sentiment comparison artifacts are generated.
- `HANDOFF.md` records whether sentiment is kept or rejected.
- Phase 11 remains blocked unless the final selected candidate meets the paper
  trading gate criteria.
