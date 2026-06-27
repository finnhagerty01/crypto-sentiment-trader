# Phase 10: Gated Sentiment Experiment

## Entry criteria

Start only when:

- The market-only baseline is reproducible.
- Core tests are green.
- A final holdout policy is established.
- Market-only fold and benchmark results are saved.
- The user explicitly wants sentiment work to proceed.

## Objective

Determine whether Reddit sentiment provides incremental out-of-sample value. Do not treat sentiment as a required production feature.

## Required structure

Add sentiment code only under:

```text
rebuild/src/trader/sentiment/
```

Suggested modules:

```text
collector.py
storage.py
scoring.py
features.py
experiments.py
```

Do not restore dependencies on legacy sentiment modules.

## Experiment order

Run one isolated addition at a time:

1. Hourly VADER mean sentiment.
2. Post count and explicit missingness.
3. Reliability-shrunk sentiment.
4. Causal 6-hour EWMA.
5. Causal 24-hour EWMA.
6. Fast-minus-slow EWMA.
7. One sentiment lag.
8. Kalman-filtered latent sentiment.

Do not add all features in one run.

## Data correctness requirements

- Preserve Reddit upvote score separately from sentiment score.
- Track post count and subreddit count.
- Distinguish no matching posts from collection failure.
- Save raw posts and derived hourly sentiment as separate versioned datasets.
- Record collection timestamps and source.
- Tests must use local fixtures.
- Do not use future engagement values for historical predictions.

## Kalman gate

Kalman filtering may be tested only after EWMA results exist.

Requirements:

- one-dimensional state initially;
- filtering, not smoothing;
- state at `t` uses observations only through `t`;
- process and measurement variance chosen on development data only;
- parameters frozen for validation and holdout;
- direct comparison against raw and EWMA sentiment.

## Evaluation

Use identical:

- market dataset;
- chronological folds;
- target;
- model class;
- probability threshold;
- costs;
- backtest behavior.

Only the sentiment feature set may change.

Keep sentiment only if it improves multiple folds and the final holdout on net performance and stability, without unacceptable turnover or drawdown.

## Acceptance criteria

- An ablation table compares every sentiment variant to market-only.
- Missingness and reliability are explicit.
- No future-aware filtering exists.
- The conclusion may validly be “sentiment does not help.”
