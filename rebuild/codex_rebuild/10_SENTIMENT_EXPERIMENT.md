# Phase 10: Gated Sentiment Experiment

## Entry criteria

Start only when:

- The market-only baseline is reproducible.
- Core tests are green.
- A final holdout policy is established.
- Market-only fold and benchmark results are saved.
- The user explicitly wants sentiment work to proceed.

## Objective

Determine whether Reddit sentiment from submissions and comments provides
incremental out-of-sample value. Do not treat sentiment as a required production
feature.

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

## Reddit ingestion scope

The sentiment experiment must explicitly ingest both Reddit submissions and
comments. Do not repeat the legacy behavior where only submissions are collected
while comment activity is represented only by `num_comments`.

Requirements:

- Collect submissions from configured subreddits.
- Collect comments associated with collected submissions.
- Store submissions and comments as separate raw datasets or separate tables in
  the same versioned dataset.
- Preserve stable Reddit IDs and parent relationships:
  - submission ID;
  - comment ID;
  - comment parent ID;
  - subreddit;
  - created UTC timestamp.
- Preserve engagement fields separately from sentiment fields:
  - submission score;
  - submission `num_comments`;
  - comment score;
  - optional author metadata when available and safe to store.
- Bound comment collection with explicit configuration such as maximum comments
  per submission, maximum comment depth, and timeout/retry limits.
- Make comment collection deterministic for tests by using local fixtures and
  fake clients.
- Record whether a submission has zero comments, comments were intentionally
  capped, comments failed to collect, or comments were not requested.
- Do not silently collapse post text and comment text into one raw field; any
  combined text used for scoring must be derived from preserved raw records.

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
- Track post count, comment count, and subreddit count.
- Track submission-derived sentiment and comment-derived sentiment separately
  before testing any combined sentiment feature.
- Distinguish no matching posts from collection failure.
- Distinguish submissions with no comments from comment collection failure or
  configured comment caps.
- Save raw submissions, raw comments, and derived hourly sentiment as separate
  versioned datasets.
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
