# Repository Review and Simplification Plan

**Reviewed:** June 24, 2026
**Scope:** What is currently implemented in this repository, how data flows through it, what ran on the former AWS server, and how to return to a small testable system.

## Executive summary

This repository currently tries to be five systems at once:

1. A Reddit collector and archive.
2. A market and derivatives data collector.
3. A sentiment and technical-feature research pipeline.
4. A tuned ensemble training and backtesting framework.
5. A paper/live trading service with portfolio risk controls.

The main problem is not simply the number of files. The problem is that collection, training, inference, portfolio simulation, and exchange execution are coupled inside one `main.py` process. Every process start performs a large training operation before it can produce one signal. The system also has parallel old and new implementations, stale documentation, ignored model artifacts, and tests that no longer match the implementation.

The correct next step is not another model or feature. It is to establish a small, deterministic research pipeline:

`saved market data -> small feature set -> simple baseline model -> chronological test -> report`

Only after that pipeline beats basic baselines after fees should Reddit sentiment, multiple assets, tuning, risk layers, and paper execution be added back one at a time.

## Current status

### Test baseline

Command run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest pytest -q
```

Result:

```text
270 passed, 57 failed, 246 warnings
```

The main failure groups are:

- All 14 archive tests fail because they expect the previous CSV API (`archive_path=...`), while `src/data/archive.py` now exposes a SQLite API (`db_path=...`).
- Most `ImprovedTradingModel` tests fail because BTC beta calculation in `prepare_features()` attempts to assign a multi-column `DataFrame` to one column.
- Tests also expect model behaviors that are absent from the current implementation, including a configurable target threshold and probability diagnostics.

This means the central feature/training path is not currently a green baseline.

### Evidence from the former AWS server

The untracked files in the repository root appear to be copied server artifacts:

- `master_reddit_server.csv`: approximately 2.2 MB.
- `reddit_archive_server.csv`: approximately 4.1 MB.
- `trade_logs_server.log`: approximately 4.0 MB.

The log indicates:

- The service was running in paper mode, not live order mode.
- It reached at least 1,835 hourly iterations.
- The simulated account began at $1,000.
- The final visible status on May 15, 2026 was approximately $973.48 with three positions.
- Recent rolling retrain validation was unstable, with examples around precision `0.49-0.57` and F1 `0.53-0.55`.
- Reddit API retries and timeouts occurred regularly.
- Trailing-stop exits are logged as `STOP LOSS`, including some profitable exits. That label is misleading for later analysis.

The server run demonstrates that the process could stay alive. It does not establish that the strategy has positive out-of-sample expectancy.

## Repository map: what is actually active

| Area | Current implementation | Runtime status |
|---|---|---|
| Entry point | `main.py` | Active |
| Configuration | `configs/data.yaml`, `src/utils/config.py` | Active, but several values are unused |
| Reddit collection | `src/data/reddit_client.py` | Active |
| Reddit archive | `src/data/archive.py` | Active SQLite implementation |
| Spot market data | `src/data/market_client.py` | Active |
| Futures funding/open interest | `src/data/market_client.py` | Active but incomplete and failure-tolerant |
| Sentiment | `src/features/sentiment_advanced.py` | Active, VADER-based |
| Technical features | `src/features/technical.py` | Active |
| Primary model | `src/models/trading_model.py` | Active but currently broken in feature preparation |
| Ensemble | `src/models/ensemble.py` | Active |
| Tuning | `src/models/tuning.py` | Active at process startup |
| Validation | `src/models/validation.py` | Active |
| Backtest | `src/backtest/*`, `scripts/run_backtest.py` | Separate research path |
| Paper portfolio/risk | `src/risk/*` | Active in `main.py` |
| Exchange execution | `src/execution/live.py` | Active and initialized even in paper mode |

### Implemented but not used by the main runtime

- `src/analysis/models.py`: older, simpler Random Forest trading model.
- `src/analysis/sentiment.py`: alternate sentiment analyzer with optional FinBERT.
- `src/data/engagement_tracker.py` and `src/analysis/engagement_validation.py`: research-only engagement snapshot system.
- `MultiTargetEnsemble` and `PurgedKFold`: implemented and tested, but not used by `main.py`.
- `append_sentiment()` and `load_sentiment()` in the archive: not used by `main.py`.
- SHAP and collinearity scripts: manual research tools only.
- Saved models under `models/`: ignored by Git and appear to come from an older pipeline with a different feature schema. Current runtime does not load them.

## Actual runtime data flow

```text
configs/data.yaml
       |
       v
TradingConfig
       |
       +-------------------+
       |                   |
       v                   v
RedditClient          MarketClient
       |                   |
       v                   +--> Binance US spot OHLCV
fresh Reddit posts         +--> Binance futures funding rates
       |                   +--> Binance futures open interest
       v
SQLite archive + 30-day CSV rolling window
       |
       v
EnhancedSentimentAnalyzer (VADER)
       |
       v
hourly sentiment by symbol
       |
       +---------------------------+
                                   v
                         ImprovedTradingModel.prepare_features()
                                   |
                                   +--> technical indicators
                                   +--> sentiment/return lags
                                   +--> BTC beta/residual
                                   +--> funding/open interest
                                   +--> next-hour binary target in training
                                   |
                                   v
                         calibrated four-model ensemble
                                   |
                                   v
                          BUY / HOLD / SELL signals
                                   |
                                   v
                      paper portfolio or Binance order
```

## What happens on every `main.py` startup

1. Logging is configured.
2. The SQLite archive is initialized and the legacy CSV is migrated.
3. Reddit, market, sentiment, model, executor, portfolio, sizing, stops, exits, and risk-budget objects are created.
4. The 30-day rolling Reddit CSV is loaded.
5. Up to 100 recent posts are fetched from each configured subreddit.
6. The complete Reddit archive and rolling CSV are combined.
7. Spot market history is fetched back to the oldest Reddit timestamp.
8. Funding and open-interest data are fetched.
9. All features are generated.
10. XGBoost and LightGBM are tuned with Optuna.
11. A four-model calibrated ensemble is fitted.
12. Feature importance is calculated by fitting additional tree models.
13. The least important 25% of features are removed.
14. The ensemble is fitted again.
15. Walk-forward validation is run on BTC.
16. Only then does the hourly live/paper loop begin.

This is too much startup work for an execution process. Training should create a versioned artifact. Execution should load that artifact and fail closed if it is unavailable or incompatible.

## Data collection details and limitations

### Reddit

`RedditClient` uses PRAW and collects submissions, not comments.

- Six subreddits are configured.
- `fetch_historical(days=30)` requests only 100 posts per subreddit despite comments and configuration suggesting 500 or 1,000.
- `reddit_limit_per_sub`, `reddit_min_score`, and `reddit_min_length` are not applied.
- Spam filtering performs additional author API calls for karma and account age. The server log shows these calls contribute retries and latency.
- A post can be assigned to multiple symbols if it contains multiple keywords.
- Posts with no configured symbol keyword are discarded from sentiment output.

### Archive

There are currently two forms of Reddit state:

- SQLite: `data/reddit_archive.db`, intended as the permanent archive.
- CSV: `data/master_reddit.csv`, used as a 30-day rolling window.

This dual representation adds synchronization and migration complexity.

There is also an implementation bug in `get_connection()`: it calculates the requested path but always connects to the global `DB_PATH`. A caller-provided `db_path` is therefore ignored.

### Market data

- Spot OHLCV comes from Binance US at one-hour resolution.
- Incomplete candles are removed.
- Fetch errors are printed and skipped, so a run may silently continue with missing symbols.
- The configured `intervals` and `market_data_limit` are not used.
- Funding and open interest come from the global Binance Futures REST API, not Binance US.
- Open-interest history is capped at 500 hourly rows, approximately 21 days, even when a longer training period is requested.
- Older missing derivatives values are filled with zero, which means “data unavailable” is treated as “neutral market state.”

## Sentiment flow

The active analyzer is `EnhancedSentimentAnalyzer`, not the FinBERT-capable analyzer in `src/analysis/sentiment.py`.

The active flow is:

1. Combine title and body.
2. Lowercase text.
3. Score unique text with VADER.
4. Assign posts to symbols using keyword matching.
5. Aggregate hourly mean sentiment, post count, and comment count.
6. Add sentiment velocity, momentum, distribution, regime, source, engagement, and cross-symbol features.

There is an important data bug:

- Raw Reddit `score` means upvotes.
- `EnhancedSentimentAnalyzer` overwrites `score` with VADER compound sentiment.
- The advanced engagement logic then cannot use the original upvote score and generally falls back to comments only.

Therefore features named as engagement-weighted sentiment do not reflect the full Reddit engagement that their names imply.

## Feature generation and target

The current primary model creates roughly these groups:

- 24 lag features: sentiment, post volume, and return at 1, 2, 3, 6, 12, 24, 36, and 48 hours.
- Technical indicators: RSI, MACD, Bollinger Bands, ATR, ADX, moving averages, volume, money flow, price position, microstructure, time, and volatility regime.
- Derivatives: funding rate and open-interest change.
- Advanced sentiment: velocity, reversal, distribution, engagement, market-relative sentiment, and z-scores.
- BTC relationship: BTC return, rolling beta, and idiosyncratic return.

The training label is:

```text
target = 1 when the next hourly return is greater than 0.1%
target = 0 otherwise
```

The model is not predicting a three-way BUY/HOLD/SELL target. It predicts the probability of the next return exceeding `0.1%`, and threshold rules translate that probability into actions:

- BUY at probability `>= 0.46`.
- SELL at probability `<= 0.32`.
- HOLD between those values.
- Confidence must also be `>= 0.46`.

This target and decision policy need empirical justification. A BUY threshold below `0.50` can be valid for a calibrated asymmetric decision problem, but only if selected from out-of-sample net returns. It should not be chosen from classification metrics alone.

## Model training

The current ensemble contains:

1. Random Forest.
2. XGBoost.
3. LightGBM.
4. Logistic Regression.

Each model is wrapped in three-fold isotonic calibration, then combined by soft voting.

When tuning is enabled:

- Optuna runs 10 trials for XGBoost.
- Optuna runs 10 trials for LightGBM.
- Tuning uses BTC only.
- The final ensemble trains on all symbols pooled together.
- Feature importance is estimated by fitting separate tree models.
- The top 75% of features are retained.
- The final ensemble is retrained.
- Walk-forward classification validation runs on BTC only.

Problems:

- This is computationally expensive relative to the amount and quality of data.
- Tuning, feature selection, calibration, and final evaluation reuse the same general history without a clearly isolated final holdout.
- Classification F1 is not the strategy objective. Net return, drawdown, turnover, and stability matter.
- Pooled rows from different assets are treated as exchangeable even though their distributions and data quality differ.
- No trained artifact from this active pipeline is saved or loaded.
- The execution service retrains from mutable external data, so identical code can produce a different model after every restart.

## Backtesting flow

`scripts/run_backtest.py`:

1. Fetches fresh market data from Binance.
2. Loads local Reddit data.
3. Builds the same feature frame.
4. Splits timestamps 60/40.
5. Trains once on the first 60%.
6. Predicts the remaining period.
7. Executes signals at the next bar open.
8. Applies fees, slippage, stop-loss, take-profit, and position limits.
9. Compares against buy-and-hold benchmarks.

Positive aspects:

- Orders are executed at the next bar open rather than the signal bar close.
- Fees and slippage are modeled.
- Stops use bar high/low.
- Basic benchmark comparisons exist.

Important mismatches:

- The backtest enables shorting; the live/paper implementation is long-only.
- The backtest strips confidence from model signals before passing them to the engine.
- The model is trained once, not retrained walk-forward in the backtest.
- Backtesting fetches mutable external data, so results are not reproducible unless inputs are saved.
- Existing reports and saved models were produced by older code and feature schemas. They are not reliable evidence for the current implementation.
- One saved model metadata file reports perfect validation folds, a strong leakage or pipeline-error warning rather than credible model quality.

## Paper and live execution flow

The hourly loop:

1. Fetches recent Reddit posts and appends them to storage.
2. Fetches 35 days of spot and derivatives data.
3. Recomputes sentiment over the full 30-day rolling Reddit window.
4. Recomputes all features.
5. Predicts only the latest timestamp.
6. Fetches current ticker prices.
7. Updates paper equity and risk budget.
8. Processes stops.
9. Opens or closes positions from signals.
10. Retrains every 24 hours on the rolling window.

### Execution defects

- `BinanceExecutor` requires API credentials even in dry-run mode because it is always instantiated.
- Paper portfolio state exists only in memory. A restart returns to $1,000 and forgets positions, stops, P&L, cooldowns, and the risk budget.
- In live mode, a successful BUY order does not add the position to `PortfolioRiskManager`. Subsequent portfolio accounting and signal exits cannot work correctly.
- Local portfolio quantity rounding is not based on exchange symbol filters.
- `SELL` attempts the entire free token balance, not necessarily the quantity opened by this strategy.
- The process does not reconcile local state with actual Binance balances or open orders.
- No idempotency key prevents duplicate orders after retries or restarts.
- Stop checks occur hourly against a current ticker price, so a stop can be crossed and recover between loops without being observed.
- The “run once” scheduler mode cannot preserve portfolio state between invocations.
- Risk parameters are initialized from a fixed starting capital, not from actual exchange account equity.

Live trading should remain disabled until these state and reconciliation issues are fixed.

## Where the repository is bloated

### Duplicate implementations

- Two sentiment analyzers.
- Two trading models.
- A SQLite archive plus a rolling CSV.
- Separate backtest risk logic and live risk logic.
- Several advanced model classes that the runtime does not use.

### Complexity without a stable baseline

- Four-model calibrated ensemble.
- Two hyperparameter searches.
- Automatic feature selection.
- More than 50 candidate runtime features.
- Twelve assets.
- Spot, funding, open interest, Reddit sentiment, engagement, cross-asset beta, microstructure, and volatility regimes.
- Portfolio, correlation, sector, stop, exit, cooldown, daily-loss, and drawdown controls.

Each item can be reasonable in isolation. Adding all of them before establishing a reproducible baseline makes it impossible to identify which part creates or destroys value.

### Excess documentation and stale claims

The repository contains more documentation lines than core runtime lines. Several documents describe intended or previous code rather than current behavior. The 14-line README does not explain how to install, test, train, backtest, or run the active system, and its installation command is malformed.

## Noise handling and filtering

Noise handling is currently insufficient. The active pipeline mostly relies on rolling technical indicators, zero-filling, feature selection, and model averaging. Those mechanisms do not adequately distinguish noisy observations, missing data, and genuine neutral market states.

Examples:

- Sparse Reddit hours can produce unstable sentiment averages from very few posts.
- Hours without matching Reddit posts are represented as zero sentiment.
- Missing funding and open-interest observations are filled with zero.
- Extreme return, volume, engagement, and open-interest changes are not consistently handled as outliers.
- The next-hour return is converted into a binary label using a fixed `0.1%` threshold, regardless of current volatility.
- Many correlated technical indicators are derived from the same underlying prices.
- Daily retraining allows a short noisy period to materially change the model.
- Configuration includes unused Kalman-filter settings, but no active Kalman filtering is present in the main feature pipeline.

### Recommended filtering order

Use simple, causal, testable controls before introducing a Kalman filter.

#### 1. Represent missing data explicitly

Do not treat unavailable data as a genuine zero observation.

For each optional data source:

- preserve missing values until feature processing;
- add flags such as `has_sentiment`, `has_funding`, and `has_open_interest`;
- forward-fill only when the value is valid between publication intervals;
- impose a maximum fill age;
- exclude or mark stale observations after that age.

For example, a funding rate may reasonably be carried forward between funding events. Open-interest data missing because an API request failed should not be interpreted as zero open interest or zero change.

#### 2. Add robust outlier handling

Apply causal clipping to variables with heavy tails:

- hourly returns;
- volume and volume changes;
- post engagement;
- comment volume;
- funding rates;
- open-interest changes.

Prefer rolling median and median absolute deviation, or rolling quantile clipping, over global mean and standard deviation. Global statistics can leak information from the future when calculated before a chronological split.

One possible causal rule is:

```text
rolling_median = median of previous N observations
rolling_mad = median absolute deviation of previous N observations
clipped_value = clip(value, median - k * MAD, median + k * MAD)
```

The rolling window must be shifted so the current observation does not influence the limits used to clip itself.

#### 3. Smooth sentiment and activity with EWMA

Use one-sided exponentially weighted moving averages for noisy Reddit features.

Candidate features:

- raw hourly sentiment;
- sentiment EWMA with a 6-hour half-life;
- sentiment EWMA with a 24-hour half-life;
- post-volume EWMA;
- difference between fast and slow sentiment EWMAs.

EWMA is preferable for the first baseline because it is:

- causal;
- easy to understand;
- inexpensive;
- deterministic;
- controlled by one interpretable parameter;
- straightforward to test for leakage.

Keep the raw value alongside the smoothed value. Smoothing can remove useful shocks, so the model should be able to distinguish the current observation from its estimated background level.

#### 4. Measure sentiment reliability

An hourly sentiment mean from one post should not be treated as equally reliable as a mean from 50 posts.

Add:

- contributing post count;
- contributing subreddit count;
- effective sample size;
- dispersion or disagreement;
- data-age/staleness;
- a low-sample flag.

Possible policies:

- require a minimum post count before calculating sentiment;
- shrink low-sample sentiment toward zero;
- retain the sentiment but provide reliability features to the model.

A simple shrinkage estimate is:

```text
reliable_sentiment = raw_sentiment * post_count / (post_count + shrinkage_constant)
```

This preserves strong evidence from active hours while reducing the effect of isolated posts.

#### 5. Denoise the target

The model should not be trained to predict price movements that are smaller than normal market noise and expected trading costs.

Use a neutral zone based on:

- estimated round-trip fees;
- expected slippage;
- a fraction of recent volatility.

For example:

```text
noise_band = round_trip_cost + volatility_multiplier * rolling_volatility
positive target when next return > noise_band
negative target when next return < -noise_band
neutral/ignored otherwise
```

For a long-or-cash binary model, neutral and negative observations can remain the non-entry class. Another option is to exclude observations inside the noise band during training and evaluate whether that creates more selective, higher-quality signals.

The target rule must be calculated using information available at the prediction timestamp, except for the future return used as the label itself.

#### 6. Use model simplicity as noise control

Regularization and feature reduction are often more effective than filtering dozens of weak variables.

For the initial baseline:

- use Logistic Regression;
- standardize features using training data only;
- limit the feature set;
- use explicit L1 or L2 regularization;
- avoid automatic feature selection until the validation design is stable.

This creates a reference point for determining whether later filters or nonlinear models actually add value.

### Kalman filtering

A Kalman filter could be useful later for estimating:

- a latent price trend;
- latent sentiment after accounting for noisy hourly observations;
- a slowly changing relationship between sentiment and returns.

It should not be the first noise-control mechanism. A Kalman model introduces process variance, measurement variance, initialization, and state-design choices. Those parameters can overfit and can make a smooth-looking series appear more useful than it is.

Any Kalman implementation must be strictly causal:

```text
state at time t = filter(observations through time t)
```

Do not use a Kalman smoother that revises past states using future observations. That would leak future information into training and backtesting.

If tested, begin with one narrowly defined experiment:

1. Apply a one-dimensional Kalman filter only to hourly sentiment.
2. Estimate or choose parameters on training data only.
3. Freeze those parameters for the validation and holdout periods.
4. Compare raw sentiment, EWMA sentiment, and Kalman-filtered sentiment using identical folds.
5. Keep Kalman filtering only if it improves net out-of-sample results and stability.

### How filtering should be evaluated

Every filtering change should be an isolated ablation:

```text
same saved dataset
same chronological folds
same model
same target
same execution assumptions
same transaction costs
only the filtering method changes
```

Evaluate:

- net return after fees and slippage;
- maximum drawdown;
- turnover and number of trades;
- precision of actual entries;
- performance across folds and market regimes;
- sensitivity to filter parameters;
- signal delay introduced by smoothing.

The objective is not the smoothest feature chart. The objective is more stable out-of-sample decisions after costs.

## Recommended minimal system

Start with one asset, one data source, one model, one target, and one evaluation path.

### Version 0: deterministic market-only baseline

Use:

- Asset: `BTCUSDT`.
- Interval: one hour.
- Data: saved OHLCV only.
- Features:
  - return over 1, 6, and 24 hours;
  - rolling volatility over 24 hours;
  - volume ratio;
  - RSI 14.
- Noise controls:
  - rolling median/MAD clipping for heavy-tailed features;
  - explicit missingness and staleness flags;
  - no centered rolling windows or future-aware smoothing.
- Model: Logistic Regression.
- Target: next-hour return greater than expected round-trip costs plus a volatility-based noise buffer.
- Evaluation: expanding walk-forward split with a final untouched holdout.
- Strategy: long or cash only.
- Costs: one explicit fee and slippage assumption.
- Outputs: predictions, trades, equity curve, benchmark comparison, and a small JSON metadata file.

Do not use Reddit, derivatives, an ensemble, feature selection, Optuna, shorting, or live execution in this version.

### Version 1: validate sentiment incrementally

After Version 0 is reproducible:

1. Add only `sentiment_mean`.
2. Run the exact same folds and holdout.
3. Compare net return, drawdown, trade count, and fold stability against Version 0.
4. Compare raw sentiment with causal 6-hour and 24-hour EWMA sentiment.
5. Add post-count and low-sample reliability features.
6. Add `post_volume`.
7. Repeat.
8. Add one lag at a time only if the previous feature survives.
9. Test Kalman-filtered sentiment only after EWMA has established a reference result.

Sentiment remains only if it improves multiple out-of-sample periods after costs.

### Version 2: paper execution

Only after the research pipeline is stable:

- Save and load model artifacts.
- Persist paper account state and orders.
- Separate `collect`, `train`, `backtest`, and `paper-trade` commands.
- Reconcile positions on startup.
- Run hourly inference without retraining.
- Retrain from a separate scheduled job.

### Version 3: live execution

Require:

- Exchange state reconciliation.
- Correct lot-size and minimum-notional handling.
- Idempotent order records.
- Persistent positions and stops.
- A kill switch.
- Alerting.
- A meaningful paper-trading observation period.
- Explicit approval before enabling live orders.

## Proposed cleanup sequence

### Standalone rebuild boundary

Implement the replacement as an independent project under `rebuild/`:

```text
rebuild/
  pyproject.toml
  README.md
  configs/
  src/trader/
  tests/
  artifacts/
```

This keeps dependencies, tests, imports, and generated artifacts separate from the current application. The existing root implementation should remain unchanged as reference material until the deterministic rebuild passes its core acceptance criteria.

After the rebuild is accepted:

1. Quarantine the existing implementation under `legacy/`.
2. Promote the contents of `rebuild/` to the repository root only with explicit approval.
3. Re-run the complete rebuild tests and offline baseline after the move.
4. Retain historical server exports and results separately from active artifacts.

### Phase 1: restore a trustworthy test baseline

1. Add `pytest` as a development dependency.
2. Decide whether the archive is SQLite or CSV. Prefer SQLite.
3. Rewrite archive tests around temporary SQLite databases.
4. Fix `get_connection()` to honor `db_path`.
5. Fix the BTC beta calculation or remove it from the baseline.
6. Reconcile `ImprovedTradingModel` tests with its public API.
7. Run focused fast tests on every change.
8. Do not continue until the suite is green.

### Phase 2: create reproducible offline data

1. Add a command that downloads and saves BTC hourly OHLCV.
2. Never fetch external data inside a unit test.
3. Give each dataset a date range, source, symbol, interval, and content hash.
4. Train and backtest against a named saved dataset.

### Phase 3: implement the small baseline

1. Build a six-feature BTC dataset.
2. Add causal rolling outlier clipping and missingness flags.
3. Define a cost- and volatility-aware target noise band.
4. Train Logistic Regression.
5. Add an expanding walk-forward evaluation.
6. Keep a final untouched holdout.
7. Compare against:
   - always cash;
   - BTC buy and hold;
   - a simple momentum rule.
8. Report net results after costs.

### Phase 4: split commands and artifacts

Suggested commands:

```text
python -m trader collect-market
python -m trader collect-reddit
python -m trader build-dataset
python -m trader train
python -m trader backtest
python -m trader paper-trade --run-once
```

Each command should have one responsibility and explicit input/output paths.

Suggested artifacts:

```text
artifacts/
  datasets/
  models/
  backtests/
  paper_state/
```

### Phase 5: reintroduce complexity by evidence

Candidate additions, in order:

1. Reddit mean sentiment.
2. Causal sentiment EWMA and reliability weighting.
3. Reddit post volume.
4. One or two sentiment lags.
5. Kalman-filtered sentiment, only if it beats EWMA.
6. ETH as a second asset.
7. A single tree model.
8. Position sizing and stops.
9. Derivatives features.
10. Tuning.
11. Ensembles.

Every addition should have an ablation result showing its incremental effect.

## What to remove or quarantine now

Do not immediately delete history. Move inactive code to a clearly named `experimental/` or rely on Git history while the baseline is rebuilt.

Strong candidates to remove from the active path:

- FinBERT, Torch, and Transformers.
- SHAP and Streamlit dependencies.
- Optuna.
- XGBoost and LightGBM.
- `MultiTargetEnsemble`.
- `PurgedKFold` unless the chosen label horizon requires it.
- Engagement snapshot analysis.
- Funding and open-interest collection.
- Sector and dynamic-correlation risk controls.
- Shorting.
- The old `src/analysis/models.py` model after its useful simple ideas are migrated.
- One of the two Reddit storage representations.

This would reduce installation size, AWS memory pressure, cold-start time, and the number of interacting failure modes.

## Immediate next iteration

The first implementation iteration should be deliberately narrow:

1. Fix the tests without changing strategy behavior.
2. Add a saved BTC OHLCV fixture/dataset.
3. Add causal outlier clipping, missingness flags, and a volatility-aware target noise band.
4. Implement a standalone market-only Logistic Regression backtest.
5. Produce one Markdown/CSV report with chronological out-of-sample metrics.
6. Compare it with buy-and-hold and cash.

Success for that iteration is not profitability. Success is:

- one command;
- deterministic inputs;
- green tests;
- no network required for tests;
- no credentials required;
- reproducible results;
- clear evidence of where each prediction came from.

That gives the repository a foundation on which sentiment can be tested rather than assumed to help.
