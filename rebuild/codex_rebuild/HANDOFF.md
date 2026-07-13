# Rebuild Handoff Log

## Current phase

Phase 10 is complete. Model Refinement 01, 02, 02.5, 03, 04, 05, and 06 are
complete. A non-phase cross-symbol fine-interval diagnostic is complete.
Phase 11 remains gated future paper-trading work.

## Completed phases

- Phase 02: Baseline and safety.
- Phase 03: Package scaffold and configuration.
- Phase 04: Market data pipeline.
- Phase 05: Features, noise handling, and target.
- Phase 06: Model and chronological validation.
- Phase 07: Backtest and reporting.
- Phase 08: CLI and reproducible baseline run.
- Phase 09: Legacy quarantine and dependency reduction.
- Phase 10: Gated sentiment experiment scaffolding.

## Current verification status

- Legacy suite observed on June 27, 2026: `270 passed, 57 failed, 246 warnings`.
- Replacement suite: `179 passed`.
- `rebuild/` is an isolated Python project with independent pytest discovery.
- Default rebuild installation excludes Reddit, transformer, tree-model,
  plotting, and research dependencies.

## Decisions

- Build the replacement under `rebuild/src/trader/` before quarantining legacy code.
- Run rebuild dependency and test commands from `rebuild/`.
- Core baseline is BTCUSDT one-hour market data only.
- Research diagnostics may evaluate other Binance.US spot symbols, but this
  does not change the core baseline or authorize paper trading.
- Core model is Logistic Regression.
- Core evaluation is chronological and long-or-cash.
- EWMA/Kalman sentiment work is deferred until the market-only baseline is accepted.
- Sentiment work is an isolated experiment package and is not part of the
  default market-only CLI path.
- Model refinement lives under `rebuild/codex_rebuild/model_refinement/` and
  must run before Phase 11 unless the user explicitly accepts the current model
  as paper-trading ready.

## User-owned and untracked files

Do not modify or delete without explicit instruction:

- `master_reddit_server.csv`
- `reddit_archive_server.csv`
- `trade_logs_server.log`

## 2026-06-24 — Standalone project initialization

### Completed

- Created `rebuild/` as an independent Python project.
- Added package, configuration, test, fixture, and artifact directory placeholders.
- Added a minimal CLI entry point so the package boundary is executable.
- Updated all rebuild instructions to target `rebuild/`.
- Kept the repository-root implementation untouched as legacy reference material.

### Files changed

- `rebuild/`
- `docs/codex_rebuild/`
- `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`
- `.gitignore`

### Commands and results

- `PYTHONPATH=rebuild/src python3 -m trader.cli --help`: passed.
- Parsed `rebuild/pyproject.toml` with `tomllib`: passed.
- `git diff --check`: passed.

### Decisions

- The rebuild has its own `pyproject.toml` and dependency boundary.
- Generated rebuild artifacts remain under `rebuild/artifacts/`.
- Phase 02 should establish the independent test baseline before feature implementation.

### Remaining blockers

- Dependencies have not been installed for the rebuild.
- No baseline configuration or implementation modules exist yet.

### Recommended next phase

- `02_BASELINE_AND_SAFETY.md`

## 2026-06-27 — Phase 02

### Completed

- Reproduced the unchanged legacy test baseline and confirmed the known failure
  categories without modifying legacy behavior.
- Confirmed `pytest` is declared in the rebuild's `dev` dependency extra and
  generated `rebuild/uv.lock` for reproducible resolution.
- Confirmed pytest discovery is restricted to `rebuild/tests/` when run from
  the standalone project.
- Added a minimal import test for the existing `trader` package scaffold.
- Documented separate commands for legacy, replacement-only, and all tests.
- Verified the unit, integration, and fixture directories exist.
- Verified generated artifact files are ignored and fixture files remain
  trackable.

### Files changed

- `rebuild/README.md`
- `rebuild/tests/unit/test_import.py`
- `rebuild/uv.lock`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest pytest -q` from the
  repository root: expected legacy failure, `270 passed, 57 failed, 246
  warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev` from `rebuild/`: passed;
  installed the locked standalone development environment.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `1 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `1 passed`.
- `git check-ignore -v --no-index rebuild/artifacts/example.json
  rebuild/tests/fixtures/example.csv`: only the hypothetical generated
  artifact was ignored; the fixture path was not ignored.
- `git diff --check`: passed.

### Decisions

- Keep root legacy tests and standalone rebuild tests as explicit sequential
  commands because they have separate project and dependency boundaries.
- Test the package import now because the package scaffold already exists.
- Preserve the known legacy failures: 14 archive tests target the retired CSV
  API, while 43 trading-model tests expose the BTC beta assignment defect and
  missing threshold/diagnostic behavior.

### Remaining blockers

- None for Phase 02.
- Baseline configuration and functional pipeline modules remain intentionally
  deferred to later phases.

### Recommended next phase

- `03_PACKAGE_SCAFFOLD_AND_CONFIG.md`

## 2026-06-27 — Phase 03

### Completed

- Added the complete market-only baseline configuration with no credentials,
  external-service settings, or speculative fields.
- Added immutable, typed dataclasses for every configuration section and the
  root configuration.
- Added strict YAML loading with explicit path resolution, type checking, and
  rejection of missing, unknown, invalid, and non-finite values.
- Added all five planned CLI subcommands as non-operational placeholders with
  clear failure messages and nonzero exit status.
- Added unit coverage for config loading, path resolution, immutability,
  schema rejection, value validation, malformed YAML, CLI help, and every
  placeholder command.
- Confirmed the package tree and existing `crypto-trader` entry point match the
  target architecture for this phase.

### Files changed

- `rebuild/configs/baseline.yaml`
- `rebuild/src/trader/config.py`
- `rebuild/src/trader/cli.py`
- `rebuild/tests/unit/test_config.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit` from
  `rebuild/`: passed, `25 passed` before the final non-finite-value test was
  added.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help` from
  `rebuild/`: passed and listed all five placeholder commands.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader collect-market`
  from `rebuild/`: expected failure with status 2 and a clear not-implemented
  message.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `26 passed`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `26 passed`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `26 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "from trader.config
  import load_config; c = load_config('configs/baseline.yaml'); assert
  c.data.symbol == 'BTCUSDT'"` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.
- Trailing-whitespace scan of Phase 03 files: passed.

### Decisions

- The public configuration API for later phases is
  `trader.config.load_config(path="configs/baseline.yaml",
  working_directory=None) -> TraderConfig`.
- Public config types are `TraderConfig`, `DataConfig`, `FeaturesConfig`,
  `TargetConfig`, `ModelConfig`, `ValidationConfig`, `CostsConfig`, and
  `BacktestConfig`; invalid input raises `trader.config.ConfigError`.
- Relative config paths resolve against the explicit `working_directory`, or
  the process working directory when it is omitted. Import location is never
  used as an implicit base.
- Baseline symbol and interval are constrained to `BTCUSDT` and `1h`.
- Costs must be finite, greater than zero, and less than one. Threshold and
  holdout fractions use the open interval `(0, 1)`.
- Placeholder CLI commands do not load configuration or import any legacy
  implementation.

### Remaining blockers

- None for Phase 03.
- Market schemas, storage, and collection remain intentionally deferred to
  Phase 04.

### Recommended next phase

- `04_MARKET_DATA_PIPELINE.md`

## 2026-07-08 — Phase 04

### Completed

- Added canonical BTCUSDT 1h OHLCV schema validation and normalization.
- Added deterministic Parquet storage with JSON metadata sidecars and content
  hashing.
- Added an explicit Binance US spot kline collector with bounded retries,
  timeouts, explicit start/end dates, incomplete-candle filtering, and an
  injectable fake-client boundary for tests.
- Wired `crypto-trader collect-market` to save versioned raw datasets under
  the requested output directory.
- Added a small deterministic BTCUSDT 1h CSV fixture and offline unit and
  integration coverage for schema validation, collection, storage, metadata,
  and CLI wiring.

### Files changed

- `rebuild/src/trader/cli.py`
- `rebuild/src/trader/data/schemas.py`
- `rebuild/src/trader/data/storage.py`
- `rebuild/src/trader/data/market.py`
- `.gitignore`
- `rebuild/tests/fixtures/btcusdt_1h.csv`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/tests/unit/data/test_schemas.py`
- `rebuild/tests/unit/data/test_storage.py`
- `rebuild/tests/unit/data/test_market.py`
- `rebuild/tests/integration/test_market_fixture_storage.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: initially failed because the new schema
  tests assigned numeric invalid values into Arrow-backed string columns before
  validation ran; fixed the test inputs to use string payloads.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `42 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `42 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help` from
  `rebuild/`: passed and listed `collect-market` plus the future placeholder
  commands.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader collect-market
  --help` from `rebuild/`: passed and showed required explicit `--start` and
  `--end` options.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.

### Decisions

- The documented baseline data source is Binance US spot klines via
  `https://api.binance.us/api/v3/klines`; no futures, derivatives, Reddit, or
  root server artifacts are used.
- `collect-market` is the only implemented command that may contact an
  external market API. All default tests remain offline and use fake clients or
  local fixtures.
- Public market-data APIs added for later phases are
  `trader.data.schemas.normalize_ohlcv`,
  `trader.data.storage.write_market_dataset`,
  `trader.data.storage.read_market_dataset`,
  `trader.data.storage.build_metadata`,
  `trader.data.storage.content_hash`, and
  `trader.data.market.collect_market_data`.
- Metadata sidecars are written as `<dataset>.parquet.metadata.json` and
  include symbol, interval, row count, UTC date range, schema version, source,
  and deterministic content hash.

### Remaining blockers

- None for Phase 04.
- Feature engineering, noise handling, and target generation remain
  intentionally deferred to Phase 05.

### Recommended next phase

- `05_FEATURES_NOISE_AND_TARGET.md`

## 2026-07-10 — Phase 05

### Completed

- Added causal BTCUSDT market feature generation for the six required
  baseline features while preserving execution OHLCV columns.
- Added explicit raw feature, clipped feature, missingness flag, execution,
  and model-feature column lists.
- Added causal rolling median/MAD clipping that shifts statistics by one bar
  so the current observation never sets its own clipping threshold.
- Added missingness flags captured before any later imputation step.
- Added volatility and cost aware `next_return`, `noise_band`, and nullable
  binary `target` construction.
- Added one pure dataset builder that performs feature, noise, and target
  construction without fetching or writing data.
- Added unit coverage for hand-calculated returns, RSI warm-up and bounds,
  constant-volume behavior, realized-volatility units, future-mutation
  leakage checks, current-extreme clipping behavior, target cost/volatility
  behavior, model-feature list safety, and deterministic output.

### Files changed

- `rebuild/src/trader/features/market.py`
- `rebuild/src/trader/features/noise.py`
- `rebuild/src/trader/features/target.py`
- `rebuild/tests/unit/features/test_market_features.py`
- `rebuild/tests/unit/features/test_noise.py`
- `rebuild/tests/unit/features/test_target.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/features`
  from `rebuild/`: passed, `13 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `55 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `55 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.

### Decisions

- `realized_volatility_24h` is the hourly standard deviation of trailing
  one-hour returns over the configured volatility window. It is not annualized
  and is not horizon-scaled.
- Model features are the clipped versions of the six raw market features plus
  their missingness flags. Raw OHLCV, `next_return`, `noise_band`, and
  `target` are excluded from `MODEL_FEATURE_COLUMNS`.
- Rolling median/MAD clipping uses a complete prior window. During warm-up, or
  when prior MAD is zero or unavailable, values are left unchanged rather than
  producing infinite or global-statistic limits.
- Rows without a future close or without a volatility estimate retain
  `target` as nullable `NA` so training can exclude them while inference rows
  remain present.

### Remaining blockers

- None for Phase 05.
- Model training and chronological validation remain intentionally deferred to
  Phase 06.

### Recommended next phase

- `06_MODEL_AND_VALIDATION.md`

## 2026-07-11 — Phase 06

### Completed

- Added the market-only Logistic Regression baseline as a scikit-learn
  pipeline with median imputation, `StandardScaler`, and a deterministic
  classifier.
- Added positive-class probability prediction, threshold-based binary signal
  generation, exposed feature names, fitted scaler diagnostics, and model
  metadata.
- Added final-holdout splitting and expanding walk-forward validation that
  reserves the configured holdout before producing development folds.
- Added per-fold classification metrics, probability diagnostics, class counts,
  and explicit single-class fold handling.
- Added joblib model persistence with JSON metadata sidecars and exact feature
  compatibility checks on load.
- Added offline unit coverage for chronological isolation, holdout exclusion,
  per-fold scaler fitting, deterministic training, bounded probabilities,
  single-class handling, artifact round trips, and feature-order rejection.

### Files changed

- `rebuild/src/trader/modeling/baseline.py`
- `rebuild/src/trader/modeling/validation.py`
- `rebuild/src/trader/modeling/artifacts.py`
- `rebuild/tests/unit/modeling/__init__.py`
- `rebuild/tests/unit/modeling/testing_data.py`
- `rebuild/tests/unit/modeling/test_baseline.py`
- `rebuild/tests/unit/modeling/test_validation.py`
- `rebuild/tests/unit/modeling/test_artifacts.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/modeling`
  from `rebuild/`: passed, `14 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `69 passed, 8 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `69 passed, 8 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.

### Decisions

- Imputation is included because Phase 05 model features can contain warm-up
  missing values; the imputer is inside the saved sklearn pipeline so fitted
  preprocessing travels with the model.
- The baseline uses `LogisticRegression(C=config.model.regularization_c,
  solver="liblinear", random_state=42, max_iter=1000)` with default L2
  regularization.
- Artifact loading validates the exact saved feature name order against the
  caller's expected feature contract. Prediction input frames are selected by
  canonical feature names so Phase 05 datasets do not need to store columns in
  model order.
- Walk-forward validation reports fold metrics only. It does not use holdout
  results and does not optimize any parameters.
- Single-class training windows are skipped by default during walk-forward
  validation with an explicit reason, and can be configured to fail.

### Remaining blockers

- None for Phase 06.
- Backtest simulation, benchmarks, and reporting remain intentionally deferred
  to Phase 07.

### Recommended next phase

- `07_BACKTEST_AND_REPORTING.md`

## 2026-07-12 — Phase 07

### Completed

- Added a transparent long-or-cash BTC backtest engine with next-bar
  execution, one position or cash, no shorting, explicit per-side fee and
  slippage accounting, no final-bar signal fill, and force-close accounting at
  the final close.
- Added strategy metrics for total return, conditional annualized return,
  max drawdown, hourly Sharpe ratio, trade count, win rate, average trade
  return, profit factor without infinities, turnover, fees, slippage, and
  exposure.
- Added comparable cash, buy-and-hold, and causal momentum benchmarks over the
  same market period and cost assumptions.
- Added a deterministic report writer that creates the required versioned
  artifact directory with config, dataset metadata, model metadata, fold
  metrics, predictions, trades, equity, metrics, benchmark metrics, and
  summary files.
- Added offline unit coverage for next-bar fills, hand-verifiable costs, no
  shorting, non-negative cash, final accounting, final-bar signal ignoring,
  drawdown, zero-trade metrics, no-loss profit factor handling, benchmark
  period alignment, and deterministic report output.

### Files changed

- `rebuild/src/trader/backtest/engine.py`
- `rebuild/src/trader/backtest/metrics.py`
- `rebuild/src/trader/backtest/benchmarks.py`
- `rebuild/src/trader/reporting/writer.py`
- `rebuild/tests/unit/backtest/test_engine.py`
- `rebuild/tests/unit/backtest/test_metrics.py`
- `rebuild/tests/unit/backtest/test_benchmarks.py`
- `rebuild/tests/unit/reporting/test_writer.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/backtest
  tests/unit/reporting` from `rebuild/`: passed, `11 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `80 passed, 8 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `80 passed, 8 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.

### Decisions

- A buy signal at bar `t` fills at bar `t+1` open with the fill price
  increased by slippage; a sell signal fills at bar `t+1` open with the fill
  price reduced by slippage.
- Entry uses all available cash after reserving the entry fee, so cash does
  not go negative. Exits liquidate the full BTC quantity and return to cash.
- Equity while long is marked as conservative liquidation value after sell-side
  slippage and fee. The final open position is force-closed at the final bar's
  close using the same sell-side cost rule.
- The annualized return is reported only for periods of at least 24 hours.
  Hourly Sharpe uses `365.25 * 24` periods per year and returns `None` for
  zero-variance or undersized return series.
- Profit factor returns `None` when there are no closed trades or no losing
  closed trades, avoiding misleading infinite values.
- The momentum benchmark is explicitly causal: it is long only when the
  trailing close-to-close return over the configured lookback is positive, and
  it shares the same next-bar execution engine and costs as the strategy.
- Report writing is deterministic for a caller-supplied `run_id`; no timestamp
  or random identifier is injected by the writer.

### Remaining blockers

- None for Phase 07.
- CLI wiring and a reproducible end-to-end run remain intentionally deferred to
  Phase 08.

### Recommended next phase

- `08_CLI_AND_REPRODUCIBLE_RUN.md`

## 2026-07-12 — Phase 08

### Completed

- Wired `build-dataset`, `train`, `backtest`, and `run-baseline` to the
  completed Phase 04-07 components.
- Added a small workflow orchestration module that writes feature datasets,
  model artifacts, report bundles, metadata sidecars, and config copies.
- Preserved `collect-market` as the only implemented command that may fetch
  external data; `run-baseline` starts from an already saved market dataset.
- Added artifact overwrite checks for generated feature datasets, models, and
  report directories.
- Added concise metric comparison output for `backtest` and `run-baseline`.
- Rewrote the root README around the rebuild workflow, test commands, real
  collection command, offline fixture verification, artifact layout, and the
  explicit live-trading disablement.
- Added an offline integration test that creates a deterministic saved market
  dataset from the BTC fixture in a temporary directory, runs the public
  `run-baseline` CLI twice, verifies dataset/model/report artifacts, reads
  predictions/metrics/benchmark files, checks deterministic predictions and
  metrics, and asserts no collection path is used.

### Files changed

- `README.md`
- `rebuild/src/trader/cli.py`
- `rebuild/src/trader/workflow.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/tests/integration/test_offline_baseline_workflow.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/test_cli.py
  tests/integration/test_offline_baseline_workflow.py` from `rebuild/`:
  passed, `4 passed, 16 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `78 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `78 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.

### Decisions

- Feature dataset filenames and model filenames include the feature dataset
  content hash prefix. Re-running into the same output directory fails clearly
  instead of overwriting.
- Report directories use an explicit `--run-id` when supplied; otherwise the
  CLI-generated run id includes feature hash, model metadata hash, and UTC
  creation time. Existing report directories are rejected before writing.
- Feature datasets store a deterministic metadata sidecar with raw dataset
  metadata, raw content hash, feature content hash, target definition, model
  feature columns, labeled row count, and configuration hash.
- The final saved model is trained on the development split only. The final
  holdout remains reserved for `backtest`.
- The offline integration test expands the tiny six-row fixture
  deterministically in a temporary directory so the model can train with both
  target classes while still using no network, credentials, or mutable data.

### Remaining blockers

- None for Phase 08.
- Legacy quarantine and dependency cleanup remain intentionally deferred to
  Phase 09.

### Recommended next phase

- `09_LEGACY_QUARANTINE_AND_DOCS.md`

## 2026-07-12 — Phase 09

### Completed

- Quarantined the tracked legacy runtime under `legacy/runtime/`, including
  the former root `main.py`, root `src/`, root tests, legacy config, and
  stale dependency manifests.
- Moved tracked legacy research scripts under `legacy/scripts/`.
- Moved old strategy documentation under `legacy/docs/`, while keeping
  `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md` active.
- Added `legacy/README.md` with a historical-status banner and an inventory
  classifying legacy modules as superseded, future experiment, data migration
  utility, or historical documentation.
- Replaced root `main.py` with a dependency-free refusal message that directs
  users to `crypto-trader` in `rebuild/`.
- Updated the root README to state that the rebuild is active and legacy files
  are quarantined.
- Removed root `pyproject.toml`, root `uv.lock`, and root `requirements.txt`
  from the active path by moving them into legacy reference.
- Added optional rebuild dependency extras for deferred `sentiment` and
  `research` work while keeping core dependencies limited to the market-only
  baseline.
- Tightened `.gitignore` coverage for generated artifacts, logs, local
  credentials, rebuild datasets, rebuild models, rebuild backtests, and paper
  state.
- Preserved untracked server artifacts in place:
  `master_reddit_server.csv`, `reddit_archive_server.csv`, and
  `trade_logs_server.log`.

### Files changed

- `.gitignore`
- `README.md`
- `main.py`
- `docs/`
- `legacy/`
- `rebuild/pyproject.toml`
- `rebuild/uv.lock`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `rg -n "(from|import) (src|analysis|models|features|risk|execution)|src\\.analysis|src\\.models|src\\.features|src\\.risk|src\\.execution" rebuild/src/trader rebuild/tests rebuild/configs README.md docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`: passed; no legacy imports in rebuild.
- `python3 - <<'PY' ... tomllib.loads(Path('rebuild/pyproject.toml').read_text()) ... PY`: passed; default dependencies are only `joblib`, `numpy`, `pandas`, `pyarrow`, `pyyaml`, and `scikit-learn`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit tests/integration` from `rebuild/`: first sandboxed attempt failed on blocked DNS while resolving packages after metadata changed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit tests/integration` from `rebuild/` with approved network access: passed, `78 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from `rebuild/`: passed, `78 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader run-baseline ... --dataset-output-dir ...`: expected CLI usage failure because the actual Phase 08 option names are `--dataset-dir`, `--model-dir`, and `--report-dir`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader run-baseline --config configs/baseline.yaml --market-data /tmp/crypto_phase09_baseline/btcusdt_1h_phase09_raw.parquet --dataset-dir /tmp/crypto_phase09_baseline/datasets --model-dir /tmp/crypto_phase09_baseline/models --report-dir /tmp/crypto_phase09_baseline/backtests --run-id phase09-baseline` from `rebuild/`: passed; saved feature dataset, model artifact, and report bundle under `/tmp/crypto_phase09_baseline/`.
- `python3 ../main.py` from `rebuild/`: expected failure status `2` with the quarantine message and rebuild CLI instructions.
- `UV_CACHE_DIR=/tmp/uv-cache uv sync --no-dev --no-default-groups --locked` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv pip list --python .venv/bin/python` from `rebuild/`: passed; core environment contains the rebuild package, `joblib`, `numpy`, `pandas`, `pyarrow`, `pyyaml`, `scikit-learn`, and their transitive dependencies only.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev --no-default-groups --locked python -c "..."` from `rebuild/`: passed; `torch`, `transformers`, `xgboost`, `lightgbm`, `optuna`, `shap`, `streamlit`, `plotly`, `praw`, and `vaderSentiment` were not importable in the core environment.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: first sandboxed attempt failed on blocked DNS while reinstalling
  `pygments` for the dev extra after the core-only install check; rerun with
  approved network access passed, `78 passed, 24 warnings`.
- Final `git diff --check`: passed.

### Decisions

- Quarantine was chosen over deleting legacy code so historical behavior and
  migration references remain available without being mistaken for the active
  system.
- Root `main.py` is intentionally non-operational; running it must not import
  Reddit, Binance, or legacy model code.
- Root dependency manifests are legacy reference only. The active source of
  truth for installation is `rebuild/pyproject.toml`.
- Future Reddit/VADER sentiment dependencies live in the `sentiment` optional
  extra. Research-heavy plotting, tuning, transformer, SHAP, and tree-model
  dependencies live in the `research` optional extra.
- The ignored, untracked root `archive/` utilities and generated data/log
  files were left untouched.

### Remaining blockers

- None for Phase 09.
- Phase 10 sentiment work should not start until the market-only baseline is
  accepted and its entry criteria are confirmed.

### Recommended next phase

- `10_SENTIMENT_EXPERIMENT.md` only after its gates are satisfied.

## 2026-07-12 — Phase 10

### Completed

- Added a gated sentiment experiment package under
  `rebuild/src/trader/sentiment/` without changing the market-only CLI.
- Added fake-client-friendly Reddit collection for submissions and bounded
  comments with stable IDs, parent relationships, UTC timestamps, engagement
  fields, collection source, and explicit comment collection statuses.
- Added versioned raw sentiment storage that saves submissions and comments as
  separate Parquet tables with metadata and content hashes.
- Added separate versioned storage for derived hourly sentiment datasets.
- Added optional VADER scoring behind the existing `sentiment` extra, plus a
  deterministic lexical scorer for offline tests.
- Added scoring that preserves Reddit upvote/comment engagement fields
  separately from `sentiment_score`.
- Added causal hourly sentiment features for the required ablation order:
  hourly mean, counts/missingness, reliability shrinkage, 6-hour EWMA, 24-hour
  EWMA, fast-minus-slow EWMA, one-hour lag, and one-dimensional Kalman
  filtering.
- Added an ablation runner that compares market-only against cumulative
  sentiment variants while reusing the same feature dataset, chronological
  folds, Logistic Regression model class, threshold, costs, and long-or-cash
  backtest behavior.
- Added offline unit coverage for collection status handling, separate
  submission/comment records, engagement preservation, causal transforms,
  storage round trips, and ablation table structure.

### Files changed

- `rebuild/src/trader/sentiment/__init__.py`
- `rebuild/src/trader/sentiment/collector.py`
- `rebuild/src/trader/sentiment/storage.py`
- `rebuild/src/trader/sentiment/scoring.py`
- `rebuild/src/trader/sentiment/features.py`
- `rebuild/src/trader/sentiment/experiments.py`
- `rebuild/tests/unit/sentiment/test_collector.py`
- `rebuild/tests/unit/sentiment/test_scoring_features.py`
- `rebuild/tests/unit/sentiment/test_storage_experiments.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/sentiment`
  from `rebuild/`: first failed on an invalid `Protocol` import from
  `collections.abc`; fixed by importing `Protocol` from `typing`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/sentiment`
  from `rebuild/`: then failed because a new test imported
  `tests.unit.modeling.testing_data` through a non-package test root; fixed by
  making the sentiment ablation test self-contained.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/sentiment`
  from `rebuild/`: passed, `9 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `87 passed, 24 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check`: passed.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/sentiment` from `rebuild/`: passed, `10 passed`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `88 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit
  tests/integration` from `rebuild/`: passed, `88 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `git diff --check`: passed.

### Decisions

- Sentiment remains outside the default CLI and workflow; Phase 10 exposes
  experiment APIs only.
- The Reddit client boundary is a small protocol so tests can use local
  fixtures and optional PRAW integration can be added later without import-time
  side effects.
- VADER is optional and imported only when `VaderSentimentScorer` is
  constructed, preserving the core dependency boundary.
- Raw submission text and raw comment text are preserved separately. Any
  combined sentiment is derived from scored records, not stored as the only raw
  representation.
- Missing sentiment hours are explicit: count features are zero, raw hourly
  mean sentiment remains missing, and downstream causal transforms use a
  neutral filled observation.
- Kalman support is filtering only, one-dimensional, and causal. The API
  exposes fixed variance parameters so future experiments can choose them on
  development data before holdout evaluation.

### Remaining blockers

- No real Reddit adapter, CLI command, or production workflow integration was
  added in Phase 10.
- No live sentiment experiment has been run against collected Reddit data, so
  the conclusion remains unset; the valid eventual conclusion may be
  “sentiment does not help.”
- Phase 11 paper trading remains gated and must not start until its own entry
  criteria are explicitly satisfied.

### Recommended next phase

- `11_PAPER_TRADING_GATE.md` only after its gates are satisfied.

## 2026-07-12 — Model refinement planning module

### Completed

- Added a separate `model_refinement/` planning track under
  `rebuild/codex_rebuild/`.
- Documented that model refinement is not Phase 11 and does not begin paper
  trading.
- Added scoped work packages for threshold diagnostics, target/horizon
  experiments, market feature groups, experiment orchestration, and sentiment
  re-evaluation.
- Updated the rebuild instruction README to direct weak-signal market-only
  work into the model refinement track before Phase 11.

### Files changed

- `rebuild/codex_rebuild/README.md`
- `rebuild/codex_rebuild/HANDOFF.md`
- `rebuild/codex_rebuild/model_refinement/README.md`
- `rebuild/codex_rebuild/model_refinement/01_THRESHOLD_SWEEP_AND_DIAGNOSTICS.md`
- `rebuild/codex_rebuild/model_refinement/02_TARGET_AND_HORIZON_EXPERIMENTS.md`
- `rebuild/codex_rebuild/model_refinement/03_MARKET_FEATURE_GROUPS.md`
- `rebuild/codex_rebuild/model_refinement/04_EXPERIMENT_ORCHESTRATION.md`
- `rebuild/codex_rebuild/model_refinement/05_SENTIMENT_REEVALUATION_GATE.md`

### Commands and results

- `git status --short`: inspected existing uncommitted Phase 10 files and left
  them intact.
- `git diff --check`: passed.

### Decisions

- Model refinement is a separate module, not a Phase 12+ sequence.
- Shorting remains out of scope until long/cash signal improves.
- Phase 11 remains gated until model refinement produces a candidate worth
  paper trading.

### Remaining blockers

- No model refinement implementation has started yet.
- The next implementation unit is
  `model_refinement/01_THRESHOLD_SWEEP_AND_DIAGNOSTICS.md`.

### Recommended next work

- Implement `rebuild/codex_rebuild/model_refinement/01_THRESHOLD_SWEEP_AND_DIAGNOSTICS.md`.

## 2026-07-12 — Model Refinement 01

### Completed

- Added threshold sweep diagnostics for the existing market-only Logistic
  Regression model without changing target, features, model class, costs, or
  long-or-cash backtest behavior.
- Added the fixed threshold candidate set from `0.10` through `0.55`.
- Refit the model once per chronological development fold, generated one
  validation probability vector per fold, and evaluated every threshold against
  that same vector.
- Added per-fold classification metrics, probability diagnostics, and trading
  metrics for each threshold.
- Added development-only threshold aggregation and deterministic selection:
  zero-trade thresholds are excluded, negative or below-cash median returns are
  excluded, then candidates are ranked by median return, drawdown magnitude,
  and turnover.
- Added holdout evaluation only after development-fold threshold selection.
- Added artifact writing helpers for `threshold_fold_metrics.csv`,
  `threshold_summary.csv`, `selected_threshold.json`, and
  `holdout_threshold_metrics.json`.
- Added `scripts/run_threshold_sweep.py` as a simple command wrapper around the
  threshold API to avoid fragile pasted heredocs.
- Added unit coverage for shared fold probabilities, deterministic selection,
  zero-trade exclusion, holdout isolation, all-fail selection, and inclusive
  `>= threshold` signal conversion.
- Ran the sweep on
  `artifacts/datasets/btcusdt_1h_features_e5ddf65b05a6.parquet`; it selected
  threshold `0.10` from development folds and wrote the four threshold
  artifacts under `artifacts/model_refinement/threshold_sweep_01/`.

### Files changed

- `rebuild/src/trader/modeling/thresholds.py`
- `rebuild/scripts/run_threshold_sweep.py`
- `rebuild/tests/unit/modeling/test_thresholds.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/modeling/test_thresholds.py` from `rebuild/`: initially failed
  while tightening synthetic price fixtures; fixed the tests so losing
  low-threshold exposure is represented explicitly.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/modeling/test_thresholds.py` from `rebuild/`: passed, `6 passed`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `94 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `git diff --check` from the repository root: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/run_threshold_sweep.py` from `rebuild/`: passed; selected threshold
  `0.10`, wrote `threshold_fold_metrics.csv`, `threshold_summary.csv`,
  `selected_threshold.json`, and `holdout_threshold_metrics.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_threshold_sweep.py` from `rebuild/`: passed.

### Decisions

- Kept the sweep as a modeling module API for this step; CLI/workflow
  orchestration can wire it into saved baseline artifacts in the later
  refinement orchestration step.
- Selection uses development summaries only. Holdout metrics are confirmation
  output and are not passed into `select_threshold`.
- `max_drawdown` is stored in the existing negative-return convention, so the
  tie-break uses lower drawdown magnitude rather than a more negative numeric
  value.
- The real baseline feature artifact sweep selected `0.10` on development
  folds. The default `0.55` and `0.50` thresholds produced zero trades across
  development folds, so the observed issue is at least partly a too-strict
  threshold rather than a complete inability to produce long signals.

### Remaining blockers

- Threshold sweep is available through `scripts/run_threshold_sweep.py`, but no
  first-class `crypto-trader` CLI subcommand has been added.
- Phase 11 remains blocked until refinement produces a development-selected
  candidate confirmed on holdout.
- Shorting remains out of scope.

### Recommended next work

- Run or wire the threshold sweep against reproduced Phase 08 baseline
  artifacts before moving to
  `model_refinement/02_TARGET_AND_HORIZON_EXPERIMENTS.md`.

## 2026-07-12 — Model Refinement 02

### Completed

- Added explicit `target.cost_buffer` configuration with allowed values
  `none`, `one_way`, and `round_trip`.
- Preserved the previous baseline target behavior by setting
  `cost_buffer: round_trip` in `configs/baseline.yaml` and offline workflow
  test configuration.
- Updated target construction so `next_return` uses the configured future
  close horizon and `noise_band` uses the selected cost buffer plus the
  configured volatility multiplier.
- Added target-distribution diagnostics covering row count, labeled count,
  positive and negative counts, positive rate, unlabeled count, and first/last
  labeled timestamps.
- Updated affected test config builders to include the new target field.
- Added unit coverage for hand-calculated labels across all cost-buffer modes,
  multi-bar horizon labeling, end-of-series unlabeled rows, baseline default
  preservation, invalid cost-buffer rejection, and positive-rate diagnostics.

### Files changed

- `rebuild/configs/baseline.yaml`
- `rebuild/src/trader/config.py`
- `rebuild/src/trader/features/target.py`
- `rebuild/tests/integration/test_offline_baseline_workflow.py`
- `rebuild/tests/unit/features/test_target.py`
- `rebuild/tests/unit/modeling/test_artifacts.py`
- `rebuild/tests/unit/modeling/test_baseline.py`
- `rebuild/tests/unit/modeling/test_thresholds.py`
- `rebuild/tests/unit/modeling/test_validation.py`
- `rebuild/tests/unit/reporting/test_writer.py`
- `rebuild/tests/unit/sentiment/test_storage_experiments.py`
- `rebuild/tests/unit/test_config.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/features/test_target.py tests/unit/test_config.py` from
  `rebuild/`: passed, `26 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: initially failed because
  `tests/integration/test_offline_baseline_workflow.py` used an inline config
  without `target.cost_buffer`; fixed the fixture to use `round_trip`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `99 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `git diff --check` from the repository root: passed.

### Decisions

- `cost_buffer` is validated in the strict config loader as a string enum
  rather than adding permissive backward compatibility for old generated
  artifacts.
- The baseline remains equivalent to the previous target because
  `round_trip` computes `2 * (fee_per_side + slippage_per_side)`.
- Target diagnostics are implemented as a pure helper in
  `trader.features.target` so later experiment orchestration can compare
  target/horizon grid candidates without involving holdout selection.
- No target/horizon candidate was selected in this step; selection remains for
  the later development-fold orchestration work.

### Remaining blockers

- Phase 11 remains blocked.
- Target/horizon grid orchestration is not wired yet.
- Feature group experiments have not started.

### Recommended next work

- Implement
  `model_refinement/03_MARKET_FEATURE_GROUPS.md` after using the new target
  diagnostics in the planned orchestration flow.

## 2026-07-12 — Model Refinement 02 grid execution

### Completed

- Added reusable target/horizon grid-search orchestration for the exact Model
  Refinement 02 grid: horizons `1, 3, 6, 12, 24`, cost buffers `none`,
  `one_way`, `round_trip`, and volatility multipliers `0.00, 0.05, 0.10`.
- Added `scripts/run_target_horizon_grid.py` with defaults pointing at the
  current saved raw BTCUSDT 1h parquet and output under
  `artifacts/model_refinement/target_horizon_grid/`.
- For each of the 45 candidates, the script rebuilds the feature dataset,
  records target-distribution diagnostics, runs the existing threshold sweep,
  and writes per-candidate artifacts.
- Added top-level grid outputs: `grid_results.csv`, `grid_results.json`,
  `selected_candidate.json`, and `target_horizon_grid_report.md`.
- Updated workflow target-definition metadata to describe configurable
  horizon and cost-buffer behavior instead of the old hardcoded round-trip
  target.
- Ran the grid on the currently available five-month dataset.

### Files changed

- `rebuild/src/trader/modeling/target_horizon_grid.py`
- `rebuild/scripts/run_target_horizon_grid.py`
- `rebuild/tests/unit/modeling/test_target_horizon_grid.py`
- `rebuild/src/trader/workflow.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/modeling/test_target_horizon_grid.py
  tests/unit/features/test_target.py tests/unit/test_config.py` from
  `rebuild/`: passed, `31 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_target_horizon_grid.py` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/run_target_horizon_grid.py` from `rebuild/`: passed; evaluated 45
  candidates, selected `h12_none_vol0p10` with threshold `0.30`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/run_target_horizon_grid.py --overwrite` from `rebuild/`: passed
  after fixing the top-level holdout exposure column name; selected the same
  candidate.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `104 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_threshold_sweep.py scripts/run_target_horizon_grid.py` from
  `rebuild/`: passed.
- Final `git diff --check` from the repository root: passed.

### Results

- Development-selected candidate: `h12_none_vol0p10`.
- Candidate definition: 12-hour target horizon, no explicit cost buffer, and
  `0.10` volatility multiplier.
- Selected probability threshold: `0.30`.
- Development median return: about `+2.41%`.
- Development median max drawdown: about `-5.38%`.
- Development median turnover: about `2.04`.
- Target positive rate: about `47.77%`.
- Holdout total return: about `-6.06%`.
- Holdout max drawdown: about `-11.65%`.
- Holdout trades: `2`.
- Holdout exposure: about `99.86%`.
- All 45 candidates produced a development-selected threshold, but the
  selected candidate did not confirm on holdout.

### Decisions

- The current five-month dataset is acceptable for a smoke/current-data
  refinement run, but not enough evidence for a robust paper-trading gate.
- Development selection still ignores holdout; holdout is reported only as
  confirmation.
- The current grid suggests the one-hour round-trip target is stricter and
  less favorable than several longer or looser targets on development folds,
  but holdout failure prevents accepting a paper-trading candidate.

### Remaining blockers

- Phase 11 remains blocked.
- No target/horizon candidate is accepted for paper trading.
- Feature group experiments remain the next refinement step, but they should
  be interpreted against the limited five-month dataset.

### Recommended next work

- Continue to `model_refinement/03_MARKET_FEATURE_GROUPS.md` using the grid
  outputs as context, while keeping Phase 11 blocked.

## 2026-07-12 — Model Refinement 02.5 recency-window diagnostic

### Completed

- Added an experiment-level train-window policy hook to the threshold sweep
  path without changing the default expanding-window behavior.
- Added supported train-window policies: `expanding`, `rolling_1000`,
  `rolling_1500`, `rolling_2000`, and `rolling_2500`.
- Kept Logistic Regression, current market-only features, long/cash
  backtesting, chronological boundaries, fixed threshold candidates, and
  development-only threshold selection unchanged.
- Added `scripts/run_recency_window_diagnostic.py` to compare all train-window
  policies on the target-grid winner configuration:
  `horizon_bars=12`, `cost_buffer=none`, `volatility_multiplier=0.10`.
- Wrote recency artifacts under
  `artifacts/model_refinement/recency_window_diagnostic/`, including
  `window_results.csv`, `window_results.json`, `selected_window.json`,
  `recency_window_report.md`, and per-window threshold sweep artifacts.
- Added unit coverage for expanding fold parity, rolling-window train caps,
  chronological train-before-validation order, invalid policy rejection,
  holdout isolation for rolling holdout fits, development-only recency
  selection, and diagnostic summary row fields.

### Files changed

- `rebuild/src/trader/modeling/thresholds.py`
- `rebuild/scripts/run_recency_window_diagnostic.py`
- `rebuild/tests/unit/modeling/test_thresholds.py`
- `rebuild/tests/unit/modeling/test_recency_windows.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/modeling/test_thresholds.py tests/unit/modeling/test_recency_windows.py`
  from `rebuild/`: passed, `14 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/run_recency_window_diagnostic.py --overwrite` from `rebuild/`:
  passed; selected `expanding` with threshold `0.30`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `112 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_recency_window_diagnostic.py` from `rebuild/`: passed.
- Final `git diff --check` from the repository root: passed.

### Results

- Development-selected train window: `expanding`.
- Selected probability threshold: `0.30`.
- Development median return: about `+2.41%`.
- Development median max drawdown: about `-5.38%`.
- Development median turnover: about `2.04`.
- Holdout total return: about `-6.06%`.
- Holdout max drawdown: about `-11.65%`.
- Holdout trades: `2`.
- Holdout turnover: about `1.94`.
- Holdout exposure: about `99.86%`.
- Rolling windows did not improve holdout return or reduce exposure versus the
  expanding baseline; all policies had the same negative holdout return and
  near-full holdout exposure.

### Decisions

- Recency dynamics are not confirmed as the primary failure mode on the
  current five-month dataset.
- Phase 11 remains blocked because the development-selected policy did not
  confirm on holdout.
- Proceed to market feature group experiments rather than changing model class,
  adding shorting, or starting paper trading.

### Remaining blockers

- Phase 11 remains blocked.
- No target/horizon/window policy is accepted for paper trading.
- Market feature groups have not started.

### Recommended next work

- Continue to `model_refinement/03_MARKET_FEATURE_GROUPS.md` using the target
  grid and recency-window artifacts as context.

## 2026-07-12 — Model Refinement 03 market feature groups

### Completed

- Added strict `features.enabled_groups` config support with default baseline
  behavior preserved by `enabled_groups: ["baseline"]`.
- Added allowed market feature groups: `trend`, `volatility`, `volume`,
  `calendar`, and `momentum_reversal`.
- Added causal feature generation for each group, with clipped model columns
  and missingness flags for non-calendar features and raw bounded encodings
  for calendar features.
- Replaced fixed model-feature selection in the model/sweep path with
  config-aware feature-name helpers while preserving the baseline
  `MODEL_FEATURE_COLUMNS` tuple for existing callers.
- Added Model Refinement 03 orchestration that evaluates baseline alone and
  baseline plus one added market feature group at a time under the selected
  target-grid winner configuration:
  `horizon_bars=12`, `cost_buffer=none`, `volatility_multiplier=0.10`.
- Wrote feature-group artifacts under
  `artifacts/model_refinement/market_feature_groups/`.

### Files changed

- `rebuild/configs/baseline.yaml`
- `rebuild/src/trader/config.py`
- `rebuild/src/trader/features/market.py`
- `rebuild/src/trader/modeling/baseline.py`
- `rebuild/src/trader/modeling/thresholds.py`
- `rebuild/src/trader/modeling/market_feature_groups.py`
- `rebuild/scripts/run_market_feature_group_experiment.py`
- `rebuild/tests/integration/test_offline_baseline_workflow.py`
- `rebuild/tests/unit/features/test_market_features.py`
- `rebuild/tests/unit/modeling/test_market_feature_groups.py`
- `rebuild/tests/unit/test_config.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/test_config.py tests/unit/features/test_market_features.py
  tests/unit/modeling/test_baseline.py
  tests/unit/modeling/test_market_feature_groups.py` from `rebuild/`:
  passed, `49 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_market_feature_group_experiment.py` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/run_market_feature_group_experiment.py --overwrite` from
  `rebuild/`: passed; selected `baseline` with threshold `0.30`.
- Initial full `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests`
  from `rebuild/`: failed because the offline integration test's inline config
  lacked the new required `features.enabled_groups`; fixed the fixture to
  declare `baseline`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `129 passed, 24 warnings`.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall
  -q src/trader` from `rebuild/`: passed.
- Final `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/run_threshold_sweep.py scripts/run_target_horizon_grid.py
  scripts/run_recency_window_diagnostic.py
  scripts/run_market_feature_group_experiment.py` from `rebuild/`: passed.
- Final `git diff --check` from the repository root: passed.

### Results

- Development-selected feature group: `baseline`.
- Selected probability threshold: `0.30`.
- Development median return: about `+2.41%`.
- Development median max drawdown: about `-5.38%`.
- Development median turnover: about `2.04`.
- Holdout total return: about `-6.06%`.
- Holdout max drawdown: about `-11.65%`.
- Holdout trades: `2`.
- Holdout turnover: about `1.94`.
- Holdout exposure: about `99.86%`.
- Added feature groups did not improve development selection versus baseline:
  `baseline__calendar` was the closest by development median return at about
  `+2.21%`, still below baseline's about `+2.41%`.
- All feature-group candidates had the same negative holdout return and
  near-full holdout exposure on the current five-month dataset.

### Decisions

- Keep the production/default baseline feature set unchanged as
  `enabled_groups: ["baseline"]`.
- Do not enable any new market feature group by default.
- Holdout remains confirmation only; selection continued to use development
  folds only.
- No sentiment, shorting, model-class change, hyperparameter tuning, paper
  trading, or live trading was added.

### Remaining blockers

- Phase 11 remains blocked.
- No target/horizon/window/feature-group candidate is accepted for paper
  trading.

### Recommended next work

- Do not start paper trading.
- Either gather a longer market-only dataset for the same refinement suite or
  define the next constrained market-only diagnostic before reconsidering the
  Phase 11 gate.

## 2026-07-12 — Model Refinement 04 experiment orchestration

### Completed

- Added `crypto-trader run-experiment-grid` for deterministic named-variant
  model-refinement comparisons from a saved raw market dataset.
- Added strict experiment YAML parsing with unique path-safe variant names,
  per-variant full config loading, dot-path overrides, clear unknown override
  failures, and duplicate variant-name failures.
- Added per-variant resolved config artifacts under
  `artifacts/experiments/<run_id>/variants/<variant_name>/resolved_config.yaml`.
- Reused `run_threshold_sweep`, config-aware `model_feature_columns(config)`,
  existing causal feature building, Logistic Regression, chronological
  development/holdout splitting, and long/cash backtesting.
- Added output artifacts:
  `experiment_config.yaml`, `variant_summary.csv`, `target_distribution.csv`,
  `threshold_summary.csv`, `fold_metrics.csv`, `holdout_metrics.csv`, and
  `benchmark_metrics.csv`.
- Ranking uses development folds only: selected threshold required,
  non-negative median development return required, median development return
  must beat cash, then ranks by median development return with drawdown and
  turnover tie-breaks. Holdout metrics are written separately and are not used
  for selection.
- Added sample experiment config
  `configs/experiments/market_signal_grid.yaml`.

### Files changed

- `rebuild/src/trader/modeling/experiments.py`
- `rebuild/src/trader/cli.py`
- `rebuild/configs/experiments/market_signal_grid.yaml`
- `rebuild/tests/integration/test_experiment_grid.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/test_cli.py tests/integration/test_experiment_grid.py` from
  `rebuild/`: passed, `8 passed, 20 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `134 passed, 44 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- No scripts were changed or added, so no script `py_compile` command was
  required for Model Refinement 04.
- `git diff --check` from the repository root: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader
  run-experiment-grid --market-data
  artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet
  --experiment-config configs/experiments/market_signal_grid.yaml
  --output-dir artifacts/experiments --run-id market-refinement-04` from
  `rebuild/`: passed and wrote
  `artifacts/experiments/market-refinement-04/`.

### Results

- Experiment artifact path:
  `rebuild/artifacts/experiments/market-refinement-04/`.
- Development-ranked selected variant: `h12_none_v010_baseline`.
- Selected threshold: `0.30`.
- Development median total return: about `+2.41%`.
- Development median cash total return: `0.00%`.
- Development median max drawdown: about `-5.38%`.
- Development median turnover: about `2.04`.
- Holdout total return: about `-6.06%`.
- Holdout max drawdown: about `-11.65%`.
- Holdout trade count: `2`.
- Holdout turnover: about `1.94`.
- Holdout exposure: about `99.86%`.
- Holdout benchmark confirmation:
  cash `0.00%`, buy-and-hold about `-6.06%`, momentum-24h about `-13.51%`.
- All named feature-group variants ranked below the baseline-only variant on
  development folds. All variants had the same poor holdout return and
  near-full holdout exposure on the current five-month dataset.

### Decisions

- Keep the selected target-grid candidate:
  `horizon_bars=12`, `cost_buffer=none`,
  `volatility_multiplier=0.10`, threshold `0.30`.
- Keep expanding train windows and baseline-only market features.
- Do not start paper trading. Holdout confirmation remains poor despite the
  development-ranked candidate beating cash in development folds.
- No sentiment, shorting, model-class change, hyperparameter tuning, paper
  trading, or live trading was added.

### Remaining blockers

- Phase 11 remains blocked.
- No model-refinement candidate is accepted for paper trading because holdout
  return remains about `-6.06%` with about `-11.65%` max drawdown and about
  `99.86%` exposure.

### Recommended next work

- Do not start paper trading.
- Either gather a longer saved market dataset and rerun the now-unified
  experiment grid, or define the next constrained market-only diagnostic before
  reconsidering the Phase 11 gate.

## 2026-07-12 — Model Refinement 05

### Completed

- Added `crypto-trader run-sentiment-gate` for the controlled sentiment
  reevaluation gate.
- The command accepts `--market-data`, `--hourly-sentiment`, `--output-dir`,
  and `--run-id`.
- The gate reads only saved market and hourly sentiment Parquet datasets. It
  does not fetch Reddit, Binance, or external data.
- The gate rebuilds market features with the fixed selected market-only
  controls:
  `horizon_bars=12`, `cost_buffer=none`,
  `volatility_multiplier=0.10`, threshold metadata `0.30`, expanding train
  windows, and baseline-only market features.
- The gate evaluates `market_only`, the Phase 10 sentiment variants in order,
  and a clearly labeled `sentiment_only` research diagnostic.
- The gate writes:
  `sentiment_gate_summary.csv`, `sentiment_target_distribution.csv`,
  `sentiment_threshold_summary.csv`, `sentiment_fold_metrics.csv`,
  `sentiment_holdout_metrics.csv`, `sentiment_benchmark_metrics.csv`,
  `sentiment_feature_diagnostics.csv`,
  `sentiment_gate_decision.json`, and per-variant resolved config and feature
  column metadata.
- Added Model Refinement 06 planning markdown for a future controlled model
  class comparison. No Model Refinement 06 implementation was added.

### Files changed

- `rebuild/src/trader/sentiment/gate.py`
- `rebuild/src/trader/cli.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/tests/unit/sentiment/test_experiment_gate.py`
- `rebuild/tests/integration/test_experiment_grid.py`
- `rebuild/codex_rebuild/model_refinement/06_MODEL_CLASS_COMPARISON.md`
- `rebuild/codex_rebuild/model_refinement/README.md`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/test_cli.py tests/unit/sentiment
  tests/integration/test_experiment_grid.py` from `rebuild/`: passed,
  `26 passed, 20 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `142 passed, 44 warnings`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check` from the repository root: passed.

### Results

- No real five-month sentiment gate artifact was produced because there is no
  saved hourly sentiment dataset under `rebuild/artifacts/`.
- Deterministic local fixture gate runs were produced only inside pytest
  temporary directories.
- The fixture decision path writes
  `sentiment_kept_for_phase11=false` and
  `phase11_remains_blocked=true`.
- `sentiment_only` is always marked research-only and cannot become the Phase
  11 candidate.
- Selected development-ranked sentiment variant from a real run: not available
  until a saved hourly sentiment dataset exists.
- Holdout confirmation from a real run: not available until a saved hourly
  sentiment dataset exists.

### Decisions

- Sentiment remains research-only until a real saved hourly sentiment dataset
  is provided and a gate run passes the strict keep policy.
- Model Refinement 06 was added as planning only and was not implemented.
- Phase 11 remains blocked.

### Remaining blockers

- A real five-month sentiment gate still requires a saved hourly sentiment
  Parquet dataset readable by `read_hourly_sentiment_dataset`.
- Phase 11 remains blocked because the current market-only holdout remains poor
  and no sentiment confirmation has been run on a real saved hourly sentiment
  dataset.

### Recommended next work

- Produce or locate the saved hourly sentiment dataset without fetching during
  the gate command, then run `crypto-trader run-sentiment-gate` against the
  existing saved market dataset.
- If sentiment fails the gate, keep it research-only and proceed only to the
  controlled Model Refinement 06 comparison if the user explicitly wants that
  diagnostic.

## Update template

## 2026-07-12 — Server CSV Sentiment Build Script

### Completed

- Added a reproducible script to convert existing server Reddit CSV files into
  the saved hourly sentiment Parquet dataset required by Model Refinement 05.
- The script reads `../reddit_archive_server.csv` and
  `../master_reddit_server.csv` by default, deduplicates by Reddit submission
  id, records the converted raw post-only dataset, scores title plus selftext,
  builds hourly sentiment features, and writes the hourly dataset.
- The script supports `--scorer vader` for the research run and
  `--scorer lexicon` for deterministic local testing.
- The available server CSVs contain posts/submissions only. No comment bodies
  were present, so the generated sentiment is posts-only; comment sentiment
  columns remain unavailable.
- Ran Model Refinement 05 against the VADER-scored posts-only hourly sentiment
  dataset.

### Files changed

- `rebuild/scripts/build_hourly_sentiment_from_server_csv.py`
- `rebuild/tests/unit/sentiment/test_server_csv_sentiment_script.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/sentiment/test_server_csv_sentiment_script.py` from `rebuild/`:
  passed, `1 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m py_compile
  scripts/build_hourly_sentiment_from_server_csv.py` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python
  scripts/build_hourly_sentiment_from_server_csv.py --scorer lexicon
  --raw-dataset-id reddit_server_csv_posts_only_raw_lexicon
  --hourly-dataset-id reddit_server_csv_posts_only_hourly_lexicon` from
  `rebuild/`: passed; wrote the lexicon proof dataset.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra sentiment python
  scripts/build_hourly_sentiment_from_server_csv.py --scorer vader
  --raw-dataset-id reddit_server_csv_posts_only_raw_vader
  --hourly-dataset-id reddit_server_csv_posts_only_hourly_vader` from
  `rebuild/`: initially failed because optional sentiment dependencies were
  not installed and network was restricted; rerun with approved network access
  passed and installed the sentiment extra dependencies.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader
  run-sentiment-gate --market-data
  artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet
  --hourly-sentiment
  artifacts/sentiment/hourly/reddit_server_csv_posts_only_hourly_vader.parquet
  --output-dir artifacts/experiments --run-id
  sentiment-gate-05-posts-only-vader` from `rebuild/`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/sentiment` from `rebuild/`: passed, `17 passed`.
- Final `git diff --check` from the repository root: passed.

### Results

- Raw VADER post-only sentiment dataset:
  `rebuild/artifacts/sentiment/raw/reddit_server_csv_posts_only_raw_vader/`.
- Hourly VADER post-only sentiment dataset:
  `rebuild/artifacts/sentiment/hourly/reddit_server_csv_posts_only_hourly_vader.parquet`.
- Hourly VADER dataset rows: `6026`.
- Hourly VADER dataset range: `2025-09-06T20:00:00Z` through
  `2026-05-15T21:00:00Z`.
- Phase 05 posts-only VADER gate artifact:
  `rebuild/artifacts/experiments/sentiment-gate-05-posts-only-vader/`.
- Gate decision: `keep_sentiment_research_only`.
- Selected development-ranked sentiment variant: none.
- Holdout confirmation: no sentiment variant improved holdout net return
  versus `market_only`; most matched the market-only holdout return of about
  `-6.06%`, while `sentiment_lag_1h` was slightly worse at about `-6.23%`.
- `sentiment_only` remained research-only and ineligible for Phase 11.

### Decisions

- Treat the generated VADER result as a posts-only sentiment diagnostic, not as
  a complete submissions-plus-comments Phase 10 result.
- Sentiment remains research-only.
- Phase 11 remains blocked.

### Remaining blockers

- True comment sentiment still requires actual comment body data.
- The existing server CSV sentiment coverage has a gap between
  `2026-01-30T21:13:09Z` and `2026-04-15T21:34:56Z`.

### Recommended next work

- Do not start paper trading.
- If comment bodies exist elsewhere, convert them into the sentiment raw
  comment schema and rerun the hourly sentiment build plus Phase 05 gate.

## 2026-07-13 — Model Refinement 06: Model Class Comparison

### Completed

- Added `crypto-trader run-model-class-comparison` for a reproducible,
  saved-market-only model-class comparison.
- Compared Logistic Regression, Random Forest, and HistGradientBoosting under
  identical fixed controls: `horizon_bars=12`, `cost_buffer=none`,
  `volatility_multiplier=0.10`, baseline-only market features, expanding
  development windows, fixed threshold candidates, final holdout separation,
  costs, and long/cash backtest behavior.
- Added dependency-gated XGBoost handling. XGBoost writes a skipped diagnostic
  row by default and is skipped cleanly if explicitly enabled without the
  optional dependency.
- Wrote per-candidate resolved config, feature columns, model metadata,
  development threshold summaries, fold metrics, holdout metrics, benchmark
  metrics, diagnostics, and decision artifacts.
- Documented probability comparison as `uncalibrated_predict_proba`; no
  calibration was introduced.
- Added unit and integration coverage for CLI help, offline execution,
  identical config/features/split signatures, development-only ranking,
  deterministic output, skipped XGBoost behavior, decision rejection, and Phase
  11 remaining blocked.

### Files changed

- `rebuild/src/trader/modeling/model_class_comparison.py`
- `rebuild/src/trader/cli.py`
- `rebuild/tests/unit/modeling/test_model_class_comparison.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/tests/integration/test_experiment_grid.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/modeling/test_model_class_comparison.py -q` from `rebuild/`:
  passed, `7 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/unit/test_cli.py
  tests/unit/modeling tests/integration/test_experiment_grid.py` from
  `rebuild/`: passed, `56 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `152 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check` from the repository root: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader
  run-model-class-comparison --market-data
  artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet
  --output-dir artifacts/experiments --run-id model-class-comparison-06` from
  `rebuild/`: passed.

### Results

- Artifact path:
  `rebuild/artifacts/experiments/model-class-comparison-06/`.
- Development-ranked selected model class: `logistic_regression`.
- Logistic Regression selected threshold: `0.30`.
- Logistic Regression median development total return: about `2.41%`.
- Random Forest median development total return: about `1.84%`.
- Gradient Boosting median development total return: about `1.84%`.
- Holdout confirmation result: no non-logistic candidate improved holdout net
  return versus Logistic Regression. The selected RF and Gradient Boosting
  grids matched Logistic Regression holdout return at about `-6.06%`, max
  drawdown about `-11.65%`, turnover about `1.94`, and exposure about
  `99.86%`.
- Decision: `model_class_change_rejected`.
- XGBoost status: skipped by default with reason
  `xgboost disabled; rerun with --enable-xgboost to evaluate`.
- Phase 11 remains blocked.

### Decisions

- Model class change does not help under the controlled Refinement 06 setup.
- Keep Logistic Regression as the baseline research model.
- Do not use holdout for model-class or hyperparameter ranking.
- Do not start paper trading.

### Remaining blockers

- The fixed market-only candidate still has poor holdout performance and near
  full exposure.
- Sentiment remains research-only from Refinement 05.
- Phase 11 remains blocked until a candidate satisfies the paper-trading gate
  and the user explicitly approves proceeding.

### Recommended next work

- Do not proceed to paper trading.
- Reassess the target/signal design or exposure-control policy before any
  further paper-trading gate attempt.

### XGBoost follow-up

- Ran `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev --with xgboost
  crypto-trader run-model-class-comparison --market-data
  artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet
  --output-dir artifacts/experiments --run-id
  model-class-comparison-06-xgboost-enabled --enable-xgboost` from
  `rebuild/` after approving transient PyPI package download.
- Artifact path:
  `rebuild/artifacts/experiments/model-class-comparison-06-xgboost-enabled/`.
- XGBoost evaluated successfully with the tiny predefined grid.
- Selected XGBoost grid: `max_depth=3`, `learning_rate=0.03`,
  `n_estimators=100`, `min_child_weight=20`.
- XGBoost development median total return: about `1.84%`, below Logistic
  Regression's about `2.41%`.
- XGBoost holdout return: about `-6.06%`, max drawdown about `-11.65%`,
  turnover about `1.94`, and exposure about `99.86%`.
- Decision remained `model_class_change_rejected`; Phase 11 remains blocked.

Append a dated entry after each phase:

```markdown
## YYYY-MM-DD — Phase NN

### Completed

- ...

### Files changed

- ...

### Commands and results

- `command`: result

### Decisions

- ...

### Remaining blockers

- ...

### Recommended next phase

- ...
```

## 2026-07-13 — Non-Phase Cross-Symbol Fine-Interval Diagnostic

### Completed

- Added `crypto-trader run-symbol-interval-grid`.
- Generalized research data paths so config validation, Binance.US collection,
  storage hashing, and candle resampling support non-BTC spot symbols.
- Added Binance.US `exchangeInfo` availability checks; unavailable symbols are
  skipped and recorded in diagnostics instead of failing the full run.
- Added fixed market-only symbol/interval orchestration for:
  `BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT,
  LTCUSDT, BCHUSDT, SHIBUSDT, UNIUSDT`.
- Added default interval sweep for `1h` through `14h`, including non-day-divisor
  intervals such as `5h`, `7h`, `11h`, and `13h`.
- Kept the model fixed to baseline Logistic Regression with baseline market
  features only, one-next-candle targets, long/cash execution, and
  development-only threshold/ranking selection.
- Added required artifacts:
  `symbol_interval_summary.csv`, `threshold_summary.csv`, `fold_metrics.csv`,
  `holdout_metrics.csv`, `benchmark_metrics.csv`,
  `dataset_diagnostics.csv`, `symbol_interval_decision.json`,
  `sentiment_provenance_audit.json`, and per-symbol/per-interval config,
  resampling metadata, and feature-column files.
- Added tests for CLI help, non-BTC config/collection/storage behavior,
  symbol-aware content hashes, unavailable-symbol diagnostics, arbitrary-hour
  resampling, incomplete-bucket accounting, interval-scaled validation windows,
  one-candle target horizon, development-only ranking, holdout confirmation
  reporting, deterministic fixture output, sentiment audit status, and Phase
  11 remaining blocked.

### Files changed

- `rebuild/src/trader/cli.py`
- `rebuild/src/trader/config.py`
- `rebuild/src/trader/data/market.py`
- `rebuild/src/trader/data/storage.py`
- `rebuild/src/trader/modeling/candle_intervals.py`
- `rebuild/src/trader/modeling/symbol_interval_grid.py`
- `rebuild/tests/unit/data/test_market.py`
- `rebuild/tests/unit/data/test_storage.py`
- `rebuild/tests/unit/modeling/test_candle_intervals.py`
- `rebuild/tests/unit/modeling/test_symbol_interval_grid.py`
- `rebuild/tests/unit/test_cli.py`
- `rebuild/tests/unit/test_config.py`
- `rebuild/tests/integration/test_experiment_grid.py`
- `rebuild/codex_rebuild/HANDOFF.md`

### Commands and results

- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
  tests/unit/test_cli.py tests/unit/data tests/unit/modeling
  tests/integration/test_experiment_grid.py` from `rebuild/`: passed,
  `97 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests` from
  `rebuild/`: passed, `179 passed`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m compileall -q
  src/trader` from `rebuild/`: passed.
- `git diff --check` from the repository root: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader
  run-symbol-interval-grid --output-dir artifacts/experiments --run-id
  symbol-interval-grid-09` from `rebuild/`: passed after live Binance.US
  network approval.

### Results

- Artifact path:
  `rebuild/artifacts/experiments/symbol-interval-grid-09/`.
- Development-ranked selected candidate: `LINKUSDT` at `11h`.
- Selected threshold: `0.30`.
- Development median total return: about `6.47%`.
- Development median cash return: `0.00%`.
- Development median buy-and-hold return: about `1.31%`.
- Development median max drawdown: about `-2.43%`.
- Development median exposure: about `62.50%`.
- Development mean precision: about `42.07%`.
- Development mean recall: about `74.81%`.
- Development mean F1: about `52.73%`.
- Holdout confirmation result: confirmed.
- Holdout strategy return: about `1.18%`.
- Holdout cash return: `0.00%`.
- Holdout buy-and-hold return: about `-0.71%`.
- Holdout max drawdown: about `-11.88%`.
- Holdout trades: `10`.
- Holdout exposure: about `74.24%`.
- Decision JSON recorded:
  `altcoin_interval_improves_over_btc_12h_on_development: true`,
  `any_candidate_confirms_on_holdout: true`,
  `holdout_used_for_ranking: false`, and `phase_11_status: blocked`.

### Decisions

- Treat `LINKUSDT` `11h` as the best current market-only research candidate,
  not as a deployable trading approval.
- Holdout remains confirmation-only. It was not used to select symbols,
  intervals, thresholds, features, or model class.
- Keep Phase 11 blocked despite the confirmed holdout result because this was a
  non-phase diagnostic, not a paper-trading gate. The result is modest, has a
  meaningful holdout drawdown, and does not validate execution readiness,
  monitoring, operational controls, or user approval for paper trading.
- Treat the prior sentiment gate as
  `inconclusive_for_bitcoin_specific_sentiment`.
- Do not interpret the old sentiment artifact as Bitcoin-specific or
  coin-specific. The audit records it as server CSV, posts-only, six subreddits,
  with no current metadata proving Bitcoin specificity.

### Remaining blockers

- Phase 11 remains blocked.
- Coin-specific sentiment has not been rebuilt or tested.
- The selected `LINKUSDT` `11h` candidate needs a separate follow-up review
  before any deployment discussion, including robustness checks, liquidity and
  cost sensitivity, symbol-specific sentiment rebuild, and paper-trading gate
  criteria.

### Recommended next work

- Rebuild sentiment as symbol-specific artifacts with explicit provenance and
  symbol labels, then evaluate whether sentiment improves the selected
  symbol/interval candidate without using holdout for selection.
- Run robustness diagnostics for `LINKUSDT` `11h`: alternate date windows,
  cost/slippage sensitivity, threshold stability, exposure controls, and
  minimum liquidity checks.
- Define an explicit Phase 11 paper-trading gate for a non-BTC candidate before
  starting any paper trading.
