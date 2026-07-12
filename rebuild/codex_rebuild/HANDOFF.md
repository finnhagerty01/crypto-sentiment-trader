# Rebuild Handoff Log

## Current phase

Phase 09 is complete. Phase 10 is next, but remains gated future sentiment work.

## Completed phases

- Phase 02: Baseline and safety.
- Phase 03: Package scaffold and configuration.
- Phase 04: Market data pipeline.
- Phase 05: Features, noise handling, and target.
- Phase 06: Model and chronological validation.
- Phase 07: Backtest and reporting.
- Phase 08: CLI and reproducible baseline run.
- Phase 09: Legacy quarantine and dependency reduction.

## Current verification status

- Legacy suite observed on June 27, 2026: `270 passed, 57 failed, 246 warnings`.
- Replacement suite: `78 passed`.
- `rebuild/` is an isolated Python project with independent pytest discovery.
- Default rebuild installation excludes Reddit, transformer, tree-model,
  plotting, and research dependencies.

## Decisions

- Build the replacement under `rebuild/src/trader/` before quarantining legacy code.
- Run rebuild dependency and test commands from `rebuild/`.
- Core baseline is BTCUSDT one-hour market data only.
- Core model is Logistic Regression.
- Core evaluation is chronological and long-or-cash.
- EWMA/Kalman sentiment work is deferred until the market-only baseline is accepted.

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

## Update template

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
