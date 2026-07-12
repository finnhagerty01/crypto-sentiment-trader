# Legacy Quarantine

This directory contains historical implementation files from the pre-rebuild
trading system. They are retained for reference and migration only. The active
system is the standalone package under `rebuild/`, invoked with
`crypto-trader`.

Do not treat files in this directory as active strategy results, supported
entry points, or current installation instructions.

## Inventory

| Path | Classification | Notes |
|---|---|---|
| `runtime/main.py` | superseded | Former coupled Reddit, market, training, paper/live loop. Root `main.py` now refuses to run it. |
| `runtime/src/analysis/models.py` | historical documentation only | Older simple model implementation; useful ideas were replaced by the tested rebuild baseline. |
| `runtime/src/analysis/sentiment.py` | future experiment | Alternate sentiment analyzer with optional transformer-era dependencies. |
| `runtime/src/analysis/engagement_validation.py` | future experiment | Engagement research utility, not part of the market-only baseline. |
| `runtime/src/backtest/` | superseded | Mutable-data backtest path replaced by deterministic rebuild backtest and reporting. |
| `runtime/src/data/archive.py` | data migration utility | Historical Reddit archive storage and migration helpers. |
| `runtime/src/data/engagement_tracker.py` | future experiment | Research-only engagement snapshot logic. |
| `runtime/src/data/market_client.py` | future experiment | Spot plus derivatives collection, excluded from the core baseline. |
| `runtime/src/data/reddit_client.py` | future experiment | Reddit collection, deferred until sentiment is tested incrementally. |
| `runtime/src/execution/live.py` | historical documentation only | Former Binance execution boundary; live trading remains disabled. |
| `runtime/src/features/sentiment_advanced.py` | future experiment | Historical VADER sentiment features, not active in the baseline. |
| `runtime/src/features/technical.py` | superseded | Broad technical feature set replaced by the small causal market-feature module. |
| `runtime/src/models/ensemble.py` | future experiment | Ensemble logic deferred until a simpler baseline earns added complexity. |
| `runtime/src/models/trading_model.py` | superseded | Former coupled feature/model path with known test failures. |
| `runtime/src/models/tuning.py` | future experiment | Optuna tuning excluded from default installation and core baseline. |
| `runtime/src/models/validation.py` | future experiment | Historical validation helpers replaced by rebuild chronological validation. |
| `runtime/src/risk/` | future experiment | Portfolio risk, stops, exits, and sizing deferred until paper-trading gates. |
| `runtime/src/utils/config.py` | superseded | Legacy config parser replaced by strict rebuild config dataclasses. |
| `runtime/configs/data.yaml` | historical documentation only | Legacy runtime config with fields outside the core baseline. |
| `runtime/tests/` | historical documentation only | Tests for the quarantined runtime; known failures are documented in `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`. |
| `runtime/pyproject.toml`, `runtime/uv.lock`, `runtime/requirements.txt` | historical documentation only | Former heavyweight dependency manifests, retained for reference only. |
| `scripts/run_backtest.py` | superseded | Old mutable-data research backtest script. |
| `scripts/collinearity_analysis.py`, `scripts/shap_analysis.py` | future experiment | Research diagnostics requiring optional heavy dependencies. |
| `docs/*.md` | historical documentation only | Old strategy docs; active rebuild docs remain outside this quarantine. |

## Active Documentation

Active project documentation remains at:

- `README.md`
- `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md`
- `rebuild/README.md`
- `rebuild/codex_rebuild/`

## User Data

Root-level server exports and local generated artifacts were not moved by this
quarantine pass. In particular, `master_reddit_server.csv`,
`reddit_archive_server.csv`, and `trade_logs_server.log` remain untouched.
