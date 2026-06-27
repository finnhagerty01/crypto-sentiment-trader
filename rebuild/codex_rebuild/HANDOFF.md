# Rebuild Handoff Log

## Current phase

Phase 03 is complete. Phase 04 is next.

## Completed phases

- Phase 02: Baseline and safety.
- Phase 03: Package scaffold and configuration.

## Current verification status

- Legacy suite observed on June 27, 2026: `270 passed, 57 failed, 246 warnings`.
- Replacement suite: `26 passed`.
- `rebuild/` is an isolated Python project with independent pytest discovery.

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
