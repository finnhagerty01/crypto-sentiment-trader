# Crypto Sentiment Trader Rebuild

This directory contains the clean replacement project.

The existing repository-root implementation is legacy reference material and must not be imported by this project.

Implementation is staged through:

- `codex_rebuild/00_MASTER_PLAN.md`
- `codex_rebuild/01_TARGET_ARCHITECTURE.md`
- `codex_rebuild/HANDOFF.md`

The first supported system will be a deterministic BTCUSDT one-hour, market-only Logistic Regression research and backtesting pipeline.

Live trading is intentionally unavailable.

## Project boundary

All new implementation code, configuration, tests, dependencies, and generated artifacts belong inside this directory:

```text
rebuild/
  pyproject.toml
  configs/
  src/trader/
  tests/
  artifacts/
```

Do not import from the repository-root `src/` package.

## Tests

Install the locked development environment and run replacement-package tests
from this directory:

```bash
uv sync --extra dev
uv run --extra dev pytest tests/unit tests/integration
```

The full replacement suite, including any tests added directly under `tests/`,
is:

```bash
uv run --extra dev pytest tests
```

Legacy tests use the repository-root project and remain intentionally separate.
Run them from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest pytest -q
```

To run all tests, run the legacy command from the repository root and then the
full replacement command from `rebuild/`. A green replacement suite does not
imply that the legacy suite is green.
