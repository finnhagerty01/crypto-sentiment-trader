# Phase 02: Baseline and Safety

## Objective

Establish a controlled starting point before reorganizing code. This phase records current behavior, creates a clean test boundary for the replacement package, and prevents accidental coupling to legacy failures.

## Required work

1. Inspect Git status and preserve all user changes and untracked server artifacts.
2. Confirm the legacy test result and categorize failures without fixing the legacy strategy.
3. Add `pytest` as a development dependency using the project’s chosen dependency mechanism.
4. Verify the initialized `rebuild/tests/unit/`, `rebuild/tests/integration/`, and `rebuild/tests/fixtures/` structure.
5. Configure test discovery so new tests can be run independently from legacy tests.
6. Add a minimal import test for `trader` only after the package scaffold exists, or document that it is deferred to Phase 03.
7. Verify generated `rebuild/artifacts/` contents are ignored while deliberate tiny fixture files under `rebuild/tests/fixtures/` remain trackable.
8. Record exact commands for:
   - legacy tests;
   - core replacement tests;
   - all tests.

## Boundaries

- Do not fix the BTC beta implementation.
- Do not rewrite the archive.
- Do not change trading behavior.
- Do not install or invoke external services.
- Do not move legacy modules.

## Suggested test commands

Choose stable commands and document them in the README or `pyproject.toml`. A likely pattern is:

```bash
cd rebuild
uv run --extra dev pytest tests/unit tests/integration
uv run --extra dev pytest tests
```

Legacy tests remain in the repository-root `tests/` directory and are outside the standalone rebuild project. Document separate root and rebuild commands; do not represent rebuild success as repair of the legacy suite.

## Acceptance criteria

- `pytest` is reproducibly available as a development dependency.
- Core test directories exist.
- Generated artifacts are ignored.
- No existing user data is changed.
- Current legacy failures are documented in `HANDOFF.md`.
- The next instance has an unambiguous command for running only replacement-package tests.
