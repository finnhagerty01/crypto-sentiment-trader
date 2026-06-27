# Phase 03: Package Scaffold and Configuration

## Objective

Complete the initialized `rebuild/src/trader/` package, typed configuration objects, and initial CLI shell without implementing data collection, modeling, or backtesting.

## Required work

1. Verify and complete the package structure defined in `01_TARGET_ARCHITECTURE.md`.
2. Add `rebuild/configs/baseline.yaml` containing only fields required by the planned baseline.
3. Implement configuration dataclasses in `rebuild/src/trader/config.py`.
4. Implement YAML loading with:
   - explicit validation;
   - clear errors for missing or unknown fields;
   - no directory creation during import;
   - no environment-variable or credential requirement.
5. Add a CLI entry point in `rebuild/src/trader/cli.py` with placeholder subcommands:
   - `collect-market`;
   - `build-dataset`;
   - `train`;
   - `backtest`;
   - `run-baseline`.
6. Placeholder commands must fail with a clear “not implemented” message rather than invoking legacy code.
7. Configure the package entry point in `pyproject.toml`, preferably:

```toml
[project.scripts]
crypto-trader = "trader.cli:main"
```

8. Add unit tests for configuration loading and validation.

## Configuration requirements

- Validate positive windows, capital, and costs.
- Validate fractions and thresholds are in meaningful ranges.
- Validate that the configured symbol is `BTCUSDT` for the baseline.
- Reject unknown keys to prevent stale configuration from appearing active.
- Paths should resolve relative to `rebuild/` or an explicit working directory, not the importing module’s current location.

## Boundaries

- Do not import legacy config classes.
- Do not create Binance or Reddit clients.
- Do not implement actual subcommand behavior.
- Do not add speculative configuration.
- Do not move old code.

## Acceptance criteria

- `import trader` succeeds without side effects.
- `crypto-trader --help` succeeds.
- `rebuild/configs/baseline.yaml` round-trips into validated dataclasses.
- Invalid and unknown configuration fields fail with useful messages.
- Unit tests pass without network or credentials.
- `HANDOFF.md` identifies the exact public config API for later phases.
