# Rebuild User Testing Guide

This guide is for manually checking that each completed rebuild phase works on
your machine. It should be updated as new phases are implemented.

Run all rebuild commands from the standalone project:

```bash
cd rebuild
```

Use the rebuild environment, not the legacy root project:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
```

Expected result after Phase 04:

```text
42 passed
```

## What Codex Tests Versus What You Should Test

Codex runs the required automated tests before finishing each phase and records
the commands in `rebuild/codex_rebuild/HANDOFF.md`.

You should still run a small smoke test after each phase to verify:

- your local dependencies resolve;
- the CLI entry point works in your shell;
- generated artifacts appear where expected;
- commands are not accidentally using legacy root code.

## Current Phase Status

| Phase | Status | User-facing check |
|---|---:|---|
| 02 Baseline and safety | Complete | Rebuild tests run independently from legacy tests. |
| 03 Package scaffold and config | Complete | `crypto-trader --help` works and config loads. |
| 04 Market data pipeline | Complete | `collect-market` can save BTCUSDT 1h raw data. |
| 05 Features, noise, and target | Not started | No manual check yet. |
| 06 Model and validation | Not started | No manual check yet. |
| 07 Backtest and reporting | Not started | No manual check yet. |
| 08 CLI and reproducible run | Not started | No manual check yet. |
| 09 Legacy quarantine and docs | Not started | No manual check yet. |

Phases 10 and 11 are gated future work. Do not test them until the handoff says
their entry criteria are satisfied.

## Baseline Smoke Tests

From `rebuild/`, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
```

Then check the CLI:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help
```

After Phase 04, the help output should list:

```text
collect-market
build-dataset
train
backtest
run-baseline
```

Only `collect-market` is operational after Phase 04. The other commands should
still fail clearly because their phases have not been implemented.

## Phase 03 Checks: Package and Config

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "from trader.config import load_config; c = load_config('configs/baseline.yaml'); print(c.data.symbol, c.data.interval)"
```

Expected output:

```text
BTCUSDT 1h
```

Run a placeholder command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader train
```

Expected result after Phase 04:

- exit status is nonzero;
- stderr says the command is not implemented in the current phase.

That is correct until the assigned phase for the command is complete.

## Phase 04 Checks: Market Data Pipeline

The automated tests cover schema validation, fake-client collection, fixture
loading, Parquet storage, metadata sidecars, and deterministic hashes.

To manually collect a small live sample from Binance US:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader collect-market \
  --start 2026-01-01T00:00:00Z \
  --end 2026-01-02T00:00:00Z
```

Expected result:

- the command prints the number of saved rows;
- it prints a `metadata content_hash`;
- a Parquet file is created in `artifacts/datasets/`;
- a matching `.metadata.json` sidecar is created next to it.

Example artifact names:

```text
artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260102T000000Z_<collected_at>.parquet
artifacts/datasets/btcusdt_1h_raw_20260101T000000Z_20260102T000000Z_<collected_at>.parquet.metadata.json
```

Generated files under `rebuild/artifacts/` are intentionally ignored by Git.

## Reading A Saved Dataset

After collecting data, replace `<dataset>` with the Parquet path printed by the
command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -c "from trader.data.storage import read_market_dataset; data = read_market_dataset('<dataset>'); print(data.head()); print(len(data))"
```

Expected result:

- the dataset loads without network access;
- columns are `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`;
- timestamps are UTC;
- symbol is `BTCUSDT`.

## What Not To Test Yet

Do not expect these commands to work until their phases are complete:

```bash
crypto-trader build-dataset
crypto-trader train
crypto-trader backtest
crypto-trader run-baseline
```

Do not use the root-level legacy commands as rebuild validation. The root
project still has known legacy test failures and is not the active rebuild
implementation.

Do not use these root server artifacts as test fixtures:

```text
master_reddit_server.csv
reddit_archive_server.csv
trade_logs_server.log
```

They are user-owned reference artifacts and should remain untouched unless you
explicitly decide otherwise.

## After Each Future Phase

After Codex completes a phase, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev crypto-trader --help
```

Then run the one user-facing command that the phase made operational. Check
`HANDOFF.md` for the exact command and expected result.

If the command produces artifacts, confirm that:

- the files appear under `rebuild/artifacts/`;
- metadata exists where documented;
- rerunning from saved artifacts does not require network access unless the
  command is explicitly a collection command.
