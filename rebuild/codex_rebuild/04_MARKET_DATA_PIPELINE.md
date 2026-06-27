# Phase 04: Market Data Pipeline

## Objective

Implement deterministic BTCUSDT one-hour OHLCV ingestion, validation, storage, and fixture support. External collection and offline loading must be separate operations.

## Required work

1. Define the canonical OHLCV schema in `rebuild/src/trader/data/schemas.py`.
2. Implement pure validation and normalization functions:
   - parse timestamps as UTC;
   - require canonical columns;
   - enforce numeric types;
   - reject duplicate `(timestamp, symbol)` rows;
   - reject non-positive prices and negative volume;
   - sort rows;
   - optionally validate hourly continuity.
3. Implement storage in `rebuild/src/trader/data/storage.py`:
   - write/read Parquet for research datasets;
   - produce a metadata JSON sidecar;
   - include symbol, interval, row count, date range, schema version, source, and content hash.
4. Implement an explicit collector in `rebuild/src/trader/data/market.py`.
5. The collector must:
   - fetch BTCUSDT one-hour candles only;
   - accept explicit start/end dates;
   - remove incomplete candles;
   - use bounded retries and timeouts;
   - return data rather than writing implicitly;
   - be replaceable with a fake client in tests.
6. Wire `collect-market` to save a versioned raw dataset.
7. Add a small deterministic fixture at `rebuild/tests/fixtures/btcusdt_1h.csv`.
8. Add unit tests using in-memory or fake responses.
9. Add an integration test that loads the local fixture, validates it, saves Parquet, reloads it, and verifies metadata/hash stability.

## Data-source decision

Use one documented source for the baseline. Do not combine Binance US spot data with global futures data. If the existing Binance library adds unnecessary weight, use a small HTTP client against the documented spot candle endpoint.

Network tests must be opt-in and excluded from the default suite.

## Boundaries

- No Reddit data.
- No funding or open interest.
- No feature engineering.
- No model training.
- No automatic fetching from backtest code.
- Do not use root-level server CSV files as test fixtures.

## Acceptance criteria

- A saved local dataset can be loaded without network access.
- Dataset metadata and content hash are deterministic.
- Incomplete and malformed candles are rejected or removed according to documented rules.
- All default tests run offline.
- `collect-market` is the only core command permitted to contact an external market API.
