# Codex Rebuild Instructions

Start with `00_MASTER_PLAN.md`.

The files in this directory are ordered work packages for separate Codex sessions. Assign one phase per session so each instance has a narrow scope and a verifiable stopping point.

## Suggested prompt for a new Codex instance

```text
Implement Phase NN from rebuild/codex_rebuild/NN_PHASE_NAME.md.

Before editing, read:
- docs/codex_rebuild/00_MASTER_PLAN.md
- docs/codex_rebuild/01_TARGET_ARCHITECTURE.md
- docs/codex_rebuild/HANDOFF.md
- docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md
- the assigned phase file

Stay within the assigned phase, preserve unrelated user changes and untracked
server artifacts, run the required tests, and update HANDOFF.md before finishing.
Do not begin the next phase. All implementation work belongs under rebuild/.
```

## Core sequence

| Phase | File | Outcome |
|---|---|---|
| 02 | `02_BASELINE_AND_SAFETY.md` | Stable test and workspace boundary |
| 03 | `03_PACKAGE_SCAFFOLD_AND_CONFIG.md` | New package, config, and CLI shell |
| 04 | `04_MARKET_DATA_PIPELINE.md` | Deterministic saved BTC OHLCV |
| 05 | `05_FEATURES_NOISE_AND_TARGET.md` | Causal features, filtering, and label |
| 06 | `06_MODEL_AND_VALIDATION.md` | Logistic Regression and chronological validation |
| 07 | `07_BACKTEST_AND_REPORTING.md` | Long-or-cash backtest and artifacts |
| 08 | `08_CLI_AND_REPRODUCIBLE_RUN.md` | One-command offline baseline |
| 09 | `09_LEGACY_QUARANTINE_AND_DOCS.md` | Clean active tree and reduced dependencies |

## Gated sequence

| Phase | File | Start condition |
|---|---|---|
| 10 | `10_SENTIMENT_EXPERIMENT.md` | Accepted market-only baseline and explicit user request |
| 11 | `11_PAPER_TRADING_GATE.md` | Stable model artifacts and explicit user request |

Do not ask an instance to implement multiple phases unless the earlier phase is already complete and its acceptance criteria are recorded in `HANDOFF.md`.
