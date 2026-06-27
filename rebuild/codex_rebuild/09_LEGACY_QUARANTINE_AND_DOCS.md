# Phase 09: Legacy Quarantine and Dependency Reduction

## Entry criteria

Do not start until:

- Phases 02 through 08 are complete.
- The offline end-to-end baseline passes.
- The root README points to the standalone `rebuild/` project.
- The user has not requested preservation of the old runtime as the default.

## Objective

Make the new package the obvious active system, quarantine historical code, and reduce default dependencies without losing user data or useful history.

## Required work

1. Inventory every legacy module and classify it:
   - superseded;
   - future experiment;
   - data migration utility;
   - historical documentation only.
2. Ensure no `rebuild/src/trader/` import references legacy modules.
3. Move legacy runtime code to an intuitively named location such as:

```text
legacy/
  runtime/
  research/
  scripts/
  docs/
```

Alternatively, delete superseded tracked code if Git history is sufficient and the user has approved that direction. Prefer quarantine for the first cleanup pass.
4. Move old strategy documentation under `legacy/docs/` or add a prominent historical-status banner.
5. Keep `docs/12_REPOSITORY_REVIEW_AND_SIMPLIFICATION.md` and `docs/codex_rebuild/` active.
6. Remove stale root entry points or replace them with a clear message directing users to `crypto-trader`.
7. Reduce default dependencies to those used by the core baseline.
8. Put future sentiment dependencies in an optional dependency group.
9. Put plotting or research dependencies in an optional development/research group.
10. Remove duplicate `requirements.txt` if `pyproject.toml` is the chosen source of truth, or generate/document it consistently.
11. Ensure `.gitignore` covers generated datasets, models, backtests, logs, local credentials, and paper state.
12. Never move, rewrite, or delete the untracked server artifacts without explicit user approval.

## Required verification

- Search for imports of old package paths.
- Run the core unit and integration suites.
- Run the offline baseline command.
- Verify installation of core dependencies does not require Torch, Transformers, XGBoost, LightGBM, Optuna, SHAP, Streamlit, Plotly, or Reddit libraries.
- Confirm historical files are clearly labeled and cannot be mistaken for active results.

## Acceptance criteria

- Repository entry points and README lead to the clean system.
- Legacy code is isolated and labeled.
- Core installation is materially smaller.
- The deterministic baseline still passes after moves.
- No user data is lost.
