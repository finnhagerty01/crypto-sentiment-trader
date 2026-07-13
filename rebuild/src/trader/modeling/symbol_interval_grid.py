"""Cross-symbol fine interval diagnostic orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from trader.backtest.benchmarks import run_benchmarks
from trader.config import TraderConfig, load_config
from trader.data.market import (
    SOURCE_NAME,
    BinanceUsSpotKlineClient,
    collect_market_data,
)
from trader.data.storage import build_metadata, read_market_dataset, write_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.modeling.baseline import BaselineLogisticModel
from trader.modeling.candle_intervals import (
    _add_buy_hold_returns,
    _concat_frames,
    _fold_signature,
    _frame_signature,
    _holdout_buy_hold_return,
    _interval_hours,
    _interval_summary_row,
    _write_csv,
    _write_json,
    _write_yaml,
    interval_comparison_config,
    rank_intervals,
    resample_ohlcv,
    select_interval_threshold,
    summarize_interval_thresholds,
)
from trader.modeling.experiments import _target_distribution_dict
from trader.modeling.thresholds import (
    DEFAULT_THRESHOLDS,
    _evaluate_holdout,
    run_threshold_sweep,
)
from trader.modeling.validation import split_final_holdout


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "SHIBUSDT",
    "UNIUSDT",
)
DEFAULT_INTERVALS = (
    "1h",
    "2h",
    "3h",
    "4h",
    "5h",
    "6h",
    "7h",
    "8h",
    "9h",
    "10h",
    "11h",
    "12h",
    "13h",
    "14h",
)
DEFAULT_START = "2026-01-01T00:00:00Z"
DEFAULT_END = "2026-06-01T00:00:00Z"


class SymbolIntervalGridError(ValueError):
    """Raised when the cross-symbol interval grid cannot be completed."""


def run_symbol_interval_grid(
    *,
    output_dir: str | Path,
    run_id: str,
    config_path: str | Path = "configs/baseline.yaml",
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    intervals: tuple[str, ...] = DEFAULT_INTERVALS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_development_exposure: float = 0.80,
    client: BinanceUsSpotKlineClient | None = None,
) -> dict[str, Any]:
    """Run the offline/collect-if-needed symbol interval diagnostic."""

    if not symbols:
        raise SymbolIntervalGridError("at least one symbol is required")
    if not intervals:
        raise SymbolIntervalGridError("at least one interval is required")
    active_client = client or BinanceUsSpotKlineClient()
    available_symbols = active_client.available_symbols()
    base_config = load_config(config_path)
    interval_hours = {interval: _interval_hours(interval) for interval in intervals}

    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing symbol interval run: {run_dir}")
    run_dir.mkdir(parents=True)
    symbols_dir = run_dir / "symbols"
    symbols_dir.mkdir()

    summary_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for symbol_order, symbol in enumerate(symbols):
        symbol_dir = symbols_dir / symbol
        symbol_dir.mkdir()
        if symbol not in available_symbols:
            diagnostic_rows.append(
                {
                    "symbol": symbol,
                    "interval": None,
                    "collection_status": "skipped_unavailable",
                    "available_on_binance_us": False,
                }
            )
            continue

        market_path = symbol_dir / f"{symbol.lower()}_1h.parquet"
        if market_path.exists():
            source_market = read_market_dataset(market_path, symbol=symbol, interval="1h")
            collection_status = "reused_saved_1h"
        else:
            source_market = collect_market_data(
                start=start,
                end=end,
                symbol=symbol,
                interval="1h",
                client=active_client,
            )
            write_market_dataset(
                source_market,
                market_path,
                source=SOURCE_NAME,
                symbol=symbol,
                interval="1h",
            )
            collection_status = "collected_1h"

        for interval_order, interval in enumerate(intervals):
            hours = interval_hours[interval]
            interval_dir = symbol_dir / "intervals" / interval
            interval_dir.mkdir(parents=True)

            market, diagnostics = resample_ohlcv(source_market, interval)
            config = _symbol_interval_config(
                base_config,
                symbol=symbol,
                interval=interval,
                interval_hours=hours,
            )
            features = build_feature_dataset(market, config)
            feature_names = model_feature_columns(config)
            split = split_final_holdout(features, config.validation)

            _write_yaml(interval_dir / "resolved_config.yaml", asdict(config))
            _write_json(
                interval_dir / "resampled_market_metadata.json",
                {
                    **build_metadata(
                        market,
                        source="resampled_from_saved_1h",
                        symbol=symbol,
                        interval=interval,
                    ).as_dict(),
                    "source_interval": "1h",
                    "interval_hours": hours,
                    **diagnostics,
                },
            )
            _write_json(
                interval_dir / "feature_columns.json",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "feature_columns": list(feature_names),
                    "feature_count": len(feature_names),
                    "feature_data_sha256": _frame_signature(features),
                    "fold_signature_sha256": _fold_signature(features, config),
                },
            )

            threshold_result = run_threshold_sweep(
                features,
                config,
                feature_names=feature_names,
                thresholds=DEFAULT_THRESHOLDS,
            )
            fold_metrics = _add_buy_hold_returns(threshold_result.fold_metrics, split, config)
            threshold_summary = summarize_interval_thresholds(
                fold_metrics,
                max_development_exposure=max_development_exposure,
            )
            selected_threshold = select_interval_threshold(threshold_summary)
            holdout_metrics = (
                _evaluate_holdout(
                    split.development,
                    split.holdout,
                    config,
                    feature_names=feature_names,
                    target_column="target",
                    threshold=selected_threshold,
                    model_factory=BaselineLogisticModel,
                    train_window_policy="expanding",
                )
                if selected_threshold is not None
                else {"status": "not_evaluated", "selected_threshold": None}
            )
            holdout_metrics = {
                **holdout_metrics,
                "buy_hold_total_return": _holdout_buy_hold_return(split.holdout, config),
            }

            threshold_summary.insert(0, "interval", interval)
            threshold_summary.insert(0, "symbol", symbol)
            threshold_frames.append(threshold_summary)
            fold_metrics = fold_metrics.copy()
            fold_metrics.insert(0, "interval", interval)
            fold_metrics.insert(0, "symbol", symbol)
            fold_frames.append(fold_metrics)
            holdout_rows.append({"symbol": symbol, "interval": interval, **holdout_metrics})
            diagnostic_rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "collection_status": collection_status,
                    "available_on_binance_us": True,
                    "interval_hours": hours,
                    **diagnostics,
                    "feature_row_count": len(features),
                    "development_row_count": len(split.development),
                    "holdout_row_count": len(split.holdout),
                    "minimum_train_bars": config.validation.minimum_train_bars,
                    "test_bars": config.validation.test_bars,
                    "step_bars": config.validation.step_bars,
                    "target_horizon_bars": config.target.horizon_bars,
                }
            )
            for benchmark_name, benchmark in run_benchmarks(
                split.holdout,
                backtest_config=config.backtest,
                costs_config=config.costs,
            ).items():
                benchmark_rows.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "benchmark": benchmark_name,
                        **benchmark.metrics,
                    }
                )

            summary_row = _interval_summary_row(
                interval=interval,
                interval_order=interval_order,
                interval_hours=hours,
                config=config,
                feature_names=feature_names,
                target_distribution=_target_distribution_dict(features),
                threshold_summary=threshold_summary,
                selected_threshold=selected_threshold,
                holdout_metrics=holdout_metrics,
            )
            summary_rows.append(
                {
                    "symbol": symbol,
                    "symbol_order": symbol_order,
                    **summary_row,
                }
            )

    symbol_interval_summary = rank_symbol_intervals(pd.DataFrame(summary_rows))
    decision = symbol_interval_decision(symbol_interval_summary)
    symbol_interval_summary = _apply_symbol_decision_flags(
        symbol_interval_summary,
        decision,
    )
    threshold_summary = _concat_frames(threshold_frames)
    fold_metrics = _concat_frames(fold_frames)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)
    dataset_diagnostics = pd.DataFrame(diagnostic_rows)

    _write_csv(symbol_interval_summary, run_dir / "symbol_interval_summary.csv")
    _write_csv(threshold_summary, run_dir / "threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "benchmark_metrics.csv")
    _write_csv(dataset_diagnostics, run_dir / "dataset_diagnostics.csv")
    _write_json(run_dir / "symbol_interval_decision.json", decision)
    _write_json(run_dir / "sentiment_provenance_audit.json", sentiment_provenance_audit())
    return {
        "run_dir": run_dir,
        "symbol_interval_summary": symbol_interval_summary,
        "decision": decision,
    }


def rank_symbol_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank symbol/interval candidates using development metrics only."""

    if summary.empty:
        return summary
    ranked = rank_intervals(summary)
    ranked = ranked.copy()
    return ranked.sort_values(
        by=["rank", "symbol_order", "interval_order"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def symbol_interval_decision(summary: pd.DataFrame) -> dict[str, Any]:
    """Return confirmation-only decision metadata for the symbol interval grid."""

    selected = summary.loc[summary["rank"] == 1] if "rank" in summary else pd.DataFrame()
    selected_row = selected.iloc[0] if not selected.empty else None
    holdout_confirmed = _row_confirms_on_holdout(selected_row)
    any_candidate_confirms = bool(
        any(_row_confirms_on_holdout(row) for _, row in summary.iterrows())
    ) if not summary.empty else False
    altcoin_improves_over_btc_12h = _altcoin_improves_over_btc_12h(summary)
    return {
        "selected_development_ranked_symbol_interval": (
            None
            if selected_row is None
            else {
                "symbol": selected_row.get("symbol"),
                "interval": selected_row.get("interval"),
            }
        ),
        "selected_threshold": (
            None if selected_row is None else _none_if_nan(selected_row.get("selected_threshold"))
        ),
        "holdout_confirmation_result": (
            "confirmed" if holdout_confirmed else "not_confirmed"
        ),
        "altcoin_interval_improves_over_btc_12h_on_development": (
            altcoin_improves_over_btc_12h
        ),
        "any_candidate_confirms_on_holdout": any_candidate_confirms,
        "holdout_used_for_ranking": False,
        "phase_11_status": "blocked",
    }


def sentiment_provenance_audit() -> dict[str, Any]:
    """Return the required audit for the prior sentiment artifact."""

    return {
        "existing_sentiment_artifact": {
            "source": "server CSV",
            "content_scope": "posts-only",
            "subreddit_scope": "six subreddits",
        },
        "bitcoin_specific_metadata_available": False,
        "metadata_assessment": (
            "no current metadata proves the existing sentiment artifact is "
            "Bitcoin-specific"
        ),
        "prior_sentiment_gate_status": (
            "inconclusive_for_bitcoin_specific_sentiment"
        ),
        "symbol_specific_sentiment_rebuild": "deferred_to_next_task",
    }


def _symbol_interval_config(
    config: TraderConfig,
    *,
    symbol: str,
    interval: str,
    interval_hours: int,
) -> TraderConfig:
    interval_config = interval_comparison_config(
        config,
        interval=interval,
        interval_hours=interval_hours,
    )
    return replace(interval_config, data=replace(interval_config.data, symbol=symbol))


def _apply_symbol_decision_flags(
    summary: pd.DataFrame,
    decision: Mapping[str, Any],
) -> pd.DataFrame:
    selected = decision.get("selected_development_ranked_symbol_interval")
    if summary.empty or not isinstance(selected, Mapping):
        return summary
    result = summary.copy()
    mask = (
        (result["symbol"] == selected.get("symbol"))
        & (result["interval"] == selected.get("interval"))
    )
    result.loc[mask, "selected_for_future_research"] = bool(
        decision.get("holdout_confirmation_result") == "confirmed"
    )
    return result


def _altcoin_improves_over_btc_12h(summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    btc = summary.loc[
        (summary["symbol"] == "BTCUSDT")
        & (summary["interval"] == "12h")
        & summary["selected_threshold"].notna()
    ]
    if btc.empty:
        return False
    btc_return = _as_float(btc.iloc[0].get("selected_median_total_return"))
    if not np.isfinite(btc_return):
        return False
    alts = summary.loc[
        (summary["symbol"] != "BTCUSDT")
        & summary["selected_threshold"].notna()
    ]
    if alts.empty:
        return False
    return bool((alts["selected_median_total_return"].astype(float) > btc_return).any())


def _row_confirms_on_holdout(row: Any) -> bool:
    if row is None:
        return False
    total = _as_float(row.get("holdout_total_return"))
    cash = _as_float(row.get("holdout_cash_total_return"))
    buy_hold = _as_float(row.get("holdout_buy_hold_total_return"))
    return bool(np.isfinite(total) and total > cash and total > buy_hold)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _none_if_nan(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value
