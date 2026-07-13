"""Offline candle interval comparison diagnostic."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from trader.backtest.benchmarks import run_benchmarks
from trader.config import TraderConfig, load_config
from trader.data.schemas import normalize_ohlcv
from trader.data.storage import build_metadata, read_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.modeling.baseline import BaselineLogisticModel
from trader.modeling.experiments import _target_distribution_dict
from trader.modeling.thresholds import (
    DEFAULT_THRESHOLDS,
    _evaluate_holdout,
    run_threshold_sweep,
    train_window_folds,
)
from trader.modeling.validation import split_final_holdout


DEFAULT_INTERVALS = ("1h", "4h", "12h", "1d")


class CandleIntervalComparisonError(ValueError):
    """Raised when the candle interval comparison cannot be completed."""


def run_candle_interval_comparison(
    *,
    market_dataset_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_path: str | Path = "configs/baseline.yaml",
    intervals: tuple[str, ...] = DEFAULT_INTERVALS,
    max_development_exposure: float = 0.80,
) -> dict[str, Any]:
    """Compare resampled BTCUSDT candle intervals from a saved 1h dataset only."""

    _validate_max_exposure(max_development_exposure)
    interval_hours = {interval: _interval_hours(interval) for interval in intervals}
    base_config = load_config(config_path)
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing candle interval run: {run_dir}")
    run_dir.mkdir(parents=True)
    intervals_dir = run_dir / "intervals"
    intervals_dir.mkdir()

    source_market = read_market_dataset(
        market_dataset_path,
        symbol=base_config.data.symbol,
        interval="1h",
    )

    summary_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for interval_order, interval in enumerate(intervals):
        hours = interval_hours[interval]
        interval_dir = intervals_dir / interval
        interval_dir.mkdir()

        market, diagnostics = resample_ohlcv(source_market, interval)
        config = interval_comparison_config(base_config, interval=interval, interval_hours=hours)
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
                    symbol=config.data.symbol,
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
        threshold_frames.append(threshold_summary)
        fold_metrics = fold_metrics.copy()
        fold_metrics.insert(0, "interval", interval)
        fold_frames.append(fold_metrics)
        holdout_rows.append({"interval": interval, **holdout_metrics})
        diagnostic_rows.append(
            {
                "interval": interval,
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
                {"interval": interval, "benchmark": benchmark_name, **benchmark.metrics}
            )

        summary_rows.append(
            _interval_summary_row(
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
        )

    interval_summary = rank_intervals(pd.DataFrame(summary_rows))
    decision = candle_interval_decision(interval_summary)
    interval_summary = _apply_decision_flags(interval_summary, decision)
    threshold_summary = _concat_frames(threshold_frames)
    fold_metrics = _concat_frames(fold_frames)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)
    dataset_diagnostics = pd.DataFrame(diagnostic_rows)

    _write_csv(interval_summary, run_dir / "interval_summary.csv")
    _write_csv(threshold_summary, run_dir / "threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "benchmark_metrics.csv")
    _write_csv(dataset_diagnostics, run_dir / "dataset_diagnostics.csv")
    _write_json(run_dir / "candle_interval_decision.json", decision)
    return {"run_dir": run_dir, "interval_summary": interval_summary, "decision": decision}


def interval_comparison_config(
    config: TraderConfig,
    *,
    interval: str,
    interval_hours: int,
) -> TraderConfig:
    """Return fixed controls for a one-next-candle interval diagnostic."""

    return replace(
        config,
        data=replace(config.data, interval=interval),
        features=replace(config.features, enabled_groups=("baseline",)),
        target=replace(config.target, horizon_bars=1),
        validation=replace(
            config.validation,
            minimum_train_bars=int(np.ceil(1000 / interval_hours)),
            test_bars=int(np.ceil(168 / interval_hours)),
            step_bars=int(np.ceil(168 / interval_hours)),
        ),
    )


def resample_ohlcv(data: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample canonical 1h OHLCV rows and drop incomplete UTC buckets."""

    hours = _interval_hours(interval)
    symbols = data["symbol"].dropna().astype(str).unique().tolist() if "symbol" in data else []
    if len(symbols) != 1:
        raise CandleIntervalComparisonError("resampling requires exactly one symbol")
    source = normalize_ohlcv(data, symbol=symbols[0], interval="1h")
    timestamps = pd.to_datetime(source["timestamp"], utc=True)
    bucket_freq = f"{hours}h"
    bucket_start = timestamps.dt.floor(bucket_freq)
    grouped = source.assign(_bucket_start=bucket_start).groupby("_bucket_start", sort=True)
    counts = grouped.size()
    complete_buckets = counts[counts == hours].index
    incomplete = counts[counts != hours]
    complete = source.assign(_bucket_start=bucket_start)
    complete = complete.loc[complete["_bucket_start"].isin(complete_buckets)]
    if complete.empty:
        raise CandleIntervalComparisonError(f"interval {interval} produced no complete buckets")

    aggregated = (
        complete.groupby("_bucket_start", sort=True)
        .agg(
            symbol=("symbol", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"_bucket_start": "timestamp"})
    )
    normalized = normalize_ohlcv(
        aggregated.loc[:, ["timestamp", "symbol", "open", "high", "low", "close", "volume"]],
        symbol=symbols[0],
        interval=interval,
    )
    diagnostics = {
        "source_row_count": len(source),
        "resampled_row_count": len(normalized),
        "dropped_source_row_count": int(incomplete.sum()) if not incomplete.empty else 0,
        "dropped_incomplete_bucket_count": int(len(incomplete)),
        "complete_bucket_count": int(len(complete_buckets)),
        "expected_source_rows_per_bucket": hours,
        "first_resampled_timestamp": _timestamp_or_none(normalized["timestamp"].iloc[0]),
        "last_resampled_timestamp": _timestamp_or_none(normalized["timestamp"].iloc[-1]),
    }
    return normalized, diagnostics


def summarize_interval_thresholds(
    fold_metrics: pd.DataFrame,
    *,
    max_development_exposure: float,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize thresholds using development folds only and interval gates."""

    rows: list[dict[str, Any]] = []
    ok_metrics = fold_metrics.loc[fold_metrics.get("status") == "ok"].copy()
    for threshold in thresholds:
        threshold_rows = ok_metrics.loc[ok_metrics["threshold"] == threshold]
        trade_count = int(threshold_rows["trade_count"].sum()) if not threshold_rows.empty else 0
        median_total_return = _median_or_nan(threshold_rows, "total_return")
        median_cash_return = _median_or_nan(threshold_rows, "cash_total_return")
        median_buy_hold_return = _median_or_nan(threshold_rows, "buy_hold_total_return")
        median_exposure = _median_or_nan(threshold_rows, "exposure_percentage")
        rows.append(
            {
                "threshold": threshold,
                "fold_count": int(len(threshold_rows)),
                "trade_count": trade_count,
                "median_total_return": median_total_return,
                "median_cash_total_return": median_cash_return,
                "median_buy_hold_total_return": median_buy_hold_return,
                "median_max_drawdown": _median_or_nan(threshold_rows, "max_drawdown"),
                "median_turnover": _median_or_nan(threshold_rows, "turnover"),
                "median_exposure_percentage": median_exposure,
                "mean_precision": _mean_or_nan(threshold_rows, "precision"),
                "mean_recall": _mean_or_nan(threshold_rows, "recall"),
                "mean_f1": _mean_or_nan(threshold_rows, "f1"),
                "passes_trade_filter": trade_count > 0,
                "passes_cash_filter": _finite(median_total_return)
                and _finite(median_cash_return)
                and median_total_return > median_cash_return,
                "passes_buy_hold_filter": _finite(median_total_return)
                and _finite(median_buy_hold_return)
                and median_total_return > median_buy_hold_return,
                "passes_exposure_filter": _finite(median_exposure)
                and median_exposure <= max_development_exposure,
            }
        )
    summary = pd.DataFrame(rows)
    summary["eligible"] = (
        summary["passes_trade_filter"]
        & summary["passes_cash_filter"]
        & summary["passes_buy_hold_filter"]
        & summary["passes_exposure_filter"]
    )
    return summary


def select_interval_threshold(summary: pd.DataFrame) -> float | None:
    """Select a threshold from development metrics only."""

    eligible = summary.loc[summary["eligible"]].copy()
    if eligible.empty:
        return None
    eligible["drawdown_magnitude"] = eligible["median_max_drawdown"].abs()
    eligible = eligible.sort_values(
        by=[
            "median_total_return",
            "drawdown_magnitude",
            "median_turnover",
            "median_exposure_percentage",
            "threshold",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return float(eligible.iloc[0]["threshold"])


def rank_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank intervals using selected development-fold metrics only."""

    if summary.empty:
        return summary
    ranked = summary.copy()
    ranked["eligible"] = ranked["selected_threshold"].notna()
    eligible = ranked.loc[ranked["eligible"]].copy()
    if not eligible.empty:
        eligible["drawdown_magnitude"] = eligible["selected_median_max_drawdown"].abs()
        eligible = eligible.sort_values(
            by=[
                "selected_median_total_return",
                "drawdown_magnitude",
                "selected_median_turnover",
                "selected_median_exposure_percentage",
                "interval_order",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        for rank, index in enumerate(eligible.index, start=1):
            ranked.loc[index, "rank"] = rank
    ranked["rank_sort"] = ranked["rank"].fillna(np.inf)
    return ranked.sort_values(
        by=["rank_sort", "interval_order"],
        ascending=[True, True],
        kind="mergesort",
    ).drop(columns=["rank_sort"])


def candle_interval_decision(summary: pd.DataFrame) -> dict[str, Any]:
    """Return the confirmation-only interval diagnostic decision."""

    selected = summary.loc[summary["rank"] == 1] if "rank" in summary else pd.DataFrame()
    selected_row = selected.iloc[0] if not selected.empty else None
    holdout_confirmed = False
    if selected_row is not None:
        holdout_confirmed = (
            _as_float(selected_row.get("holdout_total_return"))
            > max(
                _as_float(selected_row.get("holdout_cash_total_return")),
                _as_float(selected_row.get("holdout_buy_hold_total_return")),
            )
        )
    interval_change_helps = bool(
        selected_row is not None
        and selected_row.get("interval") != "1h"
        and holdout_confirmed
    )
    exposure_rejected_all = bool(
        not summary.empty
        and summary["selected_threshold"].isna().all()
        and "all_thresholds_failed_exposure_gate" in summary
        and summary["all_thresholds_failed_exposure_gate"].all()
    )
    return {
        "decision": (
            "interval_change_helps_but_phase_11_blocked"
            if interval_change_helps
            else "interval_change_does_not_help"
        ),
        "selected_development_ranked_interval": (
            None if selected_row is None else selected_row.get("interval")
        ),
        "selected_threshold": (
            None if selected_row is None else _none_if_nan(selected_row.get("selected_threshold"))
        ),
        "exposure_gate_rejected_all_intervals": exposure_rejected_all,
        "holdout_confirmation_result": (
            "confirmed" if holdout_confirmed else "not_confirmed"
        ),
        "interval_change_helps": interval_change_helps,
        "holdout_used_for_ranking": False,
        "phase_11_status": "blocked",
    }


def _interval_summary_row(
    *,
    interval: str,
    interval_order: int,
    interval_hours: int,
    config: TraderConfig,
    feature_names: tuple[str, ...],
    target_distribution: dict[str, Any],
    threshold_summary: pd.DataFrame,
    selected_threshold: float | None,
    holdout_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    selected_row: dict[str, Any] = {}
    if selected_threshold is not None:
        matching = threshold_summary.loc[threshold_summary["threshold"] == selected_threshold]
        if not matching.empty:
            selected_row = matching.iloc[0].to_dict()
    all_failed_exposure = bool(
        not threshold_summary.empty
        and (~threshold_summary["passes_exposure_filter"]).all()
    )
    buy_hold = holdout_metrics.get("buy_hold_total_return")
    return {
        "interval": interval,
        "interval_order": interval_order,
        "interval_hours": interval_hours,
        "rank": None,
        "eligible": False,
        "selected_for_future_research": False,
        "minimum_train_bars": config.validation.minimum_train_bars,
        "test_bars": config.validation.test_bars,
        "step_bars": config.validation.step_bars,
        "target_horizon_bars": config.target.horizon_bars,
        "feature_count": len(feature_names),
        "selected_threshold": selected_threshold,
        "all_thresholds_failed_exposure_gate": all_failed_exposure,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get("median_cash_total_return"),
        "selected_median_buy_hold_total_return": selected_row.get("median_buy_hold_total_return"),
        "selected_median_max_drawdown": selected_row.get("median_max_drawdown"),
        "selected_median_turnover": selected_row.get("median_turnover"),
        "selected_median_exposure_percentage": selected_row.get("median_exposure_percentage"),
        "selected_mean_precision": selected_row.get("mean_precision"),
        "selected_mean_recall": selected_row.get("mean_recall"),
        "selected_mean_f1": selected_row.get("mean_f1"),
        "holdout_status": holdout_metrics.get("status", "not_evaluated"),
        "holdout_total_return": holdout_metrics.get("total_return"),
        "holdout_cash_total_return": holdout_metrics.get("cash_total_return"),
        "holdout_buy_hold_total_return": buy_hold,
        "holdout_max_drawdown": holdout_metrics.get("max_drawdown"),
        "holdout_trade_count": holdout_metrics.get("trade_count"),
        "holdout_turnover": holdout_metrics.get("turnover"),
        "holdout_exposure_percentage": holdout_metrics.get("exposure_percentage"),
        **target_distribution,
    }


def _add_buy_hold_returns(
    fold_metrics: pd.DataFrame,
    split: Any,
    config: TraderConfig,
) -> pd.DataFrame:
    if fold_metrics.empty:
        result = fold_metrics.copy()
        result["buy_hold_total_return"] = np.nan
        return result
    folds = train_window_folds(len(split.development), config.validation)
    buy_hold_by_fold: dict[int, float] = {}
    for fold in folds:
        validation = split.development.iloc[list(fold.test_positions)]
        labeled = validation.loc[validation["target"].notna()].copy()
        if len(labeled) < 2:
            continue
        benchmark = run_benchmarks(
            labeled,
            backtest_config=config.backtest,
            costs_config=config.costs,
        )["buy_and_hold"]
        buy_hold_by_fold[fold.fold_number] = float(benchmark.metrics["total_return"])
    result = fold_metrics.copy()
    result["buy_hold_total_return"] = result["fold"].map(buy_hold_by_fold)
    return result


def _holdout_buy_hold_return(holdout: pd.DataFrame, config: TraderConfig) -> float | None:
    if len(holdout) < 2:
        return None
    benchmark = run_benchmarks(
        holdout,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )["buy_and_hold"]
    return float(benchmark.metrics["total_return"])


def _apply_decision_flags(summary: pd.DataFrame, decision: Mapping[str, Any]) -> pd.DataFrame:
    if summary.empty or decision.get("selected_development_ranked_interval") is None:
        return summary
    result = summary.copy()
    selected = result["interval"] == decision["selected_development_ranked_interval"]
    result.loc[selected, "selected_for_future_research"] = bool(
        decision.get("interval_change_helps")
    )
    return result


def _interval_hours(interval: str) -> int:
    if interval.endswith("h"):
        value = int(interval[:-1])
    elif interval.endswith("d"):
        value = int(interval[:-1]) * 24
    else:
        raise CandleIntervalComparisonError(f"unsupported interval: {interval}")
    if value <= 0:
        raise CandleIntervalComparisonError(f"interval must be positive: {interval}")
    return value


def _validate_max_exposure(value: float) -> None:
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise CandleIntervalComparisonError("--max-development-exposure must be between 0 and 1")


def _fold_signature(features: pd.DataFrame, config: TraderConfig) -> str:
    split = split_final_holdout(features, config.validation)
    folds = train_window_folds(len(split.development), config.validation)
    value = {
        "row_count": len(features),
        "holdout_start_position": split.holdout_start_position,
        "folds": [
            {
                "fold": fold.fold_number,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
            }
            for fold in folds
        ],
    }
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _frame_signature(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _median_or_nan(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return float("nan")
    return float(data[column].median())


def _mean_or_nan(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return float("nan")
    return float(data[column].mean())


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, lineterminator="\n")


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default, separators=(",", ":"))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return _timestamp_or_none(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _timestamp_or_none(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


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
