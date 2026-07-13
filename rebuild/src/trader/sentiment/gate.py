"""Sentiment reevaluation gate for Model Refinement 05."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from trader.backtest.benchmarks import run_benchmarks
from trader.config import FeaturesConfig, ModelConfig, TargetConfig, TraderConfig
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.features.target import summarize_target_distribution
from trader.modeling.thresholds import ThresholdSweepResult, run_threshold_sweep
from trader.modeling.validation import split_final_holdout
from trader.sentiment.features import SENTIMENT_VARIANT_ORDER, sentiment_feature_columns
from trader.sentiment.storage import read_hourly_sentiment_dataset


MARKET_ONLY_VARIANT = "market_only"
SENTIMENT_ONLY_VARIANT = "sentiment_only"
FIXED_HORIZON_BARS = 12
FIXED_COST_BUFFER = "none"
FIXED_VOLATILITY_MULTIPLIER = 0.10
FIXED_PROBABILITY_THRESHOLD = 0.30
MATERIAL_DRAWDOWN_TOLERANCE = 0.02
TURNOVER_MULTIPLIER_LIMIT = 3.0
MINIMUM_IMPROVED_FOLDS = 2


class SentimentGateError(ValueError):
    """Raised when the sentiment gate cannot be evaluated."""


def run_sentiment_gate(
    *,
    market_dataset_path: str | Path,
    hourly_sentiment_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    config: TraderConfig,
) -> dict[str, Any]:
    """Run the controlled sentiment reevaluation gate from saved datasets."""

    fixed_config = fixed_market_only_config(config)
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing sentiment gate run: {run_dir}")
    run_dir.mkdir(parents=True)
    variants_dir = run_dir / "variants"
    variants_dir.mkdir()

    market_data = read_market_dataset(
        market_dataset_path,
        symbol=fixed_config.data.symbol,
        interval=fixed_config.data.interval,
    )
    hourly_sentiment, sentiment_metadata = read_hourly_sentiment_dataset(
        hourly_sentiment_path
    )
    market_features = build_feature_dataset(market_data, fixed_config)
    joined = join_hourly_sentiment(market_features, hourly_sentiment)
    market_columns = model_feature_columns(fixed_config)

    summary_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    variants = [
        (MARKET_ONLY_VARIANT, market_features, market_columns, ()),
        *(
            (
                variant,
                joined,
                market_columns + sentiment_feature_columns(variant),
                sentiment_feature_columns(variant),
            )
            for variant in SENTIMENT_VARIANT_ORDER
        ),
        (
            SENTIMENT_ONLY_VARIANT,
            joined,
            sentiment_feature_columns(SENTIMENT_VARIANT_ORDER[-1]),
            sentiment_feature_columns(SENTIMENT_VARIANT_ORDER[-1]),
        ),
    ]

    for index, (variant_name, data, feature_names, sentiment_columns) in enumerate(variants):
        variant_dir = variants_dir / variant_name
        variant_dir.mkdir()
        _write_yaml(variant_dir / "resolved_config.yaml", asdict(fixed_config))
        _write_json(
            variant_dir / "feature_columns.json",
            {
                "variant_name": variant_name,
                "research_only": variant_name == SENTIMENT_ONLY_VARIANT,
                "market_feature_columns": list(market_columns),
                "sentiment_feature_columns": list(sentiment_columns),
                "feature_columns": list(feature_names),
            },
        )

        target_distribution = _target_distribution_dict(data)
        target_rows.append({"variant_name": variant_name, **target_distribution})

        threshold_result = run_threshold_sweep(
            data,
            fixed_config,
            feature_names=feature_names,
        )
        threshold_summary = threshold_result.summary.copy()
        threshold_summary.insert(0, "variant_name", variant_name)
        threshold_rows.append(threshold_summary)

        fold_metrics = threshold_result.fold_metrics.copy()
        fold_metrics.insert(0, "variant_name", variant_name)
        fold_rows.append(fold_metrics)

        holdout_metrics = threshold_result.holdout_metrics or {
            "status": "not_evaluated",
            "selected_threshold": None,
        }
        holdout_rows.append({"variant_name": variant_name, **holdout_metrics})

        split = split_final_holdout(data, fixed_config.validation)
        for benchmark_name, benchmark in run_benchmarks(
            split.holdout,
            backtest_config=fixed_config.backtest,
            costs_config=fixed_config.costs,
        ).items():
            benchmark_rows.append(
                {
                    "variant_name": variant_name,
                    "benchmark": benchmark_name,
                    **benchmark.metrics,
                }
            )

        summary_rows.append(
            _summary_row(
                variant_name=variant_name,
                variant_order=index,
                feature_names=feature_names,
                sentiment_columns=sentiment_columns,
                threshold_result=threshold_result,
            )
        )
        diagnostic_rows.append(
            _feature_diagnostics(
                variant_name=variant_name,
                data=data,
                sentiment_columns=sentiment_columns,
                sentiment_metadata=sentiment_metadata,
            )
        )

    summary = _rank_summary(pd.DataFrame(summary_rows), pd.concat(fold_rows, ignore_index=True))
    target_distribution = pd.DataFrame(target_rows)
    threshold_summary = pd.concat(threshold_rows, ignore_index=True)
    fold_metrics = pd.concat(fold_rows, ignore_index=True)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)
    feature_diagnostics = pd.DataFrame(diagnostic_rows)
    decision = _decision(
        summary,
        fold_metrics,
        feature_diagnostics,
        sentiment_metadata=sentiment_metadata,
    )

    _write_csv(summary, run_dir / "sentiment_gate_summary.csv")
    _write_csv(target_distribution, run_dir / "sentiment_target_distribution.csv")
    _write_csv(threshold_summary, run_dir / "sentiment_threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "sentiment_fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "sentiment_holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "sentiment_benchmark_metrics.csv")
    _write_csv(feature_diagnostics, run_dir / "sentiment_feature_diagnostics.csv")
    _write_json(run_dir / "sentiment_gate_decision.json", decision)

    return {
        "run_dir": run_dir,
        "summary": summary,
        "decision": decision,
    }


def fixed_market_only_config(config: TraderConfig) -> TraderConfig:
    """Return the fixed market-only candidate configuration for the gate."""

    return replace(
        config,
        features=FeaturesConfig(
            volatility_window=config.features.volatility_window,
            volume_window=config.features.volume_window,
            rsi_window=config.features.rsi_window,
            clipping_window=config.features.clipping_window,
            clipping_mad_multiplier=config.features.clipping_mad_multiplier,
            enabled_groups=("baseline",),
        ),
        target=TargetConfig(
            horizon_bars=FIXED_HORIZON_BARS,
            cost_buffer=FIXED_COST_BUFFER,
            volatility_multiplier=FIXED_VOLATILITY_MULTIPLIER,
        ),
        model=ModelConfig(
            probability_threshold=FIXED_PROBABILITY_THRESHOLD,
            regularization_c=config.model.regularization_c,
        ),
    )


def join_hourly_sentiment(
    market_features: pd.DataFrame,
    hourly_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Join saved hourly sentiment to market features by exact UTC timestamp."""

    required = set(sentiment_feature_columns(SENTIMENT_VARIANT_ORDER[-1]))
    missing = sorted(required - set(hourly_sentiment.columns))
    if missing:
        raise SentimentGateError(
            "hourly sentiment dataset is missing feature column(s): "
            + ", ".join(missing)
        )

    base = market_features.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    sentiment = hourly_sentiment.copy()
    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], utc=True)
    sentiment = sentiment.drop_duplicates("timestamp", keep="last").sort_values("timestamp")

    joined = base.merge(sentiment, on="timestamp", how="left", indicator="_sentiment_join")
    missing_join = joined["_sentiment_join"].eq("left_only")
    joined = joined.drop(columns=["_sentiment_join"])

    count_columns = [
        "submission_count",
        "comment_count",
        "subreddit_count",
        "combined_observation_count",
    ]
    for column in count_columns:
        if column in joined:
            joined.loc[missing_join, column] = 0
            joined[column] = joined[column].fillna(0).astype("int64")
    if "sentiment_missing" in joined:
        joined.loc[missing_join, "sentiment_missing"] = 1
        joined["sentiment_missing"] = joined["sentiment_missing"].fillna(1).astype("int8")
    if "sentiment_reliability" in joined:
        joined.loc[missing_join, "sentiment_reliability"] = 0.0
    if "sentiment_reliability_shrunk" in joined:
        joined.loc[missing_join, "sentiment_reliability_shrunk"] = 0.0

    joined["sentiment_join_missing"] = missing_join.astype("int8")
    return joined


def _summary_row(
    *,
    variant_name: str,
    variant_order: int,
    feature_names: tuple[str, ...],
    sentiment_columns: tuple[str, ...],
    threshold_result: ThresholdSweepResult,
) -> dict[str, Any]:
    selected_threshold = threshold_result.selected_threshold
    selected_row: dict[str, Any] = {}
    if selected_threshold is not None:
        matching = threshold_result.summary.loc[
            threshold_result.summary["threshold"] == selected_threshold
        ]
        if not matching.empty:
            selected_row = matching.iloc[0].to_dict()

    holdout_metrics = threshold_result.holdout_metrics or {}
    return {
        "variant_name": variant_name,
        "variant_order": variant_order,
        "research_only": variant_name == SENTIMENT_ONLY_VARIANT,
        "phase11_candidate": False,
        "rank": None,
        "feature_count": len(feature_names),
        "market_feature_count": len(feature_names) - len(sentiment_columns),
        "sentiment_feature_count": len(sentiment_columns),
        "selected_threshold": selected_threshold,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get("median_cash_total_return"),
        "selected_median_max_drawdown": selected_row.get("median_max_drawdown"),
        "selected_median_turnover": selected_row.get("median_turnover"),
        "selected_mean_precision": selected_row.get("mean_precision"),
        "selected_mean_recall": selected_row.get("mean_recall"),
        "selected_mean_f1": selected_row.get("mean_f1"),
        "holdout_status": holdout_metrics.get("status", "not_evaluated"),
        "holdout_total_return": holdout_metrics.get("total_return"),
        "holdout_cash_total_return": holdout_metrics.get("cash_total_return"),
        "holdout_max_drawdown": holdout_metrics.get("max_drawdown"),
        "holdout_trade_count": holdout_metrics.get("trade_count"),
        "holdout_turnover": holdout_metrics.get("turnover"),
        "holdout_exposure_percentage": holdout_metrics.get("exposure_percentage"),
    }


def _rank_summary(summary: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    ranked = summary.copy()
    market = _market_row(ranked)
    market_return = _float_or_nan(market.get("selected_median_total_return"))
    market_holdout_return = _float_or_nan(market.get("holdout_total_return"))
    market_holdout_drawdown = _float_or_nan(market.get("holdout_max_drawdown"))
    market_turnover = _float_or_nan(market.get("selected_median_turnover"))

    ranked["development_fold_improvements"] = [
        _development_fold_improvements(row["variant_name"], fold_metrics)
        for _, row in ranked.iterrows()
    ]
    ranked["median_return_delta_vs_market"] = (
        ranked["selected_median_total_return"] - market_return
    )
    ranked["holdout_return_delta_vs_market"] = (
        ranked["holdout_total_return"] - market_holdout_return
    )
    ranked["holdout_drawdown_delta_vs_market"] = (
        ranked["holdout_max_drawdown"] - market_holdout_drawdown
    )
    ranked["turnover_multiplier_vs_market"] = ranked["selected_median_turnover"].map(
        lambda value: _turnover_multiplier(value, market_turnover)
    )

    eligible = ranked.loc[
        ~ranked["variant_name"].isin([MARKET_ONLY_VARIANT, SENTIMENT_ONLY_VARIANT])
        & ranked["selected_threshold"].notna()
        & (ranked["selected_median_total_return"] > market_return)
        & (ranked["selected_median_total_return"] > ranked["selected_median_cash_total_return"])
    ].copy()
    if not eligible.empty:
        eligible["drawdown_magnitude"] = eligible["selected_median_max_drawdown"].abs()
        eligible = eligible.sort_values(
            by=[
                "selected_median_total_return",
                "development_fold_improvements",
                "drawdown_magnitude",
                "selected_median_turnover",
                "variant_order",
            ],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        for rank, index in enumerate(eligible.index, start=1):
            ranked.loc[index, "rank"] = rank

    ranked["phase11_candidate"] = False
    return ranked.sort_values(
        by=["variant_order"],
        ascending=[True],
        kind="mergesort",
    )


def _decision(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_diagnostics: pd.DataFrame,
    *,
    sentiment_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    market = _market_row(summary)
    candidates = summary.loc[
        ~summary["variant_name"].isin([MARKET_ONLY_VARIANT, SENTIMENT_ONLY_VARIANT])
    ].copy()
    ranked = candidates.sort_values(
        by=["rank"],
        na_position="last",
        kind="mergesort",
    )
    selected_variant = None
    rejection_reasons: dict[str, list[str]] = {}
    for _, row in ranked.iterrows():
        reasons = _variant_rejection_reasons(row, market, feature_diagnostics)
        rejection_reasons[str(row["variant_name"])] = reasons
        if not reasons and selected_variant is None:
            selected_variant = str(row["variant_name"])

    sentiment_only = summary.loc[summary["variant_name"].eq(SENTIMENT_ONLY_VARIANT)]
    sentiment_only_row = (
        sentiment_only.iloc[0].to_dict() if not sentiment_only.empty else None
    )
    return {
        "decision": "keep_sentiment_research_only"
        if selected_variant is None
        else "sentiment_candidate_requires_separate_phase11_approval",
        "sentiment_kept_for_phase11": False,
        "phase11_remains_blocked": True,
        "selected_development_ranked_sentiment_variant": selected_variant,
        "market_only_variant": _json_ready(market.to_dict()),
        "sentiment_only_diagnostic": _json_ready(sentiment_only_row),
        "sentiment_only_phase11_eligible": False,
        "rejection_reasons": rejection_reasons,
        "policy": {
            "minimum_improved_development_folds": MINIMUM_IMPROVED_FOLDS,
            "material_drawdown_tolerance": MATERIAL_DRAWDOWN_TOLERANCE,
            "turnover_multiplier_limit": TURNOVER_MULTIPLIER_LIMIT,
            "holdout_is_confirmation_only": True,
            "sentiment_only_is_research_only": True,
        },
        "sentiment_metadata": _json_ready(dict(sentiment_metadata)),
    }


def _variant_rejection_reasons(
    row: pd.Series,
    market: pd.Series,
    feature_diagnostics: pd.DataFrame,
) -> list[str]:
    reasons: list[str] = []
    variant_name = str(row["variant_name"])
    if _float_or_nan(row.get("development_fold_improvements")) < MINIMUM_IMPROVED_FOLDS:
        reasons.append("does_not_improve_multiple_development_folds")
    if not _greater(row.get("selected_median_total_return"), market.get("selected_median_total_return")):
        reasons.append("does_not_improve_development_median_return")
    if not _greater(row.get("selected_median_total_return"), row.get("selected_median_cash_total_return")):
        reasons.append("does_not_beat_cash_on_development_median_return")
    if not _greater(row.get("holdout_total_return"), market.get("holdout_total_return")):
        reasons.append("does_not_improve_holdout_net_return")
    if _drawdown_materially_worse(row.get("holdout_max_drawdown"), market.get("holdout_max_drawdown")):
        reasons.append("materially_worsens_holdout_drawdown")
    if _float_or_nan(row.get("turnover_multiplier_vs_market")) > TURNOVER_MULTIPLIER_LIMIT:
        reasons.append("unacceptable_turnover")
    diagnostic = feature_diagnostics.loc[
        feature_diagnostics["variant_name"].eq(variant_name)
    ]
    if diagnostic.empty or bool(diagnostic.iloc[0].get("diagnostics_preserved")) is not True:
        reasons.append("missingness_or_reliability_diagnostics_not_preserved")
    return reasons


def _development_fold_improvements(variant_name: str, fold_metrics: pd.DataFrame) -> int:
    if variant_name == MARKET_ONLY_VARIANT:
        return 0
    market = fold_metrics.loc[
        fold_metrics["variant_name"].eq(MARKET_ONLY_VARIANT)
        & fold_metrics["status"].eq("ok")
    ]
    variant = fold_metrics.loc[
        fold_metrics["variant_name"].eq(variant_name)
        & fold_metrics["status"].eq("ok")
    ]
    merged = variant.merge(
        market.loc[:, ["fold", "threshold", "total_return"]],
        on=["fold", "threshold"],
        suffixes=("", "_market"),
        how="inner",
    )
    selected = merged.loc[merged["threshold"].eq(_selected_threshold(variant))]
    if selected.empty:
        selected = merged
    return int((selected["total_return"] > selected["total_return_market"]).sum())


def _selected_threshold(rows: pd.DataFrame) -> float | None:
    ok_rows = rows.loc[rows["status"].eq("ok")]
    if ok_rows.empty:
        return None
    by_threshold = (
        ok_rows.groupby("threshold", sort=True)["total_return"]
        .median()
        .sort_values(ascending=False, kind="mergesort")
    )
    return float(by_threshold.index[0])


def _feature_diagnostics(
    *,
    variant_name: str,
    data: pd.DataFrame,
    sentiment_columns: tuple[str, ...],
    sentiment_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": variant_name,
        "sentiment_feature_count": len(sentiment_columns),
        "hourly_sentiment_dataset_id": sentiment_metadata.get("dataset_id"),
        "diagnostics_preserved": True,
    }
    if not sentiment_columns:
        row.update(
            {
                "sentiment_missing_column_present": False,
                "sentiment_missing_rate": np.nan,
                "sentiment_join_missing_rate": np.nan,
                "sentiment_reliability_column_present": False,
                "sentiment_reliability_mean": np.nan,
            }
        )
        return row

    missing_present = "sentiment_missing" in data.columns
    reliability_present = "sentiment_reliability" in data.columns
    row.update(
        {
            "sentiment_missing_column_present": missing_present,
            "sentiment_missing_rate": _mean(data, "sentiment_missing"),
            "sentiment_join_missing_rate": _mean(data, "sentiment_join_missing"),
            "sentiment_reliability_column_present": reliability_present,
            "sentiment_reliability_mean": _mean(data, "sentiment_reliability"),
            "sentiment_observation_count_mean": _mean(data, "combined_observation_count"),
        }
    )
    row["diagnostics_preserved"] = bool(missing_present and reliability_present)
    return row


def _target_distribution_dict(features: pd.DataFrame) -> dict[str, Any]:
    distribution = summarize_target_distribution(features)
    return {
        "row_count": distribution.row_count,
        "labeled_row_count": distribution.labeled_row_count,
        "positive_count": distribution.positive_count,
        "negative_count": distribution.negative_count,
        "positive_rate": distribution.positive_rate,
        "unlabeled_count": distribution.unlabeled_count,
        "first_labeled_timestamp": _timestamp_or_none(distribution.first_labeled_timestamp),
        "last_labeled_timestamp": _timestamp_or_none(distribution.last_labeled_timestamp),
    }


def _market_row(summary: pd.DataFrame) -> pd.Series:
    rows = summary.loc[summary["variant_name"].eq(MARKET_ONLY_VARIANT)]
    if rows.empty:
        raise SentimentGateError("market_only row is missing from sentiment gate summary")
    return rows.iloc[0]


def _turnover_multiplier(value: Any, market_turnover: float) -> float:
    current = _float_or_nan(value)
    if not np.isfinite(current):
        return float("nan")
    if not np.isfinite(market_turnover) or market_turnover <= 0:
        return float("inf") if current > 0 else 1.0
    return float(current / market_turnover)


def _drawdown_materially_worse(value: Any, market_value: Any) -> bool:
    current = _float_or_nan(value)
    market = _float_or_nan(market_value)
    if not np.isfinite(current) or not np.isfinite(market):
        return True
    return current < market - MATERIAL_DRAWDOWN_TOLERANCE


def _greater(left: Any, right: Any) -> bool:
    left_value = _float_or_nan(left)
    right_value = _float_or_nan(right)
    return bool(np.isfinite(left_value) and np.isfinite(right_value) and left_value > right_value)


def _mean(data: pd.DataFrame, column: str) -> float:
    if column not in data:
        return float("nan")
    return float(pd.to_numeric(data[column], errors="coerce").mean())


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _timestamp_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, lineterminator="\n")


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return _timestamp_or_none(value)
    return value
