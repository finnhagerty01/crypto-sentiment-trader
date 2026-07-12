"""Sentiment ablation experiments against the market-only baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trader.backtest.engine import run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import TraderConfig
from trader.features.market import MODEL_FEATURE_COLUMNS
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError
from trader.modeling.validation import split_final_holdout, walk_forward_validate
from trader.sentiment.features import SENTIMENT_VARIANT_ORDER, sentiment_feature_columns


@dataclass(frozen=True, slots=True)
class SentimentExperimentResult:
    """Ablation table and per-variant fold metrics."""

    ablation_table: pd.DataFrame
    fold_metrics: dict[str, list[dict[str, Any]]]


def run_sentiment_ablation(
    feature_dataset: pd.DataFrame,
    hourly_sentiment: pd.DataFrame,
    config: TraderConfig,
    *,
    variants: tuple[str, ...] = SENTIMENT_VARIANT_ORDER,
) -> SentimentExperimentResult:
    """Compare cumulative sentiment variants while holding all else fixed."""

    base_data = feature_dataset.copy()
    base_data["timestamp"] = pd.to_datetime(base_data["timestamp"], utc=True)
    sentiment = hourly_sentiment.copy()
    sentiment["timestamp"] = pd.to_datetime(sentiment["timestamp"], utc=True)
    joined = base_data.merge(sentiment, on="timestamp", how="left")
    rows: list[dict[str, Any]] = []
    folds_by_variant: dict[str, list[dict[str, Any]]] = {}

    market_result = _evaluate_variant(
        "market_only",
        base_data,
        config,
        feature_names=MODEL_FEATURE_COLUMNS,
    )
    rows.append(market_result["row"])
    folds_by_variant["market_only"] = market_result["fold_metrics"]

    for variant in variants:
        feature_names = MODEL_FEATURE_COLUMNS + sentiment_feature_columns(variant)
        result = _evaluate_variant(
            variant,
            joined,
            config,
            feature_names=feature_names,
        )
        rows.append(result["row"])
        folds_by_variant[variant] = result["fold_metrics"]

    table = pd.DataFrame(rows)
    baseline_return = float(table.loc[table["variant"].eq("market_only"), "total_return"].iloc[0])
    table["total_return_delta_vs_market"] = table["total_return"] - baseline_return
    return SentimentExperimentResult(
        ablation_table=table,
        fold_metrics=folds_by_variant,
    )


def _evaluate_variant(
    variant: str,
    data: pd.DataFrame,
    config: TraderConfig,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    split = split_final_holdout(data, config.validation)
    fold_metrics = walk_forward_validate(
        data,
        config,
        feature_names=feature_names,
    )
    model = BaselineLogisticModel(config, feature_names=feature_names)
    try:
        model.fit(split.development)
        probabilities = model.predict_positive_proba(split.holdout)
        predictions = pd.DataFrame(
            {
                "timestamp": split.holdout["timestamp"],
                "signal": (probabilities >= config.model.probability_threshold).astype("int8"),
                "probability": probabilities,
            }
        )
        backtest = run_long_cash_backtest(
            split.holdout,
            predictions,
            backtest_config=config.backtest,
            costs_config=config.costs,
        )
        metrics = calculate_backtest_metrics(backtest.equity, backtest.trades)
        status = "ok"
        reason = None
    except (ModelTrainingError, ValueError) as exc:
        metrics = {
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "trade_count": 0,
            "turnover": float("nan"),
        }
        status = "skipped"
        reason = str(exc)

    row = {
        "variant": variant,
        "status": status,
        "reason": reason,
        "feature_count": len(feature_names),
        "ok_fold_count": sum(1 for fold in fold_metrics if fold.get("status") == "ok"),
        "fold_count": len(fold_metrics),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_count": metrics.get("trade_count"),
        "turnover": metrics.get("turnover"),
    }
    return {"row": row, "fold_metrics": fold_metrics}
