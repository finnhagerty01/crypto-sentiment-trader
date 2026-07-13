"""Offline rule-regime specialist diagnostic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from trader.backtest.benchmarks import run_benchmarks
from trader.backtest.engine import run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import TraderConfig, load_config
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.features.target import TARGET_COLUMN
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError
from trader.modeling.experiments import _target_distribution_dict
from trader.modeling.thresholds import DEFAULT_THRESHOLDS, probabilities_to_signals
from trader.modeling.validation import WalkForwardFold, split_final_holdout


REGIME_COLUMN = "regime"
PAST_WEEK_RETURN_COLUMN = "past_week_return"
CANDIDATES = ("global_baseline", "regime_specialists", "regime_cash_filter")


class RegimeSpecialistComparisonError(ValueError):
    """Raised when the regime specialist comparison cannot be completed."""


@dataclass(frozen=True, slots=True)
class ProbabilityBundle:
    data: pd.DataFrame
    probabilities: np.ndarray
    diagnostics: dict[str, Any]


def run_regime_specialist_comparison(
    *,
    market_dataset_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_path: str | Path = "configs/baseline.yaml",
    lookback_bars: int = 168,
) -> dict[str, Any]:
    """Compare global, regime-specialist, and bull-only cash-filter candidates."""

    if lookback_bars <= 0:
        raise RegimeSpecialistComparisonError("--lookback-bars must be greater than zero")
    base_config = regime_specialist_config(load_config(config_path))
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing regime run: {run_dir}")
    run_dir.mkdir(parents=True)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir()

    market = read_market_dataset(
        market_dataset_path,
        symbol=base_config.data.symbol,
        interval=base_config.data.interval,
    )
    features = add_regime_labels(
        build_feature_dataset(market, base_config),
        lookback_bars=lookback_bars,
    )
    features = features.loc[features[REGIME_COLUMN].notna()].reset_index(drop=True)
    if features.empty:
        raise RegimeSpecialistComparisonError("no rows remain after regime lookback")

    feature_names = model_feature_columns(base_config)
    split = split_final_holdout(features, base_config.validation)
    target_distribution = _target_distribution_dict(features)
    fold_signature = _fold_signature(features, base_config)
    feature_signature = _frame_signature(features)

    summary_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for candidate_order, candidate_name in enumerate(CANDIDATES):
        candidate_dir = candidates_dir / candidate_name
        candidate_dir.mkdir()
        _write_yaml(candidate_dir / "resolved_config.yaml", asdict(base_config))
        _write_json(
            candidate_dir / "feature_columns.json",
            {
                "candidate_name": candidate_name,
                "feature_columns": list(feature_names),
                "feature_count": len(feature_names),
                "feature_data_sha256": feature_signature,
                "fold_signature_sha256": fold_signature,
                "regime_column": REGIME_COLUMN,
                "lookback_bars": lookback_bars,
            },
        )

        fold_metrics, diagnostics = evaluate_candidate_development(
            split.development,
            base_config,
            candidate_name=candidate_name,
            feature_names=feature_names,
        )
        threshold_summary = summarize_regime_thresholds(fold_metrics)
        selected_threshold = select_regime_threshold(threshold_summary)
        holdout_metrics = (
            evaluate_candidate_holdout(
                split.development,
                split.holdout,
                base_config,
                candidate_name=candidate_name,
                feature_names=feature_names,
                threshold=selected_threshold,
            )
            if selected_threshold is not None
            else {"status": "not_evaluated", "selected_threshold": None}
        )
        holdout_metrics = {
            **holdout_metrics,
            "buy_hold_total_return": _holdout_benchmark_return(
                split.holdout,
                base_config,
                benchmark_name="buy_and_hold",
            ),
        }

        threshold_summary.insert(0, "candidate_name", candidate_name)
        threshold_frames.append(threshold_summary)
        fold_metrics = fold_metrics.copy()
        fold_metrics.insert(0, "candidate_name", candidate_name)
        fold_frames.append(fold_metrics)
        holdout_rows.append({"candidate_name": candidate_name, **holdout_metrics})
        summary_rows.append(
            _candidate_summary_row(
                candidate_name=candidate_name,
                candidate_order=candidate_order,
                config=base_config,
                feature_names=feature_names,
                target_distribution=target_distribution,
                threshold_summary=threshold_summary,
                selected_threshold=selected_threshold,
                holdout_metrics=holdout_metrics,
                diagnostics=diagnostics,
                lookback_bars=lookback_bars,
            )
        )

    for benchmark_name, benchmark in run_benchmarks(
        split.holdout,
        backtest_config=base_config.backtest,
        costs_config=base_config.costs,
    ).items():
        benchmark_rows.append({"benchmark": benchmark_name, **benchmark.metrics})

    regime_summary = rank_regime_candidates(pd.DataFrame(summary_rows))
    decision = regime_specialist_decision(regime_summary)
    regime_summary = _apply_decision_flags(regime_summary, decision)
    threshold_summary = _concat_frames(threshold_frames)
    fold_metrics = _concat_frames(fold_frames)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)
    regime_diagnostics = regime_diagnostics_frame(
        features,
        split=split,
        lookback_bars=lookback_bars,
    )

    _write_csv(regime_summary, run_dir / "regime_summary.csv")
    _write_csv(threshold_summary, run_dir / "threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "benchmark_metrics.csv")
    _write_csv(regime_diagnostics, run_dir / "regime_diagnostics.csv")
    _write_json(run_dir / "regime_specialist_decision.json", decision)
    return {"run_dir": run_dir, "regime_summary": regime_summary, "decision": decision}


def regime_specialist_config(config: TraderConfig) -> TraderConfig:
    """Return the fixed h12 market-only setup selected before this diagnostic."""

    return replace(
        config,
        features=replace(config.features, enabled_groups=("baseline",)),
        target=replace(
            config.target,
            horizon_bars=12,
            cost_buffer="none",
            volatility_multiplier=0.10,
        ),
    )


def add_regime_labels(data: pd.DataFrame, *, lookback_bars: int = 168) -> pd.DataFrame:
    """Add causal trailing-return bull/bear regime labels."""

    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be greater than zero")
    result = data.copy()
    result[PAST_WEEK_RETURN_COLUMN] = (
        result["close"] / result["close"].shift(lookback_bars) - 1.0
    )
    regime = pd.Series(pd.NA, index=result.index, dtype="string")
    labeled = result[PAST_WEEK_RETURN_COLUMN].notna()
    regime.loc[labeled & (result[PAST_WEEK_RETURN_COLUMN] >= 0.0)] = "bull"
    regime.loc[labeled & (result[PAST_WEEK_RETURN_COLUMN] < 0.0)] = "bear"
    result[REGIME_COLUMN] = regime
    return result


def evaluate_candidate_development(
    development: pd.DataFrame,
    config: TraderConfig,
    *,
    candidate_name: str,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all default thresholds on development folds."""

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"skipped_fold_count": 0}
    for fold in _folds(len(development), config):
        train = development.iloc[list(fold.train_positions)]
        validation = development.iloc[list(fold.test_positions)]
        try:
            bundle = _candidate_probabilities(
                train,
                validation,
                config,
                candidate_name=candidate_name,
                feature_names=feature_names,
            )
        except ModelTrainingError as exc:
            diagnostics["skipped_fold_count"] += 1
            rows.extend(_skipped_threshold_rows(fold, str(exc)))
            continue
        rows.extend(
            _evaluate_thresholds(
                fold=fold,
                market_data=bundle.data,
                probabilities=bundle.probabilities,
                thresholds=DEFAULT_THRESHOLDS,
                config=config,
                candidate_name=candidate_name,
            )
        )
    return pd.DataFrame(rows), diagnostics


def evaluate_candidate_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TraderConfig,
    *,
    candidate_name: str,
    feature_names: tuple[str, ...],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one selected threshold on holdout after training on development."""

    bundle = _candidate_probabilities(
        development,
        holdout,
        config,
        candidate_name=candidate_name,
        feature_names=feature_names,
    )
    metrics = _evaluate_period(
        market_data=bundle.data,
        probabilities=bundle.probabilities,
        threshold=threshold,
        config=config,
        candidate_name=candidate_name,
    )
    return {
        "status": "ok",
        "selected_threshold": threshold,
        **metrics,
        **bundle.diagnostics,
    }


def summarize_regime_thresholds(
    fold_metrics: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Aggregate development-fold threshold metrics for regime candidates."""

    rows: list[dict[str, Any]] = []
    ok_metrics = fold_metrics.loc[fold_metrics.get("status") == "ok"].copy()
    for threshold in thresholds:
        threshold_rows = ok_metrics.loc[ok_metrics["threshold"] == threshold]
        trade_count = int(threshold_rows["trade_count"].sum()) if not threshold_rows.empty else 0
        median_total_return = _median_or_nan(threshold_rows, "total_return")
        median_cash_return = _median_or_nan(threshold_rows, "cash_total_return")
        median_buy_hold_return = _median_or_nan(threshold_rows, "buy_hold_total_return")
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
                "median_exposure_percentage": _median_or_nan(
                    threshold_rows,
                    "exposure_percentage",
                ),
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
            }
        )
    summary = pd.DataFrame(rows)
    summary["eligible"] = (
        summary["passes_trade_filter"]
        & summary["passes_cash_filter"]
        & summary["passes_buy_hold_filter"]
    )
    return summary


def select_regime_threshold(summary: pd.DataFrame) -> float | None:
    """Select a threshold using development metrics only."""

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


def rank_regime_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates from selected development-fold metrics only."""

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
                "candidate_order",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        for rank, index in enumerate(eligible.index, start=1):
            ranked.loc[index, "rank"] = rank
    ranked["rank_sort"] = ranked["rank"].fillna(np.inf)
    return ranked.sort_values(
        by=["rank_sort", "candidate_order"],
        ascending=[True, True],
        kind="mergesort",
    ).drop(columns=["rank_sort"])


def regime_specialist_decision(summary: pd.DataFrame) -> dict[str, Any]:
    """Return the confirmation-only regime diagnostic decision."""

    selected = summary.loc[summary["rank"] == 1] if "rank" in summary else pd.DataFrame()
    selected_row = selected.iloc[0] if not selected.empty else None
    baseline = summary.loc[summary["candidate_name"] == "global_baseline"]
    baseline_return = (
        _as_float(baseline.iloc[0].get("selected_median_total_return"))
        if not baseline.empty
        else float("nan")
    )
    specialist = summary.loc[summary["candidate_name"] == "regime_specialists"]
    specialist_return = (
        _as_float(specialist.iloc[0].get("selected_median_total_return"))
        if not specialist.empty
        else float("nan")
    )
    specialist_improved = bool(
        _finite(specialist_return)
        and _finite(baseline_return)
        and specialist_return > baseline_return
    )
    holdout_confirmed = False
    if selected_row is not None:
        holdout_confirmed = (
            _as_float(selected_row.get("holdout_total_return"))
            > max(
                _as_float(selected_row.get("holdout_cash_total_return")),
                _as_float(selected_row.get("holdout_buy_hold_total_return")),
            )
        )
    regime_helps = bool(
        selected_row is not None
        and selected_row.get("candidate_name") != "global_baseline"
        and specialist_improved
        and holdout_confirmed
    )
    return {
        "decision": (
            "regime_specialization_helps_but_phase_11_blocked"
            if regime_helps
            else "regime_specialization_does_not_help"
        ),
        "selected_development_ranked_candidate": (
            None if selected_row is None else selected_row.get("candidate_name")
        ),
        "selected_threshold": (
            None if selected_row is None else _none_if_nan(selected_row.get("selected_threshold"))
        ),
        "specialist_routing_improved_development": specialist_improved,
        "holdout_confirmation_result": (
            "confirmed" if holdout_confirmed else "not_confirmed"
        ),
        "regime_specialization_helps": regime_helps,
        "holdout_used_for_ranking": False,
        "phase_11_status": "blocked",
    }


def regime_diagnostics_frame(
    features: pd.DataFrame,
    *,
    split: Any,
    lookback_bars: int,
) -> pd.DataFrame:
    """Return regime label distribution diagnostics."""

    rows = []
    for period, data in (
        ("all", features),
        ("development", split.development),
        ("holdout", split.holdout),
    ):
        counts = data[REGIME_COLUMN].value_counts().to_dict()
        rows.append(
            {
                "period": period,
                "lookback_bars": lookback_bars,
                "row_count": len(data),
                "bull_row_count": int(counts.get("bull", 0)),
                "bear_row_count": int(counts.get("bear", 0)),
                "bull_fraction": (
                    float(counts.get("bull", 0) / len(data)) if len(data) else np.nan
                ),
                "first_timestamp": _timestamp_or_none(data, 0),
                "last_timestamp": _timestamp_or_none(data, -1),
            }
        )
    return pd.DataFrame(rows)


def _candidate_probabilities(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: TraderConfig,
    *,
    candidate_name: str,
    feature_names: tuple[str, ...],
) -> ProbabilityBundle:
    labeled_evaluation = evaluation.loc[
        evaluation[TARGET_COLUMN].notna() & evaluation[REGIME_COLUMN].notna()
    ].copy()
    if len(labeled_evaluation) < 2:
        raise ModelTrainingError("fewer than two labeled evaluation rows")
    if candidate_name == "global_baseline":
        model = _fit_model(train, config, feature_names=feature_names)
        return ProbabilityBundle(
            data=labeled_evaluation,
            probabilities=model.predict_positive_proba(labeled_evaluation),
            diagnostics={},
        )
    if candidate_name == "regime_cash_filter":
        model = _fit_model(train, config, feature_names=feature_names)
        probabilities = model.predict_positive_proba(labeled_evaluation)
        probabilities = np.where(
            labeled_evaluation[REGIME_COLUMN].to_numpy() == "bull",
            probabilities,
            -1.0,
        )
        return ProbabilityBundle(
            data=labeled_evaluation,
            probabilities=probabilities,
            diagnostics={"bear_rows_forced_cash": int((probabilities < 0.0).sum())},
        )
    if candidate_name == "regime_specialists":
        probabilities = np.full(len(labeled_evaluation), np.nan, dtype="float64")
        diagnostics: dict[str, Any] = {}
        for regime in ("bull", "bear"):
            train_subset = train.loc[train[REGIME_COLUMN] == regime]
            eval_mask = labeled_evaluation[REGIME_COLUMN] == regime
            eval_subset = labeled_evaluation.loc[eval_mask]
            diagnostics[f"{regime}_evaluation_row_count"] = int(len(eval_subset))
            diagnostics[f"{regime}_train_row_count"] = int(len(train_subset))
            if eval_subset.empty:
                continue
            model = _fit_model(train_subset, config, feature_names=feature_names)
            probabilities[np.flatnonzero(eval_mask.to_numpy())] = (
                model.predict_positive_proba(eval_subset)
            )
        if np.isnan(probabilities).any():
            raise ModelTrainingError("specialist routing produced missing probabilities")
        return ProbabilityBundle(
            data=labeled_evaluation,
            probabilities=probabilities,
            diagnostics=diagnostics,
        )
    raise RegimeSpecialistComparisonError(f"unknown regime candidate: {candidate_name}")


def _fit_model(
    train: pd.DataFrame,
    config: TraderConfig,
    *,
    feature_names: tuple[str, ...],
) -> BaselineLogisticModel:
    model = BaselineLogisticModel(config, feature_names=feature_names)
    model.fit(train, target_column=TARGET_COLUMN)
    return model


def _evaluate_thresholds(
    *,
    fold: WalkForwardFold,
    market_data: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...],
    config: TraderConfig,
    candidate_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "fold": fold.fold_number,
                "period": "development",
                "threshold": threshold,
                "status": "ok",
                **_evaluate_period(
                    market_data=market_data,
                    probabilities=probabilities,
                    threshold=threshold,
                    config=config,
                    candidate_name=candidate_name,
                ),
            }
        )
    return rows


def _evaluate_period(
    *,
    market_data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    config: TraderConfig,
    candidate_name: str,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype="float64")
    if len(probabilities) != len(market_data):
        raise ValueError("probability count must match market rows")
    signals = probabilities_to_signals(probabilities, threshold)
    if candidate_name == "regime_cash_filter":
        signals = np.where(market_data[REGIME_COLUMN].to_numpy() == "bull", signals, 0)
    predictions = pd.DataFrame(
        {
            "timestamp": market_data["timestamp"],
            "signal": signals,
            "probability": probabilities,
        }
    )
    backtest = run_long_cash_backtest(
        market_data,
        predictions,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    cash = run_long_cash_backtest(
        market_data,
        pd.DataFrame({"timestamp": market_data["timestamp"], "signal": 0}),
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    buy_hold_return = _holdout_benchmark_return(
        market_data,
        config,
        benchmark_name="buy_and_hold",
    )
    labeled_mask = market_data[TARGET_COLUMN].notna().to_numpy()
    return {
        **_classification_metrics(
            market_data.loc[labeled_mask, TARGET_COLUMN].astype("int8").to_numpy(),
            signals[labeled_mask],
        ),
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
        "signal_count": int(signals.sum()),
        "cash_total_return": calculate_backtest_metrics(cash.equity, cash.trades)[
            "total_return"
        ],
        "buy_hold_total_return": buy_hold_return,
        **calculate_backtest_metrics(backtest.equity, backtest.trades),
    }


def _candidate_summary_row(
    *,
    candidate_name: str,
    candidate_order: int,
    config: TraderConfig,
    feature_names: tuple[str, ...],
    target_distribution: dict[str, Any],
    threshold_summary: pd.DataFrame,
    selected_threshold: float | None,
    holdout_metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    lookback_bars: int,
) -> dict[str, Any]:
    selected_row: dict[str, Any] = {}
    if selected_threshold is not None:
        matching = threshold_summary.loc[threshold_summary["threshold"] == selected_threshold]
        if not matching.empty:
            selected_row = matching.iloc[0].to_dict()
    return {
        "candidate_name": candidate_name,
        "candidate_order": candidate_order,
        "rank": None,
        "eligible": False,
        "selected_for_future_research": False,
        "lookback_bars": lookback_bars,
        "horizon_bars": config.target.horizon_bars,
        "cost_buffer": config.target.cost_buffer,
        "volatility_multiplier": config.target.volatility_multiplier,
        "enabled_groups": ",".join(config.features.enabled_groups),
        "feature_count": len(feature_names),
        "skipped_fold_count": diagnostics.get("skipped_fold_count", 0),
        "selected_threshold": selected_threshold,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get("median_cash_total_return"),
        "selected_median_buy_hold_total_return": selected_row.get(
            "median_buy_hold_total_return"
        ),
        "selected_median_max_drawdown": selected_row.get("median_max_drawdown"),
        "selected_median_turnover": selected_row.get("median_turnover"),
        "selected_median_exposure_percentage": selected_row.get(
            "median_exposure_percentage"
        ),
        "selected_mean_precision": selected_row.get("mean_precision"),
        "selected_mean_recall": selected_row.get("mean_recall"),
        "selected_mean_f1": selected_row.get("mean_f1"),
        "holdout_status": holdout_metrics.get("status", "not_evaluated"),
        "holdout_total_return": holdout_metrics.get("total_return"),
        "holdout_cash_total_return": holdout_metrics.get("cash_total_return"),
        "holdout_buy_hold_total_return": holdout_metrics.get("buy_hold_total_return"),
        "holdout_max_drawdown": holdout_metrics.get("max_drawdown"),
        "holdout_trade_count": holdout_metrics.get("trade_count"),
        "holdout_turnover": holdout_metrics.get("turnover"),
        "holdout_exposure_percentage": holdout_metrics.get("exposure_percentage"),
        **target_distribution,
    }


def _apply_decision_flags(summary: pd.DataFrame, decision: Mapping[str, Any]) -> pd.DataFrame:
    if summary.empty or decision.get("selected_development_ranked_candidate") is None:
        return summary
    result = summary.copy()
    selected = result["candidate_name"] == decision["selected_development_ranked_candidate"]
    result.loc[selected, "selected_for_future_research"] = bool(
        decision.get("regime_specialization_helps")
    )
    return result


def _folds(row_count: int, config: TraderConfig) -> list[WalkForwardFold]:
    if row_count <= config.validation.minimum_train_bars:
        return []
    folds: list[WalkForwardFold] = []
    train_end = config.validation.minimum_train_bars
    fold_number = 1
    while train_end < row_count:
        test_end = min(train_end + config.validation.test_bars, row_count)
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        train_end += config.validation.step_bars
        fold_number += 1
    return folds


def _skipped_threshold_rows(fold: WalkForwardFold, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "fold": fold.fold_number,
            "period": "development",
            "threshold": threshold,
            "status": "skipped",
            "reason": reason,
        }
        for threshold in DEFAULT_THRESHOLDS
    ]


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def _holdout_benchmark_return(
    data: pd.DataFrame,
    config: TraderConfig,
    *,
    benchmark_name: str,
) -> float | None:
    if len(data) < 2:
        return None
    benchmark = run_benchmarks(
        data,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )[benchmark_name]
    return float(benchmark.metrics["total_return"])


def _fold_signature(features: pd.DataFrame, config: TraderConfig) -> str:
    split = split_final_holdout(features, config.validation)
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
            for fold in _folds(len(split.development), config)
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
        return _timestamp_to_string(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _timestamp_or_none(data: pd.DataFrame, position: int) -> str | None:
    if data.empty or "timestamp" not in data.columns:
        return None
    return _timestamp_to_string(data["timestamp"].iloc[position])


def _timestamp_to_string(value: object) -> str:
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
