"""Controlled model-class comparison for Model Refinement 06."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trader.backtest.benchmarks import run_benchmarks
from trader.config import TraderConfig, load_config
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError, RANDOM_STATE
from trader.modeling.experiments import _target_distribution_dict
from trader.modeling.thresholds import (
    DEFAULT_THRESHOLDS,
    ThresholdSweepResult,
    run_threshold_sweep,
    train_window_folds,
)
from trader.modeling.validation import split_final_holdout


TRAIN_WINDOW_POLICY = "expanding"
PROBABILITY_CALIBRATION = "uncalibrated_predict_proba"


class ModelClassComparisonError(ValueError):
    """Raised when the model-class comparison cannot be completed."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    model_class: str
    grid: tuple[Mapping[str, Any], ...]
    factory_builder: Callable[[Mapping[str, Any]], Callable[..., Any]] | None = None
    skipped_reason: str | None = None


class SklearnProbabilityModel:
    """Small adapter matching the threshold-sweep model protocol."""

    def __init__(
        self,
        config: TraderConfig,
        *,
        feature_names: tuple[str, ...],
        classifier: Any,
        model_type: str,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.feature_names = tuple(feature_names)
        self.model_type = model_type
        self.hyperparameters = dict(hyperparameters)
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", classifier),
            ]
        )
        self._is_fit = False

    def fit(
        self,
        data: pd.DataFrame,
        *,
        target_column: str,
    ) -> "SklearnProbabilityModel":
        if target_column not in data.columns:
            raise ModelTrainingError(f"missing target column: {target_column}")
        labeled = data.loc[data[target_column].notna()].copy()
        if labeled.empty:
            raise ModelTrainingError("no labeled rows are available for training")
        y = labeled[target_column].astype("int8")
        if y.nunique(dropna=True) != 2:
            raise ModelTrainingError(
                f"{self.model_type} requires both target classes in training data"
            )
        x = self._feature_frame(labeled)
        self.pipeline.fit(x, y)
        self._is_fit = True
        return self

    def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise ModelTrainingError(f"{self.model_type} has not been fit")
        probabilities = self.pipeline.predict_proba(self._feature_frame(data))[:, 1]
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise RuntimeError(f"{self.model_type} produced probabilities outside [0, 1]")
        return probabilities

    def _feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_names if column not in data.columns]
        if missing:
            raise ModelTrainingError(
                "missing model feature column(s): " + ", ".join(missing)
            )
        return data.loc[:, self.feature_names].astype("float64")


def run_model_class_comparison(
    *,
    market_dataset_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_path: str | Path = "configs/baseline.yaml",
    enable_xgboost: bool = False,
) -> dict[str, Any]:
    """Run the fixed Refinement 06 model-class comparison."""

    config = model_class_comparison_config(load_config(config_path))
    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing model comparison run: {run_dir}")
    run_dir.mkdir(parents=True)
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir()

    market = read_market_dataset(
        market_dataset_path,
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    features = build_feature_dataset(market, config)
    feature_names = model_feature_columns(config)
    split = split_final_holdout(features, config.validation)
    fold_signature = _fold_signature(features, config)
    feature_signature = _frame_signature(features)
    target_distribution = _target_distribution_dict(features)

    threshold_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows = _benchmark_rows(split.holdout, config)
    diagnostics_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    candidates = _candidate_specs(enable_xgboost=enable_xgboost)
    for candidate_order, candidate in enumerate(candidates):
        candidate_dir = candidates_dir / candidate.name
        candidate_dir.mkdir()
        _write_yaml(candidate_dir / "resolved_config.yaml", asdict(config))
        _write_json(
            candidate_dir / "feature_columns.json",
            {
                "candidate_name": candidate.name,
                "feature_columns": list(feature_names),
                "feature_count": len(feature_names),
                "feature_data_sha256": feature_signature,
                "fold_signature_sha256": fold_signature,
            },
        )

        if candidate.skipped_reason is not None:
            metadata = _candidate_metadata(
                candidate=candidate,
                status="skipped",
                selected_grid_index=None,
                selected_hyperparameters=None,
                reason=candidate.skipped_reason,
            )
            _write_json(candidate_dir / "model_metadata.json", metadata)
            diagnostics_rows.append(_diagnostics_row(candidate, metadata))
            summary_rows.append(
                _skipped_summary_row(candidate, candidate_order, candidate.skipped_reason)
            )
            continue

        grid_results: list[tuple[int, Mapping[str, Any], ThresholdSweepResult]] = []
        for grid_index, hyperparameters in enumerate(candidate.grid):
            assert candidate.factory_builder is not None
            result = run_threshold_sweep(
                features,
                config,
                feature_names=feature_names,
                thresholds=DEFAULT_THRESHOLDS,
                model_factory=candidate.factory_builder(hyperparameters),
                train_window_policy=TRAIN_WINDOW_POLICY,
            )
            grid_results.append((grid_index, hyperparameters, result))

            threshold = result.summary.copy()
            threshold.insert(0, "grid_index", grid_index)
            threshold.insert(0, "model_class", candidate.model_class)
            threshold.insert(0, "candidate_name", candidate.name)
            threshold["hyperparameters_json"] = _stable_json(hyperparameters)
            threshold_frames.append(threshold)

            fold_metrics = result.fold_metrics.copy()
            fold_metrics.insert(0, "grid_index", grid_index)
            fold_metrics.insert(0, "model_class", candidate.model_class)
            fold_metrics.insert(0, "candidate_name", candidate.name)
            fold_metrics["hyperparameters_json"] = _stable_json(hyperparameters)
            fold_frames.append(fold_metrics)

            holdout = result.holdout_metrics or {
                "status": "not_evaluated",
                "selected_threshold": None,
            }
            holdout_rows.append(
                {
                    "candidate_name": candidate.name,
                    "model_class": candidate.model_class,
                    "grid_index": grid_index,
                    "hyperparameters_json": _stable_json(hyperparameters),
                    **holdout,
                }
            )

        selected = _select_grid_result(grid_results)
        if selected is None:
            selected_grid_index = None
            selected_hyperparameters = None
            selected_result = None
        else:
            selected_grid_index, selected_hyperparameters, selected_result = selected

        metadata = _candidate_metadata(
            candidate=candidate,
            status="ok",
            selected_grid_index=selected_grid_index,
            selected_hyperparameters=selected_hyperparameters,
            reason=None,
        )
        _write_json(candidate_dir / "model_metadata.json", metadata)
        diagnostics_rows.append(_diagnostics_row(candidate, metadata))
        summary_rows.append(
            _candidate_summary_row(
                candidate=candidate,
                candidate_order=candidate_order,
                config=config,
                feature_names=feature_names,
                target_distribution=target_distribution,
                selected_grid_index=selected_grid_index,
                selected_hyperparameters=selected_hyperparameters,
                threshold_result=selected_result,
            )
        )

    model_class_summary = _rank_model_classes(pd.DataFrame(summary_rows))
    decision = _decision(model_class_summary)
    model_class_summary = _apply_decision_flags(model_class_summary, decision)

    threshold_summary = _concat_frames(threshold_frames)
    fold_metrics = _concat_frames(fold_frames)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)
    diagnostics = pd.DataFrame(diagnostics_rows)

    _write_csv(model_class_summary, run_dir / "model_class_summary.csv")
    _write_csv(threshold_summary, run_dir / "threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "benchmark_metrics.csv")
    _write_csv(diagnostics, run_dir / "model_diagnostics.csv")
    _write_json(run_dir / "model_class_decision.json", decision)

    return {
        "run_dir": run_dir,
        "model_class_summary": model_class_summary,
        "decision": decision,
    }


def model_class_comparison_config(config: TraderConfig) -> TraderConfig:
    """Apply the fixed controls selected before Refinement 06."""

    return replace(
        config,
        features=replace(
            config.features,
            enabled_groups=("baseline",),
        ),
        target=replace(
            config.target,
            horizon_bars=12,
            cost_buffer="none",
            volatility_multiplier=0.10,
        ),
    )


def _candidate_specs(*, enable_xgboost: bool) -> tuple[CandidateSpec, ...]:
    specs = [
        CandidateSpec(
            name="logistic_regression",
            model_class="logistic_regression",
            grid=({"regularization_c": "config.model.regularization_c"},),
            factory_builder=lambda _: BaselineLogisticModel,
        ),
        CandidateSpec(
            name="random_forest",
            model_class="random_forest",
            grid=tuple(
                {
                    "n_estimators": 100,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": RANDOM_STATE,
                    "n_jobs": 1,
                }
                for max_depth in (3, 5)
                for min_samples_leaf in (20, 50)
            ),
            factory_builder=_random_forest_factory,
        ),
        CandidateSpec(
            name="gradient_boosting",
            model_class="gradient_boosting",
            grid=tuple(
                {
                    "max_iter": 100,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "min_samples_leaf": min_samples_leaf,
                    "random_state": RANDOM_STATE,
                }
                for learning_rate in (0.03, 0.05)
                for max_leaf_nodes in (7, 15)
                for min_samples_leaf in (20, 50)
            ),
            factory_builder=_hist_gradient_boosting_factory,
        ),
    ]
    if not enable_xgboost:
        specs.append(
            CandidateSpec(
                name="xgboost",
                model_class="xgboost",
                grid=(),
                skipped_reason="xgboost disabled; rerun with --enable-xgboost to evaluate",
            )
        )
    elif importlib.util.find_spec("xgboost") is None:
        specs.append(
            CandidateSpec(
                name="xgboost",
                model_class="xgboost",
                grid=(),
                skipped_reason="xgboost dependency is unavailable",
            )
        )
    else:
        specs.append(
            CandidateSpec(
                name="xgboost",
                model_class="xgboost",
                grid=tuple(
                    {
                        "n_estimators": 100,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "min_child_weight": 20,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "random_state": RANDOM_STATE,
                        "n_jobs": 1,
                        "eval_metric": "logloss",
                    }
                    for max_depth in (3, 5)
                    for learning_rate in (0.03, 0.05)
                ),
                factory_builder=_xgboost_factory,
            )
        )
    return tuple(specs)


def _random_forest_factory(
    hyperparameters: Mapping[str, Any],
) -> Callable[..., SklearnProbabilityModel]:
    def factory(config: TraderConfig, *, feature_names: tuple[str, ...]) -> SklearnProbabilityModel:
        return SklearnProbabilityModel(
            config,
            feature_names=feature_names,
            classifier=RandomForestClassifier(**dict(hyperparameters)),
            model_type="RandomForestClassifier",
            hyperparameters=hyperparameters,
        )

    return factory


def _hist_gradient_boosting_factory(
    hyperparameters: Mapping[str, Any],
) -> Callable[..., SklearnProbabilityModel]:
    def factory(config: TraderConfig, *, feature_names: tuple[str, ...]) -> SklearnProbabilityModel:
        return SklearnProbabilityModel(
            config,
            feature_names=feature_names,
            classifier=HistGradientBoostingClassifier(**dict(hyperparameters)),
            model_type="HistGradientBoostingClassifier",
            hyperparameters=hyperparameters,
        )

    return factory


def _xgboost_factory(
    hyperparameters: Mapping[str, Any],
) -> Callable[..., SklearnProbabilityModel]:
    def factory(config: TraderConfig, *, feature_names: tuple[str, ...]) -> SklearnProbabilityModel:
        from xgboost import XGBClassifier

        return SklearnProbabilityModel(
            config,
            feature_names=feature_names,
            classifier=XGBClassifier(**dict(hyperparameters)),
            model_type="XGBClassifier",
            hyperparameters=hyperparameters,
        )

    return factory


def _select_grid_result(
    grid_results: list[tuple[int, Mapping[str, Any], ThresholdSweepResult]],
) -> tuple[int, Mapping[str, Any], ThresholdSweepResult] | None:
    eligible: list[dict[str, Any]] = []
    for grid_index, hyperparameters, result in grid_results:
        if result.selected_threshold is None:
            continue
        matching = result.summary.loc[
            result.summary["threshold"] == result.selected_threshold
        ]
        if matching.empty:
            continue
        row = matching.iloc[0]
        eligible.append(
            {
                "grid_index": grid_index,
                "hyperparameters": hyperparameters,
                "result": result,
                "median_total_return": row.get("median_total_return", np.nan),
                "median_max_drawdown": row.get("median_max_drawdown", np.nan),
                "median_turnover": row.get("median_turnover", np.nan),
            }
        )
    if not eligible:
        return None
    ranked = pd.DataFrame(eligible)
    ranked["drawdown_magnitude"] = ranked["median_max_drawdown"].abs()
    ranked = ranked.sort_values(
        by=["median_total_return", "drawdown_magnitude", "median_turnover", "grid_index"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    selected = ranked.iloc[0]
    return (
        int(selected["grid_index"]),
        selected["hyperparameters"],
        selected["result"],
    )


def _candidate_summary_row(
    *,
    candidate: CandidateSpec,
    candidate_order: int,
    config: TraderConfig,
    feature_names: tuple[str, ...],
    target_distribution: dict[str, Any],
    selected_grid_index: int | None,
    selected_hyperparameters: Mapping[str, Any] | None,
    threshold_result: ThresholdSweepResult | None,
) -> dict[str, Any]:
    selected_row: dict[str, Any] = {}
    selected_threshold = None
    holdout_metrics: dict[str, Any] = {}
    if threshold_result is not None:
        selected_threshold = threshold_result.selected_threshold
        if selected_threshold is not None:
            matching = threshold_result.summary.loc[
                threshold_result.summary["threshold"] == selected_threshold
            ]
            if not matching.empty:
                selected_row = matching.iloc[0].to_dict()
        holdout_metrics = threshold_result.holdout_metrics or {}

    return {
        "candidate_name": candidate.name,
        "model_class": candidate.model_class,
        "candidate_order": candidate_order,
        "status": "ok",
        "reason": None,
        "rank": None,
        "eligible": False,
        "passes_development_policy": False,
        "passes_holdout_confirmation": False,
        "selected_for_future_research": False,
        "selected_grid_index": selected_grid_index,
        "selected_hyperparameters_json": (
            _stable_json(selected_hyperparameters) if selected_hyperparameters is not None else None
        ),
        "horizon_bars": config.target.horizon_bars,
        "cost_buffer": config.target.cost_buffer,
        "volatility_multiplier": config.target.volatility_multiplier,
        "enabled_groups": ",".join(config.features.enabled_groups),
        "feature_count": len(feature_names),
        "train_window_policy": TRAIN_WINDOW_POLICY,
        "probability_calibration": PROBABILITY_CALIBRATION,
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
        **target_distribution,
    }


def _skipped_summary_row(
    candidate: CandidateSpec,
    candidate_order: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "candidate_name": candidate.name,
        "model_class": candidate.model_class,
        "candidate_order": candidate_order,
        "status": "skipped",
        "reason": reason,
        "rank": None,
        "eligible": False,
        "passes_development_policy": False,
        "passes_holdout_confirmation": False,
        "selected_for_future_research": False,
        "selected_grid_index": None,
        "selected_hyperparameters_json": None,
        "train_window_policy": TRAIN_WINDOW_POLICY,
        "probability_calibration": PROBABILITY_CALIBRATION,
    }


def _rank_model_classes(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    ranked = summary.copy()
    ok = ranked["status"] == "ok"
    selected = ranked["selected_threshold"].notna()
    beats_cash = (
        ranked["selected_median_total_return"].fillna(-np.inf)
        > ranked["selected_median_cash_total_return"].fillna(np.inf)
    )
    ranked["eligible"] = ok & selected & beats_cash

    eligible = ranked.loc[ranked["eligible"]].copy()
    if not eligible.empty:
        eligible["drawdown_magnitude"] = eligible["selected_median_max_drawdown"].abs()
        eligible = eligible.sort_values(
            by=[
                "selected_median_total_return",
                "drawdown_magnitude",
                "selected_median_turnover",
                "candidate_order",
            ],
            ascending=[False, True, True, True],
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


def _decision(summary: pd.DataFrame) -> dict[str, Any]:
    baseline = summary.loc[summary["candidate_name"] == "logistic_regression"]
    if baseline.empty or baseline.iloc[0].get("selected_threshold") is None:
        return {
            "decision": "model_class_change_rejected",
            "selected_model_class": "logistic_regression",
            "selected_candidate_name": None,
            "reason": "baseline logistic regression did not produce a selected development threshold",
            "phase_11_status": "blocked",
            "holdout_used_for_ranking": False,
            "probability_calibration": PROBABILITY_CALIBRATION,
        }

    baseline_row = baseline.iloc[0]
    baseline_return = _as_float(baseline_row.get("selected_median_total_return"))
    baseline_drawdown = abs(_as_float(baseline_row.get("selected_median_max_drawdown")))
    baseline_turnover = _as_float(baseline_row.get("selected_median_turnover"))
    baseline_holdout_return = _as_float(baseline_row.get("holdout_total_return"))
    baseline_holdout_drawdown = abs(_as_float(baseline_row.get("holdout_max_drawdown")))

    candidates = summary.loc[
        (summary["status"] == "ok")
        & (summary["candidate_name"] != "logistic_regression")
        & summary["selected_threshold"].notna()
    ].copy()
    if candidates.empty:
        return _rejected_decision("no non-logistic candidate selected a development threshold")

    candidates["development_improves_return"] = (
        candidates["selected_median_total_return"].astype("float64") > baseline_return
    )
    candidates["development_beats_cash"] = (
        candidates["selected_median_total_return"].astype("float64")
        > candidates["selected_median_cash_total_return"].astype("float64")
    )
    candidates["development_drawdown_ok"] = (
        candidates["selected_median_max_drawdown"].abs().astype("float64")
        <= baseline_drawdown + 0.02
    )
    candidates["development_turnover_ok"] = (
        candidates["selected_median_turnover"].astype("float64")
        <= max(baseline_turnover * 1.5, baseline_turnover + 2.0)
    )
    candidates["holdout_return_confirms"] = (
        candidates["holdout_total_return"].astype("float64") > baseline_holdout_return
    )
    candidates["holdout_drawdown_ok"] = (
        candidates["holdout_max_drawdown"].abs().astype("float64")
        <= baseline_holdout_drawdown + 0.02
    )
    candidates["holdout_turnover_ok"] = (
        candidates["holdout_turnover"].astype("float64")
        <= max(baseline_turnover * 2.0, baseline_turnover + 4.0)
    )
    candidates["passes_policy"] = (
        candidates["development_improves_return"]
        & candidates["development_beats_cash"]
        & candidates["development_drawdown_ok"]
        & candidates["development_turnover_ok"]
        & candidates["holdout_return_confirms"]
        & candidates["holdout_drawdown_ok"]
        & candidates["holdout_turnover_ok"]
    )
    passed = candidates.loc[candidates["passes_policy"]].copy()
    if passed.empty:
        return _rejected_decision(
            "no non-logistic candidate improved development results and confirmed on holdout"
        )
    passed["drawdown_magnitude"] = passed["selected_median_max_drawdown"].abs()
    passed = passed.sort_values(
        by=[
            "selected_median_total_return",
            "drawdown_magnitude",
            "selected_median_turnover",
            "candidate_order",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    selected = passed.iloc[0]
    return {
        "decision": "model_class_change_passes_refinement_06_but_phase_11_blocked",
        "selected_model_class": selected["model_class"],
        "selected_candidate_name": selected["candidate_name"],
        "selected_threshold": selected["selected_threshold"],
        "selected_hyperparameters_json": selected["selected_hyperparameters_json"],
        "reason": "non-logistic candidate passed development policy and holdout confirmation",
        "phase_11_status": "blocked",
        "holdout_used_for_ranking": False,
        "probability_calibration": PROBABILITY_CALIBRATION,
    }


def _rejected_decision(reason: str) -> dict[str, Any]:
    return {
        "decision": "model_class_change_rejected",
        "selected_model_class": "logistic_regression",
        "selected_candidate_name": "logistic_regression",
        "reason": reason,
        "phase_11_status": "blocked",
        "holdout_used_for_ranking": False,
        "probability_calibration": PROBABILITY_CALIBRATION,
    }


def _apply_decision_flags(summary: pd.DataFrame, decision: Mapping[str, Any]) -> pd.DataFrame:
    if summary.empty:
        return summary
    result = summary.copy()
    if decision.get("selected_candidate_name") in set(result["candidate_name"]):
        selected = result["candidate_name"] == decision["selected_candidate_name"]
        result.loc[selected, "selected_for_future_research"] = (
            decision["decision"]
            == "model_class_change_passes_refinement_06_but_phase_11_blocked"
        )
    non_logistic = result["candidate_name"] != "logistic_regression"
    baseline = result.loc[result["candidate_name"] == "logistic_regression"]
    if baseline.empty:
        return result
    baseline_return = _as_float(baseline.iloc[0].get("selected_median_total_return"))
    result.loc[non_logistic, "passes_development_policy"] = (
        result.loc[non_logistic, "selected_median_total_return"].astype("float64")
        > baseline_return
    )
    result.loc[non_logistic, "passes_holdout_confirmation"] = (
        result.loc[non_logistic, "holdout_total_return"].astype("float64")
        > _as_float(baseline.iloc[0].get("holdout_total_return"))
    )
    return result


def _candidate_metadata(
    *,
    candidate: CandidateSpec,
    status: str,
    selected_grid_index: int | None,
    selected_hyperparameters: Mapping[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "candidate_name": candidate.name,
        "model_class": candidate.model_class,
        "status": status,
        "reason": reason,
        "grid": list(candidate.grid),
        "selected_grid_index": selected_grid_index,
        "selected_hyperparameters": (
            dict(selected_hyperparameters) if selected_hyperparameters is not None else None
        ),
        "random_state": RANDOM_STATE,
        "threshold_candidates": list(DEFAULT_THRESHOLDS),
        "threshold_selection_scope": "development_folds_only",
        "holdout_role": "confirmation_only",
        "probability_calibration": PROBABILITY_CALIBRATION,
        "train_window_policy": TRAIN_WINDOW_POLICY,
    }


def _diagnostics_row(candidate: CandidateSpec, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": candidate.name,
        "model_class": candidate.model_class,
        "status": metadata["status"],
        "reason": metadata.get("reason"),
        "grid_size": len(candidate.grid),
        "selected_grid_index": metadata.get("selected_grid_index"),
        "probability_calibration": metadata["probability_calibration"],
        "threshold_selection_scope": metadata["threshold_selection_scope"],
        "holdout_role": metadata["holdout_role"],
    }


def _benchmark_rows(holdout: pd.DataFrame, config: TraderConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark_name, benchmark in run_benchmarks(
        holdout,
        backtest_config=config.backtest,
        costs_config=config.costs,
    ).items():
        rows.append({"benchmark": benchmark_name, **benchmark.metrics})
    return rows


def _fold_signature(features: pd.DataFrame, config: TraderConfig) -> str:
    split = split_final_holdout(features, config.validation)
    folds = train_window_folds(
        len(split.development),
        config.validation,
        train_window_policy=TRAIN_WINDOW_POLICY,
    )
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
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
