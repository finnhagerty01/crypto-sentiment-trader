"""Run Model Refinement 02.5 recency-window diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.config import TargetConfig, TraderConfig, load_config
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset
from trader.modeling.thresholds import (
    DEFAULT_THRESHOLDS,
    TRAIN_WINDOW_POLICIES,
    ModelFactory,
    ThresholdSweepResult,
    run_threshold_sweep,
    write_threshold_sweep_artifacts,
)


DEFAULT_FEATURES = Path(
    "artifacts/model_refinement/target_horizon_grid/"
    "h12_none_vol0p10/feature_dataset.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/model_refinement/recency_window_diagnostic")
TARGET_HORIZON_BARS = 12
TARGET_COST_BUFFER = "none"
TARGET_VOLATILITY_MULTIPLIER = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare expanding and recent rolling train windows."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"Feature parquet path. Default: {DEFAULT_FEATURES}",
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=None,
        help="Raw market parquet used to rebuild features if --features is absent.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Baseline config path. Default: configs/baseline.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config does not exist: {args.config}")

    config = _target_winner_config(load_config(args.config))
    features = load_or_build_feature_dataset(
        feature_dataset_path=args.features,
        raw_dataset_path=args.raw_dataset,
        config=config,
        output_dir=args.output_dir,
    )
    result = run_recency_window_diagnostic(
        features=features,
        config=config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print(f"selected_window: {result.selected_window}")
    print(f"output_dir: {result.output_dir}")
    print(f"results_csv: {result.output_dir / 'window_results.csv'}")
    print(f"selected_json: {result.output_dir / 'selected_window.json'}")
    print()
    print(result.summary.to_string(index=False))
    return 0


class RecencyWindowDiagnosticResult:
    """Aggregated diagnostic outputs."""

    def __init__(
        self,
        *,
        summary: pd.DataFrame,
        selected_window: dict[str, Any] | None,
        output_dir: Path,
    ) -> None:
        self.summary = summary
        self.selected_window = selected_window
        self.output_dir = output_dir


def load_or_build_feature_dataset(
    *,
    feature_dataset_path: Path,
    raw_dataset_path: Path | None,
    config: TraderConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Load the target-grid winner features, rebuilding from raw market data if needed."""

    if feature_dataset_path.exists():
        return pd.read_parquet(feature_dataset_path)

    raw_path = raw_dataset_path or _latest_raw_dataset(Path("artifacts/datasets"))
    if raw_path is None:
        raise FileNotFoundError(
            "feature dataset is absent and no raw BTCUSDT 1h parquet was found"
        )
    market_data = read_market_dataset(
        raw_path,
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    features = build_feature_dataset(market_data, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "feature_dataset.parquet", index=False)
    return features


def run_recency_window_diagnostic(
    *,
    features: pd.DataFrame,
    config: TraderConfig,
    output_dir: str | Path,
    overwrite: bool = False,
    model_factory: ModelFactory | None = None,
) -> RecencyWindowDiagnosticResult:
    """Evaluate all train-window policies and write diagnostic artifacts."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing recency artifacts: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for policy in TRAIN_WINDOW_POLICIES:
        kwargs: dict[str, Any] = {"train_window_policy": policy}
        if model_factory is not None:
            kwargs["model_factory"] = model_factory
        threshold_result = run_threshold_sweep(features, config, **kwargs)
        window_dir = destination / policy
        write_threshold_sweep_artifacts(threshold_result, window_dir)
        rows.append(
            summarize_window_result(
                train_window_policy=policy,
                threshold_result=threshold_result,
            )
        )

    summary = pd.DataFrame(rows)
    selected_window = select_recency_window(summary)
    summary.to_csv(destination / "window_results.csv", index=False)
    _write_json(destination / "window_results.json", {"windows": rows})
    _write_json(
        destination / "selected_window.json",
        selected_window or {"selected_window": None},
    )
    (destination / "recency_window_report.md").write_text(
        build_recency_window_report(summary, selected_window=selected_window),
        encoding="utf-8",
    )
    return RecencyWindowDiagnosticResult(
        summary=summary,
        selected_window=selected_window,
        output_dir=destination,
    )


def summarize_window_result(
    *,
    train_window_policy: str,
    threshold_result: ThresholdSweepResult,
) -> dict[str, Any]:
    """Build one policy summary row with development selection and holdout fields."""

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
        "train_window_policy": train_window_policy,
        "selected_threshold": selected_threshold,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get(
            "median_cash_total_return"
        ),
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
        "holdout_probability_min": holdout_metrics.get("probability_min"),
        "holdout_probability_mean": holdout_metrics.get("probability_mean"),
        "holdout_probability_max": holdout_metrics.get("probability_max"),
        "holdout_probability_std": holdout_metrics.get("probability_std"),
    }


def select_recency_window(summary: pd.DataFrame) -> dict[str, Any] | None:
    """Select the train-window policy from development metrics only."""

    if summary.empty:
        return None
    eligible = summary.loc[summary["selected_threshold"].notna()].copy()
    eligible = eligible.loc[
        eligible["selected_median_total_return"].notna()
        & eligible["selected_median_max_drawdown"].notna()
        & eligible["selected_median_turnover"].notna()
    ]
    if eligible.empty:
        return None

    eligible["drawdown_magnitude"] = eligible[
        "selected_median_max_drawdown"
    ].abs()
    eligible = eligible.sort_values(
        by=[
            "selected_median_total_return",
            "drawdown_magnitude",
            "selected_median_turnover",
            "train_window_policy",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return _json_ready(eligible.iloc[0].drop(labels=["drawdown_magnitude"]).to_dict())


def build_recency_window_report(
    summary: pd.DataFrame,
    *,
    selected_window: dict[str, Any] | None,
) -> str:
    """Build a concise Markdown report for the recency-window diagnostic."""

    lines = [
        "# Recency Window Diagnostic",
        "",
        "Selection uses development-fold metrics only. Holdout metrics are confirmation only.",
        "",
        "## Selected Development Policy",
        "",
    ]
    if selected_window is None:
        lines.extend(["No train-window policy produced a selectable threshold.", ""])
    else:
        lines.extend(
            [
                f"- Train window: `{selected_window['train_window_policy']}`",
                f"- Selected threshold: `{selected_window['selected_threshold']}`",
                "- Development median return: "
                f"`{_format_percent(selected_window['selected_median_total_return'])}`",
                "- Development median max drawdown: "
                f"`{_format_percent(selected_window['selected_median_max_drawdown'])}`",
                "- Development median turnover: "
                f"`{_format_number(selected_window['selected_median_turnover'])}`",
                "- Holdout return: "
                f"`{_format_percent(selected_window['holdout_total_return'])}`",
                "- Holdout max drawdown: "
                f"`{_format_percent(selected_window['holdout_max_drawdown'])}`",
                "- Holdout exposure: "
                f"`{_format_percent(selected_window['holdout_exposure_percentage'])}`",
                "",
            ]
        )

    lines.extend(["## Window Comparison", ""])
    if summary.empty:
        lines.extend(["No window results were produced.", ""])
    else:
        lines.append(
            "| window | threshold | dev return | dev drawdown | dev turnover | holdout return | holdout drawdown | holdout trades | exposure |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary.to_dict(orient="records"):
            lines.append(
                "| "
                f"`{row['train_window_policy']}` | "
                f"{_format_value(row['selected_threshold'])} | "
                f"{_format_percent(row['selected_median_total_return'])} | "
                f"{_format_percent(row['selected_median_max_drawdown'])} | "
                f"{_format_number(row['selected_median_turnover'])} | "
                f"{_format_percent(row['holdout_total_return'])} | "
                f"{_format_percent(row['holdout_max_drawdown'])} | "
                f"{_format_value(row['holdout_trade_count'])} | "
                f"{_format_percent(row['holdout_exposure_percentage'])} |"
            )
        lines.append("")

    lines.extend(["## Interpretation", ""])
    lines.append(_interpret_recency_result(summary, selected_window=selected_window))
    lines.append("")
    lines.extend(
        [
            "Phase 11 remains blocked unless a development-selected policy also confirms on holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpret_recency_result(
    summary: pd.DataFrame,
    *,
    selected_window: dict[str, Any] | None,
) -> str:
    if selected_window is None or summary.empty:
        return "Rolling recency did not produce a development-selected policy; proceed to `03_MARKET_FEATURE_GROUPS`."

    expanding = summary.loc[summary["train_window_policy"] == "expanding"]
    if expanding.empty:
        return "The expanding baseline row is absent, so holdout confirmation cannot be compared."

    selected_policy = selected_window["train_window_policy"]
    selected_holdout = selected_window.get("holdout_total_return")
    selected_exposure = selected_window.get("holdout_exposure_percentage")
    expanding_holdout = expanding.iloc[0].get("holdout_total_return")
    expanding_exposure = expanding.iloc[0].get("holdout_exposure_percentage")
    if (
        selected_policy != "expanding"
        and _finite(selected_holdout)
        and _finite(expanding_holdout)
        and float(selected_holdout) > float(expanding_holdout)
        and (
            not _finite(expanding_exposure)
            or (
                _finite(selected_exposure)
                and float(selected_exposure) < float(expanding_exposure)
            )
        )
    ):
        return (
            "A rolling window improved holdout return and reduced exposure versus "
            "the expanding baseline, so recency dynamics are likely part of the problem."
        )
    return (
        "Rolling windows did not materially improve holdout confirmation while reducing "
        "excessive exposure; proceed to `03_MARKET_FEATURE_GROUPS`."
    )


def _target_winner_config(config: TraderConfig) -> TraderConfig:
    return replace(
        config,
        target=TargetConfig(
            horizon_bars=TARGET_HORIZON_BARS,
            cost_buffer=TARGET_COST_BUFFER,
            volatility_multiplier=TARGET_VOLATILITY_MULTIPLIER,
        ),
    )


def _latest_raw_dataset(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("btcusdt_1h_raw_*.parquet"))
    if not candidates:
        return None
    return candidates[-1]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return str(value)


def _finite(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(np.isfinite(float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
