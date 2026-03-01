#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for the Trading Ensemble.

Computes SHAP values for each tree-based sub-model (RF, XGBoost, LightGBM)
and generates visualizations showing which features drive predictions.

Usage:
    pip install shap
    python scripts/shap_analysis.py

Outputs saved to outputs/shap/
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import shap
# Monkey-patch SHAP's XGBoost loader to handle XGBoost 3.x base_score format '[5E-1]'
import builtins
_original_float = builtins.float
class _PatchedFloat(_original_float):
    def __new__(cls, x=0):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            x = x.strip('[]')
        return _original_float.__new__(cls, x)
# Patch only within the SHAP tree module
_orig_xgb_loader_init = shap.explainers._tree.XGBTreeModelLoader.__init__
def _patched_xgb_loader_init(self, xgb_model):
    builtins.float = _PatchedFloat
    try:
        _orig_xgb_loader_init(self, xgb_model)
    finally:
        builtins.float = _original_float
shap.explainers._tree.XGBTreeModelLoader.__init__ = _patched_xgb_loader_init
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from src.utils.config import TradingConfig
from src.data.market_client import MarketClient
from src.data.reddit_client import RedditClient
from src.data.archive import load_archive, init_database, migrate_from_csv
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.models.trading_model import ImprovedTradingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SHAP-Analysis")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load market + sentiment data and build features using the production pipeline."""
    logger.info("Loading configuration...")
    config = TradingConfig.from_yaml(str(PROJECT_ROOT / "configs" / "data.yaml"))

    # Initialize archive
    init_database()
    migrate_from_csv()

    # Load Reddit data
    logger.info("Loading Reddit archive...")
    archive_df = load_archive()

    # Also load rolling window if available
    master_path = PROJECT_ROOT / "data" / "master_reddit.csv"
    if master_path.exists():
        master_reddit = pd.read_csv(master_path)
        master_reddit["created_utc"] = pd.to_datetime(master_reddit["created_utc"])
        if not archive_df.empty:
            reddit_data = pd.concat([archive_df, master_reddit]).drop_duplicates(subset=["id"])
        else:
            reddit_data = master_reddit
    else:
        reddit_data = archive_df

    if reddit_data.empty:
        logger.error("No Reddit data found. Run the main bot first to collect data.")
        sys.exit(1)

    reddit_data["created_utc"] = pd.to_datetime(reddit_data["created_utc"])

    # Fetch market data covering the Reddit data range
    from datetime import datetime, timezone

    min_date = reddit_data["created_utc"].min()
    days_needed = (datetime.now(timezone.utc) - min_date).days + 3
    days_needed = min(days_needed, 90)  # Cap to avoid excessive API calls
    logger.info(f"Fetching {days_needed} days of market data...")

    market_client = MarketClient(config)
    market_df = market_client.fetch_ohlcv(lookback_days=days_needed)

    if market_df.empty:
        logger.error("No market data fetched. Check Binance connectivity.")
        sys.exit(1)

    # Build sentiment features
    logger.info("Computing sentiment features...")
    sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
    sentiment_df = sentiment_analyzer.analyze(reddit_data)

    # Fetch derivatives data
    logger.info("Fetching funding rates and open interest...")
    funding_df = market_client.fetch_funding_rates(lookback_days=days_needed)
    oi_df = market_client.fetch_open_interest(lookback_days=days_needed)

    # Build features via production pipeline
    logger.info("Building features via model.prepare_features()...")
    model = ImprovedTradingModel(
        enter_threshold=config.enter_threshold,
        exit_threshold=config.exit_threshold,
        min_confidence=config.min_confidence,
    )
    train_df = model.prepare_features(market_df, sentiment_df, is_inference=False, funding_df=funding_df, oi_df=oi_df)

    if train_df.empty:
        logger.error("Feature preparation returned empty DataFrame.")
        sys.exit(1)

    logger.info(f"Dataset: {len(train_df)} rows, {len(model.features)} features")
    return model, train_df


def train_model(model, train_df):
    """Train the model using the production training pipeline."""
    logger.info("Training model (with feature selection, no tuning)...")
    result = model.train(train_df, tune=False, feature_selection=True, validate=False)
    logger.info(f"Training result: {result['status']}, {result['n_features']} features selected")
    return model


def extract_tree_estimators(model):
    """
    Extract the underlying tree-based estimators from the ensemble.

    The ensemble wraps estimators in CalibratedClassifierCV, so we need
    to dig through: VotingClassifier -> CalibratedClassifierCV -> base_estimator.
    """
    ensemble = model.ensemble
    if ensemble.ensemble_ is None:
        raise RuntimeError("Ensemble not fitted. Train the model first.")

    voting_clf = ensemble.ensemble_
    tree_models = {}

    for name, estimator in voting_clf.named_estimators_.items():
        if name == "lr":
            continue  # Skip logistic regression (not tree-based)

        # Unwrap CalibratedClassifierCV if present
        actual_model = estimator
        if hasattr(estimator, "estimator"):
            # CalibratedClassifierCV stores the base estimator
            # But the calibrated version fits internal copies, so we need
            # to fit a fresh copy for SHAP
            actual_model = estimator.estimator
        elif hasattr(estimator, "base_estimator"):
            actual_model = estimator.base_estimator

        tree_models[name] = actual_model
        logger.info(f"Extracted {name}: {type(actual_model).__name__}")

    return tree_models


def compute_shap_values(model, train_df):
    """
    Compute SHAP values for each tree-based model separately.

    We fit fresh copies of each tree model (same hyperparameters) because
    the calibration wrappers in the ensemble obscure the fitted trees.
    """
    features = model.features
    X = train_df[features].values
    y = train_df["target"].values

    # Build fresh tree estimators with the same params as the ensemble
    ensemble = model.ensemble
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

    base_estimators = ensemble._build_base_estimators(scale_pos_weight)

    # Subsample for SHAP if dataset is large (SHAP can be slow on huge datasets)
    max_samples = 5000
    if len(X) > max_samples:
        logger.info(f"Subsampling from {len(X)} to {max_samples} rows for SHAP...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_shap = X[idx]
        y_fit = y  # Still fit on full data
    else:
        X_shap = X
        y_fit = y

    shap_results = {}

    for name, est in base_estimators:
        if name == "lr":
            continue

        logger.info(f"Computing SHAP for {name}...")
        # Fit on full data
        est.fit(X, y_fit)

        # TreeExplainer is exact and fast for tree models
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(X_shap)

        # For binary classification, shap_values may be:
        #   - a list [class_0, class_1] (older SHAP / some models)
        #   - a 3D array (samples, features, classes) — e.g. RF
        #   - a 2D array (samples, features) — e.g. XGB
        # We want class 1 (BUY) SHAP values as 2D
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        shap_results[name] = {
            "values": shap_values,
            "explainer": explainer,
        }
        logger.info(f"  {name} SHAP values shape: {shap_values.shape}")

    return shap_results, X_shap, features


def plot_global_bar(shap_results, features):
    """Global bar plot: mean |SHAP| per feature, averaged across models."""
    mean_abs = []
    for name, result in shap_results.items():
        mean_abs.append(np.abs(result["values"]).mean(axis=0))

    # Average across models
    avg_importance = np.mean(mean_abs, axis=0)
    importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": avg_importance,
    }).sort_values("mean_abs_shap", ascending=True)

    # Show top 20
    top = importance_df.tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (averaged across RF, XGB, LGB)")
    ax.set_title("Global Feature Importance (Ensemble Average)")
    plt.tight_layout()

    path = OUTPUT_DIR / "global_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")

    return importance_df


def plot_per_model_comparison(shap_results, features):
    """Side-by-side bar plot comparing feature importance across models."""
    model_names = list(shap_results.keys())
    n_models = len(model_names)

    # Compute mean |SHAP| per model
    importances = {}
    for name in model_names:
        importances[name] = np.abs(shap_results[name]["values"]).mean(axis=0)

    # Get top 15 features by average importance
    avg = np.mean(list(importances.values()), axis=0)
    top_idx = np.argsort(avg)[-15:]
    top_features = [features[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_features))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, name in enumerate(model_names):
        vals = [importances[name][features.index(f)] for f in top_features]
        ax.barh(x + i * width, vals, width, label=name.upper(), color=colors[i])

    ax.set_yticks(x + width)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Per-Model Feature Importance (RF vs XGB vs LGB)")
    ax.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "per_model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_beeswarm(shap_results, X_shap, features):
    """Beeswarm plot showing feature value direction effects."""
    # Average SHAP values across models for beeswarm
    all_values = [result["values"] for result in shap_results.values()]
    avg_shap = np.mean(all_values, axis=0)

    explanation = shap.Explanation(
        values=avg_shap,
        data=X_shap,
        feature_names=features,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP Beeswarm (Ensemble Average)")
    plt.tight_layout()

    path = OUTPUT_DIR / "beeswarm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_dependence(shap_results, X_shap, features, top_n=5):
    """Scatter plots of SHAP value vs feature value for top features."""
    # Average SHAP values
    all_values = [result["values"] for result in shap_results.values()]
    avg_shap = np.mean(all_values, axis=0)

    # Find top features by mean |SHAP|
    mean_abs = np.abs(avg_shap).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    if top_n == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        feat_name = features[idx]
        ax.scatter(
            X_shap[:, idx],
            avg_shap[:, idx],
            alpha=0.3,
            s=5,
            c=X_shap[:, idx],
            cmap="coolwarm",
        )
        ax.set_xlabel(feat_name)
        ax.set_ylabel("SHAP value")
        ax.set_title(feat_name)
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)

    plt.suptitle("Feature Dependence Plots (Top 5)", y=1.02)
    plt.tight_layout()

    path = OUTPUT_DIR / "dependence_top5.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def print_feature_table(importance_df):
    """Print ranked feature importance table to console."""
    ranked = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranked.index += 1  # 1-based ranking

    print("\n" + "=" * 60)
    print("  SHAP Feature Importance Ranking (Ensemble Average)")
    print("=" * 60)
    print(f"{'Rank':<6}{'Feature':<35}{'Mean |SHAP|':<15}")
    print("-" * 60)

    for rank, row in ranked.iterrows():
        print(f"{rank:<6}{row['feature']:<35}{row['mean_abs_shap']:<15.6f}")

    print("=" * 60)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print()


def main():
    logger.info("=" * 50)
    logger.info("SHAP Feature Importance Analysis")
    logger.info("=" * 50)

    # 1. Load data and build features
    model, train_df = load_and_prepare_data()

    # 2. Train model
    model = train_model(model, train_df)

    # 3. Compute SHAP values
    shap_results, X_shap, features = compute_shap_values(model, train_df)

    if not shap_results:
        logger.error("No SHAP results computed. Exiting.")
        sys.exit(1)

    # 4. Generate plots
    logger.info("Generating plots...")
    importance_df = plot_global_bar(shap_results, features)
    plot_per_model_comparison(shap_results, features)
    plot_beeswarm(shap_results, X_shap, features)
    plot_dependence(shap_results, X_shap, features, top_n=5)

    # 5. Print table
    print_feature_table(importance_df)

    logger.info("SHAP analysis complete.")


if __name__ == "__main__":
    main()
