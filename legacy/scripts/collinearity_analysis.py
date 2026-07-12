#!/usr/bin/env python3
"""
Feature Collinearity Analysis.

Investigates whether SHAP importance splitting due to collinearity explains
why sentiment features appear weak. Generates:
  1. Clustered correlation heatmap
  2. Variance Inflation Factor (VIF) bar plot
  3. Hierarchical clustering dendrogram
  4. Collinear pairs table (|r| > 0.7)

Usage:
    python scripts/collinearity_analysis.py

Outputs saved to outputs/collinearity/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

from src.utils.config import TradingConfig
from src.data.market_client import MarketClient
from src.data.archive import load_archive, init_database, migrate_from_csv
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.models.trading_model import ImprovedTradingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Collinearity-Analysis")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "collinearity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load market + sentiment data and build features using the production pipeline."""
    logger.info("Loading configuration...")
    config = TradingConfig.from_yaml(str(PROJECT_ROOT / "configs" / "data.yaml"))

    init_database()
    migrate_from_csv()

    logger.info("Loading Reddit archive...")
    archive_df = load_archive()

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

    from datetime import datetime, timezone

    min_date = reddit_data["created_utc"].min()
    days_needed = (datetime.now(timezone.utc) - min_date).days + 3
    days_needed = min(days_needed, 90)
    logger.info(f"Fetching {days_needed} days of market data...")

    market_client = MarketClient(config)
    market_df = market_client.fetch_ohlcv(lookback_days=days_needed)

    if market_df.empty:
        logger.error("No market data fetched. Check Binance connectivity.")
        sys.exit(1)

    logger.info("Computing sentiment features...")
    sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
    sentiment_df = sentiment_analyzer.analyze(reddit_data)

    # Fetch derivatives data
    logger.info("Fetching funding rates and open interest...")
    funding_df = market_client.fetch_funding_rates(lookback_days=days_needed)
    oi_df = market_client.fetch_open_interest(lookback_days=days_needed)

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


def compute_vif(X_df):
    """
    Compute Variance Inflation Factor for each feature.

    Tries statsmodels first; falls back to manual OLS-based calculation.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_data = []
        X_arr = X_df.values.astype(np.float64)
        for i in range(X_arr.shape[1]):
            vif_data.append({
                "feature": X_df.columns[i],
                "VIF": variance_inflation_factor(X_arr, i),
            })
        return pd.DataFrame(vif_data)
    except ImportError:
        logger.info("statsmodels not found, using manual VIF calculation...")

    # Manual fallback: VIF_i = 1 / (1 - R²_i)
    vif_data = []
    X_arr = X_df.values.astype(np.float64)
    for i in range(X_arr.shape[1]):
        y = X_arr[:, i]
        X_others = np.delete(X_arr, i, axis=1)
        # Add intercept
        X_others = np.column_stack([np.ones(len(y)), X_others])
        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_others, y, rcond=None)[0]
            y_hat = X_others @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf
        vif_data.append({"feature": X_df.columns[i], "VIF": vif})

    return pd.DataFrame(vif_data)


def plot_clustered_heatmap(corr_matrix, features, linkage_matrix):
    """Generate a clustered correlation heatmap."""
    order = leaves_list(linkage_matrix)
    ordered_features = [features[i] for i in order]
    ordered_corr = corr_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(ordered_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ordered_features)))
    ax.set_yticks(range(len(ordered_features)))
    ax.set_xticklabels(ordered_features, rotation=90, fontsize=6)
    ax.set_yticklabels(ordered_features, fontsize=6)
    ax.set_title("Feature Correlation Matrix (Hierarchically Clustered)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    plt.tight_layout()

    path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_dendrogram(linkage_matrix, features):
    """Generate a hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=features,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
        color_threshold=0.3,
    )
    ax.set_title("Feature Clustering Dendrogram (distance = 1 - |correlation|)")
    ax.set_ylabel("Distance (1 - |r|)")
    ax.axhline(y=0.3, color="red", linestyle="--", linewidth=0.8, label="r=0.7 threshold")
    ax.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "dendrogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_vif(vif_df):
    """Generate a VIF bar plot."""
    vif_sorted = vif_df.sort_values("VIF", ascending=True)

    # Cap display at VIF=50 for readability
    vif_display = vif_sorted.copy()
    vif_display["VIF_display"] = vif_display["VIF"].clip(upper=50)

    colors = []
    for v in vif_display["VIF"]:
        if v > 10:
            colors.append("#d62728")  # red — severe
        elif v > 5:
            colors.append("#ff7f0e")  # orange — concerning
        else:
            colors.append("#2ca02c")  # green — acceptable

    fig, ax = plt.subplots(figsize=(10, max(8, len(vif_df) * 0.25)))
    ax.barh(vif_display["feature"], vif_display["VIF_display"], color=colors)
    ax.axvline(x=5, color="orange", linestyle="--", linewidth=0.8, label="VIF=5 (concerning)")
    ax.axvline(x=10, color="red", linestyle="--", linewidth=0.8, label="VIF=10 (severe)")
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("VIF per Feature (capped at 50 for display)")
    ax.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "vif_barplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def find_collinear_pairs(corr_matrix, features, threshold=0.7):
    """Extract all feature pairs with |correlation| > threshold."""
    pairs = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_1": features[i],
                    "feature_2": features[j],
                    "correlation": r,
                    "abs_correlation": abs(r),
                })
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return pairs_df


def print_summary(vif_df, pairs_df):
    """Print collinearity summary to console."""
    print("\n" + "=" * 70)
    print("  FEATURE COLLINEARITY ANALYSIS")
    print("=" * 70)

    # High-VIF features
    high_vif = vif_df[vif_df["VIF"] > 5].sort_values("VIF", ascending=False)
    print(f"\n--- High VIF Features (VIF > 5) --- [{len(high_vif)} features]")
    if high_vif.empty:
        print("  None — no severe collinearity detected by VIF.")
    else:
        print(f"  {'Feature':<35}{'VIF':<12}{'Severity'}")
        print("  " + "-" * 60)
        for _, row in high_vif.iterrows():
            severity = "SEVERE" if row["VIF"] > 10 else "concerning"
            vif_str = f"{row['VIF']:.1f}" if row["VIF"] < 1000 else f"{row['VIF']:.0f}"
            print(f"  {row['feature']:<35}{vif_str:<12}{severity}")

    # Collinear pairs
    print(f"\n--- Collinear Pairs (|r| > 0.7) --- [{len(pairs_df)} pairs]")
    if pairs_df.empty:
        print("  None — no highly correlated feature pairs.")
    else:
        print(f"  {'Feature 1':<30}{'Feature 2':<30}{'r':<10}")
        print("  " + "-" * 68)
        for _, row in pairs_df.iterrows():
            print(f"  {row['feature_1']:<30}{row['feature_2']:<30}{row['correlation']:<+10.3f}")

    # Sentiment-specific check
    sent_features = [f for f in vif_df["feature"] if "sent" in f.lower()]
    if sent_features:
        sent_vif = vif_df[vif_df["feature"].isin(sent_features)].sort_values("VIF", ascending=False)
        print(f"\n--- Sentiment Feature VIFs ---")
        for _, row in sent_vif.iterrows():
            vif_str = f"{row['VIF']:.1f}" if row["VIF"] < 1000 else f"{row['VIF']:.0f}"
            print(f"  {row['feature']:<35}{vif_str}")

        if not pairs_df.empty:
            sent_pairs = pairs_df[
                pairs_df["feature_1"].str.contains("sent", case=False)
                | pairs_df["feature_2"].str.contains("sent", case=False)
            ]
            if not sent_pairs.empty:
                print(f"\n--- Sentiment-Involved Collinear Pairs ---")
                for _, row in sent_pairs.iterrows():
                    print(f"  {row['feature_1']:<30}{row['feature_2']:<30}{row['correlation']:<+10.3f}")

    print("\n" + "=" * 70)
    print(f"Plots saved to: {OUTPUT_DIR}/")
    print()


def main():
    logger.info("=" * 50)
    logger.info("Feature Collinearity Analysis")
    logger.info("=" * 50)

    # 1. Load data and build features
    model, train_df = load_and_prepare_data()
    features = model.features
    X_df = train_df[features].copy()

    # Drop any columns that are constant (zero variance — can't correlate)
    const_cols = X_df.columns[X_df.std() == 0].tolist()
    if const_cols:
        logger.warning(f"Dropping {len(const_cols)} constant features: {const_cols}")
        X_df = X_df.drop(columns=const_cols)
        features = list(X_df.columns)

    logger.info(f"Analyzing {len(features)} features across {len(X_df)} rows")

    # 2. Correlation matrix
    logger.info("Computing Pearson correlation matrix...")
    corr_matrix = X_df.corr().values

    # Replace NaNs (from constant-ish columns) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 3. Hierarchical clustering (distance = 1 - |r|)
    logger.info("Computing hierarchical clustering...")
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    # Ensure symmetry and valid distances
    dist_matrix = np.clip(dist_matrix, 0, 1)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method="average")

    # 4. Generate plots
    logger.info("Generating plots...")
    plot_clustered_heatmap(corr_matrix, features, linkage_matrix)
    plot_dendrogram(linkage_matrix, features)

    # 5. Compute VIF
    logger.info("Computing Variance Inflation Factors...")
    vif_df = compute_vif(X_df)
    plot_vif(vif_df)

    # 6. Find collinear pairs
    pairs_df = find_collinear_pairs(corr_matrix, features, threshold=0.7)

    # 7. Print summary
    print_summary(vif_df, pairs_df)

    logger.info("Collinearity analysis complete.")


if __name__ == "__main__":
    main()
