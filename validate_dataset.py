import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.analysis.models import TradingModel
from sklearn.model_selection import ParameterGrid
import numpy as np
# --- SETUP ---
sys.path.append(str(Path(__file__).parent))
from src.analysis.sentiment import SentimentAnalyzer
from src.data.market_client import MarketClient
from src.utils.config import TradingConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FineTuneSweep")

def run_fine_tune(csv_filename="master_reddit_backup.csv"):
    
    # 1. LOAD DATA
    possible_paths = [Path(csv_filename), Path("..") / csv_filename, Path("data") / "master_reddit.csv"]
    data_path = next((p for p in possible_paths if p.exists()), None)
    if not data_path: return
    
    reddit_df = pd.read_csv(data_path)
    reddit_df['created_utc'] = pd.to_datetime(reddit_df['created_utc'])
    
    # 2. PREPARE BASE DATA
    config = TradingConfig.from_yaml("configs/data.yaml")
    analyzer = SentimentAnalyzer(config.symbols)
    sentiment_df = analyzer.analyze(reddit_df)
    market_client = MarketClient(config)
    market_df = market_client.fetch_ohlcv(lookback_days=45) 

    #3. TRAIN MODEL
    model = TradingModel()

    # Build unified training DF using the exact same code path as live training
    df = model.prepare_features(market_df, sentiment_df, is_inference=False)
    if df.empty:
        print("Feature prep returned empty dataframe. Check timestamp alignment / market coverage.")
        return

    feature_cols = model.features

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    # ----------------------------
    # Hyperparameter Search (Walk-forward)
    # ----------------------------

    def eval_strategy_metrics(df_slice: pd.DataFrame, preds: np.ndarray):
        """Compute trading-style metrics on a slice with columns: target_return, timestamp."""
        out = df_slice.copy()
        out["pred"] = preds

        trades = out[out["pred"] != 0]
        n_trades = len(trades)

        if n_trades == 0:
            return {
                "trades": 0,
                "win_rate": np.nan,
                "total_return": 0.0,
                "max_dd": 0.0
            }

        wins = trades[
            ((trades["pred"] == 1) & (trades["target_return"] > 0)) |
            ((trades["pred"] == -1) & (trades["target_return"] < 0))
        ]
        win_rate = len(wins) / n_trades

        out["strategy_return"] = out["pred"] * out["target_return"]
        total_return = float(out["strategy_return"].sum())

        # Drawdown from cumulative strategy returns
        cum = (1.0 + out["strategy_return"].fillna(0.0)).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = (cum / peak) - 1.0
        max_dd = float(dd.min())

        return {
            "trades": n_trades,
            "win_rate": float(win_rate),
            "total_return": total_return,
            "max_dd": max_dd
        }


    def walk_forward_splits(df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.50):
        """
        Yields (train_idx, val_idx) for walk-forward validation.
        df must be time-sorted.
        """
        n = len(df)
        min_train = int(n * min_train_frac)
        fold_size = (n - min_train) // n_folds
        if fold_size <= 0:
            raise ValueError("Not enough data for walk-forward splits. Increase lookback_days or reduce n_folds.")

        for k in range(n_folds):
            train_end = min_train + k * fold_size
            val_end = train_end + fold_size
            if val_end > n:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx


    # Ensure time-sorted (critical for walk-forward)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    X = df[feature_cols]
    y = df["target"].astype(int)

    param_grid = {
        # core capacity knobs
        "n_estimators": [100, 200, 400],
        "max_depth": [1, 2, 3, 4, 6],
        # regularization knobs
        "min_samples_leaf": [1, 5, 10, 25],
        "min_samples_split": [2, 10, 25],
        # feature subsampling
        "max_features": ["sqrt", "log2", 0.5],
        # bootstrap can help variance
        "bootstrap": [True],
    }

    print("\n" + "=" * 110)
    print("WALK-FORWARD HYPERPARAM SEARCH")
    print(f"Rows={len(df)} | Features={len(feature_cols)} | Grid size={len(list(ParameterGrid(param_grid)))}")
    print("=" * 110)
    print(f"{'n_est':>6} {'depth':>6} {'leaf':>6} {'split':>6} {'maxfeat':>8} | "
        f"{'avg_trades':>10} {'avg_win':>9} {'avg_ret%':>9} {'avg_dd%':>9} | {'score':>9}")
    print("-" * 110)

    best = None
    results = []

    for params in ParameterGrid(param_grid):
        fold_metrics = []

        for train_idx, val_idx in walk_forward_splits(df, n_folds=5, min_train_frac=0.50):
            clf = RandomForestClassifier(
                **params,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )

            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict(X.iloc[val_idx])

            m = eval_strategy_metrics(df.iloc[val_idx], preds)
            fold_metrics.append(m)

        avg_trades = np.mean([m["trades"] for m in fold_metrics])
        avg_win = np.nanmean([m["win_rate"] for m in fold_metrics])
        avg_ret = np.mean([m["total_return"] for m in fold_metrics])
        avg_dd = np.mean([m["max_dd"] for m in fold_metrics])

        # Score: reward return, penalize drawdown, lightly penalize "no trades"
        # (tune weights if you want)
        score = avg_ret - 0.5 * abs(avg_dd) - 0.0001 * (1.0 / (avg_trades + 1e-9))

        row = {
            **params,
            "avg_trades": float(avg_trades),
            "avg_win": float(avg_win) if not np.isnan(avg_win) else np.nan,
            "avg_ret": float(avg_ret),
            "avg_dd": float(avg_dd),
            "score": float(score)
        }
        results.append(row)

        print(f"{params['n_estimators']:>6} {params['max_depth']:>6} "
            f"{params['min_samples_leaf']:>6} {params['min_samples_split']:>6} {str(params['max_features']):>8} | "
            f"{avg_trades:>10.1f} {avg_win:>9.2%} {avg_ret*100:>8.2f}% {avg_dd*100:>8.2f}% | {score:>9.6f}")

        if best is None or row["score"] > best["score"]:
            best = row

    print("\nBEST PARAMS (by walk-forward score):")
    print(best)

    # Train final model on ALL data using best params (for later saving / inspection)
    final_clf = RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_leaf=best["min_samples_leaf"],
        min_samples_split=best["min_samples_split"],
        max_features=best["max_features"],
        bootstrap=best["bootstrap"],
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    final_clf.fit(X, y)

    # Optional: quick “last fold” diagnostic on the most recent segment
    last_train_idx, last_val_idx = list(walk_forward_splits(df, n_folds=5, min_train_frac=0.50))[-1]
    preds_last = final_clf.predict(X.iloc[last_val_idx])
    m_last = eval_strategy_metrics(df.iloc[last_val_idx], preds_last)

    print("\nMost recent fold performance (sanity check):")
    print(m_last)


if __name__ == "__main__":
    run_fine_tune("master_reddit_backup.csv")