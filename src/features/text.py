# src/features/text.py
"""
Goal: Turn raw Reddit posts (with VADER scores) into per-interval sentiment features
for each symbol (e.g., BTCUSDT, ETHUSDT). We will write one CSV per interval like:
  data/processed/sentiment_1h.csv.gz

You will:
- read configs/data.yaml for `symbols` and `intervals`
- define a keyword map per symbol (e.g., BTC -> ["btc","bitcoin"])
- map each post to one or more symbols based on keyword hits
- resample to fixed time bars (UTC) and aggregate sentiment + counts
"""

import re
import sys
import glob
import yaml
import math
import argparse
from pathlib import Path
from typing import Dict, List, Pattern

import numpy as np
import pandas as pd

CONFIG_PATH = "configs/data.yaml"
RAW_REDDIT_DIR = Path("data/raw/reddit")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1) SYMBOL KEYWORDS (EDIT THIS) ------------------------------------
# TODO: Expand this dictionary. Keys should be your Binance symbols from config.
# Values are a list of lowercase keywords that can identify the coin in text.
# Include both the name and ticker, and optionally common variants.
SYMBOL_KEYWORDS: Dict[str, List[str]] = {
    "BTCUSDT": ["btc", "bitcoin"],
    "ETHUSDT": ["eth", "ethereum"],
    "SOLUSDT": ["sol", "solana"],
    "BNBUSDT": ["bnb", "binance coin", "binance"],
    "XRPUSDT": ["xrp", "ripple"],
    "ADAUSDT": ["ada", "cardano"],
    "AVAXUSDT": ["avax", "avalanche"],
    "DOGEUSDT": ["doge", "dogecoin"],
    "LINKUSDT": ["link", "chainlink"],
    "MATICUSDT": ["matic", "polygon"],
    "DOTUSDT": ["dot", "polkadot"],
    "ATOMUSDT": ["atom", "cosmos"]
}

# ---------- 2) REGEX HELPERS ---------------------------------------------------
def make_patterns(keyword_map: Dict[str, List[str]]) -> Dict[str, Pattern]:
    """
    Build compiled regex per symbol to match keywords in text.

    We want to match:
      - case-insensitive words like 'bitcoin', 'ethereum'
      - cashtags like $BTC or $ETH
    Using a pattern that:
      - allows optional leading '$'
      - uses word boundaries so 'eth' in 'the' is not a match
    """
    patterns = {}
    for symbol, kws in keyword_map.items():
        # Escape keywords and join them with '|'
        # Example: r'(?<!\w)\$?(btc|bitcoin)(?!\w)'
        kw_group = "|".join([re.escape(k) for k in kws])
        pat = rf"(?<!\w)\$?(?:{kw_group})(?!\w)"
        patterns[symbol] = re.compile(pat, flags=re.IGNORECASE)
    return patterns


# ---------- 3) LOADING RAW REDDIT ---------------------------------------------
def load_all_reddit() -> pd.DataFrame:
    import glob, pandas as pd, numpy as np

    csvs = sorted(glob.glob("data/raw/reddit/reddit_*.csv.gz"))
    pars = sorted(glob.glob("data/raw/reddit/reddit_*.parquet"))
    files = csvs + pars
    if not files:
        raise FileNotFoundError("No reddit files found in data/raw/reddit/. Run ingest first.")

    dfs = []
    for f in files:
        if f.endswith(".parquet"):
            # Requires pyarrow or fastparquet
            try:
                df = pd.read_parquet(f)   # engine auto-detects
            except Exception as e:
                raise RuntimeError(
                    f"Reading {f} failed. Install a parquet engine, e.g.: "
                    "pip install --only-binary=:all: 'pyarrow>=15,<18'"
                ) from e
        else:
            df = pd.read_csv(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Ensure UTC timestamps
    if "created_utc" not in out.columns:
        raise KeyError("Expected 'created_utc' column in reddit data.")
    out["created_utc"] = pd.to_datetime(out["created_utc"], utc=True, errors="coerce")

    out = out.dropna(subset=["created_utc"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["id"]).reset_index(drop=True)
    out["text"] = (out["title"].fillna("") + " " + out["selftext"].fillna("")).astype(str)
    return out



# ---------- 4) MAP POSTS -> SYMBOLS -------------------------------------------
def tag_symbols(df: pd.DataFrame, patterns: Dict[str, Pattern]) -> pd.DataFrame:
    """
    For each post row, check which symbol patterns match the text.
    If a post mentions multiple symbols, we 'explode' to multiple rows,
    one per (post, symbol). If none match, we drop the post.

    Returns a dataframe with an added 'symbol' column and possibly more rows.
    """
    # For speed, pre-lower text once (regex is case-insensitive anyway)
    text_series = df["text"].astype(str)

    # Build list of lists: for each row, which symbols matched?
    matches_per_row = []
    for text in text_series:
        syms = [sym for sym, pat in patterns.items() if pat.search(text)]
        matches_per_row.append(syms)

    tmp = df.copy()
    tmp["symbols_matched"] = matches_per_row
    # Keep only rows that matched at least one symbol
    tmp = tmp[tmp["symbols_matched"].map(len) > 0]

    # Explode to one row per symbol
    tmp = tmp.explode("symbols_matched").rename(columns={"symbols_matched": "symbol"})
    tmp = tmp.reset_index(drop=True)
    return tmp


# ---------- 5) QUALITY FILTERS (OPTIONAL; YOU CAN TWEAK) ----------------------
def apply_quality_filters(df: pd.DataFrame, min_score: int = 10) -> pd.DataFrame:
    """
    Filter out very low-quality/noisy posts. Start simple:
      - keep posts with score >= min_score (Reddit 'score' is upvotes-downvotes)
      - drop deleted authors if you want (optional)
    """
    out = df.copy()
    if "score" in out.columns and min_score is not None:
        out = out[out["score"].fillna(0) >= min_score]
    # Example optional filter:
    if "author" in out.columns:
        out = out[out["author"].notna()]

        bad_authors = {"[deleted]", "[removed]", "automoderator"}
        out = out[~out["author"].astype(str).str.lower().isin(bad_authors)]
    return out


# ---------- 6) AGGREGATION PER INTERVAL ---------------------------------------
AGG_DEFAULTS = {
    # counts
    "post_count": ("id", "count"),
    "unique_authors": ("author", pd.Series.nunique),
    "avg_score": ("score", "mean"),
    # sentiment means
    "vader_compound_mean": ("vader_compound", "mean"),
    "vader_pos_mean": ("vader_pos", "mean"),
    "vader_neg_mean": ("vader_neg", "mean"),
    "vader_neu_mean": ("vader_neu", "mean"),
    # sentiment shares (we compute below then mean)
    "share_pos": ("is_pos", "mean"),
    "share_neg": ("is_neg", "mean"),
}

def add_sentiment_flags(df: pd.DataFrame,
                        pos_thr: float = 0.05,
                        neg_thr: float = -0.05) -> pd.DataFrame:
    """
    Create boolean flags for positive/negative using VADER compound thresholds.
    You can tune thresholds later.
    """
    out = df.copy()
    out["is_pos"] = out["vader_compound"] > pos_thr
    out["is_neg"] = out["vader_compound"] < neg_thr
    return out

def resample_aggregate(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample posts to fixed UTC bars for each symbol, aggregating VADER features.
    interval: "1h" or "4h" etc. We map to pandas offsets "1H"/"4H".
    """
    # Map "1h" -> "1H"
    rule = interval.upper()
    if not rule.endswith("H"):
        # You can add more mappings if you later use 15m, 1d, etc.
        # e.g., "15m" -> "15T", "1d" -> "1D"
        if interval.endswith("m"):
            rule = interval[:-1] + "T"  # minutes
        elif interval.endswith("d"):
            rule = interval[:-1] + "D"  # days

    # Ensure datetime index for resampling
    g = df.set_index("created_utc")

    # We group by symbol first, then resample each group
    pieces = []
    for sym, grp in g.groupby("symbol"):
        # Resample at fixed UTC boundaries; origin='start_day' gives deterministic bins
        # You can also try label='left', closed='left' if you need exact semantics.
        agg = (
            grp.resample(rule, origin="start_day")
               .agg({
                    "id": "count",
                    "author": pd.Series.nunique,
                    "score": "mean",
                    "vader_compound": "mean",
                    "vader_pos": "mean",
                    "vader_neg": "mean",
                    "vader_neu": "mean",
                    "is_pos": "mean",
                    "is_neg": "mean",
                })
               .rename(columns={
                    "id": "post_count",
                    "author": "unique_authors",
                    "score": "avg_score",
                    "vader_compound": "vader_compound_mean",
                    "vader_pos": "vader_pos_mean",
                    "vader_neg": "vader_neg_mean",
                    "vader_neu": "vader_neu_mean",
                    "is_pos": "share_pos",
                    "is_neg": "share_neg",
               })
        )
        agg["symbol"] = sym
        agg = agg.reset_index().rename(columns={"created_utc": "timestamp"})
        agg["interval"] = interval
        pieces.append(agg)

    out = pd.concat(pieces, ignore_index=True)
    # Optional: drop entirely empty rows (no posts in that bar)
    out = out.dropna(subset=["post_count"])
    return out


# ---------- 7) MAIN PIPELINE ---------------------------------------------------
def build_sentiment_features(interval: str,
                             min_score: int = 0) -> pd.DataFrame:
    """
    High-level function:
      - load raw reddit
      - tag posts with symbols via regex
      - apply quality filters
      - add pos/neg flags
      - resample+aggregate to the chosen interval
    """
    # Load configs (to get the symbols list)
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    symbols_cfg: List[str] = cfg.get("symbols", [])

    # Sanity check: ensure your keyword map covers your configured symbols
    missing = [s for s in symbols_cfg if s not in SYMBOL_KEYWORDS]
    if missing:
        print(f"[WARN] Missing keywords for symbols: {missing}. "
              "They will not get any matched posts until you add them.")
    # Create regex patterns
    pats = make_patterns(SYMBOL_KEYWORDS)

    # Load raw posts
    posts = load_all_reddit()

    # Map posts -> symbols (explode to multiple rows as needed)
    tagged = tag_symbols(posts, pats)

    # Keep only symbols we actually care about from config
    tagged = tagged[tagged["symbol"].isin(symbols_cfg)].copy()

    # Basic quality filter
    tagged = apply_quality_filters(tagged, min_score=min_score)

    # Add boolean flags for sentiment
    tagged = add_sentiment_flags(tagged)

    # Aggregate to interval
    features = resample_aggregate(tagged, interval=interval)

    return features


def save_features(df: pd.DataFrame, interval: str) -> Path:
    """
    Save the features to data/processed/sentiment_{interval}.csv.gz
    """
    out_path = OUT_DIR / f"sentiment_{interval}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Aggregate Reddit sentiment to fixed bars.")
    parser.add_argument("--interval", type=str, required=True,
                        help="Bar size, e.g., 1h or 4h (must match your configs).")
    parser.add_argument("--min_score", type=int, default=0,
                        help="Minimum Reddit post score to include (default 0).")
    args = parser.parse_args()

    df = build_sentiment_features(interval=args.interval, min_score=args.min_score)
    out = save_features(df, interval=args.interval)
    print(f"Saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
