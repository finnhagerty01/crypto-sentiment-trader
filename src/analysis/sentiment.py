import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List

class SentimentAnalyzer:
    """
    Converts raw Reddit posts into numerical sentiment scores per symbol.
    """
    def __init__(self, symbols: List[str]):
        self.analyzer = SentimentIntensityAnalyzer()
        self.symbols = symbols
        # Simple mapping: BTCUSDT -> ['btc', 'bitcoin']
        self.keywords = self._build_keywords()

    def _build_keywords(self) -> Dict[str, List[str]]:
        # You can expand this or load from config
        base_map = {
            "BTC": ["btc", "bitcoin"],
            "ETH": ["eth", "ethereum"],
            "SOL": ["sol", "solana"],
            "DOGE": ["doge", "dogecoin"],
            "ADA": ["ada", "cardano"],
            "XRP": ["xrp", "ripple"],
            "BNB": ["bnb", "binance coin"],
            "AVAX": ["avax", "avalanche"],
            "DOT": ["dot", "polkadot"],
            "LINK": ["link", "chainlink"],
            "MATIC": ["matic", "polygon"],
            "ATOM": ["atom", "cosmos"]
        }
        # Map back to your Binance symbols (e.g. BTCUSDT)
        return {s: base_map.get(s.replace("USDT", ""), []) for s in self.symbols}

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame with ['title', 'selftext', 'created_utc']
        Output: DataFrame indexed by time/symbol with sentiment scores.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        
        # 1. Combine text
        df['text'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.lower()
        
        # 2. Score Sentiment (VADER)
        # Apply only to unique text to save time
        unique_text = df['text'].unique()
        scores = {t: self.analyzer.polarity_scores(t)['compound'] for t in unique_text}
        df['score'] = df['text'].map(scores)

        # 3. Explode by Symbol
        results = []
        for symbol, keywords in self.keywords.items():
            if not keywords: continue
            
            # FIX 1: Use non-capturing group (?:...) to silence UserWarning
            pattern = r'(?<!\w)(?:' + '|'.join(keywords) + r')(?!\w)'
            
            # Find posts containing keywords
            mask = df['text'].str.contains(pattern, regex=True)
            subset = df[mask].copy()
            subset['symbol'] = symbol
            results.append(subset[['created_utc', 'symbol', 'score', 'num_comments']])

        if not results:
            return pd.DataFrame()

        final_df = pd.concat(results)
        
        # 4. Resample to Hourly Sentiment
        grouped = final_df.set_index('created_utc').groupby(
            ['symbol', pd.Grouper(freq='1h')]
        ).agg({
            'score': ['mean', 'count'], # Mean sentiment, Volume of posts
            'num_comments': 'sum'       # Total engagement
        })
        
        # Flatten columns
        grouped.columns = ['sentiment_mean', 'post_volume', 'comment_volume']
        
        # FIX 2: Rename 'created_utc' to 'timestamp' to match Market Data
        return grouped.reset_index().rename(columns={'created_utc': 'timestamp'})