import logging
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# FinBERT model configuration
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512  # Max tokens for BERT models


class SentimentAnalyzer:
    """
    Converts raw Reddit posts into numerical sentiment scores per symbol.

    Supports two analysis modes:
    - 'finbert': Uses ProsusAI/finbert transformer model (recommended for financial text)
    - 'vader': Uses VADER lexicon-based analyzer (faster, but less accurate for finance)

    FinBERT outputs probabilities for: positive, negative, neutral
    These are converted to a compound score in [-1, 1] for compatibility.
    """

    def __init__(
        self,
        symbols: List[str],
        model: str = "finbert",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            model: Sentiment model to use ('finbert' or 'vader')
            device: Device for FinBERT ('cuda', 'mps', 'cpu', or None for auto-detect)
            batch_size: Batch size for FinBERT inference
        """
        self.symbols = symbols
        self.model_type = model.lower()
        self.batch_size = batch_size
        self.keywords = self._build_keywords()

        # Initialize VADER (always available as fallback)
        self.vader = SentimentIntensityAnalyzer()

        # Initialize FinBERT if requested
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.device = None

        if self.model_type == "finbert":
            self._init_finbert(device)

    def _init_finbert(self, device: Optional[str] = None) -> None:
        """
        Initialize the FinBERT model and tokenizer.

        Args:
            device: Target device ('cuda', 'mps', 'cpu', or None for auto)
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Auto-detect device if not specified
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            self.device = device
            logger.info(f"Loading FinBERT model on {device}...")

            self.finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                FINBERT_MODEL
            )
            self.finbert_model.to(device)
            self.finbert_model.eval()

            logger.info("FinBERT model loaded successfully")

        except ImportError as e:
            logger.warning(
                f"Could not import torch/transformers: {e}. Falling back to VADER."
            )
            self.model_type = "vader"
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Falling back to VADER.")
            self.model_type = "vader"

    def _finbert_score(self, texts: List[str]) -> List[float]:
        """
        Get sentiment scores using FinBERT.

        Converts FinBERT's 3-class output (positive, negative, neutral) to a
        compound score in [-1, 1] for compatibility with VADER format.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of compound sentiment scores in [-1, 1]
        """
        import torch

        if not texts:
            return []

        scores = []

        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            # Tokenize
            inputs = self.finbert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            # Convert to compound score: positive - negative
            # (neutral is implicitly 0 contribution)
            for prob in probs:
                positive = prob[0].item()
                negative = prob[1].item()
                # Compound score: positive contribution minus negative contribution
                compound = positive - negative
                scores.append(compound)

        return scores

    def _vader_score(self, texts: List[str]) -> List[float]:
        """
        Get sentiment scores using VADER.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of compound sentiment scores in [-1, 1]
        """
        return [self.vader.polarity_scores(t)["compound"] for t in texts]

    def get_scores(self, texts: List[str]) -> List[float]:
        """
        Get sentiment scores for a list of texts.

        Uses the configured model (FinBERT or VADER).

        Args:
            texts: List of text strings to analyze

        Returns:
            List of compound sentiment scores in [-1, 1]
        """
        if self.model_type == "finbert" and self.finbert_model is not None:
            return self._finbert_score(texts)
        else:
            return self._vader_score(texts)

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
        Analyze sentiment of Reddit posts and aggregate by symbol and hour.

        Input: DataFrame with ['title', 'selftext', 'created_utc', 'num_comments']
        Output: DataFrame indexed by time/symbol with sentiment scores.

        The model used (FinBERT or VADER) depends on the `model` parameter
        passed during initialization.
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 1. Combine text (keep original case for FinBERT, lowercase for pattern matching)
        df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
        df['text_lower'] = df['text'].str.lower()

        # 2. Score Sentiment using configured model
        # Apply only to unique text to save computation
        unique_texts = df['text'].unique().tolist()
        logger.info(f"Scoring {len(unique_texts)} unique texts with {self.model_type}")

        scores_list = self.get_scores(unique_texts)
        scores_map = dict(zip(unique_texts, scores_list))
        df['score'] = df['text'].map(scores_map)

        # 3. Explode by Symbol (match keywords in lowercase text)
        results = []
        for symbol, keywords in self.keywords.items():
            if not keywords:
                continue

            # Use non-capturing group (?:...) to silence UserWarning
            pattern = r'(?<!\w)(?:' + '|'.join(keywords) + r')(?!\w)'

            # Find posts containing keywords
            mask = df['text_lower'].str.contains(pattern, regex=True)
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
            'score': ['mean', 'count'],  # Mean sentiment, Volume of posts
            'num_comments': 'sum'        # Total engagement
        })

        # Flatten columns
        grouped.columns = ['sentiment_mean', 'post_volume', 'comment_volume']

        # Rename 'created_utc' to 'timestamp' to match Market Data
        return grouped.reset_index().rename(columns={'created_utc': 'timestamp'})