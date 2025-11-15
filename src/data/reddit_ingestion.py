"""Reddit data collection utilities."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .arctic_shift import ArcticShiftClient

logger = logging.getLogger(__name__)


class RedditSentimentCollector:
    """Collect Reddit submissions and enrich them with sentiment scores."""

    PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission"

    def __init__(self, config) -> None:
        self.config = config
        self.analyzer = SentimentIntensityAnalyzer()
        self.reddit = self._init_reddit_client()
        self.arctic_shift = ArcticShiftClient(
            api_key=getattr(config, "arctic_shift_api_key", None),
            base_url=getattr(config, "arctic_shift_base_url", "https://api.arcticshift.com/v1"),
        )

    # ------------------------------------------------------------------
    def _init_reddit_client(self):
        try:
            import praw
        except ImportError:  # pragma: no cover - optional dependency path
            logger.warning("praw not installed; Reddit API collection disabled")
            return None

        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "crypto-sentiment-trader")

        if not (client_id and client_secret):
            logger.warning("Missing Reddit credentials; falling back to Pushshift/ArcticShift only")
            return None

        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )

    # ------------------------------------------------------------------
    def fetch_posts(self, lookback_hours: int = 48) -> pd.DataFrame:
        """Fetch recent Reddit posts via the official API (if available)."""
        if self.reddit is None:
            return pd.DataFrame()

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        submissions: List[dict] = []

        for subreddit in self.config.subreddits:
            logger.info("Fetching Reddit API posts for r/%s", subreddit)
            try:
                subreddit_api = self.reddit.subreddit(subreddit)
                for submission in subreddit_api.submissions(start.timestamp(), end.timestamp()):
                    record = self._submission_to_record(submission)
                    submissions.append(record)
                    if len(submissions) >= self.config.reddit_limit_per_sub * len(self.config.subreddits):
                        break
            except Exception as exc:  # pragma: no cover - API failure
                logger.error("Failed to fetch via Reddit API for r/%s: %s", subreddit, exc)

        df = pd.DataFrame(submissions)
        return self._post_process(df)

    # ------------------------------------------------------------------
    def fetch_pushshift(self, lookback_days: int) -> pd.DataFrame:
        """Fetch historical submissions using the Pushshift archive."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        params_template = {
            "sort": "desc",
            "sort_type": "created_utc",
            "size": 500,
        }

        records: List[dict] = []
        for subreddit in self.config.subreddits:
            params = params_template.copy()
            params.update({"subreddit": subreddit, "after": int(start.timestamp()), "before": int(end.timestamp())})
            logger.info("Fetching Pushshift posts for r/%s", subreddit)

            next_after = params["after"]
            while True:
                params["after"] = next_after
                try:
                    response = requests.get(self.PUSHSHIFT_URL, params=params, timeout=30)
                    response.raise_for_status()
                except requests.RequestException as exc:  # pragma: no cover - network failure
                    logger.error("Pushshift request failed: %s", exc)
                    break

                payload = response.json()
                data = payload.get("data", [])
                if not data:
                    break

                for item in data:
                    item["subreddit"] = subreddit
                records.extend(data)

                # Pushshift returns newest first; track the oldest timestamp to paginate
                next_after = data[-1]["created_utc"]
                if next_after >= params["before"]:
                    break

                if len(records) >= self.config.market_data_limit * len(self.config.subreddits):
                    break

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        if "title" not in df.columns:
            df["title"] = ""
        if "selftext" not in df.columns:
            df["selftext"] = ""
        return self._post_process(df)

    # ------------------------------------------------------------------
    def fetch_hybrid_data(self, lookback_hours: int, historical_days: int) -> pd.DataFrame:
        """Combine Arctic Shift, Pushshift and Reddit API data for a continuous history."""
        historical_df = self.fetch_arctic_shift_backfill(historical_days)
        if historical_df.empty:
            historical_df = self.fetch_pushshift(historical_days)

        recent_df = self.fetch_posts(lookback_hours)

        combined = pd.concat([historical_df, recent_df], ignore_index=True)
        return self._post_process(combined)

    # ------------------------------------------------------------------
    def fetch_arctic_shift_backfill(self, lookback_days: int) -> pd.DataFrame:
        """Use Arctic Shift to fetch a longer history matching the Reddit scraper subreddits."""
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        end = datetime.now(timezone.utc)
        df = self.arctic_shift.fetch_subreddit_posts(self.config.subreddits, start, end)
        if df.empty:
            return df

        rename_map = {
            "body": "selftext",
            "score": "score",
            "num_comments": "num_comments",
        }
        for src, dst in list(rename_map.items()):
            if src in df.columns and src != dst:
                df[dst] = df[src]
        if "title" not in df.columns:
            df["title"] = df.get("headline", "")
        if "selftext" not in df.columns:
            df["selftext"] = df.get("body", "")
        return self._post_process(df)

    # ------------------------------------------------------------------
    def _submission_to_record(self, submission) -> dict:
        data = {
            "id": submission.id,
            "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            "subreddit": str(submission.subreddit),
            "title": submission.title,
            "selftext": submission.selftext or "",
            "score": submission.score,
            "num_comments": submission.num_comments,
            "upvote_ratio": getattr(submission, "upvote_ratio", None),
            "author": str(submission.author) if submission.author else None,
            "url": submission.url,
        }
        return data

    # ------------------------------------------------------------------
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        if "created_utc" not in df.columns:
            raise KeyError("Expected 'created_utc' column in Reddit data")

        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["created_utc"]).reset_index(drop=True)

        df["title"] = df.get("title", "").fillna("")
        df["selftext"] = df.get("selftext", "").fillna("")
        df["full_text"] = (df["title"].astype(str) + " " + df["selftext"].astype(str)).str.strip()
        df["score"] = df.get("score", 0).fillna(0).astype(float)
        df["num_comments"] = df.get("num_comments", 0).fillna(0).astype(float)
        df["author"] = df.get("author", None)

        # Apply filters
        min_length = getattr(self.config, "reddit_min_length", 0)
        if min_length:
            df = df[df["full_text"].str.len() >= min_length]
        min_score = getattr(self.config, "reddit_min_score", 0)
        if min_score:
            df = df[df["score"] >= min_score]

        if df.empty:
            return df

        sentiment = df["full_text"].apply(self.analyzer.polarity_scores)
        sentiment_df = pd.DataFrame(list(sentiment))
        df = pd.concat([df, sentiment_df], axis=1)
        df.rename(columns={
            "compound": "vader_compound",
            "pos": "vader_pos",
            "neg": "vader_neg",
            "neu": "vader_neu",
        }, inplace=True)
        df["sentiment_positive"] = (df["vader_compound"] > 0.05).astype(float)
        df["sentiment_negative"] = (df["vader_compound"] < -0.05).astype(float)

        df = df.sort_values("created_utc").drop_duplicates(subset=["id"], keep="last")
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    def save_data(self, df: pd.DataFrame) -> Path:
        if df.empty:
            logger.warning("No Reddit data to save")
            raise ValueError("Cannot save empty Reddit dataframe")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.raw_reddit_dir / f"reddit_{timestamp}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved Reddit data to %s", path)
        return path


__all__ = ["RedditSentimentCollector"]
