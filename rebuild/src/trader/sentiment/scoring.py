"""Sentiment scoring that keeps engagement separate from sentiment values."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class SentimentScorer(Protocol):
    def score(self, text: str) -> float:
        ...


class VaderSentimentScorer:
    """Thin wrapper around optional VADER sentiment scoring."""

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError(
                "VADER sentiment scoring requires the rebuild sentiment extra: "
                "uv sync --extra sentiment"
            ) from exc
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return float(self._analyzer.polarity_scores(text)["compound"])


class LexiconSentimentScorer:
    """Small deterministic scorer for offline tests and fixtures."""

    _positive = frozenset({"bull", "bullish", "breakout", "gain", "gains", "good", "moon", "rally"})
    _negative = frozenset({"bear", "bearish", "bad", "crash", "dump", "loss", "losses", "risk"})

    def score(self, text: str) -> float:
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()]
        positive = sum(1 for word in words if word in self._positive)
        negative = sum(1 for word in words if word in self._negative)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total


def score_reddit_records(
    submissions: pd.DataFrame,
    comments: pd.DataFrame,
    scorer: SentimentScorer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add sentiment columns without mutating Reddit engagement scores."""

    scored_submissions = submissions.copy()
    scored_comments = comments.copy()
    scored_submissions["sentiment_score"] = [
        scorer.score(_submission_text(row))
        for row in scored_submissions.to_dict(orient="records")
    ]
    scored_comments["sentiment_score"] = [
        scorer.score(str(row.get("body", "")))
        for row in scored_comments.to_dict(orient="records")
    ]
    return scored_submissions, scored_comments


def _submission_text(row: dict[str, object]) -> str:
    title = str(row.get("title", "") or "")
    selftext = str(row.get("selftext", "") or "")
    return f"{title}\n{selftext}".strip()
