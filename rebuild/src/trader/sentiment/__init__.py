"""Deferred Reddit sentiment experiment tools.

The market-only baseline remains the default production research path. Modules
in this package are for isolated Phase 10 ablation experiments only.
"""

from trader.sentiment.collector import (
    CommentCollectionStatus,
    RedditCommentRecord,
    RedditSubmissionRecord,
    SentimentCollectionConfig,
    collect_reddit_records,
)
from trader.sentiment.experiments import SentimentExperimentResult, run_sentiment_ablation
from trader.sentiment.features import (
    SENTIMENT_VARIANT_ORDER,
    build_hourly_sentiment_features,
    sentiment_feature_columns,
)
from trader.sentiment.scoring import VaderSentimentScorer, score_reddit_records
from trader.sentiment.storage import (
    read_hourly_sentiment_dataset,
    read_sentiment_dataset,
    write_hourly_sentiment_dataset,
    write_sentiment_dataset,
)

__all__ = [
    "CommentCollectionStatus",
    "RedditCommentRecord",
    "RedditSubmissionRecord",
    "SENTIMENT_VARIANT_ORDER",
    "SentimentCollectionConfig",
    "SentimentExperimentResult",
    "VaderSentimentScorer",
    "build_hourly_sentiment_features",
    "collect_reddit_records",
    "read_sentiment_dataset",
    "read_hourly_sentiment_dataset",
    "run_sentiment_ablation",
    "score_reddit_records",
    "sentiment_feature_columns",
    "write_hourly_sentiment_dataset",
    "write_sentiment_dataset",
]
