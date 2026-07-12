"""Causal hourly sentiment feature construction."""

from __future__ import annotations

import pandas as pd


SENTIMENT_VARIANT_ORDER = (
    "hourly_mean",
    "counts_missingness",
    "reliability_shrunk",
    "ewma_6h",
    "ewma_24h",
    "ewma_fast_minus_slow",
    "sentiment_lag_1h",
    "kalman_filtered",
)


def build_hourly_sentiment_features(
    submissions: pd.DataFrame,
    comments: pd.DataFrame,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    kalman_process_variance: float = 0.01,
    kalman_measurement_variance: float = 0.25,
) -> pd.DataFrame:
    """Aggregate raw scored Reddit records into causal hourly features."""

    scored_submissions = _require_sentiment(submissions, "submissions")
    scored_comments = _require_sentiment(comments, "comments")
    records = []
    if not scored_submissions.empty:
        submission_records = scored_submissions.assign(
            hour=_hour(scored_submissions["created_utc"]),
            record_type="submission",
        )
        records.append(submission_records)
    if not scored_comments.empty:
        comment_records = scored_comments.assign(
            hour=_hour(scored_comments["created_utc"]),
            record_type="comment",
        )
        records.append(comment_records)

    if records:
        all_records = pd.concat(records, ignore_index=True, sort=False)
        observed_start = all_records["hour"].min()
        observed_end = all_records["hour"].max()
    else:
        all_records = pd.DataFrame(columns=["hour", "record_type", "sentiment_score", "subreddit"])
        observed_start = observed_end = None

    range_start = _utc_timestamp(start) if start is not None else observed_start
    range_end = _utc_timestamp(end) if end is not None else observed_end
    if range_start is None or range_end is None:
        return _empty_hourly_frame()
    hours = pd.date_range(range_start, range_end, freq="h", tz="UTC")
    result = pd.DataFrame({"timestamp": hours})

    result = result.merge(_aggregate(all_records, "submission"), on="timestamp", how="left")
    result = result.merge(_aggregate(all_records, "comment"), on="timestamp", how="left")
    result = result.merge(_combined_aggregate(all_records), on="timestamp", how="left")

    count_columns = [
        "submission_count",
        "comment_count",
        "subreddit_count",
        "combined_observation_count",
    ]
    for column in count_columns:
        result[column] = result[column].fillna(0).astype("int64")
    sentiment_columns = [
        "submission_sentiment_mean",
        "comment_sentiment_mean",
        "combined_sentiment_mean",
    ]
    for column in sentiment_columns:
        result[column] = result[column].astype("float64")

    result["sentiment_missing"] = result["combined_observation_count"].eq(0).astype("int8")
    filled_mean = result["combined_sentiment_mean"].fillna(0.0)
    count = result["combined_observation_count"].astype("float64")
    result["sentiment_reliability"] = count / (count + 5.0)
    result["sentiment_reliability_shrunk"] = filled_mean * result["sentiment_reliability"]
    result["sentiment_ewma_6h"] = filled_mean.ewm(span=6, adjust=False).mean()
    result["sentiment_ewma_24h"] = filled_mean.ewm(span=24, adjust=False).mean()
    result["sentiment_ewma_fast_minus_slow"] = (
        result["sentiment_ewma_6h"] - result["sentiment_ewma_24h"]
    )
    result["sentiment_lag_1h"] = filled_mean.shift(1)
    result["sentiment_kalman"] = _kalman_filter(
        filled_mean,
        process_variance=kalman_process_variance,
        measurement_variance=kalman_measurement_variance,
    )
    return result


def sentiment_feature_columns(variant: str) -> tuple[str, ...]:
    """Return cumulative sentiment columns for an ablation variant."""

    if variant not in SENTIMENT_VARIANT_ORDER:
        raise ValueError(f"unknown sentiment variant: {variant}")
    columns: list[str] = []
    for current in SENTIMENT_VARIANT_ORDER[: SENTIMENT_VARIANT_ORDER.index(variant) + 1]:
        columns.extend(_variant_columns(current))
    return tuple(dict.fromkeys(columns))


def _variant_columns(variant: str) -> tuple[str, ...]:
    if variant == "hourly_mean":
        return (
            "submission_sentiment_mean",
            "comment_sentiment_mean",
            "combined_sentiment_mean",
        )
    if variant == "counts_missingness":
        return (
            "submission_count",
            "comment_count",
            "subreddit_count",
            "combined_observation_count",
            "sentiment_missing",
        )
    if variant == "reliability_shrunk":
        return ("sentiment_reliability", "sentiment_reliability_shrunk")
    if variant == "ewma_6h":
        return ("sentiment_ewma_6h",)
    if variant == "ewma_24h":
        return ("sentiment_ewma_24h",)
    if variant == "ewma_fast_minus_slow":
        return ("sentiment_ewma_fast_minus_slow",)
    if variant == "sentiment_lag_1h":
        return ("sentiment_lag_1h",)
    if variant == "kalman_filtered":
        return ("sentiment_kalman",)
    raise ValueError(f"unknown sentiment variant: {variant}")


def _aggregate(records: pd.DataFrame, record_type: str) -> pd.DataFrame:
    prefix = "submission" if record_type == "submission" else "comment"
    subset = records.loc[records["record_type"].eq(record_type)].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["timestamp", f"{prefix}_sentiment_mean", f"{prefix}_count"]
        )
    return (
        subset.groupby("hour", sort=True)
        .agg(
            **{
                f"{prefix}_sentiment_mean": ("sentiment_score", "mean"),
                f"{prefix}_count": ("sentiment_score", "size"),
            }
        )
        .reset_index()
        .rename(columns={"hour": "timestamp"})
    )


def _combined_aggregate(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "combined_sentiment_mean",
                "combined_observation_count",
                "subreddit_count",
            ]
        )
    return (
        records.groupby("hour", sort=True)
        .agg(
            combined_sentiment_mean=("sentiment_score", "mean"),
            combined_observation_count=("sentiment_score", "size"),
            subreddit_count=("subreddit", "nunique"),
        )
        .reset_index()
        .rename(columns={"hour": "timestamp"})
    )


def _kalman_filter(
    observations: pd.Series,
    *,
    process_variance: float,
    measurement_variance: float,
) -> pd.Series:
    if process_variance <= 0 or measurement_variance <= 0:
        raise ValueError("Kalman variances must be positive")
    state = 0.0
    variance = 1.0
    estimates: list[float] = []
    for observation in observations.astype("float64"):
        variance = variance + process_variance
        gain = variance / (variance + measurement_variance)
        state = state + gain * (float(observation) - state)
        variance = (1.0 - gain) * variance
        estimates.append(state)
    return pd.Series(estimates, index=observations.index, dtype="float64")


def _require_sentiment(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    if "sentiment_score" not in data.columns:
        raise ValueError(f"{name} must include sentiment_score")
    result = data.copy()
    result["created_utc"] = pd.to_datetime(result["created_utc"], utc=True)
    return result


def _hour(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.floor("h")


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True)).floor("h")


def _empty_hourly_frame() -> pd.DataFrame:
    columns = [
        "timestamp",
        *sentiment_feature_columns("kalman_filtered"),
    ]
    return pd.DataFrame(columns=columns)
