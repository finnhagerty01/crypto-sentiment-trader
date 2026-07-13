"""Build saved hourly sentiment features from existing server Reddit CSVs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from trader.sentiment.features import build_hourly_sentiment_features
from trader.sentiment.scoring import (
    LexiconSentimentScorer,
    VaderSentimentScorer,
    score_reddit_records,
)
from trader.sentiment.storage import (
    COMMENT_COLUMNS,
    write_hourly_sentiment_dataset,
    write_sentiment_dataset,
)


DEFAULT_INPUTS = (
    Path("../reddit_archive_server.csv"),
    Path("../master_reddit_server.csv"),
)
DEFAULT_RAW_OUTPUT_DIR = Path("artifacts/sentiment/raw")
DEFAULT_HOURLY_OUTPUT_DIR = Path("artifacts/sentiment/hourly")
DEFAULT_RAW_DATASET_ID = "reddit_server_csv_posts_only_raw"
DEFAULT_HOURLY_DATASET_ID = "reddit_server_csv_posts_only_hourly"
ScorerName = Literal["vader", "lexicon"]


@dataclass(frozen=True, slots=True)
class BuildResult:
    raw_dataset_path: Path
    hourly_dataset_path: Path
    raw_metadata: dict[str, object]
    hourly_metadata: dict[str, object]


def build_hourly_sentiment_from_server_csv(
    *,
    input_paths: tuple[Path, ...],
    raw_output_dir: Path,
    hourly_output_dir: Path,
    raw_dataset_id: str,
    hourly_dataset_id: str,
    scorer_name: ScorerName = "vader",
) -> BuildResult:
    """Convert post-only Reddit CSVs into raw and hourly sentiment datasets."""

    submissions = _load_submission_csvs(input_paths)
    comments = _empty_comments()
    raw_path, raw_metadata = write_sentiment_dataset(
        submissions,
        comments,
        raw_output_dir,
        dataset_id=raw_dataset_id,
        source="server_csv",
        extra_metadata={
            "input_files": [str(path) for path in input_paths],
            "posts_only": True,
            "comments_source": "not_available_in_server_csv",
        },
    )
    scorer = _scorer(scorer_name)
    scored_submissions, scored_comments = score_reddit_records(
        submissions,
        comments,
        scorer,
    )
    hourly = build_hourly_sentiment_features(
        scored_submissions,
        scored_comments,
        start=scored_submissions["created_utc"].min().floor("h"),
        end=scored_submissions["created_utc"].max().floor("h"),
    )
    hourly_path, hourly_metadata = write_hourly_sentiment_dataset(
        hourly,
        hourly_output_dir,
        dataset_id=hourly_dataset_id,
        source_dataset_id=str(raw_metadata["dataset_id"]),
        variant="all_features",
    )
    return BuildResult(
        raw_dataset_path=raw_path,
        hourly_dataset_path=hourly_path,
        raw_metadata=raw_metadata,
        hourly_metadata=hourly_metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score existing server Reddit CSV posts and write the hourly sentiment "
            "Parquet dataset required by run-sentiment-gate."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        dest="inputs",
        help=(
            "Input server CSV path. May be supplied multiple times. Defaults to "
            "../reddit_archive_server.csv and ../master_reddit_server.csv."
        ),
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_DIR,
        help=f"Raw sentiment dataset directory. Default: {DEFAULT_RAW_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--hourly-output-dir",
        type=Path,
        default=DEFAULT_HOURLY_OUTPUT_DIR,
        help=f"Hourly sentiment output directory. Default: {DEFAULT_HOURLY_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--raw-dataset-id",
        default=DEFAULT_RAW_DATASET_ID,
        help=f"Raw dataset id. Default: {DEFAULT_RAW_DATASET_ID}",
    )
    parser.add_argument(
        "--hourly-dataset-id",
        default=DEFAULT_HOURLY_DATASET_ID,
        help=f"Hourly dataset id. Default: {DEFAULT_HOURLY_DATASET_ID}",
    )
    parser.add_argument(
        "--scorer",
        choices=("vader", "lexicon"),
        default="vader",
        help="Sentiment scorer. Use vader for research runs; lexicon is for tests.",
    )
    args = parser.parse_args()

    inputs = tuple(args.inputs or DEFAULT_INPUTS)
    missing = [path for path in inputs if not path.exists()]
    if missing:
        parser.error("missing input CSV(s): " + ", ".join(str(path) for path in missing))

    result = build_hourly_sentiment_from_server_csv(
        input_paths=inputs,
        raw_output_dir=args.raw_output_dir,
        hourly_output_dir=args.hourly_output_dir,
        raw_dataset_id=args.raw_dataset_id,
        hourly_dataset_id=args.hourly_dataset_id,
        scorer_name=args.scorer,
    )
    print(f"raw_dataset: {result.raw_dataset_path}")
    print(f"raw_rows: {result.raw_metadata['submission_count']}")
    print(f"hourly_dataset: {result.hourly_dataset_path}")
    print(f"hourly_rows: {result.hourly_metadata['row_count']}")
    print(f"hourly_start: {result.hourly_metadata['start']}")
    print(f"hourly_end: {result.hourly_metadata['end']}")
    return 0


def _load_submission_csvs(paths: tuple[Path, ...]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one input CSV is required")
    frames = [_normalize_submission_csv(pd.read_csv(path), source_path=path) for path in paths]
    submissions = pd.concat(frames, ignore_index=True)
    submissions = submissions.drop_duplicates("submission_id", keep="first")
    return submissions.sort_values(["created_utc", "submission_id"]).reset_index(drop=True)


def _normalize_submission_csv(data: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    required = {
        "id",
        "created_utc",
        "title",
        "selftext",
        "subreddit",
        "score",
        "num_comments",
        "source",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{source_path}: missing required column(s): {', '.join(missing)}")
    normalized = data.copy()
    normalized["created_utc"] = pd.to_datetime(normalized["created_utc"], utc=True)
    if normalized["created_utc"].isna().any():
        raise ValueError(f"{source_path}: created_utc contains invalid timestamps")
    normalized["submission_id"] = normalized["id"].astype("string")
    if normalized["submission_id"].str.len().eq(0).any():
        raise ValueError(f"{source_path}: id values must be non-empty")
    normalized["title"] = normalized["title"].fillna("").astype("string")
    normalized["selftext"] = normalized["selftext"].fillna("").astype("string")
    normalized["subreddit"] = normalized["subreddit"].fillna("").astype("string")
    normalized["score"] = pd.to_numeric(normalized["score"], errors="raise").astype("int64")
    normalized["num_comments"] = pd.to_numeric(
        normalized["num_comments"],
        errors="raise",
    ).astype("int64")
    normalized["author"] = None
    normalized["permalink"] = None
    normalized["collection_status"] = "comments_not_available"
    normalized["collected_at"] = normalized["created_utc"]
    normalized["source"] = normalized["source"].fillna("server_csv").astype("string")
    return normalized.loc[
        :,
        [
            "submission_id",
            "subreddit",
            "created_utc",
            "title",
            "selftext",
            "score",
            "num_comments",
            "author",
            "permalink",
            "collection_status",
            "collected_at",
            "source",
        ],
    ]


def _empty_comments() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMENT_COLUMNS)


def _scorer(name: ScorerName) -> VaderSentimentScorer | LexiconSentimentScorer:
    if name == "vader":
        return VaderSentimentScorer()
    if name == "lexicon":
        return LexiconSentimentScorer()
    raise ValueError(f"unknown scorer: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
