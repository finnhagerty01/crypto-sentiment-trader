from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

from trader.sentiment.storage import read_hourly_sentiment_dataset, read_sentiment_dataset


def test_builds_hourly_sentiment_from_server_csv_posts_only(tmp_path: Path) -> None:
    script = _load_script()
    first = tmp_path / "archive.csv"
    second = tmp_path / "master.csv"
    _write_server_csv(
        first,
        [
            {
                "id": "s1",
                "created_utc": "2026-01-01T00:10:00Z",
                "title": "bullish breakout",
                "selftext": "good rally",
                "subreddit": "Bitcoin",
                "score": 10,
                "num_comments": 2,
                "source": "fixture",
            },
            {
                "id": "s2",
                "created_utc": "2026-01-01T02:10:00Z",
                "title": "bearish risk",
                "selftext": "",
                "subreddit": "CryptoCurrency",
                "score": 1,
                "num_comments": 0,
                "source": "fixture",
            },
        ],
    )
    _write_server_csv(
        second,
        [
            {
                "id": "s2",
                "created_utc": "2026-01-01T02:10:00Z",
                "title": "duplicate should be dropped",
                "selftext": "",
                "subreddit": "CryptoCurrency",
                "score": 1,
                "num_comments": 0,
                "source": "fixture",
            },
            {
                "id": "s3",
                "created_utc": "2026-01-01T03:10:00Z",
                "title": "moon gains",
                "selftext": None,
                "subreddit": "Ethereum",
                "score": 5,
                "num_comments": 1,
                "source": "fixture",
            },
        ],
    )

    result = script.build_hourly_sentiment_from_server_csv(
        input_paths=(first, second),
        raw_output_dir=tmp_path / "raw",
        hourly_output_dir=tmp_path / "hourly",
        raw_dataset_id="raw-fixture",
        hourly_dataset_id="hourly-fixture",
        scorer_name="lexicon",
    )

    submissions, comments, raw_metadata = read_sentiment_dataset(result.raw_dataset_path)
    hourly, hourly_metadata = read_hourly_sentiment_dataset(result.hourly_dataset_path)

    assert raw_metadata["submission_count"] == 3
    assert raw_metadata["comment_count"] == 0
    assert raw_metadata["extra"]["posts_only"] is True
    assert comments.empty
    assert submissions["submission_id"].tolist() == ["s1", "s2", "s3"]
    assert hourly_metadata["source_dataset_id"] == "raw-fixture"
    assert hourly["timestamp"].tolist() == list(
        pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    )
    assert hourly.loc[0, "submission_count"] == 1
    assert hourly.loc[1, "combined_observation_count"] == 0
    assert hourly.loc[1, "sentiment_missing"] == 1
    assert hourly.loc[2, "combined_sentiment_mean"] < 0
    assert hourly.loc[3, "combined_sentiment_mean"] > 0


def _write_server_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _load_script():
    script_path = (
        Path(__file__).parents[3]
        / "scripts"
        / "build_hourly_sentiment_from_server_csv.py"
    )
    spec = importlib.util.spec_from_file_location("server_csv_sentiment_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
