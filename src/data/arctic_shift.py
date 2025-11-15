"""Integration with the Arctic Shift API for extended Reddit history."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class ArcticShiftClient:
    """Lightweight wrapper around the Arctic Shift REST API.

    The public documentation for Arctic Shift is intentionally sparse so this
    client focuses on the pieces we need: fetching historical Reddit posts for a
    set of subreddits.  The code is intentionally defensive – if the API key is
    missing or the service is unreachable the caller receives an empty
    DataFrame instead of an exception so the rest of the pipeline can fall back
    to the standard Reddit/Pushshift collectors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.arcticshift.com/v1",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "Arctic Shift API key not provided – falling back to Reddit/Pushshift data only."
            )

    # ------------------------------------------------------------------
    def _request(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """Issue a GET request to the API and return the decoded payload."""
        if not self.api_key:
            return None

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            logger.error("Arctic Shift request failed: %s", exc)
            return None

        try:
            return response.json()
        except ValueError as exc:
            logger.error("Failed to decode Arctic Shift response: %s", exc)
            return None

    # ------------------------------------------------------------------
    def fetch_subreddit_posts(
        self,
        subreddits: Iterable[str],
        start: datetime,
        end: datetime,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch aggregated posts from Arctic Shift for the given time window.

        Parameters
        ----------
        subreddits:
            An iterable of subreddit names.
        start, end:
            UTC datetimes delimiting the requested window.
        limit:
            Maximum number of rows per subreddit to request.  Arctic Shift
            exposes pagination through "next" cursors but in practice a single
            request with a generous limit is enough for hourly bars.
        """

        if not self.api_key:
            return pd.DataFrame()

        records: List[pd.DataFrame] = []
        start_ts = int(start.replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(end.replace(tzinfo=timezone.utc).timestamp())

        for subreddit in subreddits:
            payload = self._request(
                "reddit/submissions",
                params={
                    "subreddit": subreddit,
                    "start": start_ts,
                    "end": end_ts,
                    "limit": limit,
                    "sort": "created_utc",
                    "order": "asc",
                },
            )
            if not payload or "data" not in payload:
                continue

            df = pd.DataFrame(payload["data"])
            if df.empty:
                continue

            if "created_utc" in df.columns:
                df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
            df["subreddit"] = subreddit
            records.append(df)

        if not records:
            return pd.DataFrame()

        combined = pd.concat(records, ignore_index=True)
        combined.sort_values("created_utc", inplace=True)
        return combined


__all__ = ["ArcticShiftClient"]
