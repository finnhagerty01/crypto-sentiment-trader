import os
import re
import logging
import pandas as pd
import praw
from datetime import datetime, timezone, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

# Common spam patterns and scam domains
SPAM_PATTERNS = [
    r"dm\s*me\s*for",  # "DM me for..."
    r"check\s*my\s*profile",
    r"link\s*in\s*bio",
    r"guaranteed\s*returns",
    r"\d+x\s*gains?\s*guaranteed",
    r"free\s*crypto\s*giveaway",
    r"send\s*\d+\s*(btc|eth|sol)",  # "send 0.1 BTC"
    r"airdrop.*wallet",
    r"pump\s*(and|&)?\s*dump",
]

SCAM_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl",  # URL shorteners (often used for scams)
    "t.me/",  # Telegram links (common for pump schemes)
    "discord.gg/",  # Discord invites
    "forms.gle",  # Google forms (phishing)
]

# Excessive emoji threshold (more than this ratio is likely spam)
MAX_EMOJI_RATIO = 0.15  # 15% of text being emojis is suspicious

class RedditClient:
    """
    Unified client.
    Uses PRAW (Official API) for both historical and live data to ensure reliability.
    Includes spam filtering to remove bot posts and scam content.
    """

    # Minimum karma threshold (filter out likely bot accounts)
    MIN_KARMA = 10
    # Minimum account age in days
    MIN_ACCOUNT_AGE_DAYS = 2

    def __init__(self, subreddits: List[str], filter_spam: bool = True):
        self.subreddits = subreddits
        self.filter_spam = filter_spam
        self._spam_pattern = re.compile("|".join(SPAM_PATTERNS), flags=re.IGNORECASE)

        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "crypto_bot_v1")
            )
            # Fast check to ensure credentials work
            self.reddit.user.me()
            logger.info("Connected to Reddit Official API")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            self.reddit = None

    def _is_spam(self, post) -> bool:
        """
        Detect if a post is likely spam or from a bot account.

        Checks:
        1. Known spam text patterns (scam phrases, pump schemes)
        2. Scam domains in URLs
        3. Excessive emoji usage
        4. Low karma accounts (likely bots)
        5. New accounts (< 2 days old)

        Args:
            post: PRAW Submission object

        Returns:
            True if post appears to be spam, False otherwise
        """
        text = f"{post.title or ''} {post.selftext or ''}".lower()

        # Check 1: Known spam patterns
        if self._spam_pattern.search(text):
            logger.debug(f"Spam pattern detected in post {post.id}")
            return True

        # Check 2: Scam domains
        for domain in SCAM_DOMAINS:
            if domain in text:
                logger.debug(f"Scam domain {domain} detected in post {post.id}")
                return True

        # Check 3: Excessive emojis
        if text:
            # Count emojis using unicode ranges
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags
                "\U00002702-\U000027B0"  # dingbats
                "\U0001F900-\U0001F9FF"  # supplemental symbols
                "]+",
                flags=re.UNICODE
            )
            emoji_count = len(emoji_pattern.findall(text))
            text_len = len(text)
            if text_len > 0 and emoji_count / text_len > MAX_EMOJI_RATIO:
                logger.debug(f"Excessive emojis in post {post.id}")
                return True

        # Check 4 & 5: Author karma and account age
        try:
            author = post.author
            if author is not None:
                # Check karma (combined link + comment karma)
                total_karma = getattr(author, 'link_karma', 0) + getattr(author, 'comment_karma', 0)
                if total_karma < self.MIN_KARMA:
                    logger.debug(f"Low karma author ({total_karma}) for post {post.id}")
                    return True

                # Check account age
                created_utc = getattr(author, 'created_utc', None)
                if created_utc:
                    account_age = datetime.now(timezone.utc) - datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    if account_age.days < self.MIN_ACCOUNT_AGE_DAYS:
                        logger.debug(f"New account ({account_age.days} days) for post {post.id}")
                        return True
        except Exception as e:
            # If we can't check author (deleted, etc.), don't filter
            logger.debug(f"Could not check author for post {post.id}: {e}")

        return False

    def fetch_historical(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data via PRAW (Official API).
        LIMITATION: PRAW can only fetch the last ~1000 posts per subreddit.
        This usually covers 2-7 days for active subs, which is enough for V1 training.

        Spam posts are filtered out if filter_spam=True (default).
        """
        if not self.reddit:
            logger.error("PRAW not initialized. Cannot fetch history.")
            return pd.DataFrame()

        logger.info(f"Fetching last ~1000 posts per subreddit via PRAW...")
        all_posts = []

        # Calculate cutoff just in case 1000 posts goes back further than 'days'
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        for sub in self.subreddits:
            try:
                # fetch 'new' to get the latest history (limit=1000 is Reddit's max)
                posts = self.reddit.subreddit(sub).new(limit=100)

                count = 0
                spam_count = 0
                for post in posts:
                    post_dt = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)

                    # Stop if we went back too far (optional optimization)
                    if post_dt < cutoff_date:
                        break

                    # Filter spam if enabled
                    if self.filter_spam and self._is_spam(post):
                        spam_count += 1
                        continue

                    all_posts.append({
                        "id": post.id,
                        "created_utc": post_dt,
                        "title": post.title,
                        "selftext": post.selftext,
                        "subreddit": sub,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "source": "history_praw"
                    })
                    count += 1

                logger.info(f"Fetched {count} posts from r/{sub} (filtered {spam_count} spam)")

            except Exception as e:
                logger.error(f"Failed to fetch history for r/{sub}: {e}")

        df = pd.DataFrame(all_posts)
        return df

    def fetch_live(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch the last N posts via Official API (Real-time).

        Spam posts are filtered out if filter_spam=True (default).
        """
        if not self.reddit:
            logger.error("Cannot fetch live data: PRAW not initialized.")
            return pd.DataFrame()

        live_posts = []
        for sub in self.subreddits:
            try:
                spam_count = 0
                # Fetch new posts
                for post in self.reddit.subreddit(sub).new(limit=limit):
                    # Filter spam if enabled
                    if self.filter_spam and self._is_spam(post):
                        spam_count += 1
                        continue

                    live_posts.append({
                        "id": post.id,
                        "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        "title": post.title,
                        "selftext": post.selftext,
                        "subreddit": sub,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "source": "live"
                    })

                if spam_count > 0:
                    logger.debug(f"Filtered {spam_count} spam posts from r/{sub}")

            except Exception as e:
                logger.error(f"Error fetching live r/{sub}: {e}")

        return pd.DataFrame(live_posts)