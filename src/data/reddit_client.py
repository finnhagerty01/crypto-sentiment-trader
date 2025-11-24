import os
import logging
import pandas as pd
import praw
from datetime import datetime, timezone, timedelta
from typing import List

logger = logging.getLogger(__name__)

class RedditClient:
    """
    Unified client. 
    Uses PRAW (Official API) for both historical and live data to ensure reliability.
    """
    def __init__(self, subreddits: List[str]):
        self.subreddits = subreddits
        
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

    def fetch_historical(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data via PRAW (Official API).
        LIMITATION: PRAW can only fetch the last ~1000 posts per subreddit.
        This usually covers 2-7 days for active subs, which is enough for V1 training.
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
                posts = self.reddit.subreddit(sub).new(limit=1000)
                
                count = 0
                for post in posts:
                    post_dt = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    
                    # Stop if we went back too far (optional optimization)
                    if post_dt < cutoff_date:
                        break
                    
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
                
                logger.info(f"Fetched {count} posts from r/{sub}")
                
            except Exception as e:
                logger.error(f"Failed to fetch history for r/{sub}: {e}")

        df = pd.DataFrame(all_posts)
        return df

    def fetch_live(self, limit: int = 500) -> pd.DataFrame:
        """Fetch the last N posts via Official API (Real-time)."""
        if not self.reddit:
            logger.error("Cannot fetch live data: PRAW not initialized.")
            return pd.DataFrame()

        live_posts = []
        for sub in self.subreddits:
            try:
                # Fetch new posts
                for post in self.reddit.subreddit(sub).new(limit=limit):
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
            except Exception as e:
                logger.error(f"Error fetching live r/{sub}: {e}")

        return pd.DataFrame(live_posts)