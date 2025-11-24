# src/data/engagement_tracker.py
"""
Engagement tracking system for validation and analysis.
This tracks how post engagement evolves over time, but is NOT used for prediction.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class EngagementTracker:
    """
    Tracks post engagement metrics at multiple time points for analysis.
    
    Use cases:
    1. Validate if early engagement correlates with price movement
    2. Analyze engagement velocity (how fast posts gain traction)
    3. Identify "viral" posts that may drive sentiment cascades
    4. Compare engagement patterns across different crypto assets
    """
    
    def __init__(self, config):
        self.config = config
        self.snapshots_dir = config.data_dir / 'engagement_snapshots'
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def create_snapshot(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a snapshot of current engagement metrics.
        
        Args:
            posts_df: DataFrame with current post data
        
        Returns:
            Snapshot with engagement metrics at this point in time
        """
        snapshot = posts_df.copy()
        
        # Add snapshot metadata
        snapshot['snapshot_time'] = datetime.now(timezone.utc)
        
        # Calculate age-based metrics
        snapshot['hours_since_creation'] = (
            snapshot['snapshot_time'] - snapshot['created_utc']
        ).dt.total_seconds() / 3600
        
        # Engagement rate at this snapshot
        snapshot['engagement_rate_snapshot'] = (
            (snapshot['score'] + snapshot['num_comments']) / 
            (snapshot['hours_since_creation'] + 1)  # +1 to avoid division by zero
        )
        
        # Velocity indicators
        snapshot['engagement_velocity'] = snapshot['engagement_rate_snapshot']
        
        return snapshot
    
    def save_snapshot(self, snapshot_df: pd.DataFrame):
        """Save snapshot to disk."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filepath = self.snapshots_dir / f"snapshot_{timestamp}.parquet"
        snapshot_df.to_parquet(filepath, index=False)
        logger.info(f"Saved engagement snapshot: {filepath}")
    
    def load_snapshots(self, days_back: int = 30) -> pd.DataFrame:
        """
        Load all snapshots from the specified time period.
        
        Args:
            days_back: Number of days of snapshots to load
        
        Returns:
            Combined DataFrame of all snapshots
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        snapshot_files = sorted(self.snapshots_dir.glob("snapshot_*.parquet"))
        
        dfs = []
        for file in snapshot_files:
            # Parse timestamp from filename
            timestamp_str = file.stem.replace('snapshot_', '')
            file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            file_time = file_time.replace(tzinfo=timezone.utc)
            
            if file_time >= cutoff:
                df = pd.read_parquet(file)
                dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(dfs)} snapshots with {len(combined)} total records")
            return combined
        
        return pd.DataFrame()
    
    def calculate_engagement_progression(self, post_id: str) -> pd.DataFrame:
        """
        Calculate how a specific post's engagement evolved over time.
        
        Args:
            post_id: Reddit post ID
        
        Returns:
            DataFrame showing engagement at each snapshot
        """
        all_snapshots = self.load_snapshots()
        
        if all_snapshots.empty:
            return pd.DataFrame()
        
        # Filter to this specific post
        post_history = all_snapshots[all_snapshots['id'] == post_id].copy()
        
        if post_history.empty:
            return pd.DataFrame()
        
        # Sort by snapshot time
        post_history = post_history.sort_values('snapshot_time')
        
        # Calculate deltas between snapshots
        post_history['score_delta'] = post_history['score'].diff()
        post_history['comments_delta'] = post_history['num_comments'].diff()
        post_history['engagement_delta'] = (
            post_history['engagement_rate_snapshot'].diff()
        )
        
        return post_history[['snapshot_time', 'hours_since_creation', 
                            'score', 'num_comments', 'engagement_rate_snapshot',
                            'score_delta', 'comments_delta', 'engagement_delta']]
    
    def analyze_engagement_price_correlation(self, 
                                            sentiment_df: pd.DataFrame,
                                            market_df: pd.DataFrame,
                                            symbol: str) -> Dict:
        """
        Analyze correlation between post engagement progression and price movement.
        
        This is your validation analysis!
        
        Args:
            sentiment_df: Sentiment features with engagement metrics
            market_df: Market data with price movements
            symbol: Crypto symbol to analyze
        
        Returns:
            Dictionary with correlation analysis
        """
        # Load all engagement snapshots
        snapshots = self.load_snapshots()
        
        if snapshots.empty:
            logger.warning("No engagement snapshots available")
            return {}
        
        # Filter to relevant symbol posts
        symbol_posts = snapshots[snapshots['symbol'] == symbol].copy()
        
        if symbol_posts.empty:
            return {}
        
        # Align with market data by time windows
        market_symbol = market_df[market_df['symbol'] == symbol].copy()
        
        results = {}
        
        # Calculate correlations at different time lags
        for lag_hours in [1, 3, 6, 12, 24]:
            # For each post, find price movement lag_hours after creation
            post_price_pairs = []
            
            for _, post in symbol_posts.iterrows():
                post_time = post['created_utc']
                target_time = post_time + timedelta(hours=lag_hours)
                
                # Find nearest market data point
                time_diffs = abs(market_symbol['timestamp'] - target_time)
                if not time_diffs.empty:
                    nearest_idx = time_diffs.idxmin()
                    price_change = market_symbol.loc[nearest_idx, 'return_1']
                    
                    post_price_pairs.append({
                        'engagement_rate': post['engagement_rate_snapshot'],
                        'sentiment': post.get('vader_compound', 0),
                        'price_change': price_change,
                        'lag_hours': lag_hours
                    })
            
            if post_price_pairs:
                pairs_df = pd.DataFrame(post_price_pairs)
                
                # Calculate correlations
                engagement_corr = pairs_df['engagement_rate'].corr(pairs_df['price_change'])
                sentiment_corr = pairs_df['sentiment'].corr(pairs_df['price_change'])
                
                # Combined metric: engagement * sentiment
                pairs_df['engagement_sentiment'] = (
                    pairs_df['engagement_rate'] * pairs_df['sentiment']
                )
                combined_corr = pairs_df['engagement_sentiment'].corr(pairs_df['price_change'])
                
                results[f'{lag_hours}h'] = {
                    'engagement_correlation': engagement_corr,
                    'sentiment_correlation': sentiment_corr,
                    'combined_correlation': combined_corr,
                    'n_samples': len(pairs_df)
                }
        
        return results
    
    def identify_viral_posts(self, 
                            threshold_percentile: float = 0.9,
                            min_velocity: float = 10.0) -> pd.DataFrame:
        """
        Identify posts that went "viral" (rapid engagement growth).
        
        Args:
            threshold_percentile: Top percentile to consider viral
            min_velocity: Minimum engagement velocity
        
        Returns:
            DataFrame of viral posts
        """
        snapshots = self.load_snapshots()
        
        if snapshots.empty:
            return pd.DataFrame()
        
        # Get the latest snapshot for each post
        latest_snapshots = snapshots.sort_values('snapshot_time').groupby('id').last()
        
        # Calculate engagement velocity
        velocity_threshold = latest_snapshots['engagement_velocity'].quantile(threshold_percentile)
        
        viral_posts = latest_snapshots[
            (latest_snapshots['engagement_velocity'] >= velocity_threshold) &
            (latest_snapshots['engagement_velocity'] >= min_velocity)
        ].copy()
        
        viral_posts = viral_posts.sort_values('engagement_velocity', ascending=False)
        
        logger.info(f"Identified {len(viral_posts)} viral posts")
        
        return viral_posts[['title', 'subreddit', 'created_utc', 'score', 
                           'num_comments', 'engagement_velocity', 'vader_compound']]