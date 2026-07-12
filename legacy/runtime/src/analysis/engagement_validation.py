# src/analysis/engagement_validation.py
"""
Validation analysis to understand engagement-sentiment-price relationships.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class EngagementValidator:
    """
    Analyzes engagement patterns to validate trading hypotheses.
    
    Key Questions This Answers:
    1. Do highly engaged posts correlate with price movement?
    2. Does engagement velocity predict volatility?
    3. Do sentiment + engagement together predict better than sentiment alone?
    4. What's the optimal time lag between post and price impact?
    """
    
    def __init__(self, config):
        self.config = config
        from src.data.engagement_tracker import EngagementTracker
        self.tracker = EngagementTracker(config)
    
    def run_comprehensive_validation(self,
                                    market_df: pd.DataFrame,
                                    save_report: bool = True) -> Dict:
        """
        Run comprehensive engagement validation analysis.
        
        Args:
            market_df: Market data with price movements
            save_report: Whether to save detailed report
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'timestamp': datetime.now(timezone.utc),
            'symbols': {},
            'overall_findings': {}
        }
        
        # Analyze each symbol
        for symbol in self.config.symbols:
            logger.info(f"Analyzing engagement for {symbol}")
            
            symbol_results = self.tracker.analyze_engagement_price_correlation(
                sentiment_df=pd.DataFrame(),  # Would load from processed data
                market_df=market_df,
                symbol=symbol
            )
            
            results['symbols'][symbol] = symbol_results
        
        # Aggregate findings
        results['overall_findings'] = self._aggregate_findings(results['symbols'])
        
        if save_report:
            self._save_validation_report(results)
        
        return results
    
    def analyze_engagement_timing(self,
                                 symbol: str,
                                 market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze optimal timing: when does post engagement correlate most with price?
        
        This answers: "How quickly do highly engaged posts impact price?"
        
        Args:
            symbol: Crypto symbol
            market_df: Market data
        
        Returns:
            DataFrame with correlations at different time lags
        """
        snapshots = self.tracker.load_snapshots()

        # Return empty if no snapshots or no symbol column
        if snapshots.empty or 'symbol' not in snapshots.columns:
            return pd.DataFrame()

        symbol_posts = snapshots[snapshots['symbol'] == symbol]

        if symbol_posts.empty:
            return pd.DataFrame()
        
        # Test correlations at multiple time lags
        lag_results = []
        
        for lag_hours in range(1, 25):  # Test 1-24 hours
            correlations = []
            
            for _, post in symbol_posts.iterrows():
                post_time = post['created_utc']
                future_time = post_time + timedelta(hours=lag_hours)
                
                # Find price movement in next hour after lag
                market_window = market_df[
                    (market_df['symbol'] == symbol) &
                    (market_df['timestamp'] >= future_time) &
                    (market_df['timestamp'] < future_time + timedelta(hours=1))
                ]
                
                if not market_window.empty:
                    price_change = market_window['return_1'].mean()
                    engagement = post['engagement_rate_snapshot']
                    sentiment = post.get('vader_compound', 0)
                    
                    correlations.append({
                        'lag_hours': lag_hours,
                        'engagement': engagement,
                        'sentiment': sentiment,
                        'price_change': price_change
                    })
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                
                lag_results.append({
                    'lag_hours': lag_hours,
                    'engagement_corr': corr_df['engagement'].corr(corr_df['price_change']),
                    'sentiment_corr': corr_df['sentiment'].corr(corr_df['price_change']),
                    'n_samples': len(corr_df)
                })
        
        return pd.DataFrame(lag_results)
    
    def compare_high_vs_low_engagement(self,
                                      symbol: str,
                                      market_df: pd.DataFrame,
                                      engagement_threshold: float = 0.75) -> Dict:
        """
        Compare price movement following high-engagement vs low-engagement posts.
        
        This tests: "Do highly engaged posts lead to larger price moves?"
        
        Args:
            symbol: Crypto symbol
            market_df: Market data
            engagement_threshold: Percentile threshold for "high engagement"
        
        Returns:
            Comparison statistics
        """
        snapshots = self.tracker.load_snapshots()

        # Return empty if no snapshots or no symbol column
        if snapshots.empty or 'symbol' not in snapshots.columns:
            return {}

        symbol_posts = snapshots[snapshots['symbol'] == symbol].copy()

        if symbol_posts.empty:
            return {}
        
        # Split posts into high/low engagement
        engagement_cutoff = symbol_posts['engagement_rate_snapshot'].quantile(
            engagement_threshold
        )
        
        high_engagement = symbol_posts[
            symbol_posts['engagement_rate_snapshot'] >= engagement_cutoff
        ]
        low_engagement = symbol_posts[
            symbol_posts['engagement_rate_snapshot'] < engagement_cutoff
        ]
        
        # Calculate price movements following each group
        def get_subsequent_returns(posts_df, hours_forward=6):
            returns = []
            for _, post in posts_df.iterrows():
                post_time = post['created_utc']
                future_time = post_time + timedelta(hours=hours_forward)
                
                market_point = market_df[
                    (market_df['symbol'] == symbol) &
                    (market_df['timestamp'] >= future_time)
                ].head(1)
                
                if not market_point.empty:
                    returns.append(market_point['return_1'].iloc[0])
            
            return returns
        
        high_returns = get_subsequent_returns(high_engagement)
        low_returns = get_subsequent_returns(low_engagement)
        
        # Statistical comparison
        from scipy import stats
        
        results = {
            'high_engagement': {
                'n_posts': len(high_engagement),
                'mean_return': np.mean(high_returns) if high_returns else 0,
                'std_return': np.std(high_returns) if high_returns else 0,
                'positive_ratio': sum(r > 0 for r in high_returns) / len(high_returns) if high_returns else 0
            },
            'low_engagement': {
                'n_posts': len(low_engagement),
                'mean_return': np.mean(low_returns) if low_returns else 0,
                'std_return': np.std(low_returns) if low_returns else 0,
                'positive_ratio': sum(r > 0 for r in low_returns) / len(low_returns) if low_returns else 0
            }
        }
        
        # T-test
        if high_returns and low_returns:
            t_stat, p_value = stats.ttest_ind(high_returns, low_returns)
            results['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def _aggregate_findings(self, symbol_results: Dict) -> Dict:
        """Aggregate findings across all symbols."""
        all_lags = {}
        
        for symbol, lags in symbol_results.items():
            for lag, metrics in lags.items():
                if lag not in all_lags:
                    all_lags[lag] = {
                        'engagement_corrs': [],
                        'sentiment_corrs': [],
                        'combined_corrs': []
                    }
                
                all_lags[lag]['engagement_corrs'].append(
                    metrics.get('engagement_correlation', 0)
                )
                all_lags[lag]['sentiment_corrs'].append(
                    metrics.get('sentiment_correlation', 0)
                )
                all_lags[lag]['combined_corrs'].append(
                    metrics.get('combined_correlation', 0)
                )
        
        # Calculate averages
        findings = {}
        for lag, values in all_lags.items():
            findings[lag] = {
                'avg_engagement_corr': np.mean(values['engagement_corrs']),
                'avg_sentiment_corr': np.mean(values['sentiment_corrs']),
                'avg_combined_corr': np.mean(values['combined_corrs'])
            }
        
        # Find optimal lag if we have any findings
        if findings:
            best_lag = max(findings.items(),
                          key=lambda x: abs(x[1]['avg_combined_corr']))

            findings['optimal_lag'] = {
                'lag': best_lag[0],
                'correlation': best_lag[1]['avg_combined_corr']
            }
        else:
            findings['optimal_lag'] = {
                'lag': None,
                'correlation': 0.0
            }

        return findings
    
    def _save_validation_report(self, results: Dict):
        """Save validation report to disk."""
        report_dir = self.config.processed_dir / 'validation_reports'
        report_dir.mkdir(exist_ok=True)
        
        timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
        filepath = report_dir / f'engagement_validation_{timestamp}.json'
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved validation report: {filepath}")
    
    def plot_engagement_analysis(self, symbol: str, market_df: pd.DataFrame,
                                save_path: Path = None):
        """
        Create visualization of engagement analysis.
        
        Args:
            symbol: Crypto symbol
            market_df: Market data
            save_path: Where to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Engagement Analysis: {symbol}', fontsize=16)
        
        # 1. Correlation vs Time Lag
        timing_df = self.analyze_engagement_timing(symbol, market_df)
        if not timing_df.empty:
            ax = axes[0, 0]
            ax.plot(timing_df['lag_hours'], timing_df['engagement_corr'], 
                   label='Engagement', marker='o')
            ax.plot(timing_df['lag_hours'], timing_df['sentiment_corr'], 
                   label='Sentiment', marker='s')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Hours After Post')
            ax.set_ylabel('Correlation with Price Change')
            ax.set_title('Optimal Timing Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. High vs Low Engagement Comparison
        comparison = self.compare_high_vs_low_engagement(symbol, market_df)
        if comparison:
            ax = axes[0, 1]
            categories = ['High Engagement', 'Low Engagement']
            means = [
                comparison['high_engagement']['mean_return'],
                comparison['low_engagement']['mean_return']
            ]
            stds = [
                comparison['high_engagement']['std_return'],
                comparison['low_engagement']['std_return']
            ]
            
            ax.bar(categories, means, yerr=stds, capsize=5, 
                  color=['green' if m > 0 else 'red' for m in means])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Mean Return (6h forward)')
            ax.set_title('Return Following High vs Low Engagement Posts')
            
            if 'statistical_test' in comparison:
                p_val = comparison['statistical_test']['p_value']
                sig_text = f"p={p_val:.4f}" + (" *" if p_val < 0.05 else "")
                ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Engagement Distribution
        snapshots = self.tracker.load_snapshots()
        if not snapshots.empty and 'symbol' in snapshots.columns:
            symbol_posts = snapshots[snapshots['symbol'] == symbol]
        else:
            symbol_posts = pd.DataFrame()

        if not symbol_posts.empty:
            ax = axes[1, 0]
            ax.hist(symbol_posts['engagement_rate_snapshot'], bins=30, 
                   color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Engagement Rate')
            ax.set_ylabel('Frequency')
            ax.set_title('Engagement Rate Distribution')
            ax.axvline(symbol_posts['engagement_rate_snapshot'].median(), 
                      color='red', linestyle='--', label='Median')
            ax.legend()
        
        # 4. Sentiment vs Engagement Scatter
        if not symbol_posts.empty:
            ax = axes[1, 1]
            scatter = ax.scatter(
                symbol_posts['engagement_rate_snapshot'],
                symbol_posts.get('vader_compound', 0),
                c=symbol_posts['score'],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            ax.set_xlabel('Engagement Rate')
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Engagement vs Sentiment')
            plt.colorbar(scatter, ax=ax, label='Score')
            
            # Add correlation
            corr = symbol_posts['engagement_rate_snapshot'].corr(
                symbol_posts.get('vader_compound', pd.Series([0]))
            )
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved engagement plot: {save_path}")
        
        plt.close()