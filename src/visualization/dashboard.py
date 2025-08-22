# src/visualization/dashboard.py
"""
Interactive dashboard for monitoring trading system performance and signals.
Can be run as a web app using Streamlit or generate static reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# For web dashboard (optional)

class TradingDashboard:
    """
    Comprehensive dashboard for visualizing trading system performance.
    
    Features:
    1. Real-time signal monitoring
    2. Performance metrics visualization
    3. Risk analysis charts
    4. Feature importance plots
    5. Backtest results analysis
    """
    
    def __init__(self, config):
        """Initialize dashboard with configuration."""
        self.config = config
        self.setup_style()
        
    def setup_style(self):
        """Setup visualization style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_price_and_signals(self,
                              df: pd.DataFrame,
                              symbol: str,
                              save_path: str = None) -> go.Figure:
        """
        Plot price chart with buy/sell signals.
        
        Args:
            df: DataFrame with price and signal data
            symbol: Symbol to plot
            save_path: Optional path to save figure
        
        Returns:
            Plotly figure object
        """
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price & Signals', 
                           'Trading Probability', 
                           'Volume'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=symbol_df['timestamp'],
                open=symbol_df['open'],
                high=symbol_df['high'],
                low=symbol_df['low'],
                close=symbol_df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add buy signals
        buy_signals = symbol_df[symbol_df['position'] == 1].copy()
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['timestamp'],
                    y=buy_signals['low'] * 0.99,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        # Add sell signals
        sell_signals = symbol_df[
            (symbol_df['position'] == 0) & 
            (symbol_df['position'].shift(1) == 1)
        ].copy()
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['timestamp'],
                    y=sell_signals['high'] * 1.01,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Trading probability
        if 'probability_smooth' in symbol_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['probability_smooth'],
                    mode='lines',
                    name='Probability',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(
                y=self.config.enter_threshold,
                line_dash="dash",
                line_color="green",
                annotation_text="Enter",
                row=2, col=1
            )
            fig.add_hline(
                y=self.config.exit_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Exit",
                row=2, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=symbol_df['timestamp'],
                y=symbol_df['volume'],
                name='Volume',
                marker_color='gray'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Trading Analysis: {symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update x-axis
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_performance_metrics(self,
                                backtest_results: dict,
                                save_path: str = None) -> go.Figure:
        """
        Plot comprehensive performance metrics.
        
        Args:
            backtest_results: Dictionary with backtest results
            save_path: Optional path to save figure
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 
                          'Return Distribution',
                          'Drawdown', 
                          'Rolling Sharpe Ratio'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        df = backtest_results['data']
        
        # 1. Cumulative Returns
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['cumulative_net'],
                    mode='lines',
                    name=f'{symbol} Strategy',
                    legendgroup=symbol
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['cumulative_market'],
                    mode='lines',
                    name=f'{symbol} Buy&Hold',
                    line=dict(dash='dash'),
                    legendgroup=symbol
                ),
                row=1, col=1
            )
        
        # 2. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=df['net_return'],
                name='Strategy Returns',
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(
                x=df['market_return'],
                name='Market Returns',
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # 3. Drawdown
        cumulative = df.groupby('symbol')['net_return'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.abs()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=drawdown,
                mode='lines',
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # 4. Rolling Sharpe Ratio
        rolling_sharpe = []
        window = 24 * 7  # Weekly for hourly data
        
        for i in range(window, len(df)):
            window_returns = df['net_return'].iloc[i-window:i]
            if window_returns.std() > 0:
                sharpe = np.sqrt(252) * window_returns.mean() / window_returns.std()
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'].iloc[window:],
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Performance Metrics Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Return", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: str = None) -> go.Figure:
        """
        Plot feature importance from model.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save_path: Optional path to save figure
        
        Returns:
            Plotly figure object
        """
        top_features = importance_df.head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=top_features['importance_normalized'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance_normalized'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                )
            )
        )
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance',
            xaxis_title='Normalized Importance',
            yaxis_title='Feature',
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_sentiment_analysis(self,
                              df: pd.DataFrame,
                              symbol: str,
                              save_path: str = None) -> go.Figure:
        """
        Plot sentiment analysis over time.
        
        Args:
            df: DataFrame with sentiment data
            symbol: Symbol to analyze
            save_path: Optional path to save figure
        
        Returns:
            Plotly figure object
        """
        symbol_df = df[df['symbol'] == symbol].copy()
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Sentiment Score', 
                          'Post Volume & Engagement',
                          'Sentiment vs Price Correlation'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Sentiment scores
        if 'sentiment_mean' in symbol_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['sentiment_mean'],
                    mode='lines',
                    name='Sentiment',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add positive/negative areas
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['sentiment_mean'].clip(lower=0),
                    fill='tozeroy',
                    mode='none',
                    name='Positive',
                    fillcolor='rgba(0,255,0,0.3)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['sentiment_mean'].clip(upper=0),
                    fill='tozeroy',
                    mode='none',
                    name='Negative',
                    fillcolor='rgba(255,0,0,0.3)'
                ),
                row=1, col=1
            )
        
        # Post volume and engagement
        if 'post_count' in symbol_df.columns:
            fig.add_trace(
                go.Bar(
                    x=symbol_df['timestamp'],
                    y=symbol_df['post_count'],
                    name='Post Count',
                    marker_color='lightblue',
                    yaxis='y'
                ),
                row=2, col=1
            )
        
        if 'avg_engagement_rate' in symbol_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=symbol_df['avg_engagement_rate'],
                    mode='lines',
                    name='Engagement Rate',
                    line=dict(color='orange', width=2),
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # Rolling correlation between sentiment and returns
        if 'sentiment_mean' in symbol_df.columns and 'return_1' in symbol_df.columns:
            rolling_corr = symbol_df['sentiment_mean'].rolling(24).corr(
                symbol_df['return_1'].shift(-1)
            )
            
            fig.add_trace(
                go.Scatter(
                    x=symbol_df['timestamp'],
                    y=rolling_corr,
                    mode='lines',
                    name='Sentiment-Return Correlation',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Sentiment Analysis: {symbol}',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Post Count", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_risk_metrics(self,
                         backtest_results: dict,
                         save_path: str = None) -> go.Figure:
        """
        Plot risk analysis metrics.
        
        Args:
            backtest_results: Backtest results dictionary
            save_path: Optional path to save figure
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk (VaR)',
                          'Return vs Risk by Symbol',
                          'Win Rate Analysis',
                          'Trade Duration Distribution'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                  [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        df = backtest_results['data']
        
        # 1. Value at Risk histogram
        returns = df['net_return'].dropna()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR 95%: {var_95:.3%}",
            row=1, col=1
        )
        
        fig.add_vline(
            x=cvar_95,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"CVaR 95%: {cvar_95:.3%}",
            row=1, col=1
        )
        
        # 2. Return vs Risk scatter
        by_symbol = backtest_results['by_symbol']
        symbols = []
        returns_list = []
        risks = []
        sharpes = []
        
        for symbol, metrics in by_symbol.items():
            symbols.append(symbol)
            returns_list.append(metrics['total_return'])
            # Use downside deviation as risk measure
            symbol_returns = df[df['symbol'] == symbol]['net_return']
            downside = symbol_returns[symbol_returns < 0].std()
            risks.append(downside)
            sharpes.append(metrics['sharpe_ratio'])
        
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns_list,
                mode='markers+text',
                text=symbols,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=sharpes,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sharpe")
                ),
                name='Symbols'
            ),
            row=1, col=2
        )
        
        # 3. Win rate by symbol
        win_rates = [by_symbol[s]['win_rate'] for s in symbols]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=win_rates,
                name='Win Rate',
                marker_color=win_rates,
                marker_colorscale='RdYlGn',
                text=[f"{wr:.1%}" for wr in win_rates],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 4. Trade duration distribution
        trades_df = df[df['trade'] == 1]
        if not trades_df.empty:
            # Calculate trade durations (simplified)
            durations = []
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol]
                positions = df[df['symbol'] == symbol]['position'].values
                
                for idx in symbol_trades.index:
                    start_idx = df.index.get_loc(idx)
                    duration = 1
                    for i in range(start_idx + 1, min(start_idx + 100, len(positions))):
                        if positions[i] == 0:
                            break
                        duration += 1
                    durations.append(duration)
            
            fig.add_trace(
                go.Histogram(
                    x=durations,
                    nbinsx=20,
                    name='Trade Duration',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Risk Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_xaxes(title_text="Risk (Downside Dev)", row=1, col=2)
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_xaxes(title_text="Duration (bars)", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Total Return", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_summary_report(self,
                            dataset: pd.DataFrame,
                            backtest_results: dict,
                            signals: pd.DataFrame,
                            save_path: str = None) -> str:
        """
        Create a comprehensive HTML summary report.
        
        Args:
            dataset: Feature dataset
            backtest_results: Backtest results
            signals: Current trading signals
            save_path: Path to save HTML report
        
        Returns:
            HTML string
        """
        html_parts = []
        
        # Header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading System Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #3498db; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                .positive { color: #27ae60; }
                .negative { color: #e74c3c; }
                .signal-buy { background-color: #d4edda; }
                .signal-sell { background-color: #f8d7da; }
                .signal-hold { background-color: #fff3cd; }
            </style>
        </head>
        <body>
        """)
        
        # Title and timestamp
        html_parts.append(f"""
        <h1>Crypto Trading System Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """)
        
        # Executive Summary
        overall = backtest_results['overall']
        html_parts.append("""
        <h2>Executive Summary</h2>
        <div class="metrics">
        """)
        
        metrics = [
            ('Total Return', f"{overall['total_return']:.2%}", 
             'positive' if overall['total_return'] > 0 else 'negative'),
            ('Sharpe Ratio', f"{overall['sharpe_ratio']:.2f}", 
             'positive' if overall['sharpe_ratio'] > 0 else 'negative'),
            ('Win Rate', f"{overall['win_rate']:.1%}", 
             'positive' if overall['win_rate'] > 0.5 else 'negative'),
            ('Total Trades', f"{overall['n_trades']}", 'neutral'),
            ('Profit Factor', f"{overall['profit_factor']:.2f}", 
             'positive' if overall['profit_factor'] > 1 else 'negative'),
        ]
        
        for label, value, class_name in metrics:
            html_parts.append(f"""
            <div class="metric">
                <div class="metric-label">{label}</div>
                <div class="metric-value {class_name}">{value}</div>
            </div>
            """)
        
        html_parts.append("</div>")
        
        # Current Signals
        html_parts.append("""
        <h2>Current Trading Signals</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Action</th>
                <th>Confidence</th>
                <th>Probability</th>
                <th>Current Price</th>
                <th>24h Change</th>
            </tr>
        """)
        
        for idx, row in signals.iterrows():
            action_class = f"signal-{row['action'].lower()}"
            price_class = 'positive' if row.get('change_24h', 0) > 0 else 'negative'
            
            html_parts.append(f"""
            <tr class="{action_class}">
                <td>{idx}</td>
                <td><strong>{row['action']}</strong></td>
                <td>{row['confidence']:.1%}</td>
                <td>{row['probability_smooth']:.3f}</td>
                <td>${row.get('price', 'N/A')}</td>
                <td class="{price_class}">{row.get('change_24h', 0):.2f}%</td>
            </tr>
            """)
        
        html_parts.append("</table>")
        
        # Performance by Symbol
        html_parts.append("""
        <h2>Performance by Symbol</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Win Rate</th>
                <th>Number of Trades</th>
            </tr>
        """)
        
        for symbol, metrics in backtest_results['by_symbol'].items():
            return_class = 'positive' if metrics['total_return'] > 0 else 'negative'
            
            html_parts.append(f"""
            <tr>
                <td>{symbol}</td>
                <td class="{return_class}">{metrics['total_return']:.2%}</td>
                <td>{metrics['sharpe_ratio']:.2f}</td>
                <td>{metrics['win_rate']:.1%}</td>
                <td>{metrics['n_trades']}</td>
            </tr>
            """)
        
        html_parts.append("</table>")
        
        # Risk Metrics
        risk = backtest_results['risk']
        html_parts.append(f"""
        <h2>Risk Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Maximum Drawdown</td><td class="negative">{risk['max_drawdown']:.2%}</td></tr>
            <tr><td>Value at Risk (95%)</td><td>{risk['var_95']:.3%}</td></tr>
            <tr><td>Conditional VaR (95%)</td><td>{risk['cvar_95']:.3%}</td></tr>
            <tr><td>Return Volatility</td><td>{risk['return_volatility']:.3%}</td></tr>
            <tr><td>Downside Deviation</td><td>{risk['downside_deviation']:.3%}</td></tr>
        </table>
        """)
        
        # Dataset Information
        html_parts.append(f"""
        <h2>Dataset Information</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            <tr><td>Total Samples</td><td>{len(dataset)}</td></tr>
            <tr><td>Number of Features</td><td>{len(dataset.columns)}</td></tr>
            <tr><td>Date Range</td><td>{dataset['timestamp'].min()} to {dataset['timestamp'].max()}</td></tr>
            <tr><td>Symbols Analyzed</td><td>{', '.join(dataset['symbol'].unique())}</td></tr>
        </table>
        """)
        
        # Footer
        html_parts.append("""
        </body>
        </html>
        """)
        
        html_content = '\n'.join(html_parts)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
        
        return html_content