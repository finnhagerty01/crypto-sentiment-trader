# src/visualization/streamlit_app.py
"""
Streamlit web application for real-time monitoring.
Run with: streamlit run src/visualization/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import TradingConfig
from src.visualization.dashboard import TradingDashboard


try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    
    def run_dashboard():
        """Run Streamlit dashboard application."""
        st.set_page_config(
            page_title="Crypto Trading Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("🚀 Crypto Sentiment Trading Dashboard")
        
        # Sidebar for configuration
        st.sidebar.header("Configuration")
        
        interval = st.sidebar.selectbox(
            "Time Interval",
            options=['1h', '4h'],
            index=0
        )
        
        refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)",
            min_value=10,
            max_value=300,
            value=60
        )
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Signals", 
            "📈 Performance", 
            "🎯 Risk Analysis",
            "🔍 Feature Analysis"
        ])
        
        # Load data
        @st.cache_data(ttl=refresh_rate)
        def load_latest_data():
            """Load latest data from files."""
            # This would load from your actual data files
            config = TradingConfig.from_yaml()
            
            # Load latest predictions
            pred_file = config.processed_dir / f"preds_{interval}.csv.gz"
            if pred_file.exists():
                return pd.read_csv(pred_file, parse_dates=['timestamp'])
            return pd.DataFrame()
        
        @st.cache_data(ttl=3600)
        def load_backtest_results():
            """Load latest backtest results."""
            # Load from latest backtest file
            config = TradingConfig.from_yaml()
            files = list(config.processed_dir.glob("backtest_results_*.csv"))
            if files:
                latest = max(files, key=lambda x: x.stat().st_mtime)
                return pd.read_csv(latest, parse_dates=['timestamp'])
            return pd.DataFrame()
        
        data = load_latest_data()
        backtest_data = load_backtest_results()
        
        with tab1:
            st.header("Current Trading Signals")
            
            if not data.empty:
                # Get latest signals
                latest_signals = data.groupby('symbol').last()
                
                # Create signal cards
                cols = st.columns(3)
                for idx, (symbol, row) in enumerate(latest_signals.iterrows()):
                    col = cols[idx % 3]
                    
                    with col:
                        # Determine action
                        if row.get('probability_smooth', 0.5) > 0.6:
                            action = "BUY"
                            color = "green"
                        elif row.get('probability_smooth', 0.5) < 0.55:
                            action = "SELL"
                            color = "red"
                        else:
                            action = "HOLD"
                            color = "orange"
                        
                        st.metric(
                            label=symbol,
                            value=action,
                            delta=f"{row.get('probability_smooth', 0.5):.3f}"
                        )
                        
                        # Mini chart
                        symbol_data = data[data['symbol'] == symbol].tail(24)
                        if not symbol_data.empty:
                            st.line_chart(
                                symbol_data.set_index('timestamp')['probability_smooth'],
                                height=100
                            )
            else:
                st.warning("No data available")
        
        with tab2:
            st.header("Performance Metrics")
            
            if not backtest_data.empty:
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_return = backtest_data['net_return'].sum()
                sharpe = np.sqrt(252) * backtest_data['net_return'].mean() / backtest_data['net_return'].std()
                win_rate = (backtest_data['gross_return'] > 0).mean()
                max_dd = (backtest_data['cumulative_net'] - backtest_data['cumulative_net'].expanding().max()).min()
                
                col1.metric("Total Return", f"{total_return:.2%}")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col3.metric("Win Rate", f"{win_rate:.1%}")
                col4.metric("Max Drawdown", f"{max_dd:.2%}")
                
                # Cumulative returns chart
                st.subheader("Cumulative Returns")
                returns_data = backtest_data.pivot_table(
                    index='timestamp',
                    columns='symbol',
                    values='cumulative_net',
                    aggfunc='mean'
                )
                st.line_chart(returns_data)
                
                # Return distribution
                st.subheader("Return Distribution")
                fig = px.histogram(
                    backtest_data,
                    x='net_return',
                    nbins=50,
                    title="Strategy Return Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Risk Analysis")
            
            if not backtest_data.empty:
                # Risk metrics table
                risk_metrics = {
                    "Value at Risk (95%)": backtest_data['net_return'].quantile(0.05),
                    "Conditional VaR (95%)": backtest_data[backtest_data['net_return'] <= backtest_data['net_return'].quantile(0.05)]['net_return'].mean(),
                    "Return Volatility": backtest_data['net_return'].std(),
                    "Downside Deviation": backtest_data[backtest_data['net_return'] < 0]['net_return'].std(),
                    "Maximum Consecutive Losses": 0  # Would calculate properly
                }
                
                risk_df = pd.DataFrame(
                    list(risk_metrics.items()),
                    columns=['Metric', 'Value']
                )
                risk_df['Value'] = risk_df['Value'].apply(lambda x: f"{x:.4f}")
                
                st.table(risk_df)
                
                # Drawdown chart
                st.subheader("Drawdown Analysis")
                cumulative = backtest_data.groupby('timestamp')['net_return'].sum().cumsum()
                drawdown = cumulative - cumulative.expanding().max()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='red')
                ))
                fig.update_layout(title="Portfolio Drawdown", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Feature Analysis")
            
            # Load feature importance if available
            config = TradingConfig.from_yaml()
            importance_files = list(config.models_dir.glob("*/feature_importance.csv"))
            
            if importance_files:
                latest_importance = max(importance_files, key=lambda x: x.stat().st_mtime)
                importance_df = pd.read_csv(latest_importance)
                
                st.subheader("Feature Importance")
                
                # Top features bar chart
                top_features = importance_df.head(20)
                fig = px.bar(
                    top_features,
                    x='importance_normalized',
                    y='feature',
                    orientation='h',
                    title="Top 20 Most Important Features"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature categories
                st.subheader("Feature Categories")
                
                sentiment_features = importance_df[importance_df['feature'].str.contains('sentiment|vader|post')]
                market_features = importance_df[importance_df['feature'].str.contains('return|volume|volatility|momentum')]
                technical_features = importance_df[importance_df['feature'].str.contains('rsi|macd|bb_')]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentiment Features",
                        f"{sentiment_features['importance_normalized'].sum():.1%}"
                    )
                
                with col2:
                    st.metric(
                        "Market Features",
                        f"{market_features['importance_normalized'].sum():.1%}"
                    )
                
                with col3:
                    st.metric(
                        "Technical Features",
                        f"{technical_features['importance_normalized'].sum():.1%}"
                    )
        
        # Auto-refresh
        if st.sidebar.button("Refresh Now"):
            st.experimental_rerun()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            "💡 **Tips:**\n"
            "- Green signals indicate BUY\n"
            "- Red signals indicate SELL\n"
            "- Orange signals indicate HOLD\n"
            "- Check Risk Analysis before trading"
        )
    
    if __name__ == "__main__":
        run_dashboard()