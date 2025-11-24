import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dotenv import load_dotenv

# Import your actual project modules
sys.path.append(str(Path(__file__).parent / 'src'))
from src.utils.config import TradingConfig
from src.data.reddit_client import RedditClient
from src.data.market_client import MarketClient
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.models import TradingModel

# Load API keys
load_dotenv()

def backtest_equity_curve(df, signals, initial_capital=1000, fee=0.001):
    """
    Vectorized backtest simulation.
    """
    # Align signals with actual returns
    # 'target_return' is the return of the NEXT candle.
    market_returns = df['target_return'].values
    
    # Create a strategy return array
    # If Signal=1 (BUY), we get market_return - fees.
    # If Signal=0 (HOLD), we get 0.0.
    strategy_returns = np.where(
        signals == 1,
        market_returns - (fee * 2), # Entry + Exit fee assumption
        0.0
    )
    
    # Sanity check to prevent infinite overflow from bad data
    # Cap returns at -99% and +1000% just in case
    strategy_returns = np.clip(strategy_returns, -0.99, 10.0)
    market_returns = np.clip(market_returns, -0.99, 10.0)
    
    # Calculate Cumulative Equity
    # (1 + r1) * (1 + r2) ...
    strategy_equity = initial_capital * np.cumprod(1 + strategy_returns)
    benchmark_equity = initial_capital * np.cumprod(1 + market_returns)
    
    return strategy_equity, benchmark_equity

def main():
    print("--- 1. LOADING DATA ---")
    config = TradingConfig.from_yaml("configs/data.yaml")
    
    # Fetch Data (Same as main.py)
    reddit = RedditClient(subreddits=config.subreddits)
    market = MarketClient(config)
    sentiment = SentimentAnalyzer(symbols=config.symbols)
    model = TradingModel(enter_threshold=config.enter_threshold)

    print("Fetching historical data (this may take a moment)...")
    hist_reddit = reddit.fetch_historical(days=30)
    hist_market = market.fetch_ohlcv(lookback_days=30)
    
    print("--- 2. PREPARING FEATURES ---")
    hist_sentiment = sentiment.analyze(hist_reddit)
    df = model.prepare_features(hist_market, hist_sentiment)
    
    # Replicate the Split from models.py
    X = df[model.features]
    y = df['target']
    split = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows")
    
    # Train
    model.model.fit(X_train, y_train)
    
    # Predict
    preds = model.model.predict(X_test)
    probs = model.model.predict_proba(X_test)[:, 1]
    
    print("\n--- DEBUG: PROBABILITY DISTRIBUTION ---")
    print(f"Max Probability found: {probs.max():.4f}")
    print(f"Average Probability: {probs.mean():.4f}")
    print(f"Number of probs > 0.50: {len(probs[probs > 0.50])}")
    print(f"Number of probs > 0.55: {len(probs[probs > 0.55])}")
    print("-" * 30)
    
    # Apply Threshold (Custom Logic from models.py)
    # The default .predict() uses 0.5. We want to check our specific threshold (0.60)
    custom_preds = (probs > config.enter_threshold).astype(int)
    
    print("\n--- 3. DETAILED PERFORMANCE ---")
    print(classification_report(y_test, custom_preds, target_names=['HOLD', 'BUY']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, custom_preds)
    
    # Financial Backtest
    equity, benchmark = backtest_equity_curve(df.iloc[split:], custom_preds)
    
    print(f"Final Strategy Equity: ${equity[-1]:.2f}")
    print(f"Final Buy & Hold Equity: ${benchmark[-1]:.2f}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                xticklabels=['Pred HOLD', 'Pred BUY'],
                yticklabels=['Actual HOLD', 'Actual BUY'])
    ax[0].set_title(f'Confusion Matrix (Threshold: {config.enter_threshold})')
    ax[0].set_ylabel('True Label')
    ax[0].set_xlabel('Predicted Label')
    
    # Plot 2: Equity Curve
    ax[1].plot(equity, label='Sentiment Strategy', color='green', linewidth=2)
    ax[1].plot(benchmark, label='Buy & Hold (Benchmark)', color='gray', linestyle='--')
    ax[1].set_title('Financial Backtest (Test Set)')
    ax[1].set_xlabel('Hours')
    ax[1].set_ylabel('Portfolio Value ($)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()