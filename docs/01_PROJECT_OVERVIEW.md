# Crypto Sentiment Trader - Project Improvement Overview

## Current Architecture Analysis

### What Exists
The project has a functional end-to-end pipeline for crypto sentiment trading:

```
Reddit Posts → VADER Sentiment → Feature Engineering → Random Forest → Buy/Hold Signals → Binance Execution
     ↑                                    ↑
Binance OHLCV ────────────────────────────┘
```

### Current Feature Set (~9 features)
```python
# From models.py - prepare_features()
sentiment_lag_1, sentiment_lag_2, sentiment_lag_3     # Past sentiment
return_lag_1, return_lag_2, return_lag_3             # Past returns  
volume_lag_1, volume_lag_2, volume_lag_3             # Post volume lags
```

### Critical Gaps Identified

1. **No Technical Indicators** - Missing RSI, MACD, Bollinger Bands, ATR
2. **Limited Sentiment Features** - Only mean sentiment, missing velocity/dispersion
3. **No Volatility Modeling** - No realized vol, volatility regime detection
4. **No Cross-Asset Features** - Missing BTC dominance, correlations
5. **No On-Chain Data** - Missing exchange flows, whale tracking
6. **Basic Model** - Random Forest only, no ensemble or deep learning
7. **No Risk Management** - Fixed $100 positions, no stops, no portfolio-level risk
8. **No Backtesting Framework** - Can't validate strategy before deployment

---

## Priority Improvement Roadmap

### Phase 1: Feature Engineering (High Impact)
Add 40+ features across these categories:
- Technical indicators (15+ features)
- Enhanced sentiment features (10+ features)
- Volatility features (8+ features)
- Time-based features (5+ features)
- Cross-asset features (5+ features)

### Phase 2: Model Architecture (Medium Impact)
- Implement proper train/validation/test splits with walk-forward
- Add XGBoost and LightGBM (already in requirements.txt!)
- Implement ensemble voting
- Add proper hyperparameter tuning

### Phase 3: Risk Management (Critical for Live)
- Position sizing (Kelly Criterion, volatility-adjusted)
- Stop-loss and take-profit logic
- Portfolio-level exposure limits
- Drawdown controls

### Phase 4: Infrastructure (Production Ready)
- Proper backtesting framework
- Performance attribution
- Real-time monitoring dashboard (Streamlit exists!)
- Alert system for anomalies

---

## File Structure for New Features

```
src/
├── features/
│   ├── __init__.py
│   ├── technical.py          # RSI, MACD, Bollinger, etc.
│   ├── sentiment_advanced.py # Velocity, dispersion, engagement-weighted
│   ├── volatility.py         # Realized vol, regimes, ATR
│   ├── cross_asset.py        # BTC dominance, correlations
│   ├── time_features.py      # Hour, day, session
│   └── feature_pipeline.py   # Orchestrates all feature creation
├── models/
│   ├── __init__.py
│   ├── base.py               # Abstract base class
│   ├── random_forest.py      # Current model, refactored
│   ├── xgboost_model.py      # XGBoost implementation
│   ├── lightgbm_model.py     # LightGBM implementation
│   ├── ensemble.py           # Voting/stacking ensemble
│   └── hyperparameter.py     # Tuning utilities
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py    # Kelly, volatility-adjusted
│   ├── stop_loss.py          # Trailing stops, ATR-based
│   └── portfolio.py          # Exposure limits, correlation risk
├── backtest/
│   ├── __init__.py
│   ├── engine.py             # Walk-forward backtesting
│   ├── metrics.py            # Sharpe, Sortino, max drawdown
│   └── report.py             # Performance reporting
└── data/
    ├── onchain_client.py     # Future: Glassnode/CryptoQuant
    └── alternative_data.py   # Future: Google Trends, Fear & Greed
```

---

## Quick Wins (Implement First)

1. **Add RSI and MACD** - Proven indicators, easy to implement
2. **Sentiment velocity** - Rate of change in sentiment (already have lags!)
3. **Volatility features** - Realized vol from returns
4. **Hour-of-day features** - Crypto has clear intraday patterns
5. **Use XGBoost** - Already in requirements, likely better than RF

---

## Documents in This Package

| Document | Purpose |
|----------|---------|
| `01_PROJECT_OVERVIEW.md` | This file - high-level roadmap |
| `02_FEATURE_ENGINEERING.md` | Detailed feature specifications |
| `03_TECHNICAL_INDICATORS.md` | Implementation guide for TA |
| `04_SENTIMENT_ADVANCED.md` | Enhanced NLP/sentiment features |
| `05_MODEL_IMPROVEMENTS.md` | ML architecture upgrades |
| `06_RISK_MANAGEMENT.md` | Position sizing and risk controls |
| `07_BACKTESTING.md` | Backtesting framework design |
| `08_IMPLEMENTATION_TASKS.md` | Prioritized task list for Claude Code |
