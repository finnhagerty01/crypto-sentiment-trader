# Backtest Results Report

**Date:** January 30, 2026
**Test Period:** January 23, 2026 - January 30, 2026 (7 days)
**Initial Capital:** $10,000

---

## Executive Summary

The backtesting framework was executed against real market and sentiment data. The trading model generated **zero trades** during the test period, as it consistently predicted bearish conditions (SELL signals) while holding no positions to sell.

Paradoxically, this conservative stance **preserved capital** during a period when buy-and-hold strategies suffered 5-7% losses.

---

## Backtest Configuration

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000 |
| Fee Rate | 0.01% per side |
| Slippage Rate | 0.05% per side |
| Max Positions | 5 |
| Max Position Size | 15% of portfolio |
| Max Total Exposure | 50% of portfolio |
| Stop-Loss | 3% |
| Take-Profit | 6% |
| Training Data | 60% (first 301 samples) |
| Test Data | 40% (remaining 201 samples) |

---

## Data Summary

### Market Data
- **Total rows:** 23,749
- **Lookback period:** 90 days
- **Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, AVAXUSDT, DOGEUSDT, LINKUSDT, MATICUSDT, DOTUSDT, ATOMUSDT

### Sentiment Data
- **Total Reddit posts:** 5,004
- **Source:** master_reddit.csv

### Feature Engineering
- **Total features:** 61
  - Lag features: 24
  - Technical indicators: 21
  - Sentiment features: 16
- **Feature rows after preparation:** 502

---

## Training Results

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Negative (No Buy) | 269 | 89.4% |
| Positive (Buy) | 32 | 10.6% |

**Observation:** Severe class imbalance. Only ~11% of historical hours showed >0.5% positive returns (the threshold for a "buy" signal).

### Validation
- **Walk-forward validation:** Could not run (dataset too small)
- **Minimum required:** 668 samples
- **Available:** 301 samples

The ensemble model was trained with class weighting (scale_pos_weight=8.41) to handle imbalance.

---

## Backtest Results

### Trading Activity

| Metric | Value |
|--------|-------|
| Total Trades | 0 |
| Buy Signals Generated | 0 |
| Sell Signals Generated | 118 |
| Final Portfolio Value | $10,000 |
| Total Return | 0.00% |

### Why No Trades?

The model consistently predicted **low buy probabilities** (typically 0.01-0.25), resulting in:
- All signals classified as "SELL" (probability < 0.45 exit threshold)
- Since the portfolio started with cash only, SELL signals had nothing to close
- The model never reached the "BUY" threshold (probability >= 0.55)

Sample predictions from the test period:
```
BTCUSDT: SELL (conf=0.97, prob=0.03)
ETHUSDT: SELL (conf=0.89, prob=0.11)
BTCUSDT: SELL (conf=0.95, prob=0.05)
...
```

---

## Benchmark Comparison

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Final Value |
|----------|--------------|--------------|--------------|-------------|
| **Trading Model** | **0.00%** | **N/A** | **0.00%** | **$10,000.00** |
| Buy & Hold BTC | -6.50% | -2.28 | 15.71% | $9,349.89 |
| Buy & Hold ETH | -5.59% | -2.47 | 17.96% | $9,440.69 |
| Equal Weight Portfolio | -4.80% | -5.64 | 23.23% | $9,519.93 |

### Interpretation

The test period (Jan 23-30, 2026) was **bearish**:
- BTC declined ~6.5%
- ETH declined ~5.6%
- All benchmark strategies lost money

**The model's bearish stance, while producing no trades, effectively preserved capital.** Staying in cash during a downturn is a valid outcome, though not the intended behavior.

---

## Analysis

### Model Behavior

1. **Consistently Bearish Predictions**
   - Buy probabilities rarely exceeded 0.25
   - Model appears to have learned that positive returns (>0.5%) are rare
   - This matches the training data where only 11% of samples were positive

2. **Class Imbalance Effect**
   - Despite class weighting, the model remains conservative
   - In crypto markets, staying out during uncertain periods can be protective

3. **Threshold Sensitivity**
   - Current thresholds: BUY >= 0.55, SELL <= 0.45
   - Lowering the BUY threshold would generate more trades but potentially worse performance

### Data Limitations

1. **Insufficient Training Data**
   - Only 502 feature rows available
   - Walk-forward validation requires 668+ samples
   - More historical data needed for robust model training

2. **Sentiment Coverage**
   - 5,004 Reddit posts over 90 days
   - Consider adding more sentiment sources (Twitter, news)

---

## Recommendations

### Short-Term Improvements

1. **Increase Data Collection**
   - Run data collection for longer periods
   - Target minimum 30 days of hourly data (720+ feature rows)
   - Add alternative sentiment sources

2. **Adjust Thresholds**
   - Consider lowering BUY threshold to 0.50 during testing
   - Implement adaptive thresholds based on market regime

3. **Add Market Regime Detection**
   - Implement volatility regime indicators
   - Allow different trading behavior in bullish vs bearish markets

### Medium-Term Improvements

1. **Ensemble Calibration**
   - The current calibration may be too conservative
   - Experiment with different calibration methods

2. **Alternative Target Definition**
   - Current target: >0.5% return in next hour
   - Consider: >0% return (more balanced classes)
   - Or: Multi-class (strong sell, sell, hold, buy, strong buy)

3. **Feature Engineering**
   - Add momentum features that may predict trend continuation
   - Include market microstructure features

---

## Generated Files

| File | Description |
|------|-------------|
| `reports/backtest_report.png` | Equity curve and performance charts |
| `reports/strategy_comparison.csv` | Benchmark comparison data |

---

## Conclusion

The backtest successfully validated the backtesting framework's functionality. While the model generated no trades due to consistently bearish predictions, this behavior **accidentally preserved capital** during a market downturn.

The key issue is **insufficient training data** causing the model to be overly conservative. With more historical data and potential threshold adjustments, the system should generate meaningful trading signals.

**Next Steps:**
1. Collect more historical market and sentiment data
2. Re-run backtest with 180+ days of data
3. Experiment with lower BUY thresholds
4. Consider alternative target definitions

---

*Report generated by the Crypto Sentiment Trader Backtesting Framework*
