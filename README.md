# crypto-sentiment-trader

Research project: fuse Reddit sentiment with crypto market features to predict next-period returns and produce Buy/Hold/Sell signals. **Not financial advice.**

## Roadmap (v1)
- Ingest Reddit posts/comments for selected subreddits.
- Ingest Binance OHLCV for BTC/ETH (later more alts).
- Aggregate sentiment per bar (1h, 4h); create market features.
- Train simple models; backtest with fees/slippage.
- Report results; iterate.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt