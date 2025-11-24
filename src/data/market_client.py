import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

class MarketClient:
    def __init__(self, config):
        self.client = Client(tld='us') # Public data doesn't need API keys usually
        self.symbols = config.symbols

    def fetch_ohlcv(self, lookback_days: int = 30, interval: str = Client.KLINE_INTERVAL_1HOUR) -> pd.DataFrame:
        """
        Fetches historical data for all symbols.
        """
        start_str = f"{lookback_days} days ago UTC"
        all_dfs = []

        for symbol in self.symbols:
            try:
                klines = self.client.get_historical_klines(symbol, interval, start_str)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
                ])
                
                # Clean Types
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                all_dfs.append(df)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs).sort_values(['timestamp', 'symbol'])