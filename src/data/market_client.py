import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone

class MarketClient:
    def __init__(self, config):
        self.client = Client(tld='us') # Public data doesn't need API keys usually
        self.symbols = config.symbols

    def fetch_ohlcv(self, lookback_days: int = 30, interval: str = Client.KLINE_INTERVAL_1HOUR) -> pd.DataFrame:
        start_str = f"{lookback_days} days ago UTC"
        all_dfs = []

        for symbol in self.symbols:
            try:
                klines = self.client.get_historical_klines(symbol, interval, start_str)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
                ])

                # Keep close_time so we can drop incomplete bars
                df = df[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume']].copy()

                # Convert types
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                # Make timestamps UTC-naive but correctly based on UTC
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True).dt.tz_localize(None)

                # Drop incomplete / future-close bars (last closed candle only)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                df = df[df['close_time'] <= now_utc].copy()

                df['symbol'] = symbol
                all_dfs.append(df)

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs).sort_values(['timestamp', 'symbol'])