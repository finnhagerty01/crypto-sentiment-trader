import pandas as pd
import requests
from binance.client import Client
from datetime import datetime, timedelta, timezone

FUTURES_BASE = "https://fapi.binance.com"

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
                df = df[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'tb_base']].copy()

                # Convert types
                df[['open', 'high', 'low', 'close', 'volume', 'tb_base']] = df[['open', 'high', 'low', 'close', 'volume', 'tb_base']].astype(float)

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

    def fetch_funding_rates(self, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch historical funding rates for all symbols from Binance Futures."""
        all_dfs = []
        limit = min(lookback_days * 3, 1000)  # ~3 funding events per day (8h intervals)

        for symbol in self.symbols:
            try:
                resp = requests.get(
                    f"{FUTURES_BASE}/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": limit},
                )
                resp.raise_for_status()
                rates = resp.json()
                if not rates:
                    continue
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True).dt.tz_localize(None)
                df['symbol'] = symbol
                df['funding_rate'] = df['fundingRate'].astype(float)
                # Forward-fill funding rate to hourly timestamps
                df = df[['timestamp', 'symbol', 'funding_rate']].copy()
                all_dfs.append(df)
            except Exception as e:
                print(f"Funding rate fetch failed for {symbol} (may not have futures): {e}")

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs).sort_values(['timestamp', 'symbol'])
        # Resample to hourly so it merges cleanly with OHLCV data
        resampled = []
        for symbol in result['symbol'].unique():
            sym_df = result[result['symbol'] == symbol].set_index('timestamp')
            sym_df = sym_df[['funding_rate']].resample('1h').ffill()
            sym_df['symbol'] = symbol
            resampled.append(sym_df.reset_index())
        return pd.concat(resampled).sort_values(['timestamp', 'symbol'])

    def fetch_open_interest(self, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch historical open interest for all symbols from Binance Futures."""
        all_dfs = []
        limit = min(lookback_days * 24, 500)  # hourly data, API max 500

        for symbol in self.symbols:
            try:
                resp = requests.get(
                    f"{FUTURES_BASE}/futures/data/openInterestHist",
                    params={"symbol": symbol, "period": "1h", "limit": limit},
                )
                resp.raise_for_status()
                oi_data = resp.json()
                if not oi_data:
                    continue
                df = pd.DataFrame(oi_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
                df['symbol'] = symbol
                df['open_interest'] = df['sumOpenInterest'].astype(float)
                df['oi_change'] = df['open_interest'].pct_change().fillna(0)
                df = df[['timestamp', 'symbol', 'open_interest', 'oi_change']].copy()
                all_dfs.append(df)
            except Exception as e:
                print(f"Open interest fetch failed for {symbol} (may not have futures): {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs).sort_values(['timestamp', 'symbol'])