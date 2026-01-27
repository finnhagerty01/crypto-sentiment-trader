import os
import logging
from binance.client import Client
from binance.enums import *

logger = logging.getLogger(__name__)

class BinanceExecutor:
    """
    Strictly for EXECUTING trades. 
    """
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key:
            raise ValueError("Binance Credentials Missing")
        
        self.client = Client(api_key, api_secret, tld='us') 
        
    def get_price(self, symbol: str) -> float:
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def get_token_balance(self, symbol: str) -> float:
        """Extracts asset name (e.g. BTC) from symbol (BTCUSDT) and gets free balance."""
        asset = symbol.replace("USDT", "").replace("USD", "")
        try:
            balance_info = self.client.get_asset_balance(asset=asset)
            return float(balance_info['free'])
        except Exception as e:
            logger.error(f"Could not fetch balance for {asset}: {e}")
            return 0.0

    def execute_order(self, symbol: str, side: str, quantity: float = None):
        """
        Executes a MARKET order. 
        If side is SELL and quantity is None, sells entire available balance.
        """
        try:
            # Auto-calculate sell amount if not provided
            if side == "SELL" and quantity is None:
                qty = self.get_token_balance(symbol)
                # Simple Dust Check (Value < $10)
                price = self.get_price(symbol)
                if (qty * price) < 10.0:
                    logger.info(f"Skipping SELL for {symbol}: Value under $10 (Dust).")
                    return None
                quantity = qty

            logger.info(f"EXECUTING {side} {quantity} {symbol}")
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            return order
        except Exception as e:
            logger.critical(f"ORDER FAILED: {e}")
            return None