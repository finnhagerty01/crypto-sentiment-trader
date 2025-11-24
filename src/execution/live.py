import os
import logging
from binance.client import Client
from binance.enums import *

logger = logging.getLogger(__name__)

class BinanceExecutor:
    """
    Strictly for EXECUTING trades. 
    Logic/Strategy should be calculated before calling this class.
    """
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key:
            raise ValueError("Binance Credentials Missing")
        
        self.client = Client(api_key, api_secret, tld='us') # or 'com' depending on region
        
    def get_price(self, symbol: str) -> float:
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def execute_order(self, symbol: str, side: str, quantity: float):
        """
        Executes a MARKET order. 
        Safety: Checks min_notional filters automatically via error handling.
        """
        try:
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
            # Add emergency notification logic here (e.g., Slack/Discord webhook)
            return None