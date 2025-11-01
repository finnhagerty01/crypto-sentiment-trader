# src/trading/live_trader.py
"""
Live trading execution module with safety features and monitoring.
Handles real-time order placement and position management.
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
from binance.client import Client
from binance.exceptions import BinanceAPIException
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """
    Production-ready live trading engine with safety features.
    
    Features:
    1. Real-time order execution
    2. Position management
    3. Risk limits and safety checks
    4. Order tracking and logging
    5. Emergency stop functionality
    """
    
    def __init__(self, config, test_mode: bool = True):
        """
        Initialize trading engine.
        
        Args:
            config: Trading configuration
            test_mode: If True, simulate orders without executing
        """
        self.config = config
        self.test_mode = test_mode
        self.client = None
        self.positions = {}
        self.orders = []
        self.balance = {}
        self.trading_enabled = True
        self.max_position_size = 0.1  # Max 10% of portfolio per position
        self.max_total_exposure = 0.5  # Max 50% total exposure
        
        if not test_mode:
            self._initialize_client()
            self._update_balances()
    
    def _initialize_client(self):
        """Initialize Binance client for live trading."""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials required for live trading")
        
        self.client = Client(api_key, api_secret)
        
        # Test connection
        try:
            self.client.get_account_status()
            logger.info("Connected to Binance for live trading")
        except BinanceAPIException as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise
    
    def _update_balances(self):
        """Update account balances."""
        if self.test_mode:
            # Simulated balance for testing
            self.balance = {
                'USDT': 10000.0,
                'BTC': 0.0,
                'ETH': 0.0
            }
        else:
            account = self.client.get_account()
            self.balance = {
                asset['asset']: float(asset['free'])
                for asset in account['balances']
                if float(asset['free']) > 0
            }
    
    def execute_signals(self, signals: pd.DataFrame) -> Dict:
        """
        Execute trading signals with safety checks.
        
        Args:
            signals: DataFrame with trading signals
        
        Returns:
            Dictionary with execution results
        """
        if not self.trading_enabled:
            logger.warning("Trading is disabled")
            return {'status': 'disabled', 'orders': []}
        
        results = {
            'timestamp': datetime.now(),
            'orders': [],
            'errors': []
        }
        
        for symbol, signal in signals.iterrows():
            try:
                # Safety checks
                if not self._check_trading_conditions(symbol, signal):
                    continue
                
                # Determine order parameters
                order_params = self._calculate_order_params(symbol, signal)
                
                if order_params:
                    # Execute order
                    order_result = self._place_order(order_params)
                    results['orders'].append(order_result)
                    
            except Exception as e:
                logger.error(f"Error executing signal for {symbol}: {e}")
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Log results
        self._log_execution_results(results)
        
        return results
    
    def _check_trading_conditions(self, symbol: str, signal: Dict) -> bool:
        """
        Perform safety checks before trading.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
        
        Returns:
            True if safe to trade
        """
        # Check confidence threshold
        confidence = signal.get('confidence', 0)
        if confidence < 0.7:
            logger.info(f"Skipping {symbol}: Low confidence {confidence:.2%}")
            return False
        
        # Check maximum exposure
        current_exposure = self._calculate_total_exposure()
        if current_exposure >= self.max_total_exposure:
            logger.warning(f"Maximum exposure reached: {current_exposure:.2%}")
            return False
        
        # Check if already in position
        if symbol in self.positions and self.positions[symbol]['size'] > 0:
            if signal['action'] == 'BUY':
                logger.info(f"Already in position for {symbol}")
                return False
        
        # Check recent trade frequency (avoid overtrading)
        recent_trades = self._get_recent_trades(symbol, hours=1)
        if len(recent_trades) >= 3:
            logger.warning(f"Too many recent trades for {symbol}")
            return False
        
        return True
    
    def _calculate_order_params(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """
        Calculate order parameters based on signal and risk management.
        
        Args:
            symbol: Trading symbol
            signal: Signal dictionary
        
        Returns:
            Order parameters or None
        """
        action = signal['action']
        
        if action == 'HOLD':
            return None
        
        # Get current price
        if self.test_mode:
            price = signal.get('price', 50000)  # Default for testing
        else:
            ticker = self.client.get_ticker(symbol=symbol)
            price = float(ticker['lastPrice'])
        
        # Calculate position size based on Kelly Criterion or fixed percentage
        position_value = self._calculate_position_size(
            symbol,
            signal['probability_smooth'],
            signal.get('confidence', 0.5)
        )
        
        # Convert to quantity
        quantity = position_value / price
        
        # Get symbol info for precision
        if not self.test_mode:
            info = self.client.get_symbol_info(symbol)
            quantity = self._round_quantity(quantity, info)
        
        # Determine order type
        if action == 'BUY':
            side = 'BUY'
            # Set stop loss and take profit
            stop_loss = price * 0.98  # 2% stop loss
            take_profit = price * 1.03  # 3% take profit
        else:  # SELL
            side = 'SELL'
            # Check if we have position to sell
            if symbol not in self.positions or self.positions[symbol]['size'] == 0:
                logger.info(f"No position to sell for {symbol}")
                return None
            quantity = self.positions[symbol]['size']
            stop_loss = None
            take_profit = None
        
        return {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',  # Use market orders for immediate execution
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'signal_probability': signal['probability_smooth'],
            'signal_confidence': signal.get('confidence', 0)
        }
    
    def _calculate_position_size(self, symbol: str, probability: float, 
                            confidence: float,
                            recent_returns: Optional[pd.Series] = None) -> float:
        """
        Calculate position size using Kelly Criterion with actual edge estimation.
        
        Args:
            symbol: Trading symbol
            probability: Win probability (MUST be calibrated!)
            confidence: Signal confidence
            recent_returns: Recent returns for estimating win/loss ratio
        """
        # Estimate win/loss ratio from recent performance, not hardcoded
        if recent_returns is not None and len(recent_returns) > 20:
            winning_returns = recent_returns[recent_returns > 0]
            losing_returns = recent_returns[recent_returns < 0]
            
            if len(winning_returns) > 0 and len(losing_returns) > 0:
                avg_win = winning_returns.mean()
                avg_loss = abs(losing_returns.mean())
                b = avg_win / avg_loss if avg_loss > 0 else 1.5
            else:
                b = 1.5  # Fallback
        else:
            b = 1.5  # Fallback for insufficient data
        
        # Kelly fraction: f = (p*b - q) / b
        p = probability
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Fractional Kelly (use 25% of Kelly for safety)
        kelly_fraction *= 0.25
        
        # Apply confidence scaling
        kelly_fraction *= confidence
        
        # Apply maximum position size limit
        kelly_fraction = min(kelly_fraction, self.max_position_size)
        
        # Ensure positive and reasonable
        kelly_fraction = max(0, min(kelly_fraction, 0.10))  # Max 10% per trade with fractional Kelly
        
        # Calculate position value
        available_balance = self.balance.get('USDT', 0)
        position_value = available_balance * kelly_fraction
        
        return position_value
    
    def _place_order(self, params: Dict) -> Dict:
        """
        Place order on exchange or simulate.
        
        Args:
            params: Order parameters
        
        Returns:
            Order result
        """
        timestamp = datetime.now()
        
        if self.test_mode:
            # Simulate order
            order_result = {
                'orderId': f"TEST_{timestamp.timestamp()}",
                'symbol': params['symbol'],
                'side': params['side'],
                'type': params['type'],
                'quantity': params['quantity'],
                'price': params['price'],
                'status': 'FILLED',
                'timestamp': timestamp,
                'test_mode': True
            }
            
            # Update simulated positions
            if params['side'] == 'BUY':
                self.positions[params['symbol']] = {
                    'size': params['quantity'],
                    'entry_price': params['price'],
                    'timestamp': timestamp
                }
            else:  # SELL
                self.positions[params['symbol']] = {
                    'size': 0,
                    'entry_price': 0,
                    'timestamp': timestamp
                }
            
        else:
            # Place real order
            try:
                if params['type'] == 'MARKET':
                    order = self.client.create_order(
                        symbol=params['symbol'],
                        side=params['side'],
                        type='MARKET',
                        quantity=params['quantity']
                    )
                else:
                    order = self.client.create_order(
                        symbol=params['symbol'],
                        side=params['side'],
                        type='LIMIT',
                        timeInForce='GTC',
                        quantity=params['quantity'],
                        price=params['price']
                    )
                
                order_result = {
                    'orderId': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': float(order['origQty']),
                    'price': float(order.get('price', params['price'])),
                    'status': order['status'],
                    'timestamp': timestamp,
                    'test_mode': False
                }
                
                # Place stop loss if needed
                if params.get('stop_loss'):
                    self._place_stop_loss(params['symbol'], params['quantity'], params['stop_loss'])
                
            except BinanceAPIException as e:
                logger.error(f"Order failed: {e}")
                raise
        
        # Record order
        self.orders.append(order_result)
        logger.info(f"Order placed: {order_result}")
        
        return order_result
    
    def _place_stop_loss(self, symbol: str, quantity: float, stop_price: float):
        """Place stop loss order."""
        try:
            if not self.test_mode:
                self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='STOP_LOSS_LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=stop_price * 0.995,  # Slightly below stop
                    stopPrice=stop_price
                )
                logger.info(f"Stop loss placed for {symbol} at {stop_price}")
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure."""
        total_value = sum(self.balance.values())
        if total_value == 0:
            return 0
        
        position_value = 0
        for symbol, position in self.positions.items():
            if position['size'] > 0:
                position_value += position['size'] * position.get('entry_price', 0)
        
        return position_value / total_value if total_value > 0 else 0
    
    def _get_recent_trades(self, symbol: str, hours: int = 1) -> List[Dict]:
        """Get recent trades for a symbol."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            order for order in self.orders
            if order['symbol'] == symbol and order['timestamp'] > cutoff
        ]
    
    def _round_quantity(self, quantity: float, symbol_info: Dict) -> float:
        """Round quantity to exchange precision."""
        step_size = None
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                break
        
        if step_size:
            precision = int(round(-np.log10(step_size)))
            return round(quantity, precision)
        
        return quantity
    
    def _log_execution_results(self, results: Dict):
        """Log execution results for audit trail."""
        log_dir = Path('logs/trades')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"trades_{datetime.now():%Y%m%d}.json"
        
        # Append to daily log
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Convert datetime to string for JSON
        results_copy = results.copy()
        results_copy['timestamp'] = results_copy['timestamp'].isoformat()
        
        logs.append(results_copy)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self.positions.items():
            if position['size'] > 0:
                # Get current price
                if self.test_mode:
                    current_price = position['entry_price'] * (1 + np.random.normal(0, 0.01))
                else:
                    ticker = self.client.get_ticker(symbol=symbol)
                    current_price = float(ticker['lastPrice'])
                
                pnl = (current_price - position['entry_price']) / position['entry_price']
                
                data.append({
                    'symbol': symbol,
                    'size': position['size'],
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'value': position['size'] * current_price,
                    'timestamp': position['timestamp']
                })
        
        return pd.DataFrame(data)
    
    def emergency_stop(self):
        """Emergency stop - close all positions and disable trading."""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.trading_enabled = False
        
        # Close all positions
        for symbol, position in self.positions.items():
            if position['size'] > 0:
                try:
                    self._place_order({
                        'symbol': symbol,
                        'side': 'SELL',
                        'type': 'MARKET',
                        'quantity': position['size'],
                        'price': 0  # Market order
                    })
                except Exception as e:
                    logger.error(f"Failed to close position {symbol}: {e}")
        
        logger.info("All positions closed. Trading disabled.")