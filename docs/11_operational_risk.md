# Operational Risk & Execution Fixes

**Role:** You are an expert Crypto Quant Engineer specializing in execution algorithms and system reliability.
**Context:** We have addressed the backtesting and alpha generation flaws. Now, we must address **critical operational risks** identified in the "Senior Quant" review. These are "blocking" issues—if we go live without them, the bot will likely crash or lose money due to execution errors.
**Objective:** Implement the following three fixes to ensure execution reliability and state persistence.

---

## 🛑 Priority 1: Fix "The Precision Crash" (Blocking Bug)

**The Issue:**
In `src/execution/live.py`, we are passing raw float quantities (e.g., `0.1234567`) to the Binance API. Binance enforces strict `stepSize` (tick size) and `minQty` rules. Sending a raw float will cause the API to reject the order with a `LOT_SIZE` error.

**The Fix:**
Implement a precision normalization helper that queries `exchangeInfo` and rounds quantities down to the correct step size.

**Instructions:**
1.  **Modify `src/execution/live.py`:**
    * In `__init__`, call `self.client.get_exchange_info()`.
    * Store a mapping of `symbol -> {'stepSize': float, 'minQty': float, 'minNotional': float}`.
    * Create a helper method `_normalize_quantity(self, symbol, quantity)`.
        * It should round `quantity` **down** to the nearest multiple of `stepSize`.
        * Example: If step is `0.001` and qty is `0.1239`, return `0.123`.
        * Use `decimal.Decimal` for precise math to avoid floating point errors.
    * Update `execute_order` to call `_normalize_quantity` before sending the order.
    * Add a check: if normalized quantity < `minQty` or (quantity * price) < `minNotional`, log a warning and **do not** send the order.

---

## 🧠 Priority 2: Fix "The Amnesia Problem" (State Persistence)

**The Issue:**
The `PortfolioRiskManager` in `main.py` starts fresh every time the script runs. If the bot restarts, it forgets it owns assets, leading to potential double-buying (exceeding risk limits) or failing to sell (because it thinks it holds nothing).

**The Fix:**
"Rehydrate" the portfolio state from the actual exchange balances on startup.

**Instructions:**
1.  **Modify `src/risk/portfolio.py`:**
    * Add a method `sync_with_exchange(self, balances: Dict[str, float], current_prices: Dict[str, float])`.
    * This method should clear `self.state.positions` and repopulate it based on the actual `balances` provided.
    * *Note: We won't know the exact `entry_price` or `entry_time` from just the balance, so estimate `entry_price` as `current_price` (conservative) or load it from a database if available. for now, just ensure the **Quantity** matches.*
2.  **Modify `main.py`:**
    * In the startup sequence (before the `while True` loop), call `executor.get_token_balance()` for every symbol in your universe.
    * Pass these balances to `portfolio.sync_with_exchange()`.
    * Log the discrepancy (e.g., "Rehydrated state: Found 0.5 BTC, Risk Manager updated").

---

## 📉 Priority 3: Fix "The Flash Crash Trap" (Market Orders)

**The Issue:**
In `src/execution/live.py`, we use `ORDER_TYPE_MARKET`. In crypto, thin liquidity or "flash crashes" can result in execution prices 5-10% worse than the displayed price.

**The Fix:**
Replace Market orders with **Marketable Limit Orders** (MLOs).

**Instructions:**
1.  **Modify `src/execution/live.py`:**
    * Change the order type in `execute_order`:
        ```python
        # OLD
        type=ORDER_TYPE_MARKET
        
        # NEW
        type=ORDER_TYPE_LIMIT,
        timeInForce=TIME_IN_FORCE_GTC
        ```
    * **Logic for BUY:**
        * Fetch current `Ask` price (ticker).
        * Set limit price = `Ask * 1.01` (1% buffer). This ensures immediate execution but prevents buying at +50%.
    * **Logic for SELL:**
        * Fetch current `Bid` price.
        * Set limit price = `Bid * 0.99` (1% buffer).
    * Pass this calculated `price` to the `create_order` call.

---

## Verification Checklist
After applying these fixes, please run a script to verify:
1.  **Precision:** Call `_normalize_quantity('BTCUSDT', 0.12345678)` and confirm it matches Binance's step size.
2.  **Sync:** Restart the bot and confirm logs show "Synced portfolio with X positions found on exchange."
3.  **Safety:** Verify that a "Buy" order uses a Limit price, not a Market order.