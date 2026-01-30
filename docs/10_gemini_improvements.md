# Senior Quant Code Review & Remediation Plan

**Role:** You are an expert Python developer and quantitative trading engineer.
**Context:** I have received a critical code review from a Senior Quant at a top firm regarding my crypto trading bot. The review identified several "hidden risks" and structural flaws that need to be addressed before this system is production-ready.
**Objective:** Your task is to apply the following fixes to the codebase. Proceed step-by-step through the priorities listed below.

---

## 🚨 Priority 1: Fix Look-Ahead Bias (Critical)

**The Issue:**
The backtesting engine (`src/backtest/engine.py`) currently executes trades at the `close` price of the *same bar* that generated the signal. In live trading, the "close" is not known until the moment the bar ends, meaning we effectively traded on future information.

**The Fix:**
Refactor the execution logic to simulate trading at the **Open** of the *next* bar.

**Instructions:**
1.  **Modify `src/backtest/engine.py`:**
    * In the `run` method loop, signals generated at index `i` (using data up to timestamp `t`) must be executed using price data from index `i+1`.
    * Change the execution price logic. Instead of `price = symbol_data["close"]`, look ahead to the *next* available timestamp for that symbol and use its `open` price.
    * If no next bar exists (end of data), the trade cannot be executed.
2.  **Verify:** Ensure that feature calculation still happens on the "current" bar (row `i`), but the P&L calculation uses the "next" bar (row `i+1`).

---

## 🧠 Priority 2: Upgrade Sentiment Analysis (Alpha Generation)

**The Issue:**
The current `VADER` model is lexicon-based and struggles with crypto context (e.g., "moon" or "diamond hands"). Additionally, the `RedditClient` ingests bot spam, which skews sentiment metrics.

**The Fix:**
Implement a spam filter and replace (or augment) VADER with a financial-domain Transformer model (FinBERT) or a robust cleaning pipeline.

**Instructions:**
1.  **Modify `src/data/reddit_client.py`:**
    * Implement a `_is_spam(post)` method.
    * Filter out posts with `selftext` matching common bot patterns (e.g., duplicate templated messages, excessive emojis, or known scam domains).
    * Filter out users with very low karma (if available in the API response) or account ages < 2 days.
2.  **Modify `src/analysis/sentiment.py`:**
    * **Option A (Preferred):** Integrate `transformers` and `BertTokenizer` to use the `ProsusAI/finbert` model. Add `torch` and `transformers` to `requirements.txt`.
    * **Option B (Lightweight):** If full BERT is too heavy, implement a "Crypto Lexicon" override for VADER that properly scores terms like "ATH", "bearish", "bullish", "rekt", "moon".
    * *Note: Please ask me which option you prefer before implementing.*

---

## 🛡️ Priority 3: Dynamic Risk Management

**The Issue:**
In `src/risk/portfolio.py`, the `_get_btc_correlated_exposure` method uses a hardcoded correlation assumption (`0.7`). In reality, correlations fluctuate and converge to 1.0 during crashes.

**The Fix:**
Implement dynamic correlation tracking using a rolling window.

**Instructions:**
1.  **Modify `src/risk/portfolio.py`:**
    * Update `PortfolioRiskManager` to accept a history of prices (or access to the `MarketClient`).
    * Implement a method `calculate_correlations(lookback_days=30)`.
    * Calculate the rolling correlation matrix of all active positions against BTC.
    * Replace the hardcoded `0.7` with the actual rolling correlation coefficient for that specific asset.
    * If correlation data is unavailable (new coin), fallback to a conservative `0.9` (not 0.7).

---

## 🏗️ Priority 4: Robustness & Infrastructure

**The Issue:**
1.  The `while True` loop in `main.py` is fragile; if it crashes, the bot stops.
2.  Data is saved to CSV (`master_reddit.csv`), which is not atomic and prone to corruption during concurrent writes.

**The Fix:**
Make the script "scheduler-friendly" and move storage to SQLite.

**Instructions:**
1.  **Refactor `src/data/archive.py`:**
    * Replace CSV read/write logic with **SQLite**.
    * Create a schema with tables for `posts` (id primary key, created_utc, title, selftext, etc.) and `sentiment` (timestamp, symbol, score).
    * Ensure operations are atomic (use transactions).
2.  **Refactor `main.py`:**
    * Add a `--run-once` command-line argument.
    * If `--run-once` is passed, the script should perform one iteration of the logic (Fetch -> Analyze -> Trade) and then exit. This allows us to use `cron` or `systemd` timers for reliability.
    * Keep the `while True` loop only as the default behavior if no flag is passed.

---

## Execution Order
Please start with **Priority 1 (Fix Look-Ahead Bias)** as it invalidates all current backtest results. Once complete, wait for my confirmation to proceed to Priority 2.