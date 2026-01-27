#!/usr/bin/env python3
"""
viz_performance.py

Parse Sleek bot logs and visualize:
- Equity / USDT / RealizedPnL over time
- Fee-corrected Equity / RealizedPnL (for Binance.US Tier 0 fee mismatch)
- Fees (raw vs corrected) over time
- Trades per hour/day (if [PAPER] BUY / SELL_ALL lines are present)
- Open positions count over time

Usage examples:
  python viz_performance.py --log /path/to/bot.log
  python viz_performance.py --log bot.log --old-fee 0.001 --new-fee 0.0001
  python viz_performance.py --log bot.log --timezone "America/New_York"  # optional for display
"""

import re
import argparse
from datetime import datetime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

LEDGER_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+-\s+.*?\s+-\s+INFO\s+-\s+'
    r'\[PAPER_LEDGER\]\s+USDT=\$(?P<usdt>-?\d+(?:\.\d+)?)\s+\|\s+'
    r'Equity=\$(?P<equity>-?\d+(?:\.\d+)?)\s+\|\s+'
    r'RealizedPnL=\$(?P<realized>-?\d+(?:\.\d+)?)\s+\|\s+'
    r'Fees=\$(?P<fees>-?\d+(?:\.\d+)?)\s+\|\s+'
    r'OpenPositions=(?P<openpos>\d+)'
)

# These trade regexes are tolerant: they’ll match if you kept the suggested logging style.
BUY_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?\[PAPER\]\s+BUY\s+(?P<symbol>[A-Z0-9]+)'
)
SELL_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?\[PAPER\]\s+SELL_ALL\s+(?P<symbol>[A-Z0-9]+)'
)

def parse_logs(path: str):
    ledger_rows = []
    trades = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LEDGER_RE.search(line)
            if m:
                ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                ledger_rows.append({
                    "timestamp": ts,
                    "usdt": float(m.group("usdt")),
                    "equity": float(m.group("equity")),
                    "realized_pnl": float(m.group("realized")),
                    "fees": float(m.group("fees")),
                    "open_positions": int(m.group("openpos")),
                })
                continue

            mb = BUY_RE.search(line)
            if mb:
                ts = datetime.strptime(mb.group("ts"), "%Y-%m-%d %H:%M:%S")
                trades.append({"timestamp": ts, "side": "BUY", "symbol": mb.group("symbol")})
                continue

            ms = SELL_RE.search(line)
            if ms:
                ts = datetime.strptime(ms.group("ts"), "%Y-%m-%d %H:%M:%S")
                trades.append({"timestamp": ts, "side": "SELL", "symbol": ms.group("symbol")})
                continue

    ledger_df = pd.DataFrame(ledger_rows).sort_values("timestamp").reset_index(drop=True)
    trades_df = pd.DataFrame(trades).sort_values("timestamp").reset_index(drop=True)
    return ledger_df, trades_df


def apply_fee_correction(ledger_df: pd.DataFrame, old_fee: float, new_fee: float) -> pd.DataFrame:
    """
    Correct fees/equity/realized for fee mismatch:
      corrected_fees = fees * (new_fee/old_fee)
      fee_overcharge = fees - corrected_fees
      corrected_equity = equity + fee_overcharge
      corrected_realized_pnl = realized_pnl + fee_overcharge
    """
    df = ledger_df.copy()
    if df.empty:
        return df

    if old_fee <= 0 or new_fee < 0:
        raise ValueError("Fee rates must satisfy old_fee > 0 and new_fee >= 0")

    ratio = new_fee / old_fee
    df["fees_corrected"] = df["fees"] * ratio
    df["fee_overcharge"] = df["fees"] - df["fees_corrected"]

    df["equity_corrected"] = df["equity"] + df["fee_overcharge"]
    df["realized_pnl_corrected"] = df["realized_pnl"] + df["fee_overcharge"]

    return df


def plot_all(ledger_df: pd.DataFrame, trades_df: pd.DataFrame, title: str):
    if ledger_df.empty:
        print("No [PAPER_LEDGER] lines found. Check log path or log format.")
        return

    # --- Plot 1: Equity raw vs corrected ---
    plt.figure()
    plt.plot(ledger_df["timestamp"], ledger_df["equity"], label="Equity (raw)")
    if "equity_corrected" in ledger_df.columns:
        plt.plot(ledger_df["timestamp"], ledger_df["equity_corrected"], label="Equity (fee-corrected)")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.title(f"{title} — Equity Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Equity_raw_vs_connected.png")
    plt.close()

    # --- Plot 2: Realized PnL raw vs corrected ---
    plt.figure()
    plt.plot(ledger_df["timestamp"], ledger_df["realized_pnl"], label="RealizedPnL (raw)")
    if "realized_pnl_corrected" in ledger_df.columns:
        plt.plot(ledger_df["timestamp"], ledger_df["realized_pnl_corrected"], label="RealizedPnL (fee-corrected)")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.title(f"{title} — Realized PnL Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("PnL.png")
    plt.close()

    # --- Plot 3: USDT cash & Open positions ---
    plt.figure()
    plt.plot(ledger_df["timestamp"], ledger_df["usdt"], label="USDT (cash)")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.title(f"{title} — Cash Balance Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Cash.png")
    plt.close()

    plt.figure()
    plt.plot(ledger_df["timestamp"], ledger_df["open_positions"], label="Open Positions")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title(f"{title} — Open Positions Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Open_positions.png")
    plt.close()

    # --- Plot 4: Fees raw vs corrected (cumulative) ---
    plt.figure()
    plt.plot(ledger_df["timestamp"], ledger_df["fees"], label="Fees (raw)")
    if "fees_corrected" in ledger_df.columns:
        plt.plot(ledger_df["timestamp"], ledger_df["fees_corrected"], label="Fees (corrected)")
        plt.plot(ledger_df["timestamp"], ledger_df["fee_overcharge"], label="Fee overcharge (raw - corrected)")
    plt.xlabel("Time")
    plt.ylabel("USD")
    plt.title(f"{title} — Fees Over Time (as logged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fees.png")
    plt.close()

    # --- Trades over time (if available) ---
    if not trades_df.empty:
        # Trades per hour
        trades_df2 = trades_df.copy()
        trades_df2["hour"] = trades_df2["timestamp"].dt.floor("H")
        counts_hour = trades_df2.groupby(["hour", "side"]).size().unstack(fill_value=0)

        plt.figure()
        for col in counts_hour.columns:
            plt.plot(counts_hour.index, counts_hour[col], label=f"{col} trades/hour")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{title} — Trades Per Hour")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Trades_hours.png")
        plt.close()

        # Trades per day
        trades_df2["day"] = trades_df2["timestamp"].dt.floor("D")
        counts_day = trades_df2.groupby(["day", "side"]).size().unstack(fill_value=0)

        plt.figure()
        for col in counts_day.columns:
            plt.plot(counts_day.index, counts_day[col], label=f"{col} trades/day")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{title} — Trades Per Day")
        plt.legend()
        plt.tight_layout()
        plt.savefig("trades_days.png")
        plt.close()
    else:
        print("No [PAPER] BUY/SELL_ALL lines found — skipping trade-count plots.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to your bot log file")
    ap.add_argument("--old-fee", type=float, default=0.0010, help="Fee rate used in ledger (e.g., 0.0010)")
    ap.add_argument("--new-fee", type=float, default=0.0001, help="Correct fee rate (e.g., 0.0001 for 0.01%)")
    ap.add_argument("--title", default="Sleek V1.1", help="Title prefix for plots")
    args = ap.parse_args()

    ledger_df, trades_df = parse_logs(args.log)
    ledger_df = apply_fee_correction(ledger_df, old_fee=args.old_fee, new_fee=args.new_fee)

    # Ensure pandas datetime dtype
    ledger_df["timestamp"] = pd.to_datetime(ledger_df["timestamp"])
    if not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

    plot_all(ledger_df, trades_df, title=args.title)


if __name__ == "__main__":
    main()
