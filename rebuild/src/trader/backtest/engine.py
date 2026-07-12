"""Long-or-cash next-bar backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trader.config import BacktestConfig, CostsConfig


Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def run_long_cash_backtest(
    market_data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    costs_config: CostsConfig,
) -> BacktestResult:
    """Run a one-position BTC long-or-cash backtest.

    Signals are read after bar ``t`` closes and, when changed, are filled at
    bar ``t+1`` open. A signal on the final bar is ignored because no next bar
    exists. Any remaining BTC position is force-closed at the final close after
    sell-side slippage and fees.
    """

    market = _prepare_market(market_data)
    signals = _prepare_predictions(predictions)
    data = market.merge(signals, on="timestamp", how="left")
    data["signal"] = data["signal"].fillna(0).astype("int8").clip(lower=0, upper=1)
    if "probability" in data.columns:
        data["probability"] = data["probability"].astype("float64")

    cash = float(backtest_config.initial_capital)
    btc_quantity = 0.0
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for position in range(len(data)):
        row = data.iloc[position]
        if position < len(data) - 1:
            next_row = data.iloc[position + 1]
            desired_long = int(row["signal"]) == 1
            is_long = btc_quantity > 0.0
            if desired_long and not is_long:
                cash, btc_quantity, trade = _buy_all(
                    cash,
                    timestamp=row["timestamp"],
                    fill_timestamp=next_row["timestamp"],
                    raw_price=float(next_row["open"]),
                    costs_config=costs_config,
                )
                if trade is not None:
                    trades.append(trade)
            elif not desired_long and is_long:
                cash, btc_quantity, trade = _sell_all(
                    cash,
                    btc_quantity,
                    timestamp=row["timestamp"],
                    fill_timestamp=next_row["timestamp"],
                    raw_price=float(next_row["open"]),
                    costs_config=costs_config,
                    reason="signal_exit",
                )
                trades.append(trade)

        equity_rows.append(
            _equity_row(
                row,
                cash=cash,
                btc_quantity=btc_quantity,
                costs_config=costs_config,
            )
        )

    if btc_quantity > 0.0:
        final_row = data.iloc[-1]
        cash, btc_quantity, trade = _sell_all(
            cash,
            btc_quantity,
            timestamp=final_row["timestamp"],
            fill_timestamp=final_row["timestamp"],
            raw_price=float(final_row["close"]),
            costs_config=costs_config,
            reason="final_close",
        )
        trades.append(trade)
        equity_rows[-1] = _equity_row(
            final_row,
            cash=cash,
            btc_quantity=btc_quantity,
            costs_config=costs_config,
        )

    prediction_columns = ["timestamp", "signal"]
    if "probability" in data.columns:
        prediction_columns.append("probability")
    return BacktestResult(
        predictions=data.loc[:, prediction_columns].copy(),
        trades=pd.DataFrame(trades, columns=_trade_columns()),
        equity=pd.DataFrame(equity_rows, columns=_equity_columns()),
    )


def _prepare_market(data: pd.DataFrame) -> pd.DataFrame:
    required = ("timestamp", "open", "close")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("missing market column(s): " + ", ".join(missing))
    market = data.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market.sort_values("timestamp").reset_index(drop=True)
    if market["timestamp"].duplicated().any():
        raise ValueError("market timestamps must be unique")
    if len(market) < 2:
        raise ValueError("backtest requires at least two market rows")
    return market


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in predictions.columns or "signal" not in predictions.columns:
        raise ValueError("predictions must contain timestamp and signal columns")
    result = predictions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("prediction timestamps must be unique")
    columns = ["timestamp", "signal"]
    if "probability" in result.columns:
        columns.append("probability")
    return result.loc[:, columns]


def _buy_all(
    cash: float,
    *,
    timestamp: pd.Timestamp,
    fill_timestamp: pd.Timestamp,
    raw_price: float,
    costs_config: CostsConfig,
) -> tuple[float, float, dict[str, object] | None]:
    if cash <= 0.0:
        return cash, 0.0, None
    fill_price = raw_price * (1.0 + costs_config.slippage_per_side)
    quantity = cash / (fill_price * (1.0 + costs_config.fee_per_side))
    notional = quantity * fill_price
    fee = notional * costs_config.fee_per_side
    slippage = quantity * raw_price * costs_config.slippage_per_side
    trade = _trade_row(
        side="buy",
        signal_timestamp=timestamp,
        fill_timestamp=fill_timestamp,
        raw_price=raw_price,
        fill_price=fill_price,
        quantity=quantity,
        notional=notional,
        fee=fee,
        slippage=slippage,
        cash_after=0.0,
        btc_quantity_after=quantity,
        reason="signal_entry",
    )
    return 0.0, quantity, trade


def _sell_all(
    cash: float,
    btc_quantity: float,
    *,
    timestamp: pd.Timestamp,
    fill_timestamp: pd.Timestamp,
    raw_price: float,
    costs_config: CostsConfig,
    reason: str,
) -> tuple[float, float, dict[str, object]]:
    fill_price = raw_price * (1.0 - costs_config.slippage_per_side)
    notional = btc_quantity * fill_price
    fee = notional * costs_config.fee_per_side
    slippage = btc_quantity * raw_price * costs_config.slippage_per_side
    cash_after = cash + notional - fee
    return (
        cash_after,
        0.0,
        _trade_row(
            side="sell",
            signal_timestamp=timestamp,
            fill_timestamp=fill_timestamp,
            raw_price=raw_price,
            fill_price=fill_price,
            quantity=btc_quantity,
            notional=notional,
            fee=fee,
            slippage=slippage,
            cash_after=cash_after,
            btc_quantity_after=0.0,
            reason=reason,
        ),
    )


def _equity_row(
    row: pd.Series,
    *,
    cash: float,
    btc_quantity: float,
    costs_config: CostsConfig,
) -> dict[str, object]:
    close = float(row["close"])
    liquidation_value = (
        btc_quantity
        * close
        * (1.0 - costs_config.slippage_per_side)
        * (1.0 - costs_config.fee_per_side)
    )
    equity = cash + liquidation_value
    return {
        "timestamp": row["timestamp"],
        "cash": cash,
        "btc_quantity": btc_quantity,
        "close": close,
        "equity": equity,
        "exposure": 1.0 if btc_quantity > 0.0 else 0.0,
    }


def _trade_row(
    *,
    side: Side,
    signal_timestamp: pd.Timestamp,
    fill_timestamp: pd.Timestamp,
    raw_price: float,
    fill_price: float,
    quantity: float,
    notional: float,
    fee: float,
    slippage: float,
    cash_after: float,
    btc_quantity_after: float,
    reason: str,
) -> dict[str, object]:
    return {
        "side": side,
        "signal_timestamp": signal_timestamp,
        "fill_timestamp": fill_timestamp,
        "raw_price": raw_price,
        "fill_price": fill_price,
        "quantity": quantity,
        "notional": notional,
        "fee": fee,
        "slippage": slippage,
        "cash_after": cash_after,
        "btc_quantity_after": btc_quantity_after,
        "reason": reason,
    }


def _trade_columns() -> list[str]:
    return [
        "side",
        "signal_timestamp",
        "fill_timestamp",
        "raw_price",
        "fill_price",
        "quantity",
        "notional",
        "fee",
        "slippage",
        "cash_after",
        "btc_quantity_after",
        "reason",
    ]


def _equity_columns() -> list[str]:
    return ["timestamp", "cash", "btc_quantity", "close", "equity", "exposure"]
