from __future__ import annotations

from trader.modeling.symbol_interval_grid import (
    sentiment_provenance_audit,
    symbol_interval_decision,
)

import pandas as pd


def test_decision_can_report_no_interval_confirms_on_holdout() -> None:
    summary = pd.DataFrame(
        [
            {
                "symbol": "ETHUSDT",
                "interval": "5h",
                "rank": 1,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.04,
                "holdout_total_return": 0.01,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.02,
            }
        ]
    )

    decision = symbol_interval_decision(summary)

    assert decision["selected_development_ranked_symbol_interval"] == {
        "symbol": "ETHUSDT",
        "interval": "5h",
    }
    assert decision["holdout_confirmation_result"] == "not_confirmed"
    assert decision["any_candidate_confirms_on_holdout"] is False
    assert decision["holdout_used_for_ranking"] is False
    assert decision["phase_11_status"] == "blocked"


def test_decision_reports_altcoin_development_improvement_over_btc_12h() -> None:
    summary = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "interval": "12h",
                "rank": 2,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.02,
                "holdout_total_return": 0.0,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.0,
            },
            {
                "symbol": "ETHUSDT",
                "interval": "7h",
                "rank": 1,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.03,
                "holdout_total_return": 0.04,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.01,
            },
        ]
    )

    decision = symbol_interval_decision(summary)

    assert decision["altcoin_interval_improves_over_btc_12h_on_development"] is True
    assert decision["holdout_confirmation_result"] == "confirmed"
    assert decision["any_candidate_confirms_on_holdout"] is True


def test_sentiment_provenance_audit_marks_prior_gate_inconclusive() -> None:
    audit = sentiment_provenance_audit()

    assert audit["existing_sentiment_artifact"]["source"] == "server CSV"
    assert audit["existing_sentiment_artifact"]["content_scope"] == "posts-only"
    assert audit["bitcoin_specific_metadata_available"] is False
    assert (
        audit["prior_sentiment_gate_status"]
        == "inconclusive_for_bitcoin_specific_sentiment"
    )
