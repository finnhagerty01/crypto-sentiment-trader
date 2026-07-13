from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trader.cli import COMMANDS, main
from trader.data.storage import build_metadata


def test_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "collect-market" in capsys.readouterr().out


def test_all_phase_08_commands_are_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in COMMANDS:
        assert command in output


def test_run_experiment_grid_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-experiment-grid", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--market-data" in output
    assert "--experiment-config" in output
    assert "--output-dir" in output
    assert "--run-id" in output


def test_run_sentiment_gate_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-sentiment-gate", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--market-data" in output
    assert "--hourly-sentiment" in output
    assert "--output-dir" in output
    assert "--run-id" in output


def test_run_model_class_comparison_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-model-class-comparison", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--market-data" in output
    assert "--output-dir" in output
    assert "--run-id" in output
    assert "--enable-xgboost" in output


def test_run_candle_interval_comparison_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-candle-interval-comparison", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--market-data" in output
    assert "--output-dir" in output
    assert "--run-id" in output
    assert "--intervals" in output
    assert "--max-development-exposure" in output


def test_run_regime_specialist_comparison_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-regime-specialist-comparison", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--market-data" in output
    assert "--output-dir" in output
    assert "--run-id" in output
    assert "--lookback-bars" in output


def test_run_symbol_interval_grid_is_listed_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run-symbol-interval-grid", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--output-dir" in output
    assert "--run-id" in output
    assert "--symbols" in output
    assert "--intervals" in output
    assert "--start" in output
    assert "--end" in output
    assert "--max-development-exposure" in output


def test_collect_market_writes_dataset_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTCUSDT"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [12.0],
        }
    )
    saved_paths: list[Path] = []

    def fake_collect_market_data(**kwargs: object) -> pd.DataFrame:
        assert kwargs["start"] == "2026-01-01T00:00:00Z"
        assert kwargs["end"] == "2026-01-01T01:00:00Z"
        return rows

    def fake_write_market_dataset(
        data: pd.DataFrame,
        path: str | Path,
        *,
        source: str,
        symbol: str,
        interval: str,
    ):
        saved_paths.append(Path(path))
        return build_metadata(data, source=source, symbol=symbol, interval=interval)

    monkeypatch.chdir(Path(__file__).parents[2])
    monkeypatch.setattr("trader.cli.collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("trader.cli.write_market_dataset", fake_write_market_dataset)

    result = main(
        [
            "collect-market",
            "--start",
            "2026-01-01T00:00:00Z",
            "--end",
            "2026-01-01T01:00:00Z",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert saved_paths
    assert saved_paths[0].parent == tmp_path
    assert "saved 1 rows" in capsys.readouterr().out
