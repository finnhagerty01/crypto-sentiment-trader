from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trader.config import ConfigError, TraderConfig, load_config


PROJECT_ROOT = Path(__file__).parents[2]
BASELINE_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


@pytest.fixture
def baseline_values() -> dict[str, object]:
    with BASELINE_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_config(tmp_path: Path, values: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_loads_baseline_into_typed_immutable_config() -> None:
    config = load_config(BASELINE_PATH)

    assert isinstance(config, TraderConfig)
    assert config.data.symbol == "BTCUSDT"
    assert config.features.volatility_window == 24
    assert config.model.probability_threshold == pytest.approx(0.55)
    with pytest.raises((AttributeError, TypeError)):
        config.data.symbol = "ETHUSDT"  # type: ignore[misc]


def test_relative_path_resolves_against_explicit_working_directory() -> None:
    config = load_config("configs/baseline.yaml", working_directory=PROJECT_ROOT)

    assert config.data.interval == "1h"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("data", "symbol", "ETHUSDT", "supports only BTCUSDT"),
        ("data", "interval", "5m", "supports only 1h"),
        ("features", "rsi_window", 0, "must be greater than zero"),
        ("target", "horizon_bars", -1, "must be greater than zero"),
        ("target", "volatility_multiplier", -0.1, "greater than or equal"),
        ("model", "probability_threshold", 1.0, "between zero and one"),
        ("model", "probability_threshold", float("nan"), "must be finite"),
        ("model", "regularization_c", 0.0, "must be greater than zero"),
        ("validation", "final_holdout_fraction", 0.0, "between zero and one"),
        ("costs", "fee_per_side", 0.0, "must be greater than zero"),
        ("costs", "slippage_per_side", 1.0, "must be less than one"),
        ("backtest", "initial_capital", -1.0, "must be greater than zero"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path,
    baseline_values: dict[str, object],
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    section_values = baseline_values[section]
    assert isinstance(section_values, dict)
    section_values[field] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, baseline_values))


def test_rejects_unknown_root_field(
    tmp_path: Path, baseline_values: dict[str, object]
) -> None:
    baseline_values["reddit"] = {}

    with pytest.raises(ConfigError, match=r"config: unknown field\(s\): reddit"):
        load_config(write_config(tmp_path, baseline_values))


def test_rejects_unknown_nested_field(
    tmp_path: Path, baseline_values: dict[str, object]
) -> None:
    data = baseline_values["data"]
    assert isinstance(data, dict)
    data["api_key"] = "secret"

    with pytest.raises(ConfigError, match=r"config.data: unknown field\(s\): api_key"):
        load_config(write_config(tmp_path, baseline_values))


def test_rejects_missing_field(
    tmp_path: Path, baseline_values: dict[str, object]
) -> None:
    model = baseline_values["model"]
    assert isinstance(model, dict)
    del model["regularization_c"]

    with pytest.raises(
        ConfigError, match=r"config.model: missing field\(s\): regularization_c"
    ):
        load_config(write_config(tmp_path, baseline_values))


def test_rejects_wrong_type(
    tmp_path: Path, baseline_values: dict[str, object]
) -> None:
    features = baseline_values["features"]
    assert isinstance(features, dict)
    features["rsi_window"] = True

    with pytest.raises(ConfigError, match="expected int, got bool"):
        load_config(write_config(tmp_path, baseline_values))


def test_reports_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("data: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)
