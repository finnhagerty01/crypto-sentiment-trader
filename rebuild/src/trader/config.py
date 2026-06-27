"""Strict configuration loading for the market-only baseline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file does not match the public schema."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    symbol: str
    interval: str


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    volatility_window: int
    volume_window: int
    rsi_window: int
    clipping_window: int
    clipping_mad_multiplier: float


@dataclass(frozen=True, slots=True)
class TargetConfig:
    horizon_bars: int
    volatility_multiplier: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    probability_threshold: float
    regularization_c: float


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    minimum_train_bars: int
    test_bars: int
    step_bars: int
    final_holdout_fraction: float


@dataclass(frozen=True, slots=True)
class CostsConfig:
    fee_per_side: float
    slippage_per_side: float


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_capital: float


@dataclass(frozen=True, slots=True)
class TraderConfig:
    data: DataConfig
    features: FeaturesConfig
    target: TargetConfig
    model: ModelConfig
    validation: ValidationConfig
    costs: CostsConfig
    backtest: BacktestConfig


def load_config(
    path: str | Path = "configs/baseline.yaml",
    *,
    working_directory: str | Path | None = None,
) -> TraderConfig:
    """Load and validate a config.

    Relative paths are resolved against ``working_directory`` when supplied,
    otherwise against the caller's current working directory.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        base = Path.cwd() if working_directory is None else Path(working_directory)
        config_path = base / config_path

    try:
        with config_path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in configuration {config_path}: {exc}") from exc

    root = _require_mapping(raw, "config")
    config = _build_dataclass(TraderConfig, root, "config")
    _validate(config)
    return config


def _build_dataclass[T](
    config_type: type[T], values: Mapping[str, Any], location: str
) -> T:
    expected = {field.name for field in fields(config_type)}
    supplied = set(values)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ConfigError(f"{location}: missing field(s): {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{location}: unknown field(s): {', '.join(unknown)}")

    hints = get_type_hints(config_type)
    kwargs: dict[str, Any] = {}
    for name in expected:
        value = values[name]
        expected_type = hints[name]
        nested_type = _dataclass_type(expected_type)
        field_location = f"{location}.{name}"
        if nested_type is not None:
            kwargs[name] = _build_dataclass(
                nested_type, _require_mapping(value, field_location), field_location
            )
        else:
            _require_type(value, expected_type, field_location)
            kwargs[name] = value
    return config_type(**kwargs)


def _dataclass_type(annotation: Any) -> type[Any] | None:
    return annotation if hasattr(annotation, "__dataclass_fields__") else None


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location}: expected a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{location}: all field names must be strings")
    return value


def _require_type(value: Any, annotation: Any, location: str) -> None:
    origin = get_origin(annotation)
    accepted = get_args(annotation) if origin in (Union, UnionType) else (annotation,)

    def matches(candidate: Any) -> bool:
        if candidate is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if candidate is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, candidate)

    if not any(matches(candidate) for candidate in accepted):
        names = " or ".join(getattr(candidate, "__name__", str(candidate)) for candidate in accepted)
        raise ConfigError(f"{location}: expected {names}, got {type(value).__name__}")


def _validate(config: TraderConfig) -> None:
    if config.data.symbol != "BTCUSDT":
        raise ConfigError("config.data.symbol: baseline supports only BTCUSDT")
    if config.data.interval != "1h":
        raise ConfigError("config.data.interval: baseline supports only 1h")

    positive_integers = {
        "config.features.volatility_window": config.features.volatility_window,
        "config.features.volume_window": config.features.volume_window,
        "config.features.rsi_window": config.features.rsi_window,
        "config.features.clipping_window": config.features.clipping_window,
        "config.target.horizon_bars": config.target.horizon_bars,
        "config.validation.minimum_train_bars": config.validation.minimum_train_bars,
        "config.validation.test_bars": config.validation.test_bars,
        "config.validation.step_bars": config.validation.step_bars,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ConfigError(f"{name}: must be greater than zero")

    positive_numbers = {
        "config.features.clipping_mad_multiplier": (
            config.features.clipping_mad_multiplier
        ),
        "config.model.regularization_c": config.model.regularization_c,
        "config.costs.fee_per_side": config.costs.fee_per_side,
        "config.costs.slippage_per_side": config.costs.slippage_per_side,
        "config.backtest.initial_capital": config.backtest.initial_capital,
    }
    for name, value in positive_numbers.items():
        if not math.isfinite(value):
            raise ConfigError(f"{name}: must be finite")
        if value <= 0:
            raise ConfigError(f"{name}: must be greater than zero")

    if not math.isfinite(config.target.volatility_multiplier):
        raise ConfigError("config.target.volatility_multiplier: must be finite")
    if config.target.volatility_multiplier < 0:
        raise ConfigError(
            "config.target.volatility_multiplier: must be greater than or equal to zero"
        )
    _validate_open_fraction(
        "config.model.probability_threshold", config.model.probability_threshold
    )
    _validate_open_fraction(
        "config.validation.final_holdout_fraction",
        config.validation.final_holdout_fraction,
    )
    for name, value in {
        "config.costs.fee_per_side": config.costs.fee_per_side,
        "config.costs.slippage_per_side": config.costs.slippage_per_side,
    }.items():
        if value >= 1:
            raise ConfigError(f"{name}: must be less than one")


def _validate_open_fraction(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ConfigError(f"{name}: must be finite")
    if not 0 < value < 1:
        raise ConfigError(f"{name}: must be between zero and one (exclusive)")
