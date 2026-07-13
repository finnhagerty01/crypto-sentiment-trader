"""Strict configuration loading for the market-only baseline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from types import GenericAlias
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
    enabled_groups: tuple[str, ...] = ("baseline",)


@dataclass(frozen=True, slots=True)
class TargetConfig:
    horizon_bars: int
    cost_buffer: str
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
            kwargs[name] = _coerce_value(value, expected_type, field_location)
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
    _coerce_value(value, annotation, location)


def _coerce_value(value: Any, annotation: Any, location: str) -> Any:
    origin = get_origin(annotation)
    if origin is tuple:
        return _coerce_tuple(value, annotation, location)
    accepted = get_args(annotation) if origin in (Union, UnionType) else (annotation,)

    def matches(candidate: Any) -> bool:
        if isinstance(candidate, GenericAlias):
            return isinstance(value, get_origin(candidate))
        if candidate is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if candidate is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, candidate)

    if not any(matches(candidate) for candidate in accepted):
        names = " or ".join(getattr(candidate, "__name__", str(candidate)) for candidate in accepted)
        raise ConfigError(f"{location}: expected {names}, got {type(value).__name__}")
    return value


def _coerce_tuple(value: Any, annotation: Any, location: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{location}: expected list, got {type(value).__name__}")
    args = get_args(annotation)
    item_type = args[0] if args else Any
    if item_type is str:
        invalid = [item for item in value if not isinstance(item, str)]
        if invalid:
            raise ConfigError(f"{location}: expected list of str")
    return tuple(value)


def _validate(config: TraderConfig) -> None:
    if not config.data.symbol or not config.data.symbol.isalnum():
        raise ConfigError("config.data.symbol: must be a non-empty alphanumeric symbol")
    if not _valid_interval(config.data.interval):
        raise ConfigError("config.data.interval: must be a positive hour or day interval")

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

    allowed_feature_groups = {
        "baseline",
        "trend",
        "volatility",
        "volume",
        "calendar",
        "momentum_reversal",
    }
    if not config.features.enabled_groups:
        raise ConfigError("config.features.enabled_groups: must not be empty")
    unknown_groups = [
        group
        for group in config.features.enabled_groups
        if group not in allowed_feature_groups
    ]
    if unknown_groups:
        raise ConfigError(
            "config.features.enabled_groups: unknown group(s): "
            + ", ".join(sorted(unknown_groups))
        )
    duplicate_groups = sorted(
        {
            group
            for group in config.features.enabled_groups
            if config.features.enabled_groups.count(group) > 1
        }
    )
    if duplicate_groups:
        raise ConfigError(
            "config.features.enabled_groups: duplicate group(s): "
            + ", ".join(duplicate_groups)
        )

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
    if config.target.cost_buffer not in {"none", "one_way", "round_trip"}:
        raise ConfigError(
            "config.target.cost_buffer: must be one of none, one_way, round_trip"
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


def _valid_interval(value: str) -> bool:
    if not isinstance(value, str) or len(value) < 2:
        return False
    unit = value[-1]
    amount = value[:-1]
    if unit not in {"h", "d"} or not amount.isdigit():
        return False
    return int(amount) > 0
