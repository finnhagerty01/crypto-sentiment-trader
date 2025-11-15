# src/utils/config.py
"""
Centralized configuration management for the crypto sentiment trading system.
This module handles all configuration loading and validation.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration container with validation and defaults."""
    
    # Reddit settings
    subreddits: List[str] = field(default_factory=list)
    reddit_limit_per_sub: int = 200
    reddit_min_score: int = 10
    reddit_min_length: int = 0

    # Market settings
    symbols: List[str] = field(default_factory=list)
    intervals: List[str] = field(default_factory=list)
    market_data_limit: int = 1000
    
    # Feature engineering
    sentiment_half_life: str = "6H"
    sentiment_lookback_bars: int = 24
    
    # Trading parameters
    enter_threshold: float = 0.60
    exit_threshold: float = 0.55
    min_hold_bars: int = 0
    fee_per_side: float = 0.0010
    slippage_per_side: float = 0.0005
    
    # Paths
    data_dir: Path = Path("data")
    raw_reddit_dir: Path = Path("data/raw/reddit")
    raw_market_dir: Path = Path("data/raw/market")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # External services
    arctic_shift_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ARCTIC_SHIFT_API_KEY")
    )
    arctic_shift_base_url: str = field(
        default_factory=lambda: os.getenv("ARCTIC_SHIFT_BASE_URL", "https://api.arcticshift.com/v1")
    )

    # Data pre-processing options
    apply_kalman_filter: bool = False
    kalman_process_variance: float = 1e-5
    kalman_measurement_variance: float = 0.05
    
    def __post_init__(self):
        """Create directories and validate configuration."""
        for dir_path in [self.raw_reddit_dir, self.raw_market_dir, 
                         self.processed_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Validate thresholds
        if self.exit_threshold >= self.enter_threshold:
            raise ValueError(f"Exit threshold ({self.exit_threshold}) must be < "
                           f"enter threshold ({self.enter_threshold})")
    
    @classmethod
    def from_yaml(cls, config_path: str = "configs/data.yaml") -> 'TradingConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            valid_fields = set(cls.__dataclass_fields__.keys())
            kwargs: Dict[str, Any] = {}

            for key, value in data.items():
                if key == 'trading_params' and isinstance(value, dict):
                    for t_key, t_value in value.items():
                        if t_key in valid_fields:
                            kwargs[t_key] = t_value
                elif key in valid_fields:
                    kwargs[key] = value

            return cls(**kwargs)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def get_symbol_keywords(self) -> Dict[str, List[str]]:
        """Return keyword mappings for each symbol."""
        return {
            "BTCUSDT": ["btc", "bitcoin"],
            "ETHUSDT": ["eth", "ethereum"],
            "SOLUSDT": ["sol", "solana"],
            "BNBUSDT": ["bnb", "binance coin", "binance"],
            "XRPUSDT": ["xrp", "ripple"],
            "ADAUSDT": ["ada", "cardano"],
            "AVAXUSDT": ["avax", "avalanche"],
            "DOGEUSDT": ["doge", "dogecoin"],
            "LINKUSDT": ["link", "chainlink"],
            "MATICUSDT": ["matic", "polygon"],
            "DOTUSDT": ["dot", "polkadot"],
            "ATOMUSDT": ["atom", "cosmos"]
        }