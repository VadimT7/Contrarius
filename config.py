"""Configuration management for Contrarius betting engine."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv
import json

load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    sportradar_key: str = os.getenv("SPORTRADAR_KEY", "")
    betfair_username: str = os.getenv("BETFAIR_USERNAME", "")
    betfair_password: str = os.getenv("BETFAIR_PASSWORD", "")
    betfair_app_key: str = os.getenv("BETFAIR_APP_KEY", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: str = os.getenv("DB_TYPE", "postgresql")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "contrarius")
    db_user: str = os.getenv("DB_USER", "contrarius_user")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def connection_string(self) -> str:
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_name}.db"
        return f"{self.db_type}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    ensemble_weights: Dict[str, float] = None
    retrain_frequency_hours: int = 24
    calibration_method: str = "isotonic"
    confidence_threshold: float = 0.65
    feature_importance_threshold: float = 0.01
    max_features: int = 100
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "lightgbm": 0.25,
                "xgboost": 0.20,
                "catboost": 0.20,
                "random_forest": 0.15,
                "neural_net": 0.15,
                "logistic_regression": 0.05
            }


@dataclass
class BettingConfig:
    """Betting strategy configuration."""
    min_edge_percentage: float = float(os.getenv("MIN_EDGE_PERCENTAGE", "5.0"))
    max_bet_percentage: float = float(os.getenv("MAX_BET_PERCENTAGE", "5.0"))
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    max_daily_bets: int = int(os.getenv("MAX_DAILY_BETS", "20"))
    max_exposure_percentage: float = float(os.getenv("MAX_EXPOSURE_PERCENTAGE", "25.0"))
    bankroll: float = float(os.getenv("INITIAL_BANKROLL", "10000.0"))
    stop_loss_percentage: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "20.0"))
    take_profit_percentage: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "50.0"))


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    sports_enabled: List[str] = None
    injury_weight: float = 0.15
    weather_weight: float = 0.10
    travel_fatigue_weight: float = 0.05
    market_sentiment_weight: float = 0.20
    odds_movement_weight: float = 0.25
    sharp_money_weight: float = 0.25
    
    def __post_init__(self):
        if self.sports_enabled is None:
            self.sports_enabled = ["nfl", "nba", "mlb", "nhl", "soccer", "tennis"]


@dataclass
class SystemConfig:
    """System-wide configuration."""
    update_interval_seconds: int = int(os.getenv("UPDATE_INTERVAL_SECONDS", "60"))
    enable_notifications: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    notification_email: str = os.getenv("NOTIFICATION_EMAIL", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    timezone: str = os.getenv("TIMEZONE", "UTC")
    stealth_mode: bool = os.getenv("STEALTH_MODE", "true").lower() == "true"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.betting = BettingConfig()
        self.features = FeatureConfig()
        self.system = SystemConfig()
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.odds_api_key:
            print("Warning: No Odds API key configured")
        
        if sum(self.model.ensemble_weights.values()) != 1.0:
            print("Warning: Ensemble weights don't sum to 1.0")
            
        return True
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            "model": {
                "ensemble_weights": self.model.ensemble_weights,
                "calibration_method": self.model.calibration_method,
                "confidence_threshold": self.model.confidence_threshold
            },
            "betting": {
                "min_edge_percentage": self.betting.min_edge_percentage,
                "kelly_fraction": self.betting.kelly_fraction,
                "max_daily_bets": self.betting.max_daily_bets
            },
            "features": {
                "sports_enabled": self.features.sports_enabled,
                "injury_weight": self.features.injury_weight,
                "weather_weight": self.features.weather_weight
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load configuration from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
                
            if "model" in config_dict:
                self.model.ensemble_weights = config_dict["model"].get("ensemble_weights", self.model.ensemble_weights)
                self.model.calibration_method = config_dict["model"].get("calibration_method", self.model.calibration_method)
                
            if "betting" in config_dict:
                self.betting.min_edge_percentage = config_dict["betting"].get("min_edge_percentage", self.betting.min_edge_percentage)
                self.betting.kelly_fraction = config_dict["betting"].get("kelly_fraction", self.betting.kelly_fraction)


# Global configuration instance
config = Config() 