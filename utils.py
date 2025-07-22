"""Utility functions for Contrarius betting engine."""

import logging
from datetime import datetime, timedelta
import pytz
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import hashlib
import json
from functools import wraps
import time
import requests
from scipy import stats


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def implied_probability(odds: float, odds_format: str = "decimal") -> float:
    """Calculate implied probability from odds."""
    if odds_format == "american":
        odds = american_to_decimal(odds)
    return 1 / odds


def calculate_edge(true_prob: float, implied_prob: float) -> float:
    """Calculate betting edge percentage."""
    return ((true_prob - implied_prob) / implied_prob) * 100


def kelly_criterion(probability: float, odds: float, kelly_fraction: float = 0.25) -> float:
    """Calculate Kelly criterion stake percentage."""
    decimal_odds = odds if odds > 2 else american_to_decimal(odds)
    b = decimal_odds - 1
    p = probability
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly * kelly_fraction)


def calculate_expected_value(probability: float, odds: float, stake: float = 1) -> float:
    """Calculate expected value of a bet."""
    decimal_odds = odds if odds > 2 else american_to_decimal(odds)
    win_amount = stake * (decimal_odds - 1)
    loss_amount = stake
    
    ev = (probability * win_amount) - ((1 - probability) * loss_amount)
    return ev


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def calculate_travel_fatigue(distance_miles: float, time_zones_crossed: int, days_since_travel: int) -> float:
    """Calculate travel fatigue factor (0-1, higher is more fatigued)."""
    distance_factor = min(distance_miles / 3000, 1.0) * 0.4
    timezone_factor = min(time_zones_crossed / 3, 1.0) * 0.4
    recovery_factor = max(0, 1 - (days_since_travel / 3)) * 0.2
    
    return distance_factor + timezone_factor + recovery_factor


def parse_injury_impact(injury_description: str) -> float:
    """Parse injury description and return impact factor (0-1)."""
    severe_keywords = ["out", "doubtful", "surgery", "torn", "fracture", "break"]
    moderate_keywords = ["questionable", "strain", "sprain", "limited", "probable"]
    minor_keywords = ["probable", "minor", "day-to-day", "bruise"]
    
    injury_lower = injury_description.lower()
    
    if any(keyword in injury_lower for keyword in severe_keywords):
        return 0.8
    elif any(keyword in injury_lower for keyword in moderate_keywords):
        return 0.5
    elif any(keyword in injury_lower for keyword in minor_keywords):
        return 0.2
    else:
        return 0.1


def calculate_momentum(recent_results: List[int], weights: Optional[List[float]] = None) -> float:
    """Calculate team momentum based on recent results.
    
    Args:
        recent_results: List of recent game results (1 for win, 0 for loss)
        weights: Optional weights for each game (most recent first)
    
    Returns:
        Momentum score between -1 and 1
    """
    if not recent_results:
        return 0.0
    
    if weights is None:
        # Exponentially decaying weights for recent games
        weights = [0.5 ** i for i in range(len(recent_results))]
    
    weights = np.array(weights) / sum(weights)
    momentum = np.dot(recent_results, weights)
    
    # Scale to [-1, 1]
    return 2 * momentum - 1


def calculate_market_volatility(odds_history: List[float]) -> float:
    """Calculate market volatility from odds movement."""
    if len(odds_history) < 2:
        return 0.0
    
    returns = np.diff(odds_history) / odds_history[:-1]
    return np.std(returns)


def detect_sharp_money(odds_movement: List[Dict], volume_data: List[Dict]) -> bool:
    """Detect sharp money based on odds movement and volume."""
    if not odds_movement or not volume_data:
        return False
    
    # Look for reverse line movement (odds move against public betting)
    public_percentage = volume_data[-1].get("public_percentage", 50)
    odds_change = odds_movement[-1]["odds"] - odds_movement[0]["odds"]
    
    # If public is on one side but line moves the other way
    if (public_percentage > 65 and odds_change < -0.05) or \
       (public_percentage < 35 and odds_change > 0.05):
        return True
    
    return False


def sanitize_team_name(team_name: str) -> str:
    """Standardize team name for consistency."""
    # Remove common suffixes/prefixes
    replacements = {
        " FC": "", " SC": "", " AFC": "", " CF": "",
        "FC ": "", "SC ": "", "AFC ": "", "CF ": "",
        " United": "", " City": "", " Town": "",
        " Rovers": "", " Wanderers": ""
    }
    
    cleaned = team_name.strip()
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    return cleaned.strip()


def calculate_form_rating(results: List[Dict], venue: str = "all") -> float:
    """Calculate team form rating based on recent results."""
    if not results:
        return 0.5
    
    points = []
    for result in results:
        if venue != "all" and result.get("venue") != venue:
            continue
            
        if result["result"] == "W":
            points.append(3)
        elif result["result"] == "D":
            points.append(1)
        else:
            points.append(0)
    
    if not points:
        return 0.5
    
    # Weight recent games more heavily
    weights = [1.2 ** i for i in range(len(points))]
    weights.reverse()
    
    weighted_points = sum(p * w for p, w in zip(points, weights))
    max_points = sum(3 * w for w in weights)
    
    return weighted_points / max_points


def calculate_weather_impact(weather_data: Dict, sport: str) -> float:
    """Calculate weather impact on game outcome."""
    if not weather_data:
        return 0.0
    
    impact = 0.0
    
    # Temperature impact
    temp = weather_data.get("temperature", 70)
    if sport in ["nfl", "soccer"]:
        if temp < 32:  # Freezing
            impact += 0.15
        elif temp > 90:  # Very hot
            impact += 0.10
    
    # Wind impact
    wind_speed = weather_data.get("wind_speed", 0)
    if sport in ["nfl", "soccer"] and wind_speed > 20:
        impact += 0.20
    
    # Precipitation impact
    if weather_data.get("precipitation", False):
        if sport in ["nfl", "soccer"]:
            impact += 0.25
        elif sport == "mlb":
            impact += 0.30
    
    return min(impact, 0.5)


def generate_game_id(home_team: str, away_team: str, date: datetime) -> str:
    """Generate unique game ID."""
    date_str = date.strftime("%Y%m%d")
    teams = f"{home_team}_{away_team}_{date_str}"
    return hashlib.md5(teams.encode()).hexdigest()[:16]


def calculate_rest_advantage(home_rest_days: int, away_rest_days: int) -> float:
    """Calculate rest advantage between teams."""
    if home_rest_days == away_rest_days:
        return 0.0
    
    # Optimal rest is 2-3 days
    home_score = 1.0 if 2 <= home_rest_days <= 3 else 0.7
    away_score = 1.0 if 2 <= away_rest_days <= 3 else 0.7
    
    # Too much rest can be bad
    if home_rest_days > 7:
        home_score *= 0.9
    if away_rest_days > 7:
        away_score *= 0.9
    
    # Too little rest is very bad
    if home_rest_days < 2:
        home_score *= 0.8
    if away_rest_days < 2:
        away_score *= 0.8
    
    return (home_score - away_score) * 0.2


def calculate_calibration_error(predictions: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    """Calculate calibration error for probability predictions."""
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    calibration_errors = []
    
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        
        mask = (np.array(predictions) >= lower) & (np.array(predictions) < upper)
        if not any(mask):
            continue
        
        bin_predictions = np.array(predictions)[mask]
        bin_outcomes = np.array(outcomes)[mask]
        
        mean_prediction = np.mean(bin_predictions)
        mean_outcome = np.mean(bin_outcomes)
        
        calibration_errors.append(abs(mean_prediction - mean_outcome))
    
    return np.mean(calibration_errors) if calibration_errors else 0.0


def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate Brier score for probability predictions."""
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    return np.mean((predictions - outcomes) ** 2)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    return numerator / denominator if denominator != 0 else default


def format_currency(amount: float) -> str:
    """Format currency for display."""
    return f"${amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage for display."""
    return f"{value * 100:.{decimal_places}f}%"


def get_current_time(timezone: str = "UTC") -> datetime:
    """Get current time in specified timezone."""
    tz = pytz.timezone(timezone)
    return datetime.now(tz)


def time_until_game(game_time: datetime) -> timedelta:
    """Calculate time until game starts."""
    now = datetime.utcnow()
    if game_time.tzinfo is None:
        game_time = pytz.utc.localize(game_time)
    
    return game_time - now


def is_game_live(game_time: datetime, sport: str) -> bool:
    """Check if game is currently live."""
    time_since_start = datetime.utcnow() - game_time
    
    game_durations = {
        "nfl": timedelta(hours=3.5),
        "nba": timedelta(hours=2.5),
        "mlb": timedelta(hours=3.5),
        "nhl": timedelta(hours=3),
        "soccer": timedelta(hours=2),
        "tennis": timedelta(hours=3)
    }
    
    duration = game_durations.get(sport, timedelta(hours=3))
    return timedelta() < time_since_start < duration


def validate_odds(odds: float) -> bool:
    """Validate odds are in reasonable range."""
    # Check if decimal odds
    if 1.01 <= odds <= 100:
        return True
    # Check if American odds
    if abs(odds) >= 100 and abs(odds) <= 10000:
        return True
    return False


def calculate_roi(total_profit: float, total_staked: float) -> float:
    """Calculate return on investment."""
    return safe_divide(total_profit, total_staked, 0.0)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for returns."""
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
    
    return safe_divide(np.mean(excess_returns), np.std(excess_returns), 0.0) * np.sqrt(252) 