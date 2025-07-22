"""Feature engineering for Contrarius betting engine."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from database import db_manager, Game, Odds, Bet
from utils import (
    calculate_travel_fatigue, parse_injury_impact, calculate_momentum,
    calculate_market_volatility, calculate_form_rating, calculate_weather_impact,
    calculate_rest_advantage, sanitize_team_name
)
from config import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self):
        self.feature_modules = {
            'basic': BasicFeatures(),
            'odds': OddsFeatures(),
            'injury': InjuryFeatures(),
            'weather': WeatherFeatures(),
            'travel': TravelFeatures(),
            'sentiment': SentimentFeatures(),
            'form': FormFeatures(),
            'technical': TechnicalIndicators(),
            'sharp': SharpMoneyFeatures()
        }
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
    def extract_features(self, game: Game, historical_data: Optional[Dict] = None) -> pd.DataFrame:
        """Extract all features for a game."""
        features = {}
        
        # Extract features from each module
        for name, module in self.feature_modules.items():
            try:
                module_features = module.extract(game, historical_data)
                features.update(module_features)
            except Exception as e:
                logger.error(f"Error extracting {name} features: {e}")
        
        # Add interaction features
        interaction_features = self._create_interaction_features(features)
        features.update(interaction_features)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Remove features with low variance
        df = self._remove_low_variance_features(df)
        
        return df
    
    def _create_interaction_features(self, features: Dict) -> Dict:
        """Create interaction features between key variables."""
        interactions = {}
        
        # Form × Rest advantage
        if 'home_form_rating' in features and 'rest_advantage' in features:
            interactions['form_rest_interaction'] = (
                features['home_form_rating'] * features['rest_advantage']
            )
        
        # Injury impact × Sharp money
        if 'home_injury_impact' in features and 'sharp_money_home' in features:
            interactions['injury_sharp_interaction'] = (
                features['home_injury_impact'] * features['sharp_money_home']
            )
        
        # Weather × Travel fatigue
        if 'weather_impact' in features and 'away_travel_fatigue' in features:
            interactions['weather_travel_interaction'] = (
                features['weather_impact'] * features['away_travel_fatigue']
            )
        
        # Market volatility × Public fade
        if 'market_volatility' in features and 'public_fade_indicator' in features:
            interactions['volatility_fade_interaction'] = (
                features['market_volatility'] * features['public_fade_indicator']
            )
        
        # Momentum × Line movement
        if 'home_momentum' in features and 'line_movement_velocity' in features:
            interactions['momentum_line_interaction'] = (
                features['home_momentum'] * features['line_movement_velocity']
            )
        
        return interactions
    
    def _remove_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance."""
        if len(df) < 2:
            return df
        
        variances = df.var()
        low_variance_features = variances[variances < threshold].index.tolist()
        
        if low_variance_features:
            logger.info(f"Removing low variance features: {low_variance_features}")
            df = df.drop(columns=low_variance_features)
        
        return df
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get feature importance weights for each module."""
        return {
            'odds': config.features.odds_movement_weight,
            'sharp': config.features.sharp_money_weight,
            'sentiment': config.features.market_sentiment_weight,
            'injury': config.features.injury_weight,
            'weather': config.features.weather_weight,
            'travel': config.features.travel_fatigue_weight,
            'form': 0.15,
            'technical': 0.10,
            'basic': 0.05
        }


class BasicFeatures:
    """Extract basic game features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract basic features from game."""
        features = {}
        
        # Time-based features
        features['hour_of_day'] = game.start_time.hour
        features['day_of_week'] = game.start_time.weekday()
        features['month'] = game.start_time.month
        features['is_weekend'] = float(game.start_time.weekday() >= 5)
        
        # Sport-specific
        features[f'sport_{game.sport}'] = 1.0
        
        # Home field advantage
        features['is_home_game'] = 1.0  # Always 1 for home team perspective
        
        # Season progress (if available)
        if historical_data and 'season_game_number' in historical_data:
            features['season_progress'] = historical_data['season_game_number'] / 82  # Adjust per sport
        
        return features


class OddsFeatures:
    """Extract odds-based features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract odds movement and market features."""
        features = {}
        session = db_manager.get_session()
        
        # Get all odds for this game
        odds_data = session.query(Odds).filter(Odds.game_id == game.id).all()
        
        if not odds_data:
            session.close()
            return features
        
        # Group by market type
        moneyline_odds = [o for o in odds_data if o.market_type == 'moneyline']
        spread_odds = [o for o in odds_data if o.market_type == 'spread']
        total_odds = [o for o in odds_data if o.market_type == 'total']
        
        # Moneyline features
        if moneyline_odds:
            features.update(self._extract_market_features(moneyline_odds, 'moneyline'))
        
        # Spread features
        if spread_odds:
            features.update(self._extract_market_features(spread_odds, 'spread'))
            features['spread_value'] = self._get_consensus_line(spread_odds)
        
        # Total features
        if total_odds:
            features.update(self._extract_market_features(total_odds, 'total'))
            features['total_value'] = self._get_consensus_line(total_odds)
        
        # Cross-market correlations
        if moneyline_odds and spread_odds:
            features['ml_spread_correlation'] = self._calculate_market_correlation(
                moneyline_odds, spread_odds
            )
        
        session.close()
        return features
    
    def _extract_market_features(self, odds_list: List[Odds], market_type: str) -> Dict[str, float]:
        """Extract features for a specific market."""
        features = {}
        
        # Get odds by timestamp
        odds_by_time = sorted(odds_list, key=lambda x: x.timestamp)
        
        # Opening vs current odds
        if len(odds_by_time) >= 2:
            opening = odds_by_time[0]
            current = odds_by_time[-1]
            
            features[f'{market_type}_movement'] = current.odds - opening.odds
            features[f'{market_type}_movement_pct'] = (
                (current.odds - opening.odds) / opening.odds
            )
        
        # Market volatility
        if len(odds_by_time) >= 3:
            odds_values = [o.odds for o in odds_by_time]
            features[f'{market_type}_volatility'] = calculate_market_volatility(odds_values)
        
        # Bookmaker disagreement
        current_odds = [o for o in odds_list if 
                       (datetime.utcnow() - o.timestamp).seconds < 3600]
        
        if len(current_odds) >= 2:
            odds_values = [o.odds for o in current_odds]
            features[f'{market_type}_disagreement'] = np.std(odds_values)
        
        # Line shopping value
        if current_odds:
            best_odds = max(o.odds for o in current_odds if o.selection == 'home')
            worst_odds = min(o.odds for o in current_odds if o.selection == 'home')
            features[f'{market_type}_line_value'] = (best_odds - worst_odds) / worst_odds
        
        return features
    
    def _get_consensus_line(self, odds_list: List[Odds]) -> float:
        """Calculate consensus line from multiple bookmakers."""
        recent_odds = [o for o in odds_list if 
                      (datetime.utcnow() - o.timestamp).seconds < 3600]
        
        if not recent_odds:
            return 0.0
        
        lines = [o.line for o in recent_odds if o.line is not None]
        return np.median(lines) if lines else 0.0
    
    def _calculate_market_correlation(self, market1: List[Odds], market2: List[Odds]) -> float:
        """Calculate correlation between two markets."""
        # Match odds by timestamp
        times1 = {o.timestamp: o.odds for o in market1}
        times2 = {o.timestamp: o.odds for o in market2}
        
        common_times = set(times1.keys()) & set(times2.keys())
        
        if len(common_times) < 3:
            return 0.0
        
        values1 = [times1[t] for t in sorted(common_times)]
        values2 = [times2[t] for t in sorted(common_times)]
        
        return np.corrcoef(values1, values2)[0, 1]


class InjuryFeatures:
    """Extract injury-related features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract injury impact features."""
        features = {}
        
        if not game.injuries_data:
            return features
        
        # Calculate team injury impacts
        for team in ['home', 'away']:
            injuries = game.injuries_data.get(team, [])
            
            # Total injury impact
            total_impact = sum(parse_injury_impact(inj.get('status', '')) 
                             for inj in injuries)
            features[f'{team}_injury_impact'] = total_impact
            
            # Number of injured players
            features[f'{team}_injured_count'] = len(injuries)
            
            # Max single injury impact
            if injuries:
                max_impact = max(parse_injury_impact(inj.get('status', '')) 
                               for inj in injuries)
                features[f'{team}_max_injury'] = max_impact
            else:
                features[f'{team}_max_injury'] = 0.0
            
            # Key player injuries (would need player importance data)
            key_player_impact = sum(
                parse_injury_impact(inj.get('status', '')) * 
                inj.get('player_importance', 0.5)
                for inj in injuries
            )
            features[f'{team}_key_player_impact'] = key_player_impact
        
        # Relative injury advantage
        features['injury_advantage'] = (
            features['away_injury_impact'] - features['home_injury_impact']
        )
        
        return features


class WeatherFeatures:
    """Extract weather-related features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract weather impact features."""
        features = {}
        
        if not game.weather_data:
            return features
        
        weather = game.weather_data
        
        # Basic weather metrics
        features['temperature'] = weather.get('temperature', 70)
        features['wind_speed'] = weather.get('wind_speed', 0)
        features['humidity'] = weather.get('humidity', 50)
        features['precipitation'] = float(weather.get('precipitation', False))
        
        # Temperature extremes
        temp = weather.get('temperature', 70)
        features['extreme_cold'] = float(temp < 32)
        features['extreme_heat'] = float(temp > 90)
        
        # Wind categories
        wind = weather.get('wind_speed', 0)
        features['high_wind'] = float(wind > 20)
        features['moderate_wind'] = float(10 < wind <= 20)
        
        # Sport-specific weather impact
        features['weather_impact'] = calculate_weather_impact(weather, game.sport)
        
        # Dome/indoor adjustment
        if game.venue and 'dome' in game.venue.lower():
            features['weather_impact'] *= 0.1  # Minimal impact indoors
            
        return features


class TravelFeatures:
    """Extract travel and schedule-related features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract travel fatigue features."""
        features = {}
        
        if not game.travel_data:
            return features
        
        travel = game.travel_data
        
        # Travel distance
        for team in ['home', 'away']:
            distance = travel.get(f'{team}_distance', 0)
            time_zones = travel.get(f'{team}_time_zones', 0)
            days_since = travel.get(f'{team}_days_since_travel', 3)
            
            features[f'{team}_travel_distance'] = distance
            features[f'{team}_time_zones_crossed'] = time_zones
            features[f'{team}_travel_fatigue'] = calculate_travel_fatigue(
                distance, time_zones, days_since
            )
        
        # Rest days
        home_rest = travel.get('home_rest_days', 3)
        away_rest = travel.get('away_rest_days', 3)
        
        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['rest_advantage'] = calculate_rest_advantage(home_rest, away_rest)
        
        # Back-to-back games
        features['home_b2b'] = float(home_rest == 0)
        features['away_b2b'] = float(away_rest == 0)
        
        # Schedule density (games in last X days)
        features['home_games_last_week'] = travel.get('home_games_7d', 2)
        features['away_games_last_week'] = travel.get('away_games_7d', 2)
        
        return features


class SentimentFeatures:
    """Extract market sentiment features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract sentiment and public betting features."""
        features = {}
        
        if not game.sentiment_data:
            return features
        
        sentiment = game.sentiment_data
        
        # Public betting percentages
        public = sentiment.get('public_betting_percentage', {})
        features['public_home_pct'] = public.get('home', 50)
        features['public_away_pct'] = public.get('away', 50)
        features['public_over_pct'] = public.get('over', 50)
        features['public_under_pct'] = public.get('under', 50)
        
        # Fade the public indicators
        features['public_fade_indicator'] = float(
            features['public_home_pct'] > 70 or features['public_away_pct'] > 70
        )
        
        # Social sentiment
        social = sentiment.get('social_sentiment', {})
        if social:
            for team in ['home', 'away']:
                team_sentiment = social.get(team, {})
                total = sum(team_sentiment.values())
                
                if total > 0:
                    features[f'{team}_sentiment_positive'] = (
                        team_sentiment.get('positive', 0) / total
                    )
                    features[f'{team}_sentiment_negative'] = (
                        team_sentiment.get('negative', 0) / total
                    )
        
        # Line movement patterns
        line_movement = sentiment.get('line_movement', [])
        if len(line_movement) >= 2:
            # Movement velocity
            time_diff = (line_movement[-1]['timestamp'] - 
                        line_movement[0]['timestamp']).seconds / 3600
            line_diff = line_movement[-1]['line'] - line_movement[0]['line']
            
            features['line_movement_velocity'] = line_diff / (time_diff + 0.1)
            
            # Movement direction vs public
            features['reverse_line_movement'] = float(
                (line_diff > 0 and features['public_away_pct'] > 60) or
                (line_diff < 0 and features['public_home_pct'] > 60)
            )
        
        return features


class FormFeatures:
    """Extract team form and momentum features."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract form and momentum features."""
        features = {}
        
        if not historical_data:
            return features
        
        # Recent results
        for team in ['home', 'away']:
            results = historical_data.get(f'{team}_recent_results', [])
            
            if results:
                # Win/loss record
                wins = sum(1 for r in results if r == 1)
                features[f'{team}_recent_wins'] = wins
                features[f'{team}_win_pct_recent'] = wins / len(results)
                
                # Momentum
                features[f'{team}_momentum'] = calculate_momentum(results)
                
                # Streak
                current_streak = 0
                for r in reversed(results):
                    if r == results[-1]:
                        current_streak += 1
                    else:
                        break
                
                features[f'{team}_streak'] = current_streak * (1 if results[-1] == 1 else -1)
            
            # Form ratings
            team_results = historical_data.get(f'{team}_detailed_results', [])
            if team_results:
                features[f'{team}_form_rating'] = calculate_form_rating(team_results)
                features[f'{team}_home_form'] = calculate_form_rating(team_results, 'home')
                features[f'{team}_away_form'] = calculate_form_rating(team_results, 'away')
            
            # Scoring trends
            recent_scores = historical_data.get(f'{team}_recent_scores', [])
            if recent_scores:
                features[f'{team}_avg_points_for'] = np.mean([s['for'] for s in recent_scores])
                features[f'{team}_avg_points_against'] = np.mean([s['against'] for s in recent_scores])
                features[f'{team}_scoring_trend'] = np.polyfit(
                    range(len(recent_scores)), 
                    [s['for'] for s in recent_scores], 
                    1
                )[0]
        
        # Head-to-head
        h2h = historical_data.get('head_to_head', [])
        if h2h:
            features['h2h_home_wins'] = sum(1 for g in h2h if g['winner'] == 'home')
            features['h2h_games'] = len(h2h)
            features['h2h_avg_total'] = np.mean([g['total'] for g in h2h])
        
        return features


class TechnicalIndicators:
    """Extract technical market indicators."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract technical indicators from odds movement."""
        features = {}
        session = db_manager.get_session()
        
        # Get odds history
        odds_history = session.query(Odds).filter(
            Odds.game_id == game.id
        ).order_by(Odds.timestamp).all()
        
        if len(odds_history) < 5:
            session.close()
            return features
        
        # Calculate technical indicators for each market
        for market_type in ['moneyline', 'spread', 'total']:
            market_odds = [o for o in odds_history if o.market_type == market_type]
            
            if len(market_odds) < 5:
                continue
            
            odds_values = [o.odds for o in market_odds]
            
            # Moving averages
            features[f'{market_type}_ma5'] = np.mean(odds_values[-5:])
            features[f'{market_type}_ma10'] = np.mean(odds_values[-10:]) if len(odds_values) >= 10 else np.mean(odds_values)
            
            # RSI (Relative Strength Index)
            if len(odds_values) >= 14:
                features[f'{market_type}_rsi'] = self._calculate_rsi(odds_values)
            
            # Bollinger Bands
            if len(odds_values) >= 20:
                mean = np.mean(odds_values[-20:])
                std = np.std(odds_values[-20:])
                current = odds_values[-1]
                
                features[f'{market_type}_bb_position'] = (current - mean) / (2 * std)
                features[f'{market_type}_bb_width'] = 2 * std
            
            # Volume-weighted movement
            if hasattr(market_odds[0], 'volume') and market_odds[0].volume:
                volumes = [o.volume for o in market_odds]
                vwap = sum(o.odds * o.volume for o in market_odds) / sum(volumes)
                features[f'{market_type}_vwap'] = vwap
                features[f'{market_type}_vwap_deviation'] = (odds_values[-1] - vwap) / vwap
        
        session.close()
        return features
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        # Smooth with Wilder's method
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                up = (up * (period - 1) + delta) / period
                down = (down * (period - 1)) / period
            else:
                up = (up * (period - 1)) / period
                down = (down * (period - 1) + abs(delta)) / period
            
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
        
        return rsi


class SharpMoneyFeatures:
    """Extract sharp money indicators."""
    
    def extract(self, game: Game, historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract sharp money and professional betting features."""
        features = {}
        
        sentiment = game.sentiment_data
        if not sentiment:
            return features
        
        # Sharp money indicators
        sharp_indicators = sentiment.get('sharp_money_indicators', [])
        
        # Count and strength of sharp signals
        features['sharp_signal_count'] = len(sharp_indicators)
        
        if sharp_indicators:
            # Aggregate by side
            home_sharp = sum(1 for s in sharp_indicators if s.get('side') == 'home')
            away_sharp = sum(1 for s in sharp_indicators if s.get('side') == 'away')
            
            features['sharp_money_home'] = home_sharp
            features['sharp_money_away'] = away_sharp
            
            # Strength scores
            strength_map = {'strong': 1.0, 'moderate': 0.5, 'weak': 0.25}
            home_strength = sum(
                strength_map.get(s.get('strength', 'weak'), 0.25)
                for s in sharp_indicators if s.get('side') == 'home'
            )
            away_strength = sum(
                strength_map.get(s.get('strength', 'weak'), 0.25)
                for s in sharp_indicators if s.get('side') == 'away'
            )
            
            features['sharp_strength_home'] = home_strength
            features['sharp_strength_away'] = away_strength
            
            # Steam moves
            steam_moves = [s for s in sharp_indicators if s.get('type') == 'steam_move']
            features['steam_move_count'] = len(steam_moves)
            
            if steam_moves:
                latest_steam = max(steam_moves, key=lambda x: x.get('timestamp', 0))
                features['steam_direction'] = 1.0 if latest_steam.get('direction') == 'home' else -1.0
                features['steam_magnitude'] = latest_steam.get('movement', 0)
        
        # Professional vs public disagreement
        public = sentiment.get('public_betting_percentage', {})
        if sharp_indicators and public:
            # Check if sharp money opposes public
            public_on_home = public.get('home', 50) > 50
            sharp_on_home = features.get('sharp_money_home', 0) > features.get('sharp_money_away', 0)
            
            features['sharp_public_disagreement'] = float(public_on_home != sharp_on_home)
        
        # Line value indicators
        session = db_manager.get_session()
        current_odds = session.query(Odds).filter(
            Odds.game_id == game.id,
            Odds.timestamp > datetime.utcnow() - timedelta(hours=1)
        ).all()
        
        if current_odds:
            # Check for outlier books
            spread_odds = [o.line for o in current_odds if o.market_type == 'spread' and o.line]
            if len(spread_odds) >= 3:
                median_spread = np.median(spread_odds)
                outliers = [abs(s - median_spread) > 0.5 for s in spread_odds]
                features['outlier_book_count'] = sum(outliers)
        
        session.close()
        return features


# Global feature engineer instance
feature_engineer = FeatureEngineer() 