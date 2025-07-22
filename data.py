"""Data collection and management for Contrarius betting engine."""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tweepy
from betfairlightweight import APIClient
import time

from config import config
from database import db_manager, Game, Odds
from utils import retry_on_failure, sanitize_team_name, generate_game_id

logger = logging.getLogger(__name__)


class DataCollector:
    """Main data collection orchestrator."""
    
    def __init__(self):
        self.odds_api = OddsAPIClient()
        self.sportradar = SportradarClient()
        self.betfair = BetfairClient()
        self.weather_api = WeatherAPIClient()
        self.twitter_api = TwitterClient()
        self.web_scraper = WebScraper()
        
    async def collect_all_data(self):
        """Collect data from all sources."""
        logger.info("Starting comprehensive data collection...")
        
        # Get upcoming games
        games = await self._get_upcoming_games()
        
        # Collect data for each game
        tasks = []
        for game in games:
            tasks.append(self._collect_game_data(game))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store in database
        session = db_manager.get_session()
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error collecting game data: {result}")
                continue
            
            try:
                # Store game
                game_obj = db_manager.add_game(session, result['game'])
                
                # Store odds
                if result.get('odds'):
                    db_manager.add_odds(session, result['odds'])
                    
            except Exception as e:
                logger.error(f"Error storing game data: {e}")
                session.rollback()
        
        session.close()
        logger.info("Data collection completed")
    
    async def _get_upcoming_games(self) -> List[Dict]:
        """Get list of upcoming games across all sports."""
        games = []
        
        for sport in config.features.sports_enabled:
            try:
                sport_games = await self.odds_api.get_upcoming_games(sport)
                games.extend(sport_games)
            except Exception as e:
                logger.error(f"Error getting games for {sport}: {e}")
        
        return games
    
    async def _collect_game_data(self, game: Dict) -> Dict:
        """Collect all available data for a single game."""
        result = {'game': game, 'odds': [], 'features': {}}
        
        # Collect odds from multiple sources
        try:
            odds_data = await self._collect_odds_data(game)
            result['odds'] = odds_data
        except Exception as e:
            logger.error(f"Error collecting odds: {e}")
        
        # Collect injury data
        try:
            injuries = await self._collect_injury_data(game)
            game['injuries_data'] = injuries
        except Exception as e:
            logger.error(f"Error collecting injuries: {e}")
        
        # Collect weather data
        if game['sport'] in ['nfl', 'mlb', 'soccer']:
            try:
                weather = await self.weather_api.get_game_weather(
                    game['venue'], game['start_time']
                )
                game['weather_data'] = weather
            except Exception as e:
                logger.error(f"Error collecting weather: {e}")
        
        # Collect sentiment data
        try:
            sentiment = await self._collect_sentiment_data(game)
            game['sentiment_data'] = sentiment
        except Exception as e:
            logger.error(f"Error collecting sentiment: {e}")
        
        return result
    
    async def _collect_odds_data(self, game: Dict) -> List[Dict]:
        """Collect odds from multiple bookmakers."""
        odds_list = []
        
        # The Odds API
        try:
            odds_api_data = await self.odds_api.get_game_odds(game['external_id'])
            for bookmaker, markets in odds_api_data.items():
                for market_type, selections in markets.items():
                    for selection, odds in selections.items():
                        odds_list.append({
                            'game_id': game['id'],
                            'bookmaker': bookmaker,
                            'market_type': market_type,
                            'selection': selection,
                            'odds': odds['decimal'],
                            'implied_probability': 1 / odds['decimal'],
                            'line': odds.get('line'),
                            'timestamp': datetime.utcnow()
                        })
        except Exception as e:
            logger.error(f"Error getting odds from API: {e}")
        
        # Betfair Exchange
        try:
            betfair_odds = await self.betfair.get_market_odds(
                game['sport'], game['home_team'], game['away_team']
            )
            odds_list.extend(betfair_odds)
        except Exception as e:
            logger.error(f"Error getting Betfair odds: {e}")
        
        # Web scraping for additional bookmakers
        try:
            scraped_odds = await self.web_scraper.scrape_odds(game)
            odds_list.extend(scraped_odds)
        except Exception as e:
            logger.error(f"Error scraping odds: {e}")
        
        return odds_list
    
    async def _collect_injury_data(self, game: Dict) -> Dict:
        """Collect injury reports."""
        injuries = {'home': [], 'away': []}
        
        try:
            # Try official sources first
            if game['sport'] == 'nfl':
                injuries = await self.web_scraper.scrape_nfl_injuries(
                    game['home_team'], game['away_team']
                )
            elif game['sport'] == 'nba':
                injuries = await self.sportradar.get_nba_injuries(
                    game['home_team'], game['away_team']
                )
            
            # Supplement with Twitter
            twitter_injuries = await self.twitter_api.search_injuries(
                game['home_team'], game['away_team'], game['sport']
            )
            
            # Merge and deduplicate
            for team in ['home', 'away']:
                injuries[team] = self._merge_injury_reports(
                    injuries.get(team, []), 
                    twitter_injuries.get(team, [])
                )
                
        except Exception as e:
            logger.error(f"Error collecting injuries: {e}")
        
        return injuries
    
    async def _collect_sentiment_data(self, game: Dict) -> Dict:
        """Collect market sentiment and sharp money indicators."""
        sentiment = {
            'public_betting_percentage': {},
            'sharp_money_indicators': [],
            'social_sentiment': {},
            'line_movement': []
        }
        
        try:
            # Scrape public betting percentages
            public_data = await self.web_scraper.scrape_public_betting(game)
            sentiment['public_betting_percentage'] = public_data
            
            # Analyze line movement for sharp money
            line_history = await self._get_line_movement_history(game)
            sentiment['line_movement'] = line_history
            
            # Detect sharp money patterns
            if line_history:
                sharp_indicators = self._detect_sharp_patterns(
                    line_history, public_data
                )
                sentiment['sharp_money_indicators'] = sharp_indicators
            
            # Social media sentiment
            social = await self.twitter_api.analyze_game_sentiment(
                game['home_team'], game['away_team']
            )
            sentiment['social_sentiment'] = social
            
        except Exception as e:
            logger.error(f"Error collecting sentiment: {e}")
        
        return sentiment
    
    def _merge_injury_reports(self, official: List[Dict], twitter: List[Dict]) -> List[Dict]:
        """Merge and deduplicate injury reports."""
        merged = {}
        
        # Add official reports first (more reliable)
        for injury in official:
            key = injury['player'].lower()
            merged[key] = injury
        
        # Add Twitter reports if not duplicate
        for injury in twitter:
            key = injury['player'].lower()
            if key not in merged:
                injury['source'] = 'twitter'
                merged[key] = injury
        
        return list(merged.values())
    
    def _detect_sharp_patterns(self, line_history: List[Dict], 
                              public_betting: Dict) -> List[Dict]:
        """Detect sharp money patterns from line movement."""
        patterns = []
        
        if len(line_history) < 2:
            return patterns
        
        # Reverse line movement
        initial_line = line_history[0]['line']
        current_line = line_history[-1]['line']
        line_change = current_line - initial_line
        
        # Check each side
        for side in ['home', 'away']:
            public_pct = public_betting.get(side, 50)
            
            # Sharp pattern: line moves against public money
            if public_pct > 65 and line_change * (1 if side == 'home' else -1) < -0.5:
                patterns.append({
                    'type': 'reverse_line_movement',
                    'side': side,
                    'strength': 'strong',
                    'public_percentage': public_pct,
                    'line_movement': line_change
                })
            elif public_pct > 55 and line_change * (1 if side == 'home' else -1) < -0.25:
                patterns.append({
                    'type': 'reverse_line_movement',
                    'side': side,
                    'strength': 'moderate',
                    'public_percentage': public_pct,
                    'line_movement': line_change
                })
        
        # Steam moves (rapid line movement)
        for i in range(1, len(line_history)):
            time_diff = (line_history[i]['timestamp'] - line_history[i-1]['timestamp']).seconds
            line_diff = abs(line_history[i]['line'] - line_history[i-1]['line'])
            
            if time_diff < 300 and line_diff >= 1.0:  # 5 minutes, 1 point
                patterns.append({
                    'type': 'steam_move',
                    'timestamp': line_history[i]['timestamp'],
                    'movement': line_diff,
                    'direction': 'home' if line_history[i]['line'] > line_history[i-1]['line'] else 'away'
                })
        
        return patterns
    
    async def _get_line_movement_history(self, game: Dict) -> List[Dict]:
        """Get historical line movement data."""
        session = db_manager.get_session()
        
        odds_history = session.query(Odds).filter(
            Odds.game_id == game['id'],
            Odds.market_type == 'spread'
        ).order_by(Odds.timestamp).all()
        
        history = []
        for odds in odds_history:
            history.append({
                'timestamp': odds.timestamp,
                'line': odds.line,
                'bookmaker': odds.bookmaker
            })
        
        session.close()
        return history


class OddsAPIClient:
    """Client for The Odds API."""
    
    def __init__(self):
        self.api_key = config.api.odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"
    
    @retry_on_failure(max_retries=3)
    async def get_upcoming_games(self, sport: str) -> List[Dict]:
        """Get upcoming games for a sport."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/sports/{sport}/events"
            params = {'apiKey': self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_games(data, sport)
                else:
                    logger.error(f"Error fetching games: {response.status}")
                    return []
    
    @retry_on_failure(max_retries=3)
    async def get_game_odds(self, game_id: str) -> Dict:
        """Get odds for a specific game."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/sports/upcoming/odds"
            params = {
                'apiKey': self.api_key,
                'eventIds': game_id,
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'decimal'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_odds(data)
                else:
                    logger.error(f"Error fetching odds: {response.status}")
                    return {}
    
    def _parse_games(self, data: List[Dict], sport: str) -> List[Dict]:
        """Parse games from API response."""
        games = []
        
        for event in data:
            game = {
                'external_id': event['id'],
                'sport': sport,
                'league': event.get('sport_title', sport),
                'home_team': sanitize_team_name(event['home_team']),
                'away_team': sanitize_team_name(event['away_team']),
                'start_time': datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')),
                'venue': event.get('home_team', '')  # Often venue not provided
            }
            games.append(game)
        
        return games
    
    def _parse_odds(self, data: List[Dict]) -> Dict:
        """Parse odds from API response."""
        odds_by_bookmaker = {}
        
        for event in data:
            for bookmaker in event.get('bookmakers', []):
                bookie_name = bookmaker['key']
                odds_by_bookmaker[bookie_name] = {}
                
                for market in bookmaker['markets']:
                    market_type = market['key']
                    odds_by_bookmaker[bookie_name][market_type] = {}
                    
                    for outcome in market['outcomes']:
                        selection = outcome['name']
                        odds_by_bookmaker[bookie_name][market_type][selection] = {
                            'decimal': outcome['price'],
                            'line': outcome.get('point')
                        }
        
        return odds_by_bookmaker


class SportradarClient:
    """Client for Sportradar API."""
    
    def __init__(self):
        self.api_key = config.api.sportradar_key
        self.base_url = "https://api.sportradar.com"
    
    @retry_on_failure(max_retries=3)
    async def get_nba_injuries(self, home_team: str, away_team: str) -> Dict:
        """Get NBA injury reports."""
        # Implementation would depend on Sportradar API structure
        # This is a placeholder
        return {'home': [], 'away': []}


class BetfairClient:
    """Client for Betfair Exchange API."""
    
    def __init__(self):
        self.username = config.api.betfair_username
        self.password = config.api.betfair_password
        self.app_key = config.api.betfair_app_key
        self.client = None
        
    async def initialize(self):
        """Initialize Betfair client."""
        if not all([self.username, self.password, self.app_key]):
            logger.warning("Betfair credentials not configured")
            return
        
        self.client = APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key
        )
        self.client.login()
    
    async def get_market_odds(self, sport: str, home_team: str, away_team: str) -> List[Dict]:
        """Get Betfair exchange odds."""
        if not self.client:
            await self.initialize()
            
        if not self.client:
            return []
        
        # Placeholder - would implement actual Betfair API calls
        return []


class WeatherAPIClient:
    """Client for weather data."""
    
    def __init__(self):
        self.api_key = config.api.weather_api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    @retry_on_failure(max_retries=3)
    async def get_game_weather(self, venue: str, game_time: datetime) -> Dict:
        """Get weather forecast for game time and location."""
        # Get venue coordinates (would need geocoding in real implementation)
        # For now, return sample data
        
        weather_data = {
            'temperature': 72,
            'feels_like': 75,
            'humidity': 60,
            'wind_speed': 10,
            'wind_direction': 'NW',
            'precipitation': False,
            'conditions': 'clear'
        }
        
        return weather_data


class TwitterClient:
    """Client for Twitter sentiment analysis."""
    
    def __init__(self):
        self.bearer_token = config.api.twitter_bearer_token
        self.client = None
        
        if self.bearer_token:
            self.client = tweepy.Client(bearer_token=self.bearer_token)
    
    async def search_injuries(self, home_team: str, away_team: str, sport: str) -> Dict:
        """Search Twitter for injury updates."""
        if not self.client:
            return {'home': [], 'away': []}
        
        injuries = {'home': [], 'away': []}
        
        # Search for each team
        for team, key in [(home_team, 'home'), (away_team, 'away')]:
            query = f'"{team}" (injury OR injured OR "injury report" OR questionable OR doubtful OR out) -{is:retweet}'
            
            try:
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=50,
                    tweet_fields=['created_at', 'author_id']
                )
                
                if tweets.data:
                    # Parse tweets for injury information
                    for tweet in tweets.data:
                        injury_info = self._parse_injury_tweet(tweet.text, team)
                        if injury_info:
                            injuries[key].append(injury_info)
                            
            except Exception as e:
                logger.error(f"Error searching Twitter: {e}")
        
        return injuries
    
    async def analyze_game_sentiment(self, home_team: str, away_team: str) -> Dict:
        """Analyze Twitter sentiment for teams."""
        if not self.client:
            return {}
        
        sentiment = {
            'home': {'positive': 0, 'negative': 0, 'neutral': 0},
            'away': {'positive': 0, 'negative': 0, 'neutral': 0}
        }
        
        # Would implement actual sentiment analysis here
        # For now return sample data
        sentiment['home'] = {'positive': 45, 'negative': 30, 'neutral': 25}
        sentiment['away'] = {'positive': 40, 'negative': 35, 'neutral': 25}
        
        return sentiment
    
    def _parse_injury_tweet(self, text: str, team: str) -> Optional[Dict]:
        """Parse tweet for injury information."""
        # Simple keyword extraction - would use NLP in production
        injury_keywords = {
            'out': 0.9,
            'doubtful': 0.7,
            'questionable': 0.5,
            'probable': 0.3,
            'day-to-day': 0.2
        }
        
        text_lower = text.lower()
        
        for keyword, severity in injury_keywords.items():
            if keyword in text_lower:
                # Extract player name (simplified)
                words = text.split()
                for i, word in enumerate(words):
                    if word.lower() == team.lower() and i > 0:
                        player = words[i-1]
                        return {
                            'player': player,
                            'status': keyword,
                            'severity': severity,
                            'source': 'twitter',
                            'text': text[:200]
                        }
        
        return None


class WebScraper:
    """Web scraper for additional data sources."""
    
    def __init__(self):
        self.driver = None
        
    def _init_driver(self):
        """Initialize Selenium driver."""
        if not self.driver:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)
    
    async def scrape_odds(self, game: Dict) -> List[Dict]:
        """Scrape odds from bookmaker websites."""
        # Placeholder - would implement actual scraping
        return []
    
    async def scrape_public_betting(self, game: Dict) -> Dict:
        """Scrape public betting percentages."""
        # Would scrape from sites like ActionNetwork, VSiN, etc.
        # For now return sample data
        return {
            'home': 55,
            'away': 45,
            'over': 62,
            'under': 38
        }
    
    async def scrape_nfl_injuries(self, home_team: str, away_team: str) -> Dict:
        """Scrape NFL injury reports."""
        injuries = {'home': [], 'away': []}
        
        # Would scrape from NFL.com or ESPN
        # Placeholder implementation
        
        return injuries
    
    def __del__(self):
        """Clean up driver."""
        if self.driver:
            self.driver.quit()


# Global data collector instance
data_collector = DataCollector() 