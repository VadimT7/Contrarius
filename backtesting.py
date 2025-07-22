"""Backtesting engine for Contrarius betting system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import json

from database import db_manager, Game, Odds, Bet
from models import ModelEnsemble
from features import feature_engineer
from betting import ValueBettingEngine, BankrollManager
from utils import calculate_sharpe_ratio, calculate_roi, format_currency, format_percentage
from config import config

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.total_bets = 0
        self.winning_bets = 0
        self.losing_bets = 0
        self.total_staked = 0.0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.roi = 0.0
        self.win_rate = 0.0
        self.avg_odds = 0.0
        self.avg_edge = 0.0
        self.bankroll_history = []
        self.bet_history = []
        self.daily_returns = []
        self.monthly_returns = {}
        self.sport_performance = {}
        self.market_performance = {}
        self.feature_importance = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'losing_bets': self.losing_bets,
            'total_staked': self.total_staked,
            'total_profit': self.total_profit,
            'roi': self.roi,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_odds': self.avg_odds,
            'avg_edge': self.avg_edge,
            'sport_performance': self.sport_performance,
            'market_performance': self.market_performance
        }
    
    def summary(self) -> str:
        """Generate summary report."""
        return f"""
Backtest Results Summary
========================
Period: {len(self.daily_returns)} days
Total Bets: {self.total_bets}
Win Rate: {format_percentage(self.win_rate)}
ROI: {format_percentage(self.roi/100)}
Total Profit: {format_currency(self.total_profit)}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {format_percentage(self.max_drawdown)}
Average Odds: {self.avg_odds:.2f}
Average Edge: {format_percentage(self.avg_edge/100)}

Sport Performance:
{self._format_sport_performance()}

Market Performance:
{self._format_market_performance()}
"""
    
    def _format_sport_performance(self) -> str:
        """Format sport performance stats."""
        lines = []
        for sport, stats in self.sport_performance.items():
            lines.append(f"  {sport}: ROI={format_percentage(stats['roi']/100)}, "
                        f"Bets={stats['count']}, Win Rate={format_percentage(stats['win_rate'])}")
        return '\n'.join(lines)
    
    def _format_market_performance(self) -> str:
        """Format market performance stats."""
        lines = []
        for market, stats in self.market_performance.items():
            lines.append(f"  {market}: ROI={format_percentage(stats['roi']/100)}, "
                        f"Bets={stats['count']}, Win Rate={format_percentage(stats['win_rate'])}")
        return '\n'.join(lines)


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self):
        self.model_ensemble = ModelEnsemble()
        self.value_engine = ValueBettingEngine()
        self.data_cache = {}
        
    def run_backtest(self, start_date: datetime, end_date: datetime,
                     initial_bankroll: float = 10000,
                     strategy_params: Optional[Dict] = None) -> BacktestResult:
        """Run full backtest over specified period."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Override strategy parameters if provided
        if strategy_params:
            self._apply_strategy_params(strategy_params)
        
        # Initialize result tracking
        result = BacktestResult()
        bankroll = initial_bankroll
        
        # Get historical games
        games = self._get_historical_games(start_date, end_date)
        logger.info(f"Found {len(games)} games to backtest")
        
        # Group games by date
        games_by_date = self._group_games_by_date(games)
        
        # Simulate each day
        for date in sorted(games_by_date.keys()):
            daily_games = games_by_date[date]
            
            # Train models with data up to this date
            if self._should_retrain(date):
                self._train_models(date)
            
            # Generate predictions
            predictions = self._generate_predictions(daily_games, date)
            
            # Find value bets
            opportunities = self.value_engine.find_value_bets(predictions)
            
            # Simulate bet placement
            bets_placed = self._simulate_bets(opportunities, bankroll, result)
            
            # Simulate bet resolution
            daily_profit = self._resolve_bets(bets_placed, daily_games)
            
            # Update tracking
            bankroll += daily_profit
            result.bankroll_history.append({
                'date': date,
                'bankroll': bankroll,
                'profit': daily_profit,
                'bets_placed': len(bets_placed)
            })
            
            result.daily_returns.append(daily_profit)
        
        # Calculate final metrics
        self._calculate_metrics(result, initial_bankroll)
        
        logger.info("Backtest completed")
        return result
    
    def run_walk_forward_optimization(self, start_date: datetime, end_date: datetime,
                                    optimization_window: int = 90,
                                    test_window: int = 30,
                                    param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Run walk-forward optimization to find best parameters."""
        logger.info("Starting walk-forward optimization")
        
        results = []
        current_date = start_date
        
        while current_date + timedelta(days=optimization_window + test_window) <= end_date:
            # Optimization period
            opt_start = current_date
            opt_end = current_date + timedelta(days=optimization_window)
            
            # Test period
            test_start = opt_end
            test_end = test_start + timedelta(days=test_window)
            
            logger.info(f"Optimizing {opt_start} to {opt_end}, testing {test_start} to {test_end}")
            
            # Find best parameters
            best_params = self._optimize_parameters(opt_start, opt_end, param_grid)
            
            # Test on out-of-sample data
            test_result = self.run_backtest(test_start, test_end, strategy_params=best_params)
            
            results.append({
                'period': {'test_start': test_start, 'test_end': test_end},
                'best_params': best_params,
                'test_performance': test_result.to_dict()
            })
            
            # Move forward
            current_date += timedelta(days=test_window)
        
        # Aggregate results
        return self._aggregate_walk_forward_results(results)
    
    def run_monte_carlo_simulation(self, historical_bets: List[Dict],
                                 num_simulations: int = 1000,
                                 initial_bankroll: float = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation on historical bet outcomes."""
        logger.info(f"Running {num_simulations} Monte Carlo simulations")
        
        results = []
        
        for i in range(num_simulations):
            # Randomly sample with replacement
            sampled_bets = np.random.choice(historical_bets, size=len(historical_bets), replace=True)
            
            # Simulate bankroll progression
            bankroll = initial_bankroll
            bankroll_history = [bankroll]
            
            for bet in sampled_bets:
                if bet['won']:
                    bankroll += bet['stake'] * (bet['odds'] - 1)
                else:
                    bankroll -= bet['stake']
                bankroll_history.append(bankroll)
            
            # Calculate metrics
            total_profit = bankroll - initial_bankroll
            roi = (total_profit / initial_bankroll) * 100
            max_drawdown = self._calculate_max_drawdown(bankroll_history)
            
            results.append({
                'final_bankroll': bankroll,
                'total_profit': total_profit,
                'roi': roi,
                'max_drawdown': max_drawdown
            })
        
        # Analyze distribution
        return self._analyze_monte_carlo_results(results, initial_bankroll)
    
    def backtest_strategy_variants(self, start_date: datetime, end_date: datetime,
                                 strategies: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple strategy variants."""
        logger.info(f"Backtesting {len(strategies)} strategy variants")
        
        results = []
        
        for name, params in strategies.items():
            logger.info(f"Testing strategy: {name}")
            
            result = self.run_backtest(start_date, end_date, strategy_params=params)
            
            results.append({
                'strategy': name,
                'roi': result.roi,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'total_bets': result.total_bets,
                'params': params
            })
        
        return pd.DataFrame(results).sort_values('sharpe_ratio', ascending=False)
    
    def _get_historical_games(self, start_date: datetime, end_date: datetime) -> List[Game]:
        """Get historical games for backtesting."""
        session = db_manager.get_session()
        
        games = session.query(Game).filter(
            Game.start_time >= start_date,
            Game.start_time <= end_date
        ).order_by(Game.start_time).all()
        
        session.close()
        return games
    
    def _group_games_by_date(self, games: List[Game]) -> Dict[datetime, List[Game]]:
        """Group games by date."""
        games_by_date = {}
        
        for game in games:
            date = game.start_time.date()
            if date not in games_by_date:
                games_by_date[date] = []
            games_by_date[date].append(game)
        
        return games_by_date
    
    def _should_retrain(self, date: datetime) -> bool:
        """Check if models should be retrained."""
        # Retrain weekly
        return date.weekday() == 0  # Monday
    
    def _train_models(self, cutoff_date: datetime):
        """Train models with data up to cutoff date."""
        logger.info(f"Training models with data up to {cutoff_date}")
        
        # Get training data
        session = db_manager.get_session()
        
        # Get games from last 180 days
        train_start = cutoff_date - timedelta(days=180)
        train_games = session.query(Game).filter(
            Game.start_time >= train_start,
            Game.start_time < cutoff_date
        ).all()
        
        if len(train_games) < 100:
            logger.warning("Insufficient training data")
            return
        
        # Extract features and labels
        X_list = []
        y_list = []
        
        for game in train_games:
            # Get historical data up to game time
            historical = self._get_historical_context(game, cutoff_date)
            
            # Extract features
            features_df = feature_engineer.extract_features(game, historical)
            X_list.append(features_df)
            
            # Get outcome (simplified - would need actual results)
            y = self._get_game_outcome(game)
            y_list.append(y)
        
        X = pd.concat(X_list, ignore_index=True)
        y = np.array(y_list)
        
        # Train ensemble
        self.model_ensemble.train_all(X, y)
        
        session.close()
    
    def _generate_predictions(self, games: List[Game], date: datetime) -> List[Dict]:
        """Generate predictions for games."""
        predictions = []
        
        for game in games:
            # Get historical context
            historical = self._get_historical_context(game, date)
            
            # Extract features
            features_df = feature_engineer.extract_features(game, historical)
            
            # Get predictions
            ensemble_pred = self.model_ensemble.predict_proba(features_df)
            
            # Format prediction
            prediction = {
                'game_id': game.id,
                'moneyline_probabilities': {
                    'home': ensemble_pred['probability'][0],
                    'away': 1 - ensemble_pred['probability'][0]
                },
                'confidence_score': ensemble_pred['confidence'][0],
                'feature_importance': ensemble_pred['feature_importance'],
                'sharp_indicators': self._extract_sharp_indicators(game)
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def _simulate_bets(self, opportunities: List, bankroll: float, 
                      result: BacktestResult) -> List[Dict]:
        """Simulate bet placement."""
        bets_placed = []
        
        for opp in opportunities:
            # Check if we have enough bankroll
            if opp.recommended_stake > bankroll * 0.1:  # Max 10% per bet
                opp.recommended_stake = bankroll * 0.1
            
            if opp.recommended_stake < 10:  # Min bet size
                continue
            
            # Record bet
            bet = {
                'game_id': opp.game_id,
                'market_type': opp.market_type,
                'selection': opp.selection,
                'odds': opp.best_odds,
                'stake': opp.recommended_stake,
                'edge': opp.edge_percentage,
                'model_prob': opp.model_probability
            }
            
            bets_placed.append(bet)
            result.bet_history.append(bet)
        
        return bets_placed
    
    def _resolve_bets(self, bets: List[Dict], games: List[Game]) -> float:
        """Resolve bets and calculate profit/loss."""
        daily_profit = 0.0
        
        for bet in bets:
            # Get game outcome
            game = next((g for g in games if g.id == bet['game_id']), None)
            if not game:
                continue
            
            # Simulate outcome (would use actual results in production)
            won = self._simulate_bet_outcome(bet, game)
            
            if won:
                profit = bet['stake'] * (bet['odds'] - 1)
                bet['won'] = True
            else:
                profit = -bet['stake']
                bet['won'] = False
            
            bet['profit'] = profit
            daily_profit += profit
        
        return daily_profit
    
    def _simulate_bet_outcome(self, bet: Dict, game: Game) -> bool:
        """Simulate bet outcome based on model probability."""
        # Use model probability with some noise
        true_prob = bet['model_prob']
        
        # Add realistic noise to simulate prediction error
        noise = np.random.normal(0, 0.05)
        actual_prob = np.clip(true_prob + noise, 0, 1)
        
        # Simulate outcome
        return np.random.random() < actual_prob
    
    def _calculate_metrics(self, result: BacktestResult, initial_bankroll: float):
        """Calculate performance metrics."""
        if not result.bet_history:
            return
        
        # Basic metrics
        result.total_bets = len(result.bet_history)
        result.winning_bets = sum(1 for b in result.bet_history if b.get('won', False))
        result.losing_bets = result.total_bets - result.winning_bets
        result.win_rate = result.winning_bets / result.total_bets if result.total_bets > 0 else 0
        
        # Financial metrics
        result.total_staked = sum(b['stake'] for b in result.bet_history)
        result.total_profit = sum(b.get('profit', 0) for b in result.bet_history)
        result.roi = (result.total_profit / result.total_staked * 100) if result.total_staked > 0 else 0
        
        # Risk metrics
        bankroll_values = [initial_bankroll] + [h['bankroll'] for h in result.bankroll_history]
        result.max_drawdown = self._calculate_max_drawdown(bankroll_values)
        result.sharpe_ratio = calculate_sharpe_ratio(result.daily_returns)
        
        # Average metrics
        result.avg_odds = np.mean([b['odds'] for b in result.bet_history])
        result.avg_edge = np.mean([b['edge'] for b in result.bet_history])
        
        # Performance by segment
        self._calculate_segment_performance(result)
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if len(bankroll_history) < 2:
            return 0.0
        
        peak = bankroll_history[0]
        max_dd = 0.0
        
        for value in bankroll_history[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_segment_performance(self, result: BacktestResult):
        """Calculate performance by sport and market."""
        # By sport
        sport_stats = {}
        market_stats = {}
        
        session = db_manager.get_session()
        
        for bet in result.bet_history:
            # Get game info
            game = session.query(Game).filter(Game.id == bet['game_id']).first()
            if not game:
                continue
            
            # Sport performance
            sport = game.sport
            if sport not in sport_stats:
                sport_stats[sport] = {'count': 0, 'staked': 0, 'profit': 0, 'wins': 0}
            
            sport_stats[sport]['count'] += 1
            sport_stats[sport]['staked'] += bet['stake']
            sport_stats[sport]['profit'] += bet.get('profit', 0)
            sport_stats[sport]['wins'] += 1 if bet.get('won', False) else 0
            
            # Market performance
            market = bet['market_type']
            if market not in market_stats:
                market_stats[market] = {'count': 0, 'staked': 0, 'profit': 0, 'wins': 0}
            
            market_stats[market]['count'] += 1
            market_stats[market]['staked'] += bet['stake']
            market_stats[market]['profit'] += bet.get('profit', 0)
            market_stats[market]['wins'] += 1 if bet.get('won', False) else 0
        
        session.close()
        
        # Calculate ROI and win rate
        for stats_dict in [sport_stats, market_stats]:
            for key, stats in stats_dict.items():
                stats['roi'] = (stats['profit'] / stats['staked'] * 100) if stats['staked'] > 0 else 0
                stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        
        result.sport_performance = sport_stats
        result.market_performance = market_stats
    
    def _optimize_parameters(self, start_date: datetime, end_date: datetime,
                           param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize parameters using grid search."""
        best_sharpe = -float('inf')
        best_params = {}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Test each combination
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {}
            
            for params in param_combinations:
                future = executor.submit(
                    self.run_backtest, start_date, end_date, 
                    strategy_params=params
                )
                futures[future] = params
            
            for future in as_completed(futures):
                params = futures[future]
                try:
                    result = future.result()
                    if result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_params = params
                except Exception as e:
                    logger.error(f"Error testing params {params}: {e}")
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _apply_strategy_params(self, params: Dict):
        """Apply strategy parameters to config."""
        if 'min_edge_percentage' in params:
            config.betting.min_edge_percentage = params['min_edge_percentage']
        if 'kelly_fraction' in params:
            config.betting.kelly_fraction = params['kelly_fraction']
        if 'max_bet_percentage' in params:
            config.betting.max_bet_percentage = params['max_bet_percentage']
        if 'confidence_threshold' in params:
            config.model.confidence_threshold = params['confidence_threshold']
    
    def _get_historical_context(self, game: Game, cutoff_date: datetime) -> Dict:
        """Get historical context for feature extraction."""
        # Cache key
        cache_key = f"{game.id}_{cutoff_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Generate historical data (simplified)
        context = {
            'home_recent_results': [1, 0, 1, 1, 0],  # Would fetch actual results
            'away_recent_results': [0, 1, 0, 1, 1],
            'season_game_number': 30,
            'h2h_history': []
        }
        
        self.data_cache[cache_key] = context
        return context
    
    def _get_game_outcome(self, game: Game) -> int:
        """Get actual game outcome for training."""
        # Simplified - would fetch actual results
        # For now, simulate based on odds
        session = db_manager.get_session()
        
        odds = session.query(Odds).filter(
            Odds.game_id == game.id,
            Odds.market_type == 'moneyline'
        ).first()
        
        session.close()
        
        if odds and odds.selection == 'home':
            # Use implied probability as proxy
            return 1 if np.random.random() < odds.implied_probability else 0
        
        return np.random.randint(0, 2)
    
    def _extract_sharp_indicators(self, game: Game) -> Dict:
        """Extract sharp money indicators."""
        if game.sentiment_data:
            return {
                'sharp_money_home': len([
                    s for s in game.sentiment_data.get('sharp_money_indicators', [])
                    if s.get('side') == 'home'
                ]),
                'sharp_money_away': len([
                    s for s in game.sentiment_data.get('sharp_money_indicators', [])
                    if s.get('side') == 'away'
                ])
            }
        return {'sharp_money_home': 0, 'sharp_money_away': 0}
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward optimization results."""
        # Extract performance metrics
        rois = [r['test_performance']['roi'] for r in results]
        sharpes = [r['test_performance']['sharpe_ratio'] for r in results]
        
        # Find most common parameters
        param_counts = {}
        for r in results:
            params_str = json.dumps(r['best_params'], sort_keys=True)
            param_counts[params_str] = param_counts.get(params_str, 0) + 1
        
        most_common_params = json.loads(
            max(param_counts.items(), key=lambda x: x[1])[0]
        )
        
        return {
            'average_roi': np.mean(rois),
            'average_sharpe': np.mean(sharpes),
            'roi_std': np.std(rois),
            'sharpe_std': np.std(sharpes),
            'most_stable_params': most_common_params,
            'detailed_results': results
        }
    
    def _analyze_monte_carlo_results(self, results: List[Dict], 
                                   initial_bankroll: float) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        final_bankrolls = [r['final_bankroll'] for r in results]
        rois = [r['roi'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        
        return {
            'expected_final_bankroll': np.mean(final_bankrolls),
            'bankroll_std': np.std(final_bankrolls),
            'bankroll_percentiles': {
                '5%': np.percentile(final_bankrolls, 5),
                '25%': np.percentile(final_bankrolls, 25),
                '50%': np.percentile(final_bankrolls, 50),
                '75%': np.percentile(final_bankrolls, 75),
                '95%': np.percentile(final_bankrolls, 95)
            },
            'probability_of_profit': sum(1 for b in final_bankrolls if b > initial_bankroll) / len(final_bankrolls),
            'expected_roi': np.mean(rois),
            'roi_std': np.std(rois),
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_case_drawdown': np.percentile(max_drawdowns, 95)
        }


# Global backtest engine instance
backtest_engine = BacktestEngine() 