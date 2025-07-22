"""Value betting and risk management for Contrarius betting engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from database import db_manager, Game, Odds, Bet, BankrollHistory
from utils import (
    kelly_criterion, calculate_edge, calculate_expected_value,
    implied_probability, american_to_decimal, format_currency,
    format_percentage
)
from config import config

logger = logging.getLogger(__name__)


@dataclass
class BettingOpportunity:
    """Represents a potential betting opportunity."""
    game_id: int
    market_type: str
    selection: str
    best_odds: float
    bookmaker: str
    model_probability: float
    implied_probability: float
    edge_percentage: float
    expected_value: float
    kelly_stake: float
    recommended_stake: float
    confidence_score: float
    feature_importance: Dict[str, float]
    sharp_agreement: bool
    risk_factors: List[str]


class ValueBettingEngine:
    """Main value betting detection and execution engine."""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.bet_selector = BetSelector()
        self.stake_calculator = StakeCalculator()
        self.portfolio_optimizer = PortfolioOptimizer()
        
    def find_value_bets(self, predictions: List[Dict]) -> List[BettingOpportunity]:
        """Find all value betting opportunities from predictions."""
        opportunities = []
        
        for pred in predictions:
            game_id = pred['game_id']
            
            # Get current odds
            session = db_manager.get_session()
            current_odds = db_manager.get_latest_odds(session, game_id)
            session.close()
            
            if not current_odds:
                continue
            
            # Check each market
            for market_type in ['moneyline', 'spread', 'total']:
                market_opps = self._evaluate_market(
                    pred, current_odds, market_type
                )
                opportunities.extend(market_opps)
        
        # Filter by edge threshold
        min_edge = config.betting.min_edge_percentage
        opportunities = [o for o in opportunities if o.edge_percentage >= min_edge]
        
        # Apply risk filters
        opportunities = self.risk_manager.filter_opportunities(opportunities)
        
        # Optimize portfolio allocation
        opportunities = self.portfolio_optimizer.optimize_allocation(opportunities)
        
        # Sort by expected value
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return opportunities
    
    def _evaluate_market(self, prediction: Dict, odds_list: List[Odds], 
                        market_type: str) -> List[BettingOpportunity]:
        """Evaluate betting opportunities in a specific market."""
        opportunities = []
        
        # Get model predictions for this market
        model_probs = prediction.get(f'{market_type}_probabilities', {})
        confidence = prediction.get('confidence_score', 0.5)
        
        if not model_probs:
            return opportunities
        
        # Find best odds for each selection
        market_odds = [o for o in odds_list if o.market_type == market_type]
        
        for selection, model_prob in model_probs.items():
            # Find best odds
            selection_odds = [o for o in market_odds if o.selection == selection]
            if not selection_odds:
                continue
            
            best_odds_obj = max(selection_odds, key=lambda x: x.odds)
            best_odds = best_odds_obj.odds
            
            # Convert to decimal if needed
            if best_odds < 0 or best_odds > 100:
                best_odds = american_to_decimal(best_odds)
            
            # Calculate value
            implied_prob = implied_probability(best_odds)
            edge = calculate_edge(model_prob, implied_prob)
            
            if edge < config.betting.min_edge_percentage:
                continue
            
            # Calculate stakes
            kelly_stake = kelly_criterion(
                model_prob, best_odds, config.betting.kelly_fraction
            )
            
            # Expected value per unit
            ev = calculate_expected_value(model_prob, best_odds, 1.0)
            
            # Check sharp money agreement
            sharp_agreement = self._check_sharp_agreement(
                prediction, market_type, selection
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(
                prediction, market_type, selection
            )
            
            opportunity = BettingOpportunity(
                game_id=prediction['game_id'],
                market_type=market_type,
                selection=selection,
                best_odds=best_odds,
                bookmaker=best_odds_obj.bookmaker,
                model_probability=model_prob,
                implied_probability=implied_prob,
                edge_percentage=edge,
                expected_value=ev,
                kelly_stake=kelly_stake,
                recommended_stake=0,  # Set later by stake calculator
                confidence_score=confidence,
                feature_importance=prediction.get('feature_importance', {}),
                sharp_agreement=sharp_agreement,
                risk_factors=risk_factors
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def _check_sharp_agreement(self, prediction: Dict, market_type: str, 
                              selection: str) -> bool:
        """Check if sharp money agrees with our selection."""
        sharp_indicators = prediction.get('sharp_indicators', {})
        
        if market_type == 'moneyline':
            sharp_on_home = sharp_indicators.get('sharp_money_home', 0) > \
                          sharp_indicators.get('sharp_money_away', 0)
            return (selection == 'home' and sharp_on_home) or \
                   (selection == 'away' and not sharp_on_home)
        
        return sharp_indicators.get(f'sharp_{market_type}_{selection}', False)
    
    def _identify_risk_factors(self, prediction: Dict, market_type: str, 
                              selection: str) -> List[str]:
        """Identify risk factors for a bet."""
        risk_factors = []
        
        # Low confidence
        if prediction.get('confidence_score', 0) < 0.6:
            risk_factors.append('low_model_confidence')
        
        # High variance in individual model predictions
        if prediction.get('prediction_variance', 0) > 0.1:
            risk_factors.append('high_model_disagreement')
        
        # Against public
        public_on_selection = prediction.get(f'public_{selection}_pct', 50)
        if public_on_selection < 30:
            risk_factors.append('against_heavy_public')
        
        # Injury concerns
        if selection in ['home', 'away']:
            injury_impact = prediction.get(f'{selection}_injury_impact', 0)
            if injury_impact > 0.5:
                risk_factors.append('significant_injuries')
        
        # Recent poor form
        if selection in ['home', 'away']:
            momentum = prediction.get(f'{selection}_momentum', 0)
            if momentum < -0.3:
                risk_factors.append('poor_recent_form')
        
        return risk_factors
    
    def place_bets(self, opportunities: List[BettingOpportunity], 
                   manual_approval: bool = True) -> List[Bet]:
        """Place bets based on opportunities."""
        placed_bets = []
        session = db_manager.get_session()
        
        # Get current bankroll
        current_bankroll = db_manager.get_current_bankroll(session)
        
        # Calculate stakes
        opportunities = self.stake_calculator.calculate_stakes(
            opportunities, current_bankroll
        )
        
        # Apply daily limits
        opportunities = self._apply_daily_limits(opportunities, session)
        
        for opp in opportunities:
            if opp.recommended_stake <= 0:
                continue
            
            # Create bet record
            bet_data = {
                'game_id': opp.game_id,
                'market_type': opp.market_type,
                'selection': opp.selection,
                'odds': opp.best_odds,
                'stake': opp.recommended_stake,
                'expected_value': opp.expected_value * opp.recommended_stake,
                'edge_percentage': opp.edge_percentage,
                'kelly_stake': opp.kelly_stake,
                'bookmaker': opp.bookmaker,
                'is_manual': manual_approval
            }
            
            # Add to database
            bet = db_manager.add_bet(session, bet_data)
            placed_bets.append(bet)
            
            # Update bankroll
            db_manager.update_bankroll(
                session, -opp.recommended_stake, 'bet_placed', bet.id
            )
            
            logger.info(f"Placed bet: {bet.id} - {opp.market_type} {opp.selection} "
                       f"@ {opp.best_odds} for {format_currency(opp.recommended_stake)}")
        
        session.close()
        return placed_bets
    
    def _apply_daily_limits(self, opportunities: List[BettingOpportunity], 
                           session) -> List[BettingOpportunity]:
        """Apply daily betting limits."""
        # Get today's bets
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        today_bets = session.query(Bet).filter(
            Bet.placed_at >= today_start
        ).all()
        
        # Check bet count limit
        max_bets = config.betting.max_daily_bets
        remaining_bets = max_bets - len(today_bets)
        
        if remaining_bets <= 0:
            logger.warning("Daily bet limit reached")
            return []
        
        # Check exposure limit
        current_exposure = sum(b.stake for b in today_bets if b.status == 'pending')
        current_bankroll = db_manager.get_current_bankroll(session)
        max_exposure = current_bankroll * (config.betting.max_exposure_percentage / 100)
        
        remaining_exposure = max_exposure - current_exposure
        
        # Filter opportunities
        filtered = []
        total_stake = 0
        
        for opp in opportunities[:remaining_bets]:
            if total_stake + opp.recommended_stake <= remaining_exposure:
                filtered.append(opp)
                total_stake += opp.recommended_stake
        
        return filtered


class RiskManager:
    """Risk management and filtering."""
    
    def filter_opportunities(self, opportunities: List[BettingOpportunity]) -> List[BettingOpportunity]:
        """Apply risk filters to opportunities."""
        filtered = []
        
        for opp in opportunities:
            if self._passes_risk_checks(opp):
                filtered.append(opp)
            else:
                logger.info(f"Filtered out bet due to risk: {opp.game_id} "
                           f"{opp.market_type} {opp.selection}")
        
        return filtered
    
    def _passes_risk_checks(self, opp: BettingOpportunity) -> bool:
        """Check if opportunity passes risk thresholds."""
        # Minimum confidence
        if opp.confidence_score < config.model.confidence_threshold:
            return False
        
        # Too many risk factors
        if len(opp.risk_factors) >= 3:
            return False
        
        # Critical risk factors
        critical_risks = ['significant_injuries', 'high_model_disagreement']
        if any(risk in opp.risk_factors for risk in critical_risks):
            # Require higher edge for risky bets
            if opp.edge_percentage < config.betting.min_edge_percentage * 1.5:
                return False
        
        # Sharp money disagreement on high confidence bets
        if opp.confidence_score > 0.7 and not opp.sharp_agreement:
            # Be more cautious
            if opp.edge_percentage < config.betting.min_edge_percentage * 1.25:
                return False
        
        return True
    
    def calculate_var(self, portfolio: List[Dict], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk for portfolio."""
        if not portfolio:
            return 0.0
        
        # Simulate outcomes
        simulations = 10000
        returns = []
        
        for _ in range(simulations):
            total_return = 0
            for bet in portfolio:
                prob = bet['probability']
                stake = bet['stake']
                odds = bet['odds']
                
                # Simulate outcome
                if np.random.random() < prob:
                    total_return += stake * (odds - 1)
                else:
                    total_return -= stake
            
            returns.append(total_return)
        
        # Calculate VaR
        returns.sort()
        var_index = int((1 - confidence_level) * simulations)
        return -returns[var_index]
    
    def check_correlation_risk(self, opportunities: List[BettingOpportunity]) -> Dict[str, float]:
        """Check for correlated bets."""
        correlation_matrix = {}
        
        # Group by game
        games = {}
        for opp in opportunities:
            if opp.game_id not in games:
                games[opp.game_id] = []
            games[opp.game_id].append(opp)
        
        # Same game correlation
        same_game_exposure = sum(
            sum(o.recommended_stake for o in opps)
            for opps in games.values() if len(opps) > 1
        )
        
        # Sport concentration
        sport_exposure = {}
        session = db_manager.get_session()
        
        for game_id, opps in games.items():
            game = session.query(Game).filter(Game.id == game_id).first()
            if game:
                sport = game.sport
                sport_exposure[sport] = sport_exposure.get(sport, 0) + \
                                      sum(o.recommended_stake for o in opps)
        
        session.close()
        
        total_exposure = sum(sport_exposure.values())
        max_sport_concentration = max(sport_exposure.values()) / total_exposure if total_exposure > 0 else 0
        
        return {
            'same_game_exposure': same_game_exposure,
            'max_sport_concentration': max_sport_concentration,
            'correlation_score': (same_game_exposure / total_exposure) * 0.5 + \
                               max_sport_concentration * 0.5 if total_exposure > 0 else 0
        }


class StakeCalculator:
    """Calculate optimal stake sizes."""
    
    def calculate_stakes(self, opportunities: List[BettingOpportunity], 
                        bankroll: float) -> List[BettingOpportunity]:
        """Calculate recommended stakes for all opportunities."""
        # Apply Kelly criterion with fractional adjustment
        for opp in opportunities:
            kelly_stake_pct = opp.kelly_stake
            
            # Apply maximum bet percentage
            max_stake_pct = config.betting.max_bet_percentage / 100
            stake_pct = min(kelly_stake_pct, max_stake_pct)
            
            # Adjust for confidence
            confidence_adj = self._confidence_adjustment(opp.confidence_score)
            stake_pct *= confidence_adj
            
            # Adjust for sharp money
            if opp.sharp_agreement:
                stake_pct *= 1.1  # 10% increase
            else:
                stake_pct *= 0.9  # 10% decrease
            
            # Calculate actual stake
            opp.recommended_stake = bankroll * stake_pct
            
            # Round to reasonable amount
            opp.recommended_stake = self._round_stake(opp.recommended_stake)
        
        return opportunities
    
    def _confidence_adjustment(self, confidence: float) -> float:
        """Adjust stake based on confidence score."""
        # Linear scaling from 0.5 at min confidence to 1.0 at high confidence
        min_conf = config.model.confidence_threshold
        
        if confidence < min_conf:
            return 0.0
        elif confidence >= 0.8:
            return 1.0
        else:
            return 0.5 + 0.5 * ((confidence - min_conf) / (0.8 - min_conf))
    
    def _round_stake(self, stake: float) -> float:
        """Round stake to reasonable betting amount."""
        if stake < 10:
            return round(stake, 0)
        elif stake < 100:
            return round(stake / 5) * 5
        elif stake < 1000:
            return round(stake / 10) * 10
        else:
            return round(stake / 50) * 50


class PortfolioOptimizer:
    """Optimize bet portfolio allocation."""
    
    def optimize_allocation(self, opportunities: List[BettingOpportunity]) -> List[BettingOpportunity]:
        """Optimize portfolio allocation across opportunities."""
        if len(opportunities) <= 1:
            return opportunities
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(opportunities)
        
        # Mean-variance optimization
        returns = np.array([o.expected_value for o in opportunities])
        
        # Simple allocation based on Sharpe ratio
        # (Would implement full Markowitz optimization in production)
        sharpe_ratios = []
        
        for i, opp in enumerate(opportunities):
            expected_return = opp.expected_value
            
            # Estimate variance based on model probability
            p = opp.model_probability
            variance = p * (1 - p)  # Bernoulli variance
            
            # Adjust for correlations
            correlation_adj = np.mean([correlation_matrix[i][j] 
                                     for j in range(len(opportunities)) if i != j])
            variance *= (1 + correlation_adj)
            
            # Calculate Sharpe ratio
            sharpe = expected_return / np.sqrt(variance) if variance > 0 else 0
            sharpe_ratios.append(sharpe)
        
        # Allocate based on Sharpe ratios
        total_sharpe = sum(max(0, s) for s in sharpe_ratios)
        
        if total_sharpe > 0:
            for i, opp in enumerate(opportunities):
                if sharpe_ratios[i] > 0:
                    allocation_pct = sharpe_ratios[i] / total_sharpe
                    opp.recommended_stake *= allocation_pct
                else:
                    opp.recommended_stake = 0
        
        return opportunities
    
    def _calculate_correlation_matrix(self, opportunities: List[BettingOpportunity]) -> np.ndarray:
        """Calculate correlation matrix for opportunities."""
        n = len(opportunities)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                correlation = self._calculate_bet_correlation(
                    opportunities[i], opportunities[j]
                )
                correlation_matrix[i][j] = correlation
                correlation_matrix[j][i] = correlation
        
        return correlation_matrix
    
    def _calculate_bet_correlation(self, bet1: BettingOpportunity, 
                                  bet2: BettingOpportunity) -> float:
        """Calculate correlation between two bets."""
        # Same game bets are highly correlated
        if bet1.game_id == bet2.game_id:
            if bet1.market_type == bet2.market_type:
                # Opposite selections are negatively correlated
                if bet1.selection != bet2.selection:
                    return -0.8
                else:
                    return 1.0
            else:
                # Different markets in same game
                return 0.3
        
        # Different games - check for other correlations
        # (Would implement more sophisticated correlation in production)
        return 0.0


class BetSelector:
    """Select best bets from opportunities."""
    
    def select_bets(self, opportunities: List[BettingOpportunity], 
                   selection_criteria: str = 'expected_value') -> List[BettingOpportunity]:
        """Select best bets based on criteria."""
        if selection_criteria == 'expected_value':
            return sorted(opportunities, key=lambda x: x.expected_value, reverse=True)
        elif selection_criteria == 'edge':
            return sorted(opportunities, key=lambda x: x.edge_percentage, reverse=True)
        elif selection_criteria == 'confidence':
            return sorted(opportunities, key=lambda x: x.confidence_score, reverse=True)
        elif selection_criteria == 'sharp_aligned':
            # Prioritize bets that align with sharp money
            sharp_aligned = [o for o in opportunities if o.sharp_agreement]
            others = [o for o in opportunities if not o.sharp_agreement]
            return sharp_aligned + others
        else:
            return opportunities


class BankrollManager:
    """Manage bankroll and track performance."""
    
    def __init__(self):
        self.stop_loss_triggered = False
        self.take_profit_triggered = False
        
    def check_limits(self) -> Dict[str, bool]:
        """Check if stop loss or take profit limits are hit."""
        session = db_manager.get_session()
        
        # Get initial and current bankroll
        initial = config.betting.bankroll
        current = db_manager.get_current_bankroll(session)
        
        # Calculate change
        change_pct = ((current - initial) / initial) * 100
        
        # Check stop loss
        if change_pct <= -config.betting.stop_loss_percentage:
            self.stop_loss_triggered = True
            logger.warning(f"Stop loss triggered at {format_percentage(change_pct/100)}")
        
        # Check take profit
        if change_pct >= config.betting.take_profit_percentage:
            self.take_profit_triggered = True
            logger.info(f"Take profit triggered at {format_percentage(change_pct/100)}")
        
        session.close()
        
        return {
            'stop_loss': self.stop_loss_triggered,
            'take_profit': self.take_profit_triggered,
            'current_bankroll': current,
            'change_percentage': change_pct
        }
    
    def get_performance_metrics(self, days: int = 30) -> Dict[str, float]:
        """Calculate performance metrics."""
        session = db_manager.get_session()
        
        # Get bets from last N days
        cutoff = datetime.utcnow() - timedelta(days=days)
        bets = session.query(Bet).filter(
            Bet.placed_at >= cutoff
        ).all()
        
        if not bets:
            session.close()
            return {}
        
        # Calculate metrics
        total_staked = sum(b.stake for b in bets)
        completed_bets = [b for b in bets if b.status in ['won', 'lost']]
        
        wins = [b for b in completed_bets if b.status == 'won']
        losses = [b for b in completed_bets if b.status == 'lost']
        
        total_profit = sum(b.result for b in completed_bets if b.result)
        
        metrics = {
            'total_bets': len(bets),
            'completed_bets': len(completed_bets),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(completed_bets) if completed_bets else 0,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
            'average_stake': total_staked / len(bets) if bets else 0,
            'average_odds': np.mean([b.odds for b in bets]),
            'average_edge': np.mean([b.edge_percentage for b in bets])
        }
        
        session.close()
        return metrics


# Global instances
value_betting_engine = ValueBettingEngine()
bankroll_manager = BankrollManager() 