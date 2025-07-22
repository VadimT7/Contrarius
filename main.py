"""Contrarius - Elite Sports Betting Engine

Main entry point for the betting system.
"""

import asyncio
import logging
import sys
import schedule
import time
from datetime import datetime
import argparse
import warnings
from typing import List, Dict
from datetime import timedelta

from database import db_manager
from data import data_collector
from models import ModelEnsemble
from features import feature_engineer
from betting import value_betting_engine, bankroll_manager
from config import config
from utils import setup_logging

# Suppress warnings
warnings.filterwarnings('ignore')

logger = setup_logging(config.system.log_level)


class ContrariusEngine:
    """Main betting engine orchestrator."""
    
    def __init__(self):
        self.model_ensemble = ModelEnsemble()
        self.is_running = False
        self.last_model_train = None
        
    def initialize(self):
        """Initialize the system."""
        logger.info("Initializing Contrarius betting engine...")
        
        # Validate configuration
        if not config.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
            
        # Initialize database
        logger.info("Setting up database...")
        db_manager.create_tables()
        
        # Load saved models if available
        try:
            logger.info("Loading saved models...")
            self.model_ensemble.load_models("models/")
        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")
            logger.info("Models will be trained on first run")
            
        logger.info("Initialization complete")
        
    async def run_cycle(self):
        """Run one complete betting cycle."""
        try:
            logger.info("Starting betting cycle...")
            
            # Check bankroll limits
            limits = bankroll_manager.check_limits()
            if limits['stop_loss']:
                logger.error("Stop loss triggered - suspending operations")
                return
            elif limits['take_profit']:
                logger.info("Take profit reached - consider withdrawing")
                
            # Collect latest data
            logger.info("Collecting latest data...")
            await data_collector.collect_all_data()
            
            # Check if models need retraining
            if self._should_retrain_models():
                self._train_models()
                
            # Generate predictions
            predictions = self._generate_predictions()
            
            # Find value bets
            opportunities = value_betting_engine.find_value_bets(predictions)
            logger.info(f"Found {len(opportunities)} value betting opportunities")
            
            # Place bets (if auto-betting enabled)
            if opportunities and config.system.stealth_mode:
                # In stealth mode, just log opportunities
                for opp in opportunities[:5]:  # Top 5
                    logger.info(f"Value bet: {opp.selection} @ {opp.best_odds} "
                               f"(edge: {opp.edge_percentage:.1f}%)")
            else:
                # Place bets if not in stealth mode
                placed_bets = value_betting_engine.place_bets(
                    opportunities,
                    manual_approval=False
                )
                logger.info(f"Placed {len(placed_bets)} bets")
                
            # Update bet results
            self._update_bet_results()
            
            # Log performance
            self._log_performance()
            
            logger.info("Betting cycle completed")
            
        except Exception as e:
            logger.error(f"Error in betting cycle: {e}", exc_info=True)
            
    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained."""
        if self.last_model_train is None:
            return True
            
        hours_since_train = (datetime.utcnow() - self.last_model_train).seconds / 3600
        return hours_since_train >= config.model.retrain_frequency_hours
        
    def _train_models(self):
        """Train all models with latest data."""
        logger.info("Training models...")
        
        # Get training data
        session = db_manager.get_session()
        
        # This is simplified - in production would have proper data pipeline
        logger.warning("Model training not fully implemented - using mock data")
        
        # Mock training
        import pandas as pd
        import numpy as np
        
        # Generate mock features
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = np.random.randint(0, 2, n_samples)
        
        # Train ensemble
        self.model_ensemble.train_all(X, y)
        
        # Save models
        self.model_ensemble.save_models("models/")
        
        self.last_model_train = datetime.utcnow()
        session.close()
        
        logger.info("Model training completed")
        
    def _generate_predictions(self) -> List[Dict]:
        """Generate predictions for upcoming games."""
        session = db_manager.get_session()
        
        # Get upcoming games
        upcoming_games = db_manager.get_upcoming_games(session, hours=24)
        logger.info(f"Generating predictions for {len(upcoming_games)} games")
        
        predictions = []
        
        for game in upcoming_games:
            try:
                # Extract features
                features = feature_engineer.extract_features(game)
                
                # Get ensemble prediction
                pred = self.model_ensemble.predict_proba(features)
                
                # Format prediction
                prediction = {
                    'game_id': game.id,
                    'moneyline_probabilities': {
                        'home': pred['probability'][0],
                        'away': 1 - pred['probability'][0]
                    },
                    'confidence_score': pred['confidence'][0],
                    'feature_importance': pred['feature_importance']
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting game {game.id}: {e}")
                
        session.close()
        return predictions
        
    def _update_bet_results(self):
        """Update results for completed bets."""
        session = db_manager.get_session()
        
        # Get pending bets
        pending_bets = db_manager.get_pending_bets(session)
        
        for bet in pending_bets:
            game = bet.game
            
            # Check if game has finished
            if datetime.utcnow() > game.start_time + timedelta(hours=4):
                # In production, would fetch actual results
                # For now, simulate based on our prediction
                import random
                
                # Simulate win based on our predicted probability
                won = random.random() < 0.52  # Slight edge
                
                if won:
                    bet.status = 'won'
                    bet.result = bet.stake * (bet.odds - 1)
                else:
                    bet.status = 'lost'
                    bet.result = -bet.stake
                    
                bet.resolved_at = datetime.utcnow()
                
                # Update bankroll
                db_manager.update_bankroll(
                    session, bet.result, 
                    'bet_won' if won else 'bet_lost', 
                    bet.id
                )
                
                logger.info(f"Bet {bet.id} resolved: {'WON' if won else 'LOST'} "
                           f"{bet.result:+.2f}")
                
        session.commit()
        session.close()
        
    def _log_performance(self):
        """Log system performance metrics."""
        metrics = bankroll_manager.get_performance_metrics(7)
        
        if metrics:
            logger.info(
                f"Weekly Performance - "
                f"ROI: {metrics['roi']:.1f}%, "
                f"Win Rate: {metrics['win_rate']*100:.1f}%, "
                f"Profit: ${metrics['total_profit']:.2f}"
            )
            
    def start(self):
        """Start the betting engine."""
        logger.info("Starting Contrarius engine...")
        self.is_running = True
        
        # Schedule tasks
        schedule.every(config.system.update_interval_seconds).seconds.do(
            lambda: asyncio.run(self.run_cycle())
        )
        
        # Run initial cycle
        asyncio.run(self.run_cycle())
        
        # Main loop
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
            
    def stop(self):
        """Stop the betting engine."""
        logger.info("Stopping Contrarius engine...")
        self.is_running = False


def run_dashboard():
    """Run the Streamlit dashboard."""
    import subprocess
    logger.info("Starting dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])


def run_backtest(args):
    """Run backtesting mode."""
    from backtesting import backtest_engine
    
    logger.info("Running backtest...")
    
    result = backtest_engine.run_backtest(
        args.start_date,
        args.end_date,
        args.bankroll
    )
    
    print(result.summary())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Contrarius Sports Betting Engine")
    parser.add_argument(
        "--mode", 
        choices=["engine", "dashboard", "backtest", "both"],
        default="dashboard",
        help="Operation mode"
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=10000,
        help="Initial bankroll for backtesting"
    )
    
    args = parser.parse_args()
    
    if args.mode == "engine":
        # Run engine only
        engine = ContrariusEngine()
        engine.initialize()
        
        try:
            engine.start()
        except KeyboardInterrupt:
            engine.stop()
            
    elif args.mode == "dashboard":
        # Run dashboard only
        run_dashboard()
        
    elif args.mode == "backtest":
        # Run backtest
        if not args.start_date or not args.end_date:
            print("Error: --start-date and --end-date required for backtest")
            sys.exit(1)
            
        run_backtest(args)
        
    elif args.mode == "both":
        # Run both engine and dashboard
        import threading
        
        # Start engine in background thread
        engine = ContrariusEngine()
        engine.initialize()
        
        engine_thread = threading.Thread(target=engine.start)
        engine_thread.daemon = True
        engine_thread.start()
        
        # Run dashboard in main thread
        run_dashboard()
        
        # Stop engine when dashboard closes
        engine.stop()


if __name__ == "__main__":
    main() 