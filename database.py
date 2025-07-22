"""Database management for Contrarius betting engine."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
import logging
from typing import List, Dict, Optional
from config import config

Base = declarative_base()
logger = logging.getLogger(__name__)


class Game(Base):
    """Game/match information."""
    __tablename__ = 'games'
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String(100), unique=True, index=True)
    sport = Column(String(50), index=True)
    league = Column(String(100))
    home_team = Column(String(200))
    away_team = Column(String(200))
    start_time = Column(DateTime, index=True)
    venue = Column(String(200))
    weather_data = Column(JSON)
    injuries_data = Column(JSON)
    lineup_data = Column(JSON)
    travel_data = Column(JSON)
    sentiment_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    odds = relationship("Odds", back_populates="game", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="game", cascade="all, delete-orphan")
    bets = relationship("Bet", back_populates="game", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_game_sport_time', 'sport', 'start_time'),
    )


class Odds(Base):
    """Odds information from various bookmakers."""
    __tablename__ = 'odds'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), index=True)
    bookmaker = Column(String(100))
    market_type = Column(String(50))  # moneyline, spread, total
    selection = Column(String(100))  # home, away, over, under
    odds = Column(Float)
    implied_probability = Column(Float)
    line = Column(Float, nullable=True)  # for spreads and totals
    volume = Column(Float, nullable=True)
    sharp_money = Column(Boolean, default=False)
    movement_velocity = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    game = relationship("Game", back_populates="odds")
    
    __table_args__ = (
        Index('idx_odds_game_time', 'game_id', 'timestamp'),
        Index('idx_odds_market', 'market_type', 'bookmaker'),
    )


class Prediction(Base):
    """Model predictions for games."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), index=True)
    model_name = Column(String(50))
    market_type = Column(String(50))
    selection = Column(String(100))
    raw_probability = Column(Float)
    calibrated_probability = Column(Float)
    confidence_score = Column(Float)
    feature_importance = Column(JSON)
    model_version = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    game = relationship("Game", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_pred_game_model', 'game_id', 'model_name'),
    )


class Bet(Base):
    """Placed bets and their outcomes."""
    __tablename__ = 'bets'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), index=True)
    market_type = Column(String(50))
    selection = Column(String(100))
    odds = Column(Float)
    stake = Column(Float)
    expected_value = Column(Float)
    edge_percentage = Column(Float)
    kelly_stake = Column(Float)
    status = Column(String(20), default='pending')  # pending, won, lost, void, cancelled
    result = Column(Float, nullable=True)
    placed_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    bookmaker = Column(String(100))
    is_manual = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    game = relationship("Game", back_populates="bets")
    
    __table_args__ = (
        Index('idx_bet_status_time', 'status', 'placed_at'),
    )


class ModelPerformance(Base):
    """Track model performance over time."""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), index=True)
    date = Column(DateTime, index=True)
    sport = Column(String(50))
    market_type = Column(String(50))
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(Float)
    log_loss = Column(Float)
    brier_score = Column(Float)
    roi = Column(Float)
    profit_loss = Column(Float)
    calibration_error = Column(Float)
    feature_importance = Column(JSON)
    
    __table_args__ = (
        Index('idx_perf_model_date', 'model_name', 'date'),
    )


class FeatureStore(Base):
    """Store computed features for reuse."""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), index=True)
    feature_set = Column(String(100))
    features = Column(JSON)
    version = Column(String(20))
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feature_game_set', 'game_id', 'feature_set'),
    )


class BankrollHistory(Base):
    """Track bankroll changes over time."""
    __tablename__ = 'bankroll_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    balance = Column(Float)
    change = Column(Float)
    change_type = Column(String(50))  # bet_placed, bet_won, bet_lost, deposit, withdrawal
    bet_id = Column(Integer, ForeignKey('bets.id'), nullable=True)
    notes = Column(Text, nullable=True)


class SystemLog(Base):
    """System activity and error logs."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(20))
    component = Column(String(50))
    message = Column(Text)
    details = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('idx_log_time_level', 'timestamp', 'level'),
    )


class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self):
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def _create_engine(self):
        """Create database engine with appropriate settings."""
        connect_args = {}
        if config.database.db_type == "sqlite":
            connect_args = {
                "check_same_thread": False,
                "connect_args": {"check_same_thread": False}
            }
        
        return create_engine(
            config.database.connection_string,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            **connect_args
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def add_game(self, session: Session, game_data: Dict) -> Game:
        """Add new game to database."""
        game = Game(**game_data)
        session.add(game)
        session.commit()
        session.refresh(game)
        return game
    
    def add_odds(self, session: Session, odds_data: List[Dict]):
        """Add odds data to database."""
        odds_objects = [Odds(**data) for data in odds_data]
        session.bulk_save_objects(odds_objects)
        session.commit()
    
    def add_prediction(self, session: Session, prediction_data: Dict) -> Prediction:
        """Add model prediction."""
        prediction = Prediction(**prediction_data)
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction
    
    def add_bet(self, session: Session, bet_data: Dict) -> Bet:
        """Record a placed bet."""
        bet = Bet(**bet_data)
        session.add(bet)
        session.commit()
        session.refresh(bet)
        return bet
    
    def get_pending_bets(self, session: Session) -> List[Bet]:
        """Get all pending bets."""
        return session.query(Bet).filter(Bet.status == 'pending').all()
    
    def get_upcoming_games(self, session: Session, hours: int = 24) -> List[Game]:
        """Get upcoming games within specified hours."""
        cutoff_time = datetime.utcnow() + timedelta(hours=hours)
        return session.query(Game).filter(
            Game.start_time > datetime.utcnow(),
            Game.start_time < cutoff_time
        ).all()
    
    def get_latest_odds(self, session: Session, game_id: int) -> List[Odds]:
        """Get latest odds for a game."""
        subquery = session.query(
            Odds.bookmaker,
            Odds.market_type,
            Odds.selection,
            func.max(Odds.timestamp).label('max_timestamp')
        ).filter(
            Odds.game_id == game_id
        ).group_by(
            Odds.bookmaker,
            Odds.market_type,
            Odds.selection
        ).subquery()
        
        return session.query(Odds).join(
            subquery,
            and_(
                Odds.bookmaker == subquery.c.bookmaker,
                Odds.market_type == subquery.c.market_type,
                Odds.selection == subquery.c.selection,
                Odds.timestamp == subquery.c.max_timestamp,
                Odds.game_id == game_id
            )
        ).all()
    
    def update_bankroll(self, session: Session, amount: float, change_type: str, bet_id: Optional[int] = None):
        """Update bankroll history."""
        current_balance = self.get_current_bankroll(session)
        new_balance = current_balance + amount
        
        history = BankrollHistory(
            balance=new_balance,
            change=amount,
            change_type=change_type,
            bet_id=bet_id
        )
        session.add(history)
        session.commit()
    
    def get_current_bankroll(self, session: Session) -> float:
        """Get current bankroll balance."""
        latest = session.query(BankrollHistory).order_by(
            BankrollHistory.timestamp.desc()
        ).first()
        
        return latest.balance if latest else config.betting.bankroll
    
    def log_system_event(self, session: Session, level: str, component: str, message: str, details: Optional[Dict] = None):
        """Log system event."""
        log = SystemLog(
            level=level,
            component=component,
            message=message,
            details=details
        )
        session.add(log)
        session.commit()


# Global database manager instance
db_manager = DatabaseManager()


from datetime import timedelta
from sqlalchemy import func, and_ 