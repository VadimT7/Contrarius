"""Database setup script for Contrarius."""

import sys
import os
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db_manager, BankrollHistory
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database():
    """Initialize database with tables and initial data."""
    logger.info("Setting up Contrarius database...")
    
    try:
        # Create all tables
        logger.info("Creating database tables...")
        db_manager.create_tables()
        logger.info("Tables created successfully")
        
        # Initialize bankroll
        session = db_manager.get_session()
        
        # Check if bankroll already initialized
        existing = session.query(BankrollHistory).first()
        
        if not existing:
            logger.info("Initializing bankroll...")
            initial_bankroll = BankrollHistory(
                timestamp=datetime.utcnow(),
                balance=config.betting.bankroll,
                change=0,
                change_type='initial_deposit',
                notes='Initial bankroll deposit'
            )
            session.add(initial_bankroll)
            session.commit()
            logger.info(f"Bankroll initialized with ${config.betting.bankroll}")
        else:
            logger.info("Bankroll already initialized")
            
        session.close()
        
        # Create necessary directories
        directories = ['models', 'logs', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
                
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        sys.exit(1)


def reset_database():
    """Reset database (WARNING: This will delete all data!)."""
    response = input("WARNING: This will delete all data! Are you sure? (yes/no): ")
    
    if response.lower() != 'yes':
        logger.info("Database reset cancelled")
        return
        
    logger.info("Resetting database...")
    
    try:
        # Drop all tables
        from database import Base
        Base.metadata.drop_all(bind=db_manager.engine)
        logger.info("All tables dropped")
        
        # Recreate tables
        setup_database()
        
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contrarius Database Setup")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (WARNING: Deletes all data!)"
    )
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        setup_database()


if __name__ == "__main__":
    main() 