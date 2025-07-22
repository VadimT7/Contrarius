# Contrarius - Elite Sports Betting Engine 🎯

A state-of-the-art, ML-powered sports prediction and value betting system that ruthlessly exploits market inefficiencies through advanced ensemble modeling, real-time data aggregation, and sophisticated risk management.

## ⚠️ Legal Disclaimer

**IMPORTANT**: This software is provided for informational and research purposes only. Sports betting may be illegal in your jurisdiction. Users are responsible for complying with all local laws and regulations. Never bet more than you can afford to lose. If you have a gambling problem, seek help.

## 🚀 Features

### Machine Learning Excellence
- **6-Model Ensemble**: LightGBM, XGBoost, CatBoost, Random Forest, Neural Networks, and Logistic Regression
- **Dynamic Model Weighting**: Automatically adjusts model weights based on recent performance
- **Probability Calibration**: Isotonic/Platt scaling for razor-sharp probability estimates
- **Self-Improving System**: Continuous learning from outcomes

### Alpha-Grade Data & Features
- **Multi-Source Data Aggregation**: The Odds API, Sportradar, Betfair Exchange, web scraping
- **Unique Feature Engineering**:
  - Real-time injury tracking and impact analysis
  - Weather conditions and game-time factors
  - Travel fatigue and rest advantage calculations
  - Sharp money detection and line movement analysis
  - Market sentiment from social media
  - Technical indicators (RSI, Bollinger Bands, VWAP)

### Value Betting Engine
- **Edge Detection**: Only bets when model probability significantly exceeds implied odds
- **Kelly Criterion Staking**: Optimal bet sizing with configurable fractional Kelly
- **Portfolio Optimization**: Correlation-aware bet selection
- **Risk Management**: Stop-loss, take-profit, and maximum exposure controls

### Professional Dashboard
- **Real-Time Monitoring**: Live bet tracking, P&L visualization, and performance metrics
- **Value Bet Alerts**: Instant notification of profitable opportunities
- **Manual Override**: Approve/reject bets before execution
- **Comprehensive Analytics**: ROI by sport/market, model performance tracking

### Backtesting Suite
- **Historical Analysis**: Test strategies on past data
- **Walk-Forward Optimization**: Rolling parameter optimization
- **Monte Carlo Simulation**: Risk assessment through 1000s of scenarios
- **Strategy Comparison**: A/B test different approaches

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL or SQLite
- API Keys:
  - [The Odds API](https://the-odds-api.com/)
  - [Sportradar](https://sportradar.com/) (optional)
  - [Betfair](https://www.betfair.com/exchange/) (optional)
  - [OpenWeatherMap](https://openweathermap.org/api) (optional)
  - Twitter Bearer Token (optional)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/contrarius.git
cd contrarius
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your API keys and configuration
```

5. **Initialize database**
```bash
python setup.py
```

## ⚙️ Configuration

Edit `.env` file with your settings:

```env
# Database
DB_TYPE=postgresql  # or sqlite for local testing
DB_HOST=localhost
DB_NAME=contrarius
DB_USER=your_user
DB_PASSWORD=your_password

# API Keys
ODDS_API_KEY=your_key_here
# ... other API keys

# Betting Parameters
MIN_EDGE_PERCENTAGE=5.0    # Minimum edge to bet
KELLY_FRACTION=0.25        # Fraction of Kelly criterion
MAX_BET_PERCENTAGE=5.0     # Max bet as % of bankroll
INITIAL_BANKROLL=10000.0   # Starting bankroll
```

## 🚦 Usage

### Dashboard Mode (Recommended)
```bash
python main.py --mode dashboard
```
Access the dashboard at http://localhost:8501

### Automated Engine Mode
```bash
python main.py --mode engine
```

### Both Dashboard + Engine
```bash
python main.py --mode both
```

### Backtesting
```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --bankroll 10000
```

## 📊 Dashboard Features

### Main Dashboard
- Current bankroll and P&L tracking
- Active bet monitoring
- Performance metrics and charts
- Recent activity log

### Value Bets
- Live opportunity detection
- Filtering by edge, sport, confidence
- One-click bet placement
- Risk indicator warnings

### Performance Analytics
- ROI tracking by time period
- Win rate analysis by sport/market
- Model performance comparison
- Profit timeline visualization

### Backtesting Lab
- Historical strategy testing
- Walk-forward optimization
- Monte Carlo risk analysis
- Parameter sensitivity testing

## 🏗️ Architecture

```
contrarius/
├── main.py              # Entry point and orchestration
├── models.py            # ML model implementations
├── data.py              # Data collection and APIs
├── features.py          # Feature engineering pipeline
├── betting.py           # Value betting logic
├── database.py          # Database models and operations
├── dashboard.py         # Streamlit web interface
├── backtesting.py       # Backtesting engine
├── utils.py             # Utility functions
├── config.py            # Configuration management
└── setup.py             # Database initialization
```

## 🔧 Advanced Configuration

### Model Ensemble Weights
Adjust in `config.py` or via dashboard:
```python
'lightgbm': 0.25,
'xgboost': 0.20,
'catboost': 0.20,
'random_forest': 0.15,
'neural_net': 0.15,
'logistic_regression': 0.05
```

### Feature Importance Weights
```python
'odds_movement': 0.25,
'sharp_money': 0.25,
'market_sentiment': 0.20,
'injuries': 0.15,
'weather': 0.10,
'travel_fatigue': 0.05
```

## 📈 Performance Optimization

1. **Database Indexing**: Ensure proper indexes on frequently queried columns
2. **Model Caching**: Models are saved to disk and loaded on startup
3. **Parallel Processing**: Backtesting uses multiprocessing
4. **Data Caching**: Historical data is cached to reduce API calls

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check credentials in `.env`
   - Try SQLite for local testing

2. **API Rate Limits**
   - Reduce `UPDATE_INTERVAL_SECONDS`
   - Implement request caching
   - Use multiple API keys

3. **Model Training Fails**
   - Ensure sufficient historical data
   - Check feature engineering pipeline
   - Verify data quality

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is provided AS-IS for educational purposes. Use at your own risk.

## 🙏 Acknowledgments

- Betting exchanges for market data
- Open-source ML libraries
- Sports data providers

---

**Remember**: Bet responsibly. The house always has an edge. This system aims to find value, but cannot guarantee profits. 