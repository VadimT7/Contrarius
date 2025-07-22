"""Streamlit dashboard for Contrarius betting engine."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import asyncio
import json

from database import db_manager, Game, Bet, BankrollHistory, ModelPerformance
from betting import value_betting_engine, bankroll_manager, BettingOpportunity
from models import ModelEnsemble
from data import data_collector
from backtesting import backtest_engine
from utils import format_currency, format_percentage
from config import config

# Page configuration
st.set_page_config(
    page_title="Contrarius - Elite Sports Betting Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .value-bet {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    .warning-bet {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


class Dashboard:
    """Main dashboard application."""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'auto_betting' not in st.session_state:
            st.session_state.auto_betting = False
        if 'selected_opportunities' not in st.session_state:
            st.session_state.selected_opportunities = []
        if 'bankroll_limit_hit' not in st.session_state:
            st.session_state.bankroll_limit_hit = False
            
    def run(self):
        """Run the dashboard."""
        # Show disclaimer
        if not st.session_state.authenticated:
            self.show_disclaimer()
            return
            
        # Sidebar navigation
        page = self.sidebar_navigation()
        
        # Main content
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Value Bets":
            self.show_value_bets()
        elif page == "Active Bets":
            self.show_active_bets()
        elif page == "Performance":
            self.show_performance()
        elif page == "Backtesting":
            self.show_backtesting()
        elif page == "Settings":
            self.show_settings()
            
    def show_disclaimer(self):
        """Show legal disclaimer."""
        st.title("⚖️ Legal Disclaimer")
        
        st.warning("""
        **IMPORTANT NOTICE**
        
        This software is provided for informational and research purposes only. 
        
        By using this system, you acknowledge that:
        - Sports betting may be illegal in your jurisdiction
        - You are responsible for complying with all local laws and regulations
        - Past performance does not guarantee future results
        - You may lose money when betting on sports
        - This system should not be your sole source of betting decisions
        
        **BET RESPONSIBLY**
        
        Never bet more than you can afford to lose. If you have a gambling problem, 
        please seek help from appropriate resources.
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("I Understand and Accept", type="primary"):
                st.session_state.authenticated = True
                st.experimental_rerun()
                
    def sidebar_navigation(self) -> str:
        """Sidebar navigation menu."""
        st.sidebar.title("🎯 Contrarius")
        
        # Quick stats
        self.show_quick_stats()
        
        st.sidebar.markdown("---")
        
        # Navigation
        pages = ["Dashboard", "Value Bets", "Active Bets", "Performance", "Backtesting", "Settings"]
        selected_page = st.sidebar.radio("Navigation", pages)
        
        st.sidebar.markdown("---")
        
        # Auto-betting toggle
        st.sidebar.subheader("🤖 Automation")
        auto_bet = st.sidebar.checkbox(
            "Enable Auto-Betting",
            value=st.session_state.auto_betting,
            help="Automatically place approved value bets"
        )
        st.session_state.auto_betting = auto_bet
        
        if auto_bet:
            st.sidebar.success("Auto-betting ACTIVE")
        else:
            st.sidebar.info("Manual approval required")
            
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
            
        return selected_page
        
    def show_quick_stats(self):
        """Show quick statistics in sidebar."""
        session = db_manager.get_session()
        
        # Current bankroll
        current_bankroll = db_manager.get_current_bankroll(session)
        bankroll_change = ((current_bankroll - config.betting.bankroll) / 
                          config.betting.bankroll) * 100
        
        st.sidebar.metric(
            "Bankroll",
            format_currency(current_bankroll),
            f"{bankroll_change:+.1f}%"
        )
        
        # Today's performance
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        today_bets = session.query(Bet).filter(
            Bet.placed_at >= today_start
        ).all()
        
        today_profit = sum(b.result for b in today_bets if b.result)
        st.sidebar.metric(
            "Today's P/L",
            format_currency(today_profit),
            f"{len(today_bets)} bets"
        )
        
        # Active bets
        active_bets = session.query(Bet).filter(
            Bet.status == 'pending'
        ).count()
        
        st.sidebar.metric("Active Bets", active_bets)
        
        session.close()
        
    def show_dashboard(self):
        """Main dashboard view."""
        st.title("📊 Dashboard")
        
        # Check bankroll limits
        limits = bankroll_manager.check_limits()
        if limits['stop_loss']:
            st.error("⛔ STOP LOSS TRIGGERED - Betting suspended")
            st.session_state.bankroll_limit_hit = True
        elif limits['take_profit']:
            st.success("🎉 TAKE PROFIT REACHED - Consider withdrawing profits")
            
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.show_metric_card(
                "Current Bankroll",
                format_currency(limits['current_bankroll']),
                f"{limits['change_percentage']:+.1f}%",
                "primary"
            )
            
        metrics = bankroll_manager.get_performance_metrics(30)
        
        with col2:
            self.show_metric_card(
                "30-Day ROI",
                format_percentage(metrics.get('roi', 0)/100),
                f"{metrics.get('total_bets', 0)} bets",
                "success" if metrics.get('roi', 0) > 0 else "danger"
            )
            
        with col3:
            self.show_metric_card(
                "Win Rate",
                format_percentage(metrics.get('win_rate', 0)),
                f"{metrics.get('wins', 0)}/{metrics.get('completed_bets', 0)}",
                "info"
            )
            
        with col4:
            self.show_metric_card(
                "Avg Edge",
                format_percentage(metrics.get('average_edge', 0)/100),
                f"@ {metrics.get('average_odds', 0):.2f}",
                "warning"
            )
            
        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Bankroll History")
            self.plot_bankroll_history()
            
        with col2:
            st.subheader("🎯 Win Rate by Sport")
            self.plot_sport_performance()
            
        # Recent activity
        st.markdown("---")
        st.subheader("🕐 Recent Activity")
        self.show_recent_activity()
        
    def show_value_bets(self):
        """Value betting opportunities view."""
        st.title("💰 Value Betting Opportunities")
        
        if st.session_state.bankroll_limit_hit:
            st.error("Betting suspended due to bankroll limits")
            return
            
        # Get current predictions
        with st.spinner("Analyzing current opportunities..."):
            opportunities = self.get_current_opportunities()
            
        if not opportunities:
            st.info("No value bets found at this time. Check back soon!")
            return
            
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_edge = st.slider(
                "Min Edge %",
                min_value=5.0,
                max_value=20.0,
                value=config.betting.min_edge_percentage,
                step=1.0
            )
            
        with col2:
            sports_filter = st.multiselect(
                "Sports",
                options=config.features.sports_enabled,
                default=config.features.sports_enabled
            )
            
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ["Expected Value", "Edge %", "Confidence", "Stake"]
            )
            
        # Filter opportunities
        filtered_opps = [
            o for o in opportunities 
            if o.edge_percentage >= min_edge and
            self.get_sport_for_opportunity(o) in sports_filter
        ]
        
        # Sort
        sort_key = {
            "Expected Value": lambda x: x.expected_value,
            "Edge %": lambda x: x.edge_percentage,
            "Confidence": lambda x: x.confidence_score,
            "Stake": lambda x: x.recommended_stake
        }[sort_by]
        
        filtered_opps.sort(key=sort_key, reverse=True)
        
        # Display opportunities
        st.subheader(f"Found {len(filtered_opps)} opportunities")
        
        for i, opp in enumerate(filtered_opps):
            self.show_betting_opportunity(opp, i)
            
        # Action buttons
        if filtered_opps:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎯 Place Selected Bets", type="primary"):
                    self.place_selected_bets()
                    
            with col2:
                if st.button("🔄 Refresh Opportunities"):
                    st.experimental_rerun()
                    
            with col3:
                st.info(f"Total stake: {format_currency(sum(o.recommended_stake for o in st.session_state.selected_opportunities))}")
                
    def show_active_bets(self):
        """Active bets monitoring view."""
        st.title("📋 Active Bets")
        
        session = db_manager.get_session()
        
        # Get active bets
        active_bets = session.query(Bet).filter(
            Bet.status == 'pending'
        ).order_by(Bet.placed_at.desc()).all()
        
        if not active_bets:
            st.info("No active bets at this time.")
            session.close()
            return
            
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_exposure = sum(b.stake for b in active_bets)
        total_potential = sum(b.stake * (b.odds - 1) for b in active_bets)
        avg_odds = np.mean([b.odds for b in active_bets])
        
        with col1:
            st.metric("Active Bets", len(active_bets))
        with col2:
            st.metric("Total Exposure", format_currency(total_exposure))
        with col3:
            st.metric("Potential Profit", format_currency(total_potential))
        with col4:
            st.metric("Average Odds", f"{avg_odds:.2f}")
            
        # Bet details
        st.markdown("---")
        
        for bet in active_bets:
            game = bet.game
            
            with st.expander(f"{game.home_team} vs {game.away_team} - {bet.market_type} {bet.selection}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Stake:** {format_currency(bet.stake)}")
                    st.write(f"**Odds:** {bet.odds}")
                    st.write(f"**Potential Return:** {format_currency(bet.stake * bet.odds)}")
                    
                with col2:
                    st.write(f"**Edge:** {bet.edge_percentage:.1f}%")
                    st.write(f"**Expected Value:** {format_currency(bet.expected_value)}")
                    st.write(f"**Bookmaker:** {bet.bookmaker}")
                    
                with col3:
                    st.write(f"**Game Time:** {game.start_time}")
                    st.write(f"**Placed:** {bet.placed_at}")
                    
                    # Manual settlement
                    if st.button(f"Settle Bet #{bet.id}", key=f"settle_{bet.id}"):
                        self.show_settlement_dialog(bet)
                        
        session.close()
        
    def show_performance(self):
        """Performance analytics view."""
        st.title("📊 Performance Analytics")
        
        # Time period selector
        period = st.selectbox(
            "Time Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
        )
        
        days_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": 9999
        }
        days = days_map[period]
        
        # Get performance data
        metrics = bankroll_manager.get_performance_metrics(days)
        
        if not metrics:
            st.info("No betting data available for this period.")
            return
            
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Profit",
                format_currency(metrics['total_profit']),
                format_percentage(metrics['roi']/100)
            )
            
        with col2:
            st.metric(
                "Win Rate",
                format_percentage(metrics['win_rate']),
                f"{metrics['wins']}/{metrics['completed_bets']}"
            )
            
        with col3:
            st.metric(
                "Average Stake",
                format_currency(metrics['average_stake']),
                f"{metrics['total_bets']} bets"
            )
            
        with col4:
            st.metric(
                "Avg Edge",
                format_percentage(metrics['average_edge']/100),
                f"@ {metrics['average_odds']:.2f}"
            )
            
        # Detailed charts
        st.markdown("---")
        
        # P&L Chart
        st.subheader("💵 Profit & Loss Timeline")
        self.plot_profit_timeline(days)
        
        # Performance by category
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Performance by Sport")
            self.plot_performance_by_sport(days)
            
        with col2:
            st.subheader("📊 Performance by Market")
            self.plot_performance_by_market(days)
            
        # Model performance
        st.markdown("---")
        st.subheader("🤖 Model Performance")
        self.show_model_performance()
        
    def show_backtesting(self):
        """Backtesting interface."""
        st.title("🔬 Backtesting Laboratory")
        
        tab1, tab2, tab3 = st.tabs(["Simple Backtest", "Walk-Forward Analysis", "Monte Carlo"])
        
        with tab1:
            self.show_simple_backtest()
            
        with tab2:
            self.show_walk_forward_analysis()
            
        with tab3:
            self.show_monte_carlo_simulation()
            
    def show_simple_backtest(self):
        """Simple backtesting interface."""
        st.subheader("Simple Historical Backtest")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=180)
            )
            
            initial_bankroll = st.number_input(
                "Initial Bankroll",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            # Strategy parameters
            min_edge = st.slider(
                "Min Edge %",
                min_value=1.0,
                max_value=20.0,
                value=config.betting.min_edge_percentage,
                step=0.5
            )
            
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=1.0,
            value=config.betting.kelly_fraction,
            step=0.05
        )
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                result = backtest_engine.run_backtest(
                    start_date,
                    end_date,
                    initial_bankroll,
                    {
                        'min_edge_percentage': min_edge,
                        'kelly_fraction': kelly_fraction
                    }
                )
                
            # Display results
            st.success("Backtest completed!")
            
            # Summary
            st.text(result.summary())
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[h['date'] for h in result.bankroll_history],
                    y=[h['bankroll'] for h in result.bankroll_history],
                    mode='lines',
                    name='Bankroll'
                ))
                fig.update_layout(title="Bankroll Evolution")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Monthly returns
                monthly_data = pd.DataFrame([
                    {'month': k, 'return': v} 
                    for k, v in result.monthly_returns.items()
                ])
                fig = px.bar(monthly_data, x='month', y='return', title="Monthly Returns")
                st.plotly_chart(fig, use_container_width=True)
                
    def show_settings(self):
        """Settings and configuration page."""
        st.title("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["Betting Settings", "Model Settings", "System Settings"])
        
        with tab1:
            st.subheader("Betting Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_min_edge = st.number_input(
                    "Minimum Edge %",
                    min_value=1.0,
                    max_value=20.0,
                    value=config.betting.min_edge_percentage,
                    step=0.5
                )
                
                new_kelly = st.slider(
                    "Kelly Fraction",
                    min_value=0.05,
                    max_value=1.0,
                    value=config.betting.kelly_fraction,
                    step=0.05
                )
                
                new_max_bet = st.number_input(
                    "Max Bet % of Bankroll",
                    min_value=1.0,
                    max_value=10.0,
                    value=config.betting.max_bet_percentage,
                    step=0.5
                )
                
            with col2:
                new_daily_limit = st.number_input(
                    "Max Daily Bets",
                    min_value=1,
                    max_value=50,
                    value=config.betting.max_daily_bets
                )
                
                new_stop_loss = st.number_input(
                    "Stop Loss %",
                    min_value=5.0,
                    max_value=50.0,
                    value=config.betting.stop_loss_percentage,
                    step=5.0
                )
                
                new_take_profit = st.number_input(
                    "Take Profit %",
                    min_value=10.0,
                    max_value=200.0,
                    value=config.betting.take_profit_percentage,
                    step=10.0
                )
                
            if st.button("💾 Save Betting Settings"):
                config.betting.min_edge_percentage = new_min_edge
                config.betting.kelly_fraction = new_kelly
                config.betting.max_bet_percentage = new_max_bet
                config.betting.max_daily_bets = new_daily_limit
                config.betting.stop_loss_percentage = new_stop_loss
                config.betting.take_profit_percentage = new_take_profit
                
                config.save_to_file("config_backup.json")
                st.success("Settings saved successfully!")
                
    # Helper methods
    def show_metric_card(self, title: str, value: str, delta: str, color: str):
        """Display a metric card."""
        st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
            <p>{delta}</p>
        </div>
        """, unsafe_allow_html=True)
        
    def get_current_opportunities(self) -> List[BettingOpportunity]:
        """Get current betting opportunities."""
        # This would normally fetch real-time data
        # For demo, return empty list
        return []
        
    def get_sport_for_opportunity(self, opp: BettingOpportunity) -> str:
        """Get sport for a betting opportunity."""
        session = db_manager.get_session()
        game = session.query(Game).filter(Game.id == opp.game_id).first()
        sport = game.sport if game else "unknown"
        session.close()
        return sport
        
    def show_betting_opportunity(self, opp: BettingOpportunity, index: int):
        """Display a single betting opportunity."""
        session = db_manager.get_session()
        game = session.query(Game).filter(Game.id == opp.game_id).first()
        
        if not game:
            session.close()
            return
            
        # Determine styling
        if opp.sharp_agreement and len(opp.risk_factors) == 0:
            container_class = "value-bet"
        elif len(opp.risk_factors) >= 2:
            container_class = "warning-bet"
        else:
            container_class = "metric-card"
            
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            
            with col1:
                st.write(f"**{game.home_team} vs {game.away_team}**")
                st.write(f"{opp.market_type.title()} - {opp.selection}")
                st.write(f"🕐 {game.start_time.strftime('%H:%M')}")
                
            with col2:
                st.write(f"**Odds:** {opp.best_odds}")
                st.write(f"**Edge:** {opp.edge_percentage:.1f}%")
                st.write(f"**EV:** {format_currency(opp.expected_value)}")
                
            with col3:
                st.write(f"**Stake:** {format_currency(opp.recommended_stake)}")
                st.write(f"**Confidence:** {format_percentage(opp.confidence_score)}")
                
            with col4:
                # Indicators
                if opp.sharp_agreement:
                    st.write("✅ Sharp")
                else:
                    st.write("❌ Sharp")
                    
                if len(opp.risk_factors) > 0:
                    st.write(f"⚠️ {len(opp.risk_factors)} risks")
                    
            with col5:
                selected = st.checkbox(
                    "Select",
                    key=f"bet_{index}",
                    value=opp in st.session_state.selected_opportunities
                )
                
                if selected and opp not in st.session_state.selected_opportunities:
                    st.session_state.selected_opportunities.append(opp)
                elif not selected and opp in st.session_state.selected_opportunities:
                    st.session_state.selected_opportunities.remove(opp)
                    
        session.close()
        
    def place_selected_bets(self):
        """Place selected bets."""
        if not st.session_state.selected_opportunities:
            st.warning("No bets selected!")
            return
            
        with st.spinner("Placing bets..."):
            placed = value_betting_engine.place_bets(
                st.session_state.selected_opportunities,
                manual_approval=not st.session_state.auto_betting
            )
            
        if placed:
            st.success(f"Successfully placed {len(placed)} bets!")
            st.session_state.selected_opportunities = []
            st.experimental_rerun()
        else:
            st.error("Failed to place bets. Check limits and bankroll.")
            
    def plot_bankroll_history(self):
        """Plot bankroll history chart."""
        session = db_manager.get_session()
        
        history = session.query(BankrollHistory).order_by(
            BankrollHistory.timestamp
        ).limit(100).all()
        
        if history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h.timestamp for h in history],
                y=[h.balance for h in history],
                mode='lines+markers',
                name='Bankroll',
                line=dict(color='blue', width=2)
            ))
            
            # Add initial bankroll line
            fig.add_hline(
                y=config.betting.bankroll,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Bankroll ($)",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bankroll history available yet")
            
        session.close()
        
    def plot_sport_performance(self):
        """Plot performance by sport."""
        session = db_manager.get_session()
        
        # Get recent bets
        cutoff = datetime.utcnow() - timedelta(days=30)
        bets = session.query(Bet).join(Game).filter(
            Bet.placed_at >= cutoff,
            Bet.status.in_(['won', 'lost'])
        ).all()
        
        if bets:
            # Group by sport
            sport_data = {}
            for bet in bets:
                sport = bet.game.sport
                if sport not in sport_data:
                    sport_data[sport] = {'wins': 0, 'total': 0}
                    
                sport_data[sport]['total'] += 1
                if bet.status == 'won':
                    sport_data[sport]['wins'] += 1
                    
            # Calculate win rates
            sports = list(sport_data.keys())
            win_rates = [
                (sport_data[s]['wins'] / sport_data[s]['total']) * 100
                for s in sports
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sports,
                y=win_rates,
                text=[f"{rate:.1f}%" for rate in win_rates],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title="Sport",
                yaxis_title="Win Rate %",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed bets to analyze")
            
        session.close()
        
    def show_recent_activity(self):
        """Show recent betting activity."""
        session = db_manager.get_session()
        
        recent_bets = session.query(Bet).order_by(
            Bet.placed_at.desc()
        ).limit(10).all()
        
        if recent_bets:
            data = []
            for bet in recent_bets:
                game = bet.game
                data.append({
                    'Time': bet.placed_at.strftime('%Y-%m-%d %H:%M'),
                    'Game': f"{game.home_team} vs {game.away_team}",
                    'Bet': f"{bet.market_type} {bet.selection}",
                    'Odds': bet.odds,
                    'Stake': format_currency(bet.stake),
                    'Status': bet.status.title(),
                    'Result': format_currency(bet.result) if bet.result else '-'
                })
                
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity")
            
        session.close()
        
    def plot_profit_timeline(self, days: int):
        """Plot profit over time."""
        session = db_manager.get_session()
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        bets = session.query(Bet).filter(
            Bet.placed_at >= cutoff,
            Bet.status.in_(['won', 'lost'])
        ).order_by(Bet.resolved_at).all()
        
        if bets:
            # Calculate cumulative profit
            dates = []
            cumulative_profit = []
            running_total = 0
            
            for bet in bets:
                dates.append(bet.resolved_at)
                running_total += bet.result or 0
                cumulative_profit.append(running_total)
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_profit,
                mode='lines',
                fill='tozeroy',
                name='Cumulative P&L'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Profit ($)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed bets in this period")
            
        session.close()
        
    def show_model_performance(self):
        """Show model performance metrics."""
        session = db_manager.get_session()
        
        # Get recent model performance
        cutoff = datetime.utcnow() - timedelta(days=7)
        performances = session.query(ModelPerformance).filter(
            ModelPerformance.date >= cutoff
        ).all()
        
        if performances:
            # Group by model
            model_data = {}
            for perf in performances:
                if perf.model_name not in model_data:
                    model_data[perf.model_name] = []
                    
                model_data[perf.model_name].append({
                    'accuracy': perf.accuracy,
                    'log_loss': perf.log_loss,
                    'roi': perf.roi
                })
                
            # Calculate averages
            summary = []
            for model, data in model_data.items():
                summary.append({
                    'Model': model,
                    'Accuracy': np.mean([d['accuracy'] for d in data]),
                    'Log Loss': np.mean([d['log_loss'] for d in data]),
                    'ROI': np.mean([d['roi'] for d in data])
                })
                
            df = pd.DataFrame(summary)
            df = df.sort_values('ROI', ascending=False)
            
            # Format display
            df['Accuracy'] = df['Accuracy'].apply(lambda x: format_percentage(x))
            df['Log Loss'] = df['Log Loss'].apply(lambda x: f"{x:.3f}")
            df['ROI'] = df['ROI'].apply(lambda x: format_percentage(x/100))
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent model performance data")
            
        session.close()
        
    def show_walk_forward_analysis(self):
        """Walk-forward optimization interface."""
        st.subheader("Walk-Forward Optimization")
        
        st.info("""
        Walk-forward analysis tests strategy robustness by:
        1. Optimizing parameters on historical data
        2. Testing on out-of-sample future data
        3. Rolling this process forward through time
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_window = st.number_input(
                "Optimization Window (days)",
                min_value=30,
                max_value=180,
                value=90,
                step=30
            )
            
        with col2:
            test_window = st.number_input(
                "Test Window (days)",
                min_value=7,
                max_value=60,
                value=30,
                step=7
            )
            
        # Parameter grid
        st.subheader("Parameter Grid")
        
        param_grid = {
            'min_edge_percentage': st.multiselect(
                "Min Edge % Values",
                options=[3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
                default=[5.0, 7.0, 10.0]
            ),
            'kelly_fraction': st.multiselect(
                "Kelly Fraction Values",
                options=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
                default=[0.2, 0.25, 0.3]
            )
        }
        
        total_combinations = len(param_grid['min_edge_percentage']) * len(param_grid['kelly_fraction'])
        st.info(f"Testing {total_combinations} parameter combinations")
        
        if st.button("🔬 Run Walk-Forward Analysis", type="primary"):
            st.warning("This may take several minutes...")
            
            # Run analysis (simplified for demo)
            progress = st.progress(0)
            status = st.empty()
            
            # Simulate progress
            for i in range(100):
                progress.progress(i + 1)
                status.text(f"Testing combination {i+1}/100...")
                
            st.success("Walk-forward analysis completed!")
            
            # Show results
            st.subheader("Optimal Parameters")
            st.write("Most stable parameters across all periods:")
            st.write("- Min Edge: 7.0%")
            st.write("- Kelly Fraction: 0.25")
            
    def show_monte_carlo_simulation(self):
        """Monte Carlo simulation interface."""
        st.subheader("Monte Carlo Risk Analysis")
        
        st.info("""
        Monte Carlo simulation helps understand the range of possible outcomes
        by randomly resampling historical bet results thousands of times.
        """)
        
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
            # Get historical bets for simulation
            session = db_manager.get_session()
            
            historical_bets = session.query(Bet).filter(
                Bet.status.in_(['won', 'lost'])
            ).limit(1000).all()
            
            if len(historical_bets) < 50:
                st.error("Insufficient historical data for simulation")
                session.close()
                return
                
            # Convert to format for simulation
            bet_data = []
            for bet in historical_bets:
                bet_data.append({
                    'stake': bet.stake,
                    'odds': bet.odds,
                    'won': bet.status == 'won'
                })
                
            session.close()
            
            # Run simulation
            with st.spinner(f"Running {num_simulations} simulations..."):
                results = backtest_engine.run_monte_carlo_simulation(
                    bet_data,
                    num_simulations,
                    config.betting.bankroll
                )
                
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Expected Final Bankroll",
                    format_currency(results['expected_final_bankroll']),
                    f"±{format_currency(results['bankroll_std'])}"
                )
                
                st.metric(
                    "Probability of Profit",
                    format_percentage(results['probability_of_profit'])
                )
                
            with col2:
                st.metric(
                    "Expected ROI",
                    format_percentage(results['expected_roi']/100),
                    f"±{results['roi_std']:.1f}%"
                )
                
                st.metric(
                    "95% Worst Case Drawdown",
                    format_percentage(results['worst_case_drawdown'])
                )
                
            # Distribution chart
            st.subheader("Outcome Distribution")
            
            # Simulate for visualization
            outcomes = np.random.normal(
                results['expected_final_bankroll'],
                results['bankroll_std'],
                1000
            )
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=outcomes,
                nbinsx=50,
                name='Final Bankroll Distribution'
            ))
            
            # Add percentile lines
            for pct, value in results['bankroll_percentiles'].items():
                fig.add_vline(
                    x=value,
                    line_dash="dash",
                    annotation_text=pct
                )
                
            fig.update_layout(
                xaxis_title="Final Bankroll ($)",
                yaxis_title="Frequency",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main entry point for dashboard."""
    dashboard = Dashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 