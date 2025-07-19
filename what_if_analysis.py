"""
What-If Portfolio Analysis System
================================
Interactive portfolio simulation and optimization tool
for testing portfolio changes before committing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

@dataclass
class PortfolioChange:
    """Represents a proposed portfolio change"""
    action: str  # 'add', 'remove', 'modify'
    symbol: str
    shares: float = 0
    target_weight: float = 0
    notes: str = ""

@dataclass
class PortfolioMetrics:
    """Portfolio analysis metrics"""
    total_value: float
    total_score: float
    average_score: float
    symbol_count: int
    sector_breakdown: Dict[str, float]
    risk_metrics: Dict[str, float]
    top_performers: List[Dict]
    diversification_score: float
    volatility_estimate: float

class WhatIfAnalyzer:
    def __init__(self):
        """Initialize What-If analyzer"""
        self.base_portfolio = {}
        self.proposed_changes = []
        self.market_data_cache = {}
    
    def set_base_portfolio(self, portfolio_data: Dict[str, Any]):
        """Set the current portfolio as baseline"""
        self.base_portfolio = portfolio_data.copy()
    
    def add_proposed_change(self, change: PortfolioChange):
        """Add a proposed change to the simulation"""
        self.proposed_changes.append(change)
    
    def clear_proposed_changes(self):
        """Clear all proposed changes"""
        self.proposed_changes = []
    
    def calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if not portfolio:
            return PortfolioMetrics(
                total_value=0, total_score=0, average_score=0, symbol_count=0,
                sector_breakdown={}, risk_metrics={}, top_performers=[],
                diversification_score=0, volatility_estimate=0
            )
        
        # Get current market data for all symbols
        portfolio_data = []
        total_value = 0
        total_weighted_score = 0
        sector_values = {}
        
        for symbol, holding in portfolio.items():
            try:
                market_data = self._get_market_data(symbol)
                shares = holding.get('shares', 0)
                current_price = market_data.get('current_price', 0)
                position_value = shares * current_price
                
                # Get or calculate stock score
                stock_score = holding.get('score', 0)
                if stock_score == 0:
                    stock_score = self._calculate_stock_score(symbol, market_data)
                
                # Sector classification
                sector = market_data.get('sector', 'Unknown')
                sector_values[sector] = sector_values.get(sector, 0) + position_value
                
                portfolio_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': current_price,
                    'position_value': position_value,
                    'score': stock_score,
                    'sector': sector,
                    'market_data': market_data
                })
                
                total_value += position_value
                total_weighted_score += stock_score * position_value
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")
                continue
        
        # Calculate metrics
        average_score = total_weighted_score / total_value if total_value > 0 else 0
        symbol_count = len(portfolio_data)
        
        # Sector breakdown as percentages
        sector_breakdown = {}
        if total_value > 0:
            sector_breakdown = {
                sector: (value / total_value) * 100 
                for sector, value in sector_values.items()
            }
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_data, total_value)
        
        # Top performers
        top_performers = sorted(
            portfolio_data, 
            key=lambda x: x['score'], 
            reverse=True
        )[:5]
        
        # Diversification score
        diversification_score = self._calculate_diversification_score(
            sector_breakdown, symbol_count
        )
        
        # Volatility estimate
        volatility_estimate = self._estimate_portfolio_volatility(portfolio_data)
        
        return PortfolioMetrics(
            total_value=total_value,
            total_score=total_weighted_score,
            average_score=average_score,
            symbol_count=symbol_count,
            sector_breakdown=sector_breakdown,
            risk_metrics=risk_metrics,
            top_performers=top_performers,
            diversification_score=diversification_score,
            volatility_estimate=volatility_estimate
        )
    
    def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol with caching"""
        if symbol not in self.market_data_cache:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                self.market_data_cache[symbol] = {
                    'current_price': info.get('currentPrice', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0)
                }
            except Exception as e:
                print(f"⚠️ Error fetching data for {symbol}: {e}")
                self.market_data_cache[symbol] = {
                    'current_price': 0, 'sector': 'Unknown', 'industry': 'Unknown',
                    'market_cap': 0, 'beta': 1.0, 'pe_ratio': 0, 'dividend_yield': 0
                }
        
        return self.market_data_cache[symbol]
    
    def _calculate_stock_score(self, symbol: str, market_data: Dict) -> float:
        """Calculate basic stock score if not available"""
        # Simplified scoring based on available data
        score = 5.0  # Start with neutral
        
        try:
            pe_ratio = market_data.get('pe_ratio', 0)
            if 0 < pe_ratio < 15:
                score += 1
            elif 15 <= pe_ratio < 25:
                score += 0.5
            elif pe_ratio > 30:
                score -= 0.5
            
            dividend_yield = market_data.get('dividend_yield', 0)
            if dividend_yield > 0.03:  # > 3%
                score += 1
            elif dividend_yield > 0.01:  # > 1%
                score += 0.5
            
            # Market cap consideration
            market_cap = market_data.get('market_cap', 0)
            if market_cap > 100e9:  # Large cap
                score += 0.5
            
        except Exception:
            pass
        
        return max(0, min(10, score))
    
    def _calculate_risk_metrics(self, portfolio_data: List[Dict], total_value: float) -> Dict:
        """Calculate portfolio risk metrics"""
        if not portfolio_data or total_value == 0:
            return {}
        
        # Calculate weighted beta
        weighted_beta = 0
        for stock in portfolio_data:
            weight = stock['position_value'] / total_value
            beta = stock['market_data'].get('beta', 1.0)
            weighted_beta += weight * beta
        
        # Calculate concentration risk
        max_position_pct = max(
            stock['position_value'] / total_value * 100 
            for stock in portfolio_data
        )
        
        # Calculate low-score exposure
        low_score_exposure = sum(
            stock['position_value'] 
            for stock in portfolio_data 
            if stock['score'] < 5
        ) / total_value * 100
        
        return {
            'portfolio_beta': weighted_beta,
            'max_position_concentration': max_position_pct,
            'low_score_exposure': low_score_exposure,
            'risk_level': self._assess_risk_level(weighted_beta, max_position_pct)
        }
    
    def _assess_risk_level(self, beta: float, concentration: float) -> str:
        """Assess overall portfolio risk level"""
        risk_score = 0
        
        # Beta risk
        if beta > 1.3:
            risk_score += 2
        elif beta > 1.1:
            risk_score += 1
        
        # Concentration risk
        if concentration > 20:
            risk_score += 2
        elif concentration > 10:
            risk_score += 1
        
        if risk_score >= 3:
            return "High"
        elif risk_score >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_diversification_score(self, sector_breakdown: Dict, symbol_count: int) -> float:
        """Calculate diversification score (0-10)"""
        if symbol_count == 0:
            return 0
        
        # Base score from symbol count
        if symbol_count >= 20:
            count_score = 4
        elif symbol_count >= 10:
            count_score = 3
        elif symbol_count >= 5:
            count_score = 2
        else:
            count_score = 1
        
        # Sector diversification score
        if not sector_breakdown:
            return count_score
        
        # Calculate sector concentration
        max_sector_pct = max(sector_breakdown.values()) if sector_breakdown else 100
        sector_count = len(sector_breakdown)
        
        if max_sector_pct < 30 and sector_count >= 5:
            sector_score = 6
        elif max_sector_pct < 40 and sector_count >= 4:
            sector_score = 4
        elif max_sector_pct < 50 and sector_count >= 3:
            sector_score = 3
        elif sector_count >= 2:
            sector_score = 2
        else:
            sector_score = 1
        
        return min(10, count_score + sector_score)
    
    def _estimate_portfolio_volatility(self, portfolio_data: List[Dict]) -> float:
        """Estimate portfolio volatility based on beta and diversification"""
        if not portfolio_data:
            return 0
        
        # Simple volatility estimate based on weighted beta
        total_value = sum(stock['position_value'] for stock in portfolio_data)
        weighted_volatility = 0
        
        for stock in portfolio_data:
            weight = stock['position_value'] / total_value
            beta = stock['market_data'].get('beta', 1.0)
            # Assume market volatility of ~20% annually
            stock_volatility = beta * 20
            weighted_volatility += weight * stock_volatility
        
        # Adjust for diversification (more stocks = lower volatility)
        diversification_factor = max(0.5, 1 - (len(portfolio_data) - 1) * 0.05)
        
        return weighted_volatility * diversification_factor
    
    def simulate_portfolio_changes(self) -> Tuple[PortfolioMetrics, PortfolioMetrics]:
        """Simulate proposed changes and return before/after metrics"""
        
        # Calculate current portfolio metrics
        current_metrics = self.calculate_portfolio_metrics(self.base_portfolio)
        
        # Apply proposed changes to create simulated portfolio
        simulated_portfolio = self.base_portfolio.copy()
        
        for change in self.proposed_changes:
            if change.action == 'add':
                # Add new position or increase existing
                if change.symbol in simulated_portfolio:
                    current_shares = simulated_portfolio[change.symbol].get('shares', 0)
                    simulated_portfolio[change.symbol]['shares'] = current_shares + change.shares
                else:
                    simulated_portfolio[change.symbol] = {
                        'shares': change.shares,
                        'score': 0  # Will be calculated
                    }
            
            elif change.action == 'remove':
                # Remove position or reduce shares
                if change.symbol in simulated_portfolio:
                    if change.shares == 0:  # Remove completely
                        del simulated_portfolio[change.symbol]
                    else:
                        current_shares = simulated_portfolio[change.symbol].get('shares', 0)
                        new_shares = max(0, current_shares - change.shares)
                        if new_shares == 0:
                            del simulated_portfolio[change.symbol]
                        else:
                            simulated_portfolio[change.symbol]['shares'] = new_shares
            
            elif change.action == 'modify':
                # Modify existing position
                if change.symbol in simulated_portfolio:
                    simulated_portfolio[change.symbol]['shares'] = change.shares
        
        # Calculate simulated portfolio metrics
        simulated_metrics = self.calculate_portfolio_metrics(simulated_portfolio)
        
        return current_metrics, simulated_metrics
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        current_metrics, simulated_metrics = self.simulate_portfolio_changes()
        
        # Calculate changes
        value_change = simulated_metrics.total_value - current_metrics.total_value
        score_change = simulated_metrics.average_score - current_metrics.average_score
        diversification_change = simulated_metrics.diversification_score - current_metrics.diversification_score
        volatility_change = simulated_metrics.volatility_estimate - current_metrics.volatility_estimate
        
        # Sector changes
        sector_changes = {}
        all_sectors = set(list(current_metrics.sector_breakdown.keys()) + 
                         list(simulated_metrics.sector_breakdown.keys()))
        
        for sector in all_sectors:
            current_pct = current_metrics.sector_breakdown.get(sector, 0)
            simulated_pct = simulated_metrics.sector_breakdown.get(sector, 0)
            sector_changes[sector] = simulated_pct - current_pct
        
        return {
            'current_metrics': current_metrics,
            'simulated_metrics': simulated_metrics,
            'changes': {
                'value_change': value_change,
                'value_change_pct': (value_change / current_metrics.total_value * 100) if current_metrics.total_value > 0 else 0,
                'score_change': score_change,
                'diversification_change': diversification_change,
                'volatility_change': volatility_change,
                'sector_changes': sector_changes
            },
            'recommendations': self._generate_recommendations(current_metrics, simulated_metrics)
        }
    
    def _generate_recommendations(self, current: PortfolioMetrics, simulated: PortfolioMetrics) -> List[str]:
        """Generate recommendations based on simulation"""
        recommendations = []
        
        # Score improvement
        score_diff = simulated.average_score - current.average_score
        if score_diff > 0.5:
            recommendations.append(f"✅ Score improves by {score_diff:.1f} points - Good change!")
        elif score_diff < -0.5:
            recommendations.append(f"⚠️ Score decreases by {abs(score_diff):.1f} points - Consider alternatives")
        
        # Diversification
        div_diff = simulated.diversification_score - current.diversification_score
        if div_diff > 1:
            recommendations.append("✅ Diversification improves significantly")
        elif div_diff < -1:
            recommendations.append("⚠️ Diversification decreases - Consider adding different sectors")
        
        # Risk assessment
        current_risk = current.risk_metrics.get('risk_level', 'Unknown')
        simulated_risk = simulated.risk_metrics.get('risk_level', 'Unknown')
        
        if current_risk != simulated_risk:
            if simulated_risk == 'High':
                recommendations.append("🔴 Risk level increases to HIGH - Review position sizes")
            elif simulated_risk == 'Low' and current_risk != 'Low':
                recommendations.append("✅ Risk level decreases - More conservative portfolio")
        
        # Concentration warnings
        max_concentration = simulated.risk_metrics.get('max_position_concentration', 0)
        if max_concentration > 15:
            recommendations.append(f"⚠️ Largest position is {max_concentration:.1f}% - Consider reducing concentration")
        
        return recommendations

class StreamlitWhatIfInterface:
    """Streamlit interface for What-If analysis"""
    
    def __init__(self):
        self.analyzer = WhatIfAnalyzer()
    
    def render_what_if_interface(self, current_portfolio: Dict[str, Any]):
        """Render the complete What-If analysis interface"""
        
        st.subheader("🔮 What-If Portfolio Analysis")
        st.markdown("Test portfolio changes before committing them!")
        
        # Set base portfolio
        self.analyzer.set_base_portfolio(current_portfolio)
        
        # Proposed changes interface
        with st.expander("📝 Propose Portfolio Changes", expanded=True):
            self._render_change_interface()
        
        # Analysis results
        if self.analyzer.proposed_changes:
            with st.expander("📊 Analysis Results", expanded=True):
                self._render_analysis_results()
        
        # Quick scenarios
        with st.expander("⚡ Quick Scenarios"):
            self._render_quick_scenarios()
    
    def _render_change_interface(self):
        """Render interface for proposing changes"""
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            action = st.selectbox("Action", ["Add", "Remove", "Modify"])
        
        with col3:
            shares = st.number_input("Shares", min_value=0.0, value=0.0, step=1.0)
        
        with col4:
            if st.button("Add Change"):
                if symbol and shares > 0:
                    change = PortfolioChange(
                        action=action.lower(),
                        symbol=symbol,
                        shares=shares
                    )
                    self.analyzer.add_proposed_change(change)
                    st.success(f"Added: {action} {shares} shares of {symbol}")
                    st.rerun()
        
        # Show current proposed changes
        if self.analyzer.proposed_changes:
            st.markdown("**Proposed Changes:**")
            changes_data = []
            for i, change in enumerate(self.analyzer.proposed_changes):
                changes_data.append({
                    'Action': change.action.title(),
                    'Symbol': change.symbol,
                    'Shares': change.shares,
                    'Remove': i
                })
            
            changes_df = pd.DataFrame(changes_data)
            st.dataframe(changes_df[['Action', 'Symbol', 'Shares']], hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear All Changes"):
                    self.analyzer.clear_proposed_changes()
                    st.rerun()
    
    def _render_analysis_results(self):
        """Render analysis comparison results"""
        
        report = self.analyzer.generate_comparison_report()
        current = report['current_metrics']
        simulated = report['simulated_metrics']
        changes = report['changes']
        
        # Key metrics comparison
        st.markdown("### 📈 Key Metrics Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${simulated.total_value:,.0f}",
                f"${changes['value_change']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Average Score",
                f"{simulated.average_score:.1f}",
                f"{changes['score_change']:+.1f}"
            )
        
        with col3:
            st.metric(
                "Diversification",
                f"{simulated.diversification_score:.1f}/10",
                f"{changes['diversification_change']:+.1f}"
            )
        
        with col4:
            st.metric(
                "Est. Volatility",
                f"{simulated.volatility_estimate:.1f}%",
                f"{changes['volatility_change']:+.1f}%"
            )
        
        # Sector allocation comparison
        st.markdown("### 🏭 Sector Allocation Changes")
        
        if current.sector_breakdown or simulated.sector_breakdown:
            self._render_sector_comparison(current.sector_breakdown, simulated.sector_breakdown)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = report['recommendations']
        if recommendations:
            for rec in recommendations:
                if "✅" in rec:
                    st.success(rec)
                elif "⚠️" in rec or "🔴" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
        else:
            st.info("No specific recommendations - changes appear neutral")
    
    def _render_sector_comparison(self, current_sectors: Dict, simulated_sectors: Dict):
        """Render sector allocation comparison chart"""
        
        all_sectors = set(list(current_sectors.keys()) + list(simulated_sectors.keys()))
        
        comparison_data = []
        for sector in all_sectors:
            current_pct = current_sectors.get(sector, 0)
            simulated_pct = simulated_sectors.get(sector, 0)
            
            comparison_data.append({
                'Sector': sector,
                'Current': current_pct,
                'Simulated': simulated_pct,
                'Change': simulated_pct - current_pct
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=df['Sector'],
            y=df['Current'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Simulated',
            x=df['Sector'],
            y=df['Simulated'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Sector Allocation: Current vs Simulated",
            xaxis_title="Sector",
            yaxis_title="Allocation (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_quick_scenarios(self):
        """Render quick scenario testing"""
        
        st.markdown("**Quick Scenario Tests:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛡️ Add Defensive Stock"):
                # Add a defensive stock like KO or PG
                defensive_stocks = ['KO', 'PG', 'JNJ', 'MCD']
                import random
                stock = random.choice(defensive_stocks)
                change = PortfolioChange(action='add', symbol=stock, shares=10)
                self.analyzer.add_proposed_change(change)
                st.rerun()
        
        with col2:
            if st.button("🚀 Add Growth Stock"):
                # Add a growth stock
                growth_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
                import random
                stock = random.choice(growth_stocks)
                change = PortfolioChange(action='add', symbol=stock, shares=5)
                self.analyzer.add_proposed_change(change)
                st.rerun()
        
        with col3:
            if st.button("🏦 Add Financial Stock"):
                # Add a financial stock
                financial_stocks = ['JPM', 'BAC', 'WFC', 'GS']
                import random
                stock = random.choice(financial_stocks)
                change = PortfolioChange(action='add', symbol=stock, shares=15)
                self.analyzer.add_proposed_change(change)
                st.rerun()

# Global interface instance
what_if_interface = StreamlitWhatIfInterface()
