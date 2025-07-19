"""
Enhanced Features Integration Module
===================================
Integration layer for SQLite Portfolio, Advanced Caching, 
Async Data Loading, and What-If Analysis.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Import our new modules
from portfolio_database import PortfolioDatabase
from advanced_caching import AdvancedCache
from async_data_loader import AsyncStockDataLoader
from what_if_analysis import WhatIfAnalyzer

class EnhancedFeaturesManager:
    """Manager class for all enhanced features"""
    
    def __init__(self):
        """Initialize enhanced features manager"""
        try:
            self.portfolio_db = PortfolioDatabase()
            self.async_loader = AsyncStockDataLoader()
            self.cache = AdvancedCache()
            self.what_if_analyzer = WhatIfAnalyzer()
            self.cache_stats = {}
        except Exception as e:
            print(f"Warning: Could not initialize some enhanced features: {e}")
            self.portfolio_db = None
            self.async_loader = None
            self.cache = None
            self.what_if_analyzer = None
            self.cache_stats = {}
    
    def initialize_all_systems(self):
        """Initialize all enhanced feature systems"""
        try:
            # Initialize session state
            self.initialize_session_state()
            
            # Verify all systems are ready
            systems_status = {
                'portfolio_db': self.portfolio_db is not None,
                'async_loader': self.async_loader is not None,
                'cache': self.cache is not None,
                'what_if_analyzer': self.what_if_analyzer is not None
            }
            
            # Log initialization status
            for system, status in systems_status.items():
                if status:
                    print(f"✅ {system}: Initialized")
                else:
                    print(f"❌ {system}: Failed to initialize")
            
            return all(systems_status.values())
            
        except Exception as e:
            print(f"❌ Enhanced features initialization failed: {e}")
            return False
    
    def initialize_session_state(self):
        """Initialize session state for enhanced features"""
        
        # Migration flag
        if "portfolio_migrated" not in st.session_state:
            st.session_state.portfolio_migrated = False
        
        # Async loading preferences
        if "use_async_loading" not in st.session_state:
            st.session_state.use_async_loading = True
        
        # Cache preferences
        if "cache_enabled" not in st.session_state:
            st.session_state.cache_enabled = True
        
        # What-if analysis state
        if "what_if_active" not in st.session_state:
            st.session_state.what_if_active = False
    
    def migrate_legacy_portfolio(self):
        """Migrate from JSON portfolio to SQLite"""
        
        if st.session_state.portfolio_migrated:
            return True
        
        st.markdown("### 🔄 Portfolio System Upgrade")
        
        # Check if legacy portfolio exists
        legacy_portfolio = st.session_state.get("portfolio", [])
        
        if legacy_portfolio:
            st.info(f"Found {len(legacy_portfolio)} stocks in legacy portfolio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Upgrade to SQLite Database"):
                    
                    with st.spinner("Migrating portfolio..."):
                        success_count = 0
                        
                        for symbol in legacy_portfolio:
                            if self.portfolio_db.add_to_portfolio(symbol):
                                success_count += 1
                        
                        st.success(f"✅ Successfully migrated {success_count}/{len(legacy_portfolio)} stocks")
                        st.session_state.portfolio_migrated = True
                        
                        # Backup legacy portfolio
                        st.session_state.portfolio_backup = legacy_portfolio.copy()
                        
                        st.rerun()
            
            with col2:
                if st.button("📋 Keep Current System"):
                    st.session_state.portfolio_migrated = True
                    st.info("Continuing with current portfolio system")
                    st.rerun()
        
        else:
            # No legacy portfolio, start fresh with SQLite
            st.session_state.portfolio_migrated = True
            st.success("✅ Starting fresh with enhanced SQLite portfolio system")
            
        return st.session_state.portfolio_migrated
    
    def get_enhanced_portfolio(self) -> List[str]:
        """Get portfolio symbols from SQLite database"""
        try:
            return self.portfolio_db.get_portfolio_symbols()
        except Exception as e:
            st.error(f"Error accessing portfolio database: {e}")
            # Fallback to legacy portfolio
            return st.session_state.get("portfolio", [])
    
    def add_to_enhanced_portfolio(self, symbol: str, shares: float = 0, 
                                 cost_per_share: float = 0) -> bool:
        """Add stock to enhanced portfolio"""
        try:
            success = self.portfolio_db.add_to_portfolio(
                symbol=symbol,
                shares=shares,
                cost_per_share=cost_per_share
            )
            
            if success:
                # Clear relevant caches
                self.invalidate_portfolio_caches()
                
            return success
        except Exception as e:
            st.error(f"Error adding to portfolio: {e}")
            return False
    
    def remove_from_enhanced_portfolio(self, symbol: str) -> bool:
        """Remove stock from enhanced portfolio"""
        try:
            success = self.portfolio_db.remove_from_portfolio(symbol)
            
            if success:
                # Clear relevant caches
                self.invalidate_portfolio_caches()
                
            return success
        except Exception as e:
            st.error(f"Error removing from portfolio: {e}")
            return False
    
    def analyze_portfolio_async(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze portfolio using async loading"""
        
        if not st.session_state.use_async_loading or len(symbols) < 3:
            # Use synchronous loading for small portfolios
            return self._analyze_portfolio_sync(symbols)
        
        try:
            return self.async_loader.fetch_multiple_stocks_threaded(symbols)
        except Exception as e:
            st.error(f"Async analysis failed: {e}")
            return self._analyze_portfolio_sync(symbols)
    
    def _analyze_portfolio_sync(self, symbols: List[str]) -> Dict[str, Any]:
        """Fallback synchronous portfolio analysis"""
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Use cached analysis if available
                from AS_MH_v6 import calculate_scores_yahoo, get_industry_pe
                
                industry_pe = get_industry_pe(info)
                scores = calculate_scores_yahoo(info, industry_pe)
                
                if scores:
                    total_score = sum(scores.values()) / len(scores)
                    
                    results[symbol] = {
                        'success': True,
                        'analysis': {
                            'scores': scores,
                            'total_score': total_score,
                            'current_price': info.get('currentPrice', 0),
                            'sector': info.get('sector', 'Unknown'),
                        }
                    }
                
            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def run_enhanced_screener(self, symbols: List[str], min_score: float = 5.0) -> pd.DataFrame:
        """Run screener with async loading"""
        
        if st.session_state.use_async_loading and len(symbols) > 5:
            try:
                return self.async_loader.fetch_multiple_stocks_threaded(symbols)
            except Exception as e:
                st.error(f"Async screener failed: {e}")
                return pd.DataFrame()
        else:
            # Synchronous fallback
            return self._run_screener_sync(symbols, min_score)
    
    def _run_screener_sync(self, symbols: List[str], min_score: float) -> pd.DataFrame:
        """Fallback synchronous screener"""
        # Implementation would be similar to existing screener
        # but with progress tracking
        return pd.DataFrame()
    
    def invalidate_portfolio_caches(self):
        """Clear portfolio-related caches"""
        if st.session_state.cache_enabled:
            # Clear Streamlit caches
            st.cache_data.clear()
            
            # Clear advanced caches
            if self.cache:
                self.cache.invalidate_cache('portfolio')
                self.cache.invalidate_cache('current_price')
    
    def render_enhanced_portfolio_tab(self):
        """Render enhanced portfolio tab with new features"""
        
        st.header("📊 Enhanced Portfolio Analysis")
        
        # Portfolio management section
        with st.expander("🛠️ Portfolio Management", expanded=False):
            self._render_portfolio_management()
        
        # Get current portfolio
        portfolio_symbols = self.get_enhanced_portfolio()
        
        if not portfolio_symbols:
            st.info("📝 Add stocks to your portfolio to see analysis")
            return
        
        # Portfolio analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Analyze Portfolio"):
                results = self.analyze_portfolio_async(portfolio_symbols)
                st.session_state.portfolio_analysis_results = results
        
        with col2:
            use_async = st.checkbox(
                "⚡ Use Async Loading", 
                value=st.session_state.use_async_loading,
                help="Faster analysis for large portfolios"
            )
            st.session_state.use_async_loading = use_async
        
        with col3:
            if st.button("🔮 What-If Analysis"):
                st.session_state.what_if_active = True
        
        # Display portfolio analysis results
        if hasattr(st.session_state, 'portfolio_analysis_results'):
            self._display_portfolio_results(st.session_state.portfolio_analysis_results)
        
        # What-If Analysis
        if st.session_state.what_if_active:
            st.markdown("---")
            current_portfolio = self._get_current_portfolio_dict()
            self.what_if_interface.render_what_if_interface(current_portfolio)
    
    def _render_portfolio_management(self):
        """Render portfolio management interface"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Add Stock**")
            new_symbol = st.text_input("Symbol", key="add_symbol").upper()
            shares = st.number_input("Shares (optional)", min_value=0.0, key="add_shares")
            cost = st.number_input("Cost per Share (optional)", min_value=0.0, key="add_cost")
            
            if st.button("Add to Portfolio", key="add_button"):
                if new_symbol:
                    if self.add_to_enhanced_portfolio(new_symbol, shares, cost):
                        st.success(f"✅ Added {new_symbol} to portfolio")
                        st.rerun()
        
        with col2:
            st.markdown("**Remove Stock**")
            portfolio_symbols = self.get_enhanced_portfolio()
            
            if portfolio_symbols:
                remove_symbol = st.selectbox("Select stock to remove", portfolio_symbols)
                
                if st.button("Remove from Portfolio", key="remove_button"):
                    if self.remove_from_enhanced_portfolio(remove_symbol):
                        st.success(f"✅ Removed {remove_symbol} from portfolio")
                        st.rerun()
        
        # Portfolio summary
        portfolio_holdings = self.portfolio_db.get_portfolio()
        if portfolio_holdings:
            st.markdown("**Current Portfolio**")
            df = pd.DataFrame(portfolio_holdings)
            st.dataframe(df[['symbol', 'shares', 'average_cost', 'sector']], 
                        hide_index=True, use_container_width=True)
    
    def _display_portfolio_results(self, results: Dict[str, Any]):
        """Display portfolio analysis results"""
        
        if not results:
            st.info("No analysis results available")
            return
        
        # Summary metrics
        successful_analyses = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_analyses:
            total_score = sum(
                r['analysis']['total_score'] 
                for r in successful_analyses.values() 
                if 'analysis' in r
            )
            avg_score = total_score / len(successful_analyses)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stocks Analyzed", len(successful_analyses))
            
            with col2:
                st.metric("Average Score", f"{avg_score:.1f}/10")
            
            with col3:
                fetch_times = [r.get('fetch_time', 0) for r in results.values()]
                avg_fetch_time = sum(fetch_times) / len(fetch_times) if fetch_times else 0
                st.metric("Avg Fetch Time", f"{avg_fetch_time:.1f}s")
            
            # Detailed results table
            portfolio_data = []
            for symbol, result in successful_analyses.items():
                if 'analysis' in result:
                    analysis = result['analysis']
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Score': analysis['total_score'],
                        'Price': analysis['current_price'],
                        'Sector': analysis['sector']
                    })
            
            if portfolio_data:
                df = pd.DataFrame(portfolio_data)
                df = df.sort_values('Score', ascending=False)
                st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Show errors if any
        errors = {k: v for k, v in results.items() if not v.get('success', False)}
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)} stocks)"):
                for symbol, error_result in errors.items():
                    st.error(f"{symbol}: {error_result.get('error', 'Unknown error')}")
    
    def _get_current_portfolio_dict(self) -> Dict[str, Any]:
        """Get current portfolio as dictionary for what-if analysis"""
        holdings = self.portfolio_db.get_portfolio()
        
        portfolio_dict = {}
        for holding in holdings:
            portfolio_dict[holding['symbol']] = {
                'shares': holding['shares'],
                'average_cost': holding['average_cost'],
                'sector': holding['sector'],
                'score': 0  # Will be calculated in what-if analysis
            }
        
        return portfolio_dict
    
    def render_system_status(self):
        """Render system status and health dashboard"""
        
        st.subheader("🔧 System Status & Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Database Status**")
            db_stats = self.portfolio_db.get_database_stats()
            
            if db_stats:
                st.metric("Portfolio Holdings", db_stats.get('portfolio_holdings', 0))
                st.metric("Total Transactions", db_stats.get('total_transactions', 0))
                st.metric("Database Size", f"{db_stats.get('db_size_mb', 0):.2f} MB")
            
        with col2:
            st.markdown("**Cache Performance**")
            cache_health = self.cache.get_cache_health() if self.cache else {}
            
            st.metric("Cache Efficiency", cache_health.get('cache_efficiency', 'Unknown'))
            st.metric("Cache Size", f"{cache_health.get('total_cache_size_mb', 0):.2f} MB")
            
            if st.button("🗑️ Clear All Caches"):
                if self.cache:
                    self.cache.clear_all_caches()
                st.success("Caches cleared!")
                st.rerun()
        
        # Feature status
        st.markdown("**Enhanced Features Status**")
        
        feature_status = {
            "SQLite Portfolio": "✅ Active",
            "Advanced Caching": "✅ Active" if st.session_state.cache_enabled else "⏸️ Disabled",
            "Async Data Loading": "✅ Active" if st.session_state.use_async_loading else "⏸️ Disabled",
            "What-If Analysis": "✅ Ready"
        }
        
        cols = st.columns(len(feature_status))
        for i, (feature, status) in enumerate(feature_status.items()):
            with cols[i]:
                st.metric(feature, status)

# Global enhanced features manager
enhanced_features = EnhancedFeaturesManager()
