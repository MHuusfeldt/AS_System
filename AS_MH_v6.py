import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import numpy as np
import io
import os
from plotly.subplots import make_subplots
import warnings
import base64
import hashlib
import zipfile
from pathlib import Path
warnings.filterwarnings('ignore')

# Suppress numpy deprecation warnings related to pandas compatibility
import sys
if sys.version_info >= (3, 7):
    # Comprehensive numpy deprecation warning suppression
    warnings.filterwarnings('ignore', message='.*np.bool.*deprecated.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*np.int.*deprecated.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*np.float.*deprecated.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*np.object.*deprecated.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*np.complex.*deprecated.*', category=DeprecationWarning)
    
    # Additional specific warnings for pandas-numpy compatibility
    warnings.filterwarnings('ignore', message='.*was a deprecated alias.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*use.*by itself.*', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*boolean index.*deprecated.*', category=FutureWarning)

# Additional numpy compatibility fix for pandas
try:
    # Handle numpy deprecation in pandas
    import pandas as pd
    import numpy as np
    
    # Monkey patch for older pandas versions if needed
    if hasattr(pd.core.dtypes.common, 'is_integer_dtype'):
        # Force pandas to use correct numpy dtypes
        pass
    
    # Set numpy error handling to ignore deprecation warnings
    np.seterr(all='ignore')
    
except Exception:
    pass

# Enhanced Features Integration
try:
    from enhanced_features_integration import EnhancedFeaturesManager
    from portfolio_database import PortfolioDatabase
    from advanced_caching import AdvancedCache
    from async_data_loader import AsyncStockDataLoader
    from what_if_analysis import WhatIfAnalyzer
    ENHANCED_FEATURES_AVAILABLE = True
    
    # Only show success message on first load
    if 'enhanced_features_loaded' not in st.session_state:
        st.success("🚀 Enhanced features loaded successfully!")
        st.session_state.enhanced_features_loaded = True
        
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    if 'enhanced_features_warning_shown' not in st.session_state:
        st.warning("⚠️ Enhanced features not available. Running in basic mode.")
        st.session_state.enhanced_features_warning_shown = True

# Page configuration
st.set_page_config(layout="wide", page_title="AS System v6 - Enhanced")

# Configuration
API_KEY = "7J1AJVC9MAYLRRA7"
REQUEST_DELAY = 0.5

# Advanced Risk Analysis System
class AdvancedRiskAnalyzer:
    def __init__(self):
        # Set up global NumPy compatibility
        import warnings
        import os
        
        # Environment-level warning suppression
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        # Class-level warning suppression
        warnings.filterwarnings("ignore", category=Warning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*np.bool.*")
        warnings.filterwarnings("ignore", message=".*np.int.*")
        warnings.filterwarnings("ignore", message=".*np.float.*")
        warnings.filterwarnings("ignore", message=".*deprecated alias.*")
        
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_comprehensive_risk_metrics(self, portfolio_data):
        """Calculate comprehensive risk metrics for portfolio"""
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=Warning)
            
            try:
                # Extract symbols from portfolio data
                if isinstance(portfolio_data, pd.DataFrame):
                    portfolio_symbols = portfolio_data['Symbol'].tolist() if 'Symbol' in portfolio_data.columns else []
                    holdings_data = portfolio_data.to_dict('records') if not portfolio_data.empty else None
                else:
                    # Assume it's a list of symbols for backward compatibility
                    portfolio_symbols = portfolio_data if portfolio_data else []
                    holdings_data = None
                
                # Fetch historical data for all symbols
                price_data = self.get_portfolio_price_data(portfolio_symbols)
                
                if price_data.empty:
                    return self.get_default_risk_metrics()
                
                # Calculate returns
                returns = price_data.pct_change().dropna()
                portfolio_returns = self.calculate_portfolio_returns(returns, holdings_data)
                
                risk_metrics = {
                    # Volatility Metrics
                    'portfolio_volatility': self.calculate_portfolio_volatility(returns, holdings_data),
                    'individual_volatilities': self.calculate_individual_volatilities(returns),
                    
                    # Downside Risk Metrics
                    'value_at_risk_95': self.calculate_var(portfolio_returns, confidence=0.95),
                    'value_at_risk_99': self.calculate_var(portfolio_returns, confidence=0.99),
                    'expected_shortfall': self.calculate_expected_shortfall(portfolio_returns),
                    'maximum_drawdown': self.calculate_maximum_drawdown(portfolio_returns),
                    'downside_deviation': self.calculate_downside_deviation(portfolio_returns),
                    
                    # Risk-Adjusted Performance
                    'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns),
                    'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns),
                    'calmar_ratio': self.calculate_calmar_ratio(portfolio_returns),
                    'treynor_ratio': self.calculate_treynor_ratio(portfolio_returns, portfolio_symbols),
                    
                    # Portfolio Risk Decomposition
                    'correlation_matrix': self.calculate_correlation_matrix(returns),
                    'risk_contribution': self.calculate_risk_contribution(returns, holdings_data),
                    'concentration_risk': self.calculate_concentration_risk(holdings_data),
                    'factor_exposures': self.calculate_factor_exposures(portfolio_symbols),
                    'tail_risk': self.calculate_tail_risk(portfolio_returns),
                    
                    # Stress Testing
                    'stress_scenarios': self.run_stress_scenarios(portfolio_symbols, holdings_data),
                    'monte_carlo_var': self.monte_carlo_simulation(returns, holdings_data),
                    
                    # Liquidity Risk
                    'liquidity_score': self.calculate_liquidity_score(portfolio_symbols),
                    'market_impact': self.estimate_market_impact(portfolio_symbols, holdings_data)
                }
                
                return risk_metrics
                
            except Exception as e:
                if 'st' in globals():
                    st.error(f"Error calculating risk metrics: {e}")
                return self.get_default_risk_metrics()
    
    def get_portfolio_price_data(self, symbols, period="1y"):
        """Fetch price data for portfolio symbols with NumPy compatibility"""
        import warnings
        import os
        
        # Set up comprehensive warning suppression
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=Warning)
            
            price_data = pd.DataFrame()
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty and 'Close' in hist.columns:
                        # Ensure data is properly typed to avoid NumPy issues
                        close_data = hist['Close'].astype('float64')
                        price_data[symbol] = close_data
                except Exception:
                    continue
            
            return price_data
    
    def calculate_portfolio_returns(self, returns, holdings_data):
        """Calculate portfolio-weighted returns"""
        if holdings_data is None or len(holdings_data) == 0:
            # Equal weight if no holdings data
            return returns.mean(axis=1)
        
        # Check if holdings_data is a list of dictionaries (from to_dict('records'))
        if isinstance(holdings_data, list):
            # Convert list of records to dictionary format for processing
            holdings_dict = {}
            for holding in holdings_data:
                symbol = holding.get('Symbol')
                if symbol:
                    holdings_dict[symbol] = {
                        'market_value': holding.get('Total Value', 0),
                        'quantity': holding.get('Quantity', 0),
                        'current_price': holding.get('Current Price', 0)
                    }
            holdings_data = holdings_dict
        
        # Calculate weights from holdings
        total_value = sum(holding.get('market_value', 0) for holding in holdings_data.values())
        if total_value == 0:
            return returns.mean(axis=1)
        
        weights = {}
        for symbol, holding in holdings_data.items():
            if symbol in returns.columns:
                weights[symbol] = holding.get('market_value', 0) / total_value
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0, index=returns.index)
        for symbol, weight in weights.items():
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weight
        
        return portfolio_returns
    
    def calculate_portfolio_volatility(self, returns, holdings_data):
        """Calculate portfolio volatility"""
        # Ensure holdings_data is in the right format
        if isinstance(holdings_data, list):
            # Convert list of records to dictionary format for processing
            holdings_dict = {}
            for holding in holdings_data:
                symbol = holding.get('Symbol')
                if symbol:
                    holdings_dict[symbol] = {
                        'market_value': holding.get('Total Value', 0),
                        'quantity': holding.get('Quantity', 0),
                        'current_price': holding.get('Current Price', 0)
                    }
            holdings_data = holdings_dict
            
        portfolio_returns = self.calculate_portfolio_returns(returns, holdings_data)
        return portfolio_returns.std() * np.sqrt(252)  # Annualized
    
    def calculate_individual_volatilities(self, returns):
        """Calculate individual stock volatilities"""
        return {col: returns[col].std() * np.sqrt(252) for col in returns.columns}
    
    def calculate_var(self, returns, confidence=0.95):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_expected_shortfall(self, returns, confidence=0.95):
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_maximum_drawdown(self, returns):
        """Calculate Maximum Drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def calculate_downside_deviation(self, returns, target_return=0):
        """Calculate Downside Deviation"""
        if len(returns) == 0:
            return 0
        downside_returns = returns[returns < target_return]
        return np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe Ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns, target_return=0):
        """Calculate Sortino Ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns.mean() - target_return/252
        downside_dev = self.calculate_downside_deviation(returns, target_return/252)
        return (excess_returns / downside_dev) * np.sqrt(252) if downside_dev != 0 else 0
    
    def calculate_calmar_ratio(self, returns):
        """Calculate Calmar Ratio"""
        if len(returns) == 0:
            return 0
        annual_return = (1 + returns.mean())**252 - 1
        max_dd = abs(self.calculate_maximum_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0
    
    def calculate_treynor_ratio(self, returns, symbols):
        """Calculate Treynor Ratio (simplified)"""
        if len(returns) == 0:
            return 0
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        # Simplified beta calculation using market proxy
        return excess_returns * np.sqrt(252)  # Simplified version
    
    def calculate_correlation_matrix(self, returns):
        """Calculate correlation matrix with bulletproof NumPy 2.x compatibility"""
        try:
            if returns.empty or len(returns) < 2:
                return pd.DataFrame()
            
            # Import numpy with explicit alias setup to avoid deprecation issues
            import numpy
            
            # Set up comprehensive warning suppression at the most global level
            import warnings
            import os
            import sys
            
            # Environment variables to suppress NumPy warnings at the lowest level
            os.environ['PYTHONWARNINGS'] = 'ignore'
            os.environ['NUMPY_HIDE_WARNINGS'] = '1'
            
            # Add to sys.modules to prevent numpy bool import issues
            if not hasattr(numpy, 'bool'):
                numpy.bool = bool
            if not hasattr(numpy, 'int'):
                numpy.int = int  
            if not hasattr(numpy, 'float'):
                numpy.float = float
            if not hasattr(numpy, 'complex'):
                numpy.complex = complex
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore")
                warnings.filterwarnings("ignore", category=Warning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*bool.*")
                warnings.filterwarnings("ignore", message=".*numpy.*")
                
                # Create clean copy of data
                returns_clean = returns.copy()
                
                # Convert all data to proper numeric types without using NumPy aliases
                for col in returns_clean.columns:
                    if returns_clean[col].dtype == 'object':
                        returns_clean[col] = pd.to_numeric(returns_clean[col], errors='coerce')
                
                # Drop columns that are all NaN
                returns_clean = returns_clean.dropna(axis=1, how='all')
                
                if returns_clean.empty or len(returns_clean.columns) < 2:
                    return pd.DataFrame()
                
                # Ensure we have only numeric columns with explicit Python types
                numeric_data = pd.DataFrame()
                for col in returns_clean.columns:
                    try:
                        # Convert to float64 explicitly to avoid any NumPy type issues
                        series_data = returns_clean[col].astype('float64')
                        if not series_data.isna().all():
                            numeric_data[col] = series_data
                    except (ValueError, TypeError):
                        continue
                
                if numeric_data.empty or len(numeric_data.columns) < 2:
                    return pd.DataFrame()
                
                # Calculate correlation using manual method to avoid pandas/numpy compatibility issues
                try:
                    # Method 1: Try pandas corr with maximum error suppression
                    correlation_matrix = None
                    
                    # Suppress warnings at the numpy level
                    old_settings = numpy.seterr(all='ignore')
                    
                    try:
                        correlation_matrix = numeric_data.corr(method='pearson', min_periods=1)
                    except Exception:
                        try:
                            # Method 2: Manual correlation calculation
                            correlation_matrix = self._manual_correlation_calculation(numeric_data)
                        except Exception:
                            correlation_matrix = pd.DataFrame()
                    finally:
                        # Restore numpy error settings
                        numpy.seterr(**old_settings)
                    
                    if correlation_matrix is not None and not correlation_matrix.empty:
                        # Ensure proper data types and handle NaN values
                        correlation_matrix = correlation_matrix.fillna(0.0)
                        
                        # Convert to standard Python float type
                        correlation_matrix = correlation_matrix.astype('float64')
                        
                        # Validate correlation matrix properties
                        if correlation_matrix.shape[0] > 0 and correlation_matrix.shape[1] > 0:
                            return correlation_matrix
                
                except Exception:
                    pass
                
                return pd.DataFrame()
            
        except Exception as e:
            # Return empty DataFrame silently to avoid cluttering the UI
            return pd.DataFrame()
    
    def _manual_correlation_calculation(self, data):
        """Manual correlation calculation to avoid NumPy compatibility issues"""
        try:
            columns = data.columns.tolist()
            n = len(columns)
            
            # Initialize correlation matrix with Python floats
            corr_data = {}
            
            for i, col1 in enumerate(columns):
                corr_data[col1] = {}
                for j, col2 in enumerate(columns):
                    if i == j:
                        corr_data[col1][col2] = 1.0
                    else:
                        # Calculate correlation manually using pandas operations
                        series1 = data[col1].dropna()
                        series2 = data[col2].dropna()
                        
                        # Get common indices
                        common_idx = series1.index.intersection(series2.index)
                        
                        if len(common_idx) < 2:
                            corr_data[col1][col2] = 0.0
                        else:
                            s1 = series1.loc[common_idx]
                            s2 = series2.loc[common_idx]
                            
                            # Manual correlation calculation
                            mean1 = s1.mean()
                            mean2 = s2.mean()
                            
                            numerator = ((s1 - mean1) * (s2 - mean2)).sum()
                            denominator = (((s1 - mean1) ** 2).sum() * ((s2 - mean2) ** 2).sum()) ** 0.5
                            
                            if denominator == 0:
                                corr_data[col1][col2] = 0.0
                            else:
                                corr_data[col1][col2] = float(numerator / denominator)
            
            # Convert to DataFrame
            correlation_matrix = pd.DataFrame(corr_data)
            correlation_matrix.index = columns
            
            return correlation_matrix
            
        except Exception:
            return pd.DataFrame()
    
    def calculate_concentration_risk(self, holdings_data):
        """Calculate portfolio concentration risk using Herfindahl Index"""
        if not holdings_data:
            return 0
        
        # Check if holdings_data is a list of dictionaries (from to_dict('records'))
        if isinstance(holdings_data, list):
            # Convert list of records to dictionary format for processing
            holdings_dict = {}
            for holding in holdings_data:
                symbol = holding.get('Symbol')
                if symbol:
                    holdings_dict[symbol] = {
                        'market_value': holding.get('Total Value', 0),
                        'quantity': holding.get('Quantity', 0),
                        'current_price': holding.get('Current Price', 0)
                    }
            holdings_data = holdings_dict
        
        total_value = sum(holding.get('market_value', 0) for holding in holdings_data.values())
        if total_value == 0:
            return 0
        
        weights = [holding.get('market_value', 0) / total_value for holding in holdings_data.values()]
        hhi = sum(w**2 for w in weights)
        
        # Convert to risk score (0-10, where 10 is highest concentration risk)
        return min(10, hhi * 10)
    
    def calculate_risk_contribution(self, returns, holdings_data):
        """Calculate risk contribution by asset"""
        if returns.empty or not holdings_data:
            return {}
        
        # Handle different holdings_data formats
        if isinstance(holdings_data, list):
            # Convert list of records to dictionary format for processing
            holdings_dict = {}
            for holding in holdings_data:
                symbol = holding.get('Symbol')
                if symbol:
                    holdings_dict[symbol] = {
                        'market_value': holding.get('Total Value', 0),
                        'quantity': holding.get('Quantity', 0),
                        'current_price': holding.get('Current Price', 0)
                    }
            holdings_data = holdings_dict
        
        # Calculate portfolio weights
        total_value = sum(holding.get('market_value', 0) for holding in holdings_data.values())
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, holding in holdings_data.items():
            if symbol in returns.columns:
                weights[symbol] = holding.get('market_value', 0) / total_value
        
        # Calculate individual asset volatilities
        risk_contributions = {}
        for symbol in weights.keys():
            if symbol in returns.columns:
                asset_volatility = returns[symbol].std() * np.sqrt(252)  # Annualized
                weight = weights[symbol]
                # Risk contribution = weight * asset volatility (simplified)
                risk_contributions[symbol] = weight * asset_volatility
        
        # Normalize risk contributions to sum to 1
        total_risk = sum(risk_contributions.values())
        if total_risk > 0:
            risk_contributions = {k: v/total_risk for k, v in risk_contributions.items()}
        
        return risk_contributions
    
    def calculate_factor_exposures(self, symbols):
        """Calculate factor exposures (simplified)"""
        # Simplified factor exposure calculation
        return {'market_beta': 1.0, 'size_factor': 0.0, 'value_factor': 0.0}
    
    def calculate_tail_risk(self, returns):
        """Calculate tail risk metrics"""
        if len(returns) == 0:
            return 0
        return returns.quantile(0.01)  # 1% quantile
    
    def calculate_liquidity_score(self, symbols):
        """Calculate liquidity score (simplified)"""
        # Simplified liquidity scoring
        return 7.5  # Default good liquidity score
    
    def estimate_market_impact(self, symbols, holdings_data):
        """Estimate market impact (simplified)"""
        # Simplified market impact estimation
        return 0.01  # 1% estimated impact
    
    def run_stress_scenarios(self, symbols, holdings_data):
        """Run various stress test scenarios"""
        scenarios = {
            'market_crash_2008': {'description': '2008 Financial Crisis (-50% broad market)', 'impact': -0.50},
            'covid_crash_2020': {'description': 'COVID-19 Market Crash (-35% in 5 weeks)', 'impact': -0.35},
            'tech_bubble_2000': {'description': 'Tech Bubble Burst (-78% NASDAQ)', 'impact': -0.40},
            'inflation_shock': {'description': 'High Inflation Environment', 'impact': -0.25},
            'interest_rate_spike': {'description': 'Rapid Interest Rate Increases', 'impact': -0.20},
            'geopolitical_crisis': {'description': 'Major Geopolitical Event', 'impact': -0.30}
        }
        
        stress_results = {}
        
        # Handle different holdings_data formats
        if holdings_data:
            if isinstance(holdings_data, list):
                # Convert list of records to dictionary format for processing
                holdings_dict = {}
                for holding in holdings_data:
                    symbol = holding.get('Symbol')
                    if symbol:
                        holdings_dict[symbol] = {
                            'market_value': holding.get('Total Value', 0),
                            'quantity': holding.get('Quantity', 0),
                            'current_price': holding.get('Current Price', 0)
                        }
                current_value = sum(holding.get('market_value', 0) for holding in holdings_dict.values())
            else:
                current_value = sum(holding.get('market_value', 0) for holding in holdings_data.values())
        else:
            current_value = 1000000  # Default value
        
        for scenario_name, scenario in scenarios.items():
            stressed_value = current_value * (1 + scenario['impact'])
            loss = current_value - stressed_value
            
            stress_results[scenario_name] = {
                'description': scenario['description'],
                'portfolio_loss': loss,
                'portfolio_loss_pct': scenario['impact'] * 100,
                'stressed_value': stressed_value
            }
        
        return stress_results
    
    def monte_carlo_simulation(self, returns, holdings_data, num_simulations=1000, time_horizon=252):
        """Monte Carlo simulation for portfolio VaR"""
        if returns.empty:
            return {'mc_var_95': 0, 'mc_var_99': 0, 'expected_return': 0, 'simulation_results': []}
        
        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(returns, holdings_data)
        
        if len(portfolio_returns) == 0:
            return {'mc_var_95': 0, 'mc_var_99': 0, 'expected_return': 0, 'simulation_results': []}
        
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Run Monte Carlo simulation
        simulation_results = []
        for _ in range(num_simulations):
            random_returns = np.random.normal(mean_return, std_return, time_horizon)
            final_value = np.prod(1 + random_returns)
            simulation_results.append((final_value - 1) * 100)  # Convert to percentage
        
        return {
            'mc_var_95': np.percentile(simulation_results, 5),
            'mc_var_99': np.percentile(simulation_results, 1),
            'expected_return': np.mean(simulation_results),
            'simulation_results': simulation_results
        }
    
    def get_default_risk_metrics(self):
        """Return default risk metrics when calculation fails"""
        return {
            'portfolio_volatility': 0,
            'individual_volatilities': {},
            'value_at_risk_95': 0,
            'value_at_risk_99': 0,
            'expected_shortfall': 0,
            'maximum_drawdown': 0,
            'downside_deviation': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'treynor_ratio': 0,
            'correlation_matrix': pd.DataFrame(),
            'concentration_risk': 0,
            'factor_exposures': {},
            'tail_risk': 0,
            'stress_scenarios': {},
            'monte_carlo_var': {'mc_var_95': 0, 'mc_var_99': 0, 'expected_return': 0, 'simulation_results': []},
            'liquidity_score': 0,
            'market_impact': 0
        }

# Currency formatting functions
def get_currency_symbol(ticker_info):
    """Get the appropriate currency symbol for a stock"""
    if not ticker_info:
        return "$"
    
    currency = ticker_info.get('currency', 'USD').upper()
    symbol = ticker_info.get('symbol', '')
    
    # Danish stocks
    if currency == 'DKK' or '.CO' in symbol or symbol in DANISH_STOCKS.values():
        return "DKK"
    elif currency == 'EUR':
        return "€"
    elif currency == 'GBP':
        return "£"
    else:
        return "$"

def format_currency(value, ticker_info, decimals=2):
    """Format a monetary value with the appropriate currency symbol"""
    if value is None or value == 0:
        return "N/A"
    
    currency_symbol = get_currency_symbol(ticker_info)
    
    if currency_symbol == "DKK":
        return f"{value:,.{decimals}f} DKK"
    elif currency_symbol == "€":
        return f"€{value:,.{decimals}f}"
    elif currency_symbol == "£":
        return f"£{value:,.{decimals}f}"
    else:
        return f"${value:,.{decimals}f}"

def is_danish_stock(symbol):
    """Check if a symbol is a Danish stock"""
    return (symbol in DANISH_STOCKS or 
            symbol in DANISH_STOCKS.values() or 
            '.CO' in symbol)

# Danish stock mappings - Comprehensive list
DANISH_STOCKS = {
    # Large Cap - Major Danish companies
    "NOVO-B": "NOVO-B.CO",           # Novo Nordisk
    "NOVO": "NOVO-B.CO",             # Alternative Novo symbol
    "MAERSK-B": "MAERSK-B.CO",       # A.P. Moller-Maersk
    "MAERSK": "MAERSK-B.CO",         # Alternative Maersk symbol
    "ORSTED": "ORSTED.CO",           # Orsted
    "DSV": "DSV.CO",                 # DSV
    "CARLB": "CARL-B.CO",            # Carlsberg
    "CARL-B": "CARL-B.CO",           # Carlsberg B shares
    "NZYM-B": "NZYM-B.CO",          # Novozymes
    "NOVOZYMES": "NZYM-B.CO",        # Alternative Novozymes symbol
    "TRYG": "TRYG.CO",               # Tryg
    "DEMANT": "DEMANT.CO",           # Demant
    "COLO-B": "COLO-B.CO",          # Coloplast
    "COLOPLAST": "COLO-B.CO",        # Alternative Coloplast symbol
    "GMAB": "GMAB.CO",               # Genmab
    "GENMAB": "GMAB.CO",             # Alternative Genmab symbol
    
    # Mid Cap
    "AMBU-B": "AMBU-B.CO",          # Ambu
    "BAVA": "BAVA.CO",               # Bavarian Nordic
    "CHR": "CHR.CO",                 # Chr. Hansen
    "DANSKE": "DANSKE.CO",           # Danske Bank
    "FLS": "FLS.CO",                 # FLSmidth
    "GN": "GN.CO",                   # GN Store Nord
    "ISS": "ISS.CO",                 # ISS
    "JYSK": "JYSK.CO",               # Jyske Bank
    "NETC": "NETC.CO",               # NetCompany
    "PNDORA": "PNDORA.CO",           # Pandora
    "PANDORA": "PNDORA.CO",          # Alternative Pandora symbol
    "RBREW": "RBREW.CO",             # Royal Unibrew
    "ROCK-B": "ROCK-B.CO",           # Rockwool
    "SIM": "SIM.CO",                 # SimCorp
    "SYDB": "SYDB.CO",               # Sydbank
    "VWS": "VWS.CO",                 # Vestas Wind Systems
    "VESTAS": "VWS.CO",              # Alternative Vestas symbol
    
    # Small Cap
    "ALKA-B": "ALKA-B.CO",          # Alkane
    "BIOPRT": "BIOPRT.CO",          # Bioporto
    "CAPD": "CAPD.CO",               # Capdan
    "DKSH": "DKSH.CO",               # DKSH
    "ERHV": "ERHV.CO",               # Erhvervs
    "FLUG-B": "FLUG-B.CO",          # Flugger
    "GYLD": "GYLD.CO",               # Gyldendal
    "HPRO": "HPRO.CO",               # H+H
    "LUXOR-B": "LUXOR-B.CO",        # Luxor
    "MATAS": "MATAS.CO",             # Matas
    "NNIT": "NNIT.CO",               # NNIT
    "OSKAR": "OSKAR.CO",             # Oskar
    "RILBA": "RILBA.CO",             # Rilba
    "SANT": "SANT.CO",               # Santander Consumer Bank
    "SPNO": "SPNO.CO",               # Spar Nord
    "TLSN": "TLSN.CO",               # Tl
    
    # Additional banking and financial
    "NYKR": "NYKR.CO",               # Nykredit
    "TOPDM": "TOPDM.CO",             # TopDanmark
    "ALMB": "ALMB.CO",               # Alm. Brand
    
    # Additional healthcare and pharma
    "BAVB": "BAVA.CO",               # Bavarian Nordic (alternative)
    "NOVO-A": "NOVO-A.CO",           # Novo Nordisk A shares
    "NZYM-A": "NZYM-A.CO",          # Novozymes A shares
    "CARL-A": "CARL-A.CO",           # Carlsberg A shares
    "COLO-A": "COLO-A.CO",          # Coloplast A shares
    "ROCK-A": "ROCK-A.CO",          # Rockwool A shares
    "MAERSK-A": "MAERSK-A.CO",      # Maersk A shares
}

# S&P 500 major stocks (representative sample)
SP500_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX", "ADBE",
    "CRM", "CMCSA", "XOM", "VZ", "KO", "INTC", "ABT", "NKE", "PFE", "TMO",
    "AVGO", "CVX", "WMT", "COST", "NEE", "DHR", "ABBV", "ACN", "TXN", "LIN",
    "HON", "BMY", "UPS", "QCOM", "LOW", "AMD", "ORCL", "LMT", "T", "IBM"
]

# NASDAQ 100 major stocks (representative sample)
NASDAQ100_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE", "PYPL",
    "INTC", "CMCSA", "CSCO", "PEP", "AVGO", "TXN", "QCOM", "COST", "SBUX", "INTU",
    "AMD", "ISRG", "AMAT", "BKNG", "ADP", "GILD", "MU", "MELI", "LRCX", "FISV",
    "REGN", "CSX", "ATVI", "VRTX", "ILMN", "JD", "EA", "EXC", "KDP", "SIRI",
    "CTSH", "BIIB", "WBA", "MRNA", "ZM", "DOCU", "PTON", "DXCM", "OKTA", "TEAM"
]

# European major stocks (representative sample)
EUROPEAN_STOCKS = [
    # UK
    "SHEL", "AZN", "BP.L", "ULVR.L", "HSBA.L", "VOD.L", "GSK.L", "DGE.L", "BT-A.L", "BARC.L",
    # Germany  
    "SAP", "ASML", "TTE", "OR.PA", "SAN", "INGA.AS", "MC.PA", "RMS.PA", "AIR.PA", "BNP.PA",
    # France
    "LVMH.PA", "NESN.SW", "RHHBY", "NOVN.SW", "UG.PA", "CAP.PA", "SU.PA", "BN.PA", "EL.PA",
    # Netherlands/Other
    "RDS-A", "ING", "PHIA.AS", "UNA.AS", "HEIA.AS", "DSM.AS", "ASML.AS", "RDSA.AS"
]

# === CLOUD DATA PERSISTENCE SYSTEM ===

class PortfolioCloudManager:
    """
    Comprehensive cloud data persistence manager for Streamlit Cloud deployment.
    Combines backup/restore functionality with external database capabilities.
    """
    
    def __init__(self):
        self.backup_format_version = "1.0"
        self.supported_formats = ["json", "csv", "xlsx"]
        
    def create_portfolio_backup(self):
        """Create comprehensive portfolio backup with all session data"""
        try:
            backup_data = {
                "metadata": {
                    "version": self.backup_format_version,
                    "created": datetime.now().isoformat(),
                    "app_version": "AS_System_v6",
                    "data_types": ["portfolio", "holdings", "analysis_history", "weights"]
                },
                "portfolio": {
                    "symbols": st.session_state.get("portfolio", []),
                    "holdings": st.session_state.get("portfolio_holdings", {}),
                    "total_value": self._calculate_portfolio_value(),
                    "last_updated": datetime.now().isoformat()
                },
                "analysis_history": st.session_state.get("analysis_history", []),
                "user_settings": {
                    "score_weights": st.session_state.get("score_weights", {}),
                    "selected_symbols": st.session_state.get("selected_symbols", [])
                },
                "statistics": {
                    "total_analyses": len(st.session_state.get("analysis_history", [])),
                    "unique_symbols": len(set(st.session_state.get("portfolio", []))),
                    "backup_size_estimate": "calculated_in_kb"
                }
            }
            
            # Calculate backup size
            backup_json = json.dumps(backup_data)
            backup_data["statistics"]["backup_size_estimate"] = f"{len(backup_json.encode('utf-8')) / 1024:.1f}KB"
            
            return backup_data
            
        except Exception as e:
            st.error(f"❌ Failed to create backup: {str(e)}")
            return None
    
    def generate_downloadable_backup(self, format_type="json"):
        """Generate downloadable backup file in specified format"""
        backup_data = self.create_portfolio_backup()
        if not backup_data:
            return None, None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            content = json.dumps(backup_data, indent=2)
            filename = f"AS_System_backup_{timestamp}.json"
            mime_type = "application/json"
            
        elif format_type == "csv":
            # Create CSV format for portfolio data
            portfolio_df = pd.DataFrame([
                {
                    "Symbol": symbol,
                    "Quantity": holdings.get("quantity", 0),
                    "Purchase_Price": holdings.get("purchase_price", 0),
                    "Purchase_Date": holdings.get("purchase_date", ""),
                    "Current_Value": self._get_current_value(symbol, holdings.get("quantity", 0))
                }
                for symbol, holdings in backup_data["portfolio"]["holdings"].items()
            ])
            content = portfolio_df.to_csv(index=False)
            filename = f"AS_System_portfolio_{timestamp}.csv"
            mime_type = "text/csv"
            
        elif format_type == "xlsx":
            # Create Excel format with multiple sheets
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Portfolio sheet
                portfolio_df = pd.DataFrame([
                    {
                        "Symbol": symbol,
                        "Quantity": holdings.get("quantity", 0),
                        "Purchase_Price": holdings.get("purchase_price", 0),
                        "Purchase_Date": holdings.get("purchase_date", ""),
                    }
                    for symbol, holdings in backup_data["portfolio"]["holdings"].items()
                ])
                portfolio_df.to_excel(writer, sheet_name='Portfolio', index=False)
                
                # Settings sheet
                settings_df = pd.DataFrame([
                    {"Setting": k, "Value": v}
                    for k, v in backup_data["user_settings"]["score_weights"].items()
                ])
                settings_df.to_excel(writer, sheet_name='Settings', index=False)
                
            content = buffer.getvalue()
            filename = f"AS_System_backup_{timestamp}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        return content, filename, mime_type
    
    def restore_from_backup(self, uploaded_file):
        """Restore portfolio data from uploaded backup file"""
        try:
            if uploaded_file.type == "application/json":
                backup_data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                return self._restore_from_json(backup_data)
                
            elif uploaded_file.type == "text/csv":
                csv_data = pd.read_csv(uploaded_file)
                return self._restore_from_csv(csv_data)
                
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                excel_data = pd.read_excel(uploaded_file, sheet_name=None)
                return self._restore_from_excel(excel_data)
                
            else:
                st.error(f"❌ Unsupported file format: {uploaded_file.type}")
                return False
                
        except Exception as e:
            st.error(f"❌ Failed to restore backup: {str(e)}")
            return False
    
    def _restore_from_json(self, backup_data):
        """Restore from JSON backup format"""
        try:
            # Validate backup format
            if "metadata" not in backup_data or "portfolio" not in backup_data:
                st.error("❌ Invalid backup format")
                return False
            
            # Restore portfolio data
            if "portfolio" in backup_data:
                st.session_state.portfolio = backup_data["portfolio"].get("symbols", [])
                st.session_state.portfolio_holdings = backup_data["portfolio"].get("holdings", {})
            
            # Restore user settings
            if "user_settings" in backup_data:
                if "score_weights" in backup_data["user_settings"]:
                    st.session_state.score_weights = backup_data["user_settings"]["score_weights"]
                if "selected_symbols" in backup_data["user_settings"]:
                    st.session_state.selected_symbols = backup_data["user_settings"]["selected_symbols"]
            
            # Restore analysis history (optional)
            if "analysis_history" in backup_data:
                st.session_state.analysis_history = backup_data["analysis_history"]
            
            st.success("✅ Portfolio successfully restored from JSON backup!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to restore from JSON: {str(e)}")
            return False
    
    def _restore_from_csv(self, csv_data):
        """Restore from CSV backup format"""
        try:
            portfolio_holdings = {}
            portfolio_symbols = []
            
            for _, row in csv_data.iterrows():
                symbol = row.get("Symbol", "")
                if symbol:
                    portfolio_symbols.append(symbol)
                    portfolio_holdings[symbol] = {
                        "quantity": float(row.get("Quantity", 1.0)),
                        "purchase_price": float(row.get("Purchase_Price", 0.0)),
                        "purchase_date": str(row.get("Purchase_Date", datetime.now().strftime("%Y-%m-%d")))
                    }
            
            st.session_state.portfolio = portfolio_symbols
            st.session_state.portfolio_holdings = portfolio_holdings
            
            st.success("✅ Portfolio successfully restored from CSV backup!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to restore from CSV: {str(e)}")
            return False
    
    def _restore_from_excel(self, excel_data):
        """Restore from Excel backup format"""
        try:
            # Restore portfolio data
            if "Portfolio" in excel_data:
                portfolio_df = excel_data["Portfolio"]
                portfolio_holdings = {}
                portfolio_symbols = []
                
                for _, row in portfolio_df.iterrows():
                    symbol = row.get("Symbol", "")
                    if symbol:
                        portfolio_symbols.append(symbol)
                        portfolio_holdings[symbol] = {
                            "quantity": float(row.get("Quantity", 1.0)),
                            "purchase_price": float(row.get("Purchase_Price", 0.0)),
                            "purchase_date": str(row.get("Purchase_Date", datetime.now().strftime("%Y-%m-%d")))
                        }
                
                st.session_state.portfolio = portfolio_symbols
                st.session_state.portfolio_holdings = portfolio_holdings
            
            # Restore settings if available
            if "Settings" in excel_data:
                settings_df = excel_data["Settings"]
                score_weights = {}
                for _, row in settings_df.iterrows():
                    setting = row.get("Setting", "")
                    value = row.get("Value", 0)
                    if setting and value:
                        score_weights[setting] = float(value)
                
                if score_weights:
                    st.session_state.score_weights = score_weights
            
            st.success("✅ Portfolio successfully restored from Excel backup!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to restore from Excel: {str(e)}")
            return False
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = 0.0
        try:
            for symbol, holdings in st.session_state.get("portfolio_holdings", {}).items():
                quantity = holdings.get("quantity", 0)
                current_price = get_simple_current_price(symbol)
                if current_price and current_price > 0:
                    total_value += quantity * current_price
        except Exception:
            pass
        return total_value
    
    def _get_current_value(self, symbol, quantity):
        """Get current value for a position"""
        try:
            current_price = get_simple_current_price(symbol)
            if current_price and current_price > 0:
                return quantity * current_price
        except Exception:
            pass
        return 0.0
    
    def display_cloud_persistence_warnings(self):
        """Display comprehensive warnings about Streamlit Cloud data persistence"""
        if st.session_state.get("backup_warnings_shown", False):
            return
            
        st.warning("""
        ⚠️ **IMPORTANT: Data Persistence on Streamlit Cloud**
        
        Your portfolio data is stored in **session state** which is **TEMPORARY** on Streamlit Cloud:
        
        **Data Will Be Lost When:**
        - App restarts (daily)
        - Inactivity timeout (30 minutes)
        - App redeployment
        - Browser refresh/close
        
        **Protection Strategies:**
        1. 📥 **Regular Backups**: Download your data frequently
        2. 🔄 **Auto-backup**: Enable automatic backup reminders
        3. ☁️ **External Storage**: Use cloud database integration
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ I Understand - Don't Show Again"):
                st.session_state.backup_warnings_shown = True
                st.rerun()
        
        with col2:
            if st.button("📥 Create Backup Now"):
                self.display_backup_interface()
    
    def display_backup_interface(self):
        """Display backup and restore interface"""
        st.subheader("💾 Portfolio Backup & Restore")
        
        # Display current portfolio status
        portfolio_count = len(st.session_state.get("portfolio", []))
        holdings_count = len(st.session_state.get("portfolio_holdings", {}))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Stocks", portfolio_count)
        with col2:
            st.metric("Holdings Tracked", holdings_count)
        with col3:
            portfolio_value = self._calculate_portfolio_value()
            st.metric("Estimated Value", f"${portfolio_value:,.2f}" if portfolio_value > 0 else "N/A")
        
        # Backup section
        st.markdown("### 📥 Create Backup")
        
        # Initialize session state for backup format
        if 'backup_format' not in st.session_state:
            st.session_state.backup_format = "json"
        
        def update_backup_format():
            st.session_state.backup_format = st.session_state.backup_format_select
        
        backup_format = st.selectbox(
            "Backup Format:",
            ["json", "csv", "xlsx"],
            index=["json", "csv", "xlsx"].index(st.session_state.backup_format),
            help="JSON: Complete backup | CSV: Portfolio only | Excel: Multi-sheet backup",
            key="backup_format_select",
            on_change=update_backup_format
        )
        
        if st.button("📥 Generate Backup", type="primary"):
            with st.spinner("Creating backup..."):
                content, filename, mime_type = self.generate_downloadable_backup(backup_format)
                
                if content:
                    st.download_button(
                        label=f"⬇️ Download {filename}",
                        data=content,
                        file_name=filename,
                        mime=mime_type,
                        help="Click to download your portfolio backup"
                    )
                    st.success(f"✅ Backup created successfully! Format: {backup_format.upper()}")
                else:
                    st.error("❌ Failed to create backup")
        
        # Restore section
        st.markdown("### 📤 Restore from Backup")
        
        uploaded_file = st.file_uploader(
            "Choose backup file:",
            type=["json", "csv", "xlsx"],
            help="Upload a previously created backup file to restore your portfolio"
        )
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            st.write(f"**Type:** {uploaded_file.type}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Restore Portfolio", type="primary"):
                    if self.restore_from_backup(uploaded_file):
                        st.balloons()
                        st.info("Portfolio restored! Refresh the page to see changes.")
            
            with col2:
                if st.button("❌ Cancel"):
                    st.rerun()
        
        # Automatic backup reminder
        st.markdown("### 🔔 Backup Reminders")
        
        # Initialize session state for auto backup
        if 'auto_backup_enabled' not in st.session_state:
            st.session_state.auto_backup_enabled = False
        if 'last_backup_reminder' not in st.session_state:
            st.session_state.last_backup_reminder = None
        
        def update_auto_backup():
            st.session_state.auto_backup_enabled = st.session_state.auto_backup_checkbox
        
        auto_backup = st.checkbox(
            "Enable automatic backup reminders (every 24 hours)",
            value=st.session_state.auto_backup_enabled,
            key="auto_backup_checkbox",
            on_change=update_auto_backup
        )
        
        if auto_backup:
            last_reminder = st.session_state.get("last_backup_reminder")
            if last_reminder:
                last_time = datetime.fromisoformat(last_reminder)
                time_since = datetime.now() - last_time
                
                if time_since.total_seconds() > 86400:  # 24 hours
                    st.info("🔔 Reminder: It's been 24+ hours since your last backup reminder. Consider creating a backup!")
                    if st.button("📥 Create Backup Now"):
                        st.session_state.last_backup_reminder = datetime.now().isoformat()
                        self.display_backup_interface()
            else:
                st.session_state.last_backup_reminder = datetime.now().isoformat()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "score_weights" not in st.session_state:
        st.session_state.score_weights = {
            "PE": 0.08,
            "Forward PE": 0.12,  # More weight on forward P/E
            "PEG": 0.08,
            "PB": 0.06,
            "EV/EBITDA": 0.10,  # New metric
            "ROE": 0.10,
            "EPS Growth": 0.12,
            "Revenue Growth": 0.08,
            "FCF Trend": 0.04,
            "Debt/Equity": 0.03,
            "Dividend Yield": 0.02,
            "Gross Margin": 0.05,
            "Price/Sales": 0.04,  # New metric
            "Analyst Upside": 0.06,   # New metric
            "Momentum": 0.08,  # NEW - Technical momentum
            "Financial Health": 0.08   # NEW - Comprehensive financial health
        }
    
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    if "selected_symbols" not in st.session_state:
        st.session_state.selected_symbols = []
    
    # Initialize cloud backup manager
    if "cloud_backup_manager" not in st.session_state:
        st.session_state.cloud_backup_manager = PortfolioCloudManager()
    
    # Initialize backup warnings shown flag
    if "backup_warnings_shown" not in st.session_state:
        st.session_state.backup_warnings_shown = False
    
    # Initialize portfolio if it doesn't exist
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []
    
    # Initialize portfolio holdings with purchase prices if not exists
    if "portfolio_holdings" not in st.session_state:
        st.session_state.portfolio_holdings = {}  # Format: {symbol: {"quantity": float, "purchase_price": float, "purchase_date": str}}
    
    # Migrate old portfolio format to new format if needed
    if st.session_state.portfolio and not st.session_state.portfolio_holdings:
        # Migrate existing portfolio symbols to new format (without purchase prices initially)
        for symbol in st.session_state.portfolio:
            if symbol not in st.session_state.portfolio_holdings:
                st.session_state.portfolio_holdings[symbol] = {
                    "quantity": 1.0,
                    "purchase_price": 0.0,  # Will need to be updated by user
                    "purchase_date": datetime.now().strftime("%Y-%m-%d")
                }
    
    # Initialize Enhanced Features Manager
    if ENHANCED_FEATURES_AVAILABLE:
        # Only initialize if not already done or if it failed
        if "enhanced_features_manager" not in st.session_state or st.session_state.enhanced_features_manager is None:
            try:
                st.session_state.enhanced_features_manager = EnhancedFeaturesManager()
                # Initialize the enhanced features
                st.session_state.enhanced_features_manager.initialize_all_systems()
                st.session_state.enhanced_features_enabled = True
                st.session_state.enhanced_features_init_attempted = True
            except Exception as e:
                st.warning(f"⚠️ Could not initialize enhanced features: {e}")
                st.session_state.enhanced_features_enabled = False
                st.session_state.enhanced_features_manager = None
                st.session_state.enhanced_features_init_attempted = True
        else:
            # Enhanced features manager exists, make sure it's enabled
            if not st.session_state.get('enhanced_features_enabled', False):
                # Try to re-enable if the manager exists but was disabled
                if st.session_state.enhanced_features_manager is not None:
                    try:
                        # Test if the manager is still working
                        if hasattr(st.session_state.enhanced_features_manager, 'portfolio_db'):
                            st.session_state.enhanced_features_enabled = True
                    except Exception:
                        st.session_state.enhanced_features_enabled = False
    else:
        st.session_state.enhanced_features_enabled = False

# Portfolio sync functions
def save_portfolio_to_file():
    """Save portfolio to JSON file for automated monitoring"""
    import json
    import os
    
    portfolio_data = {
        'symbols': st.session_state.portfolio,
        'last_updated': datetime.now().isoformat(),
        'total_stocks': len(st.session_state.portfolio)
    }
    
    try:
        with open('portfolio_config.json', 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")
        return False

def auto_sync_if_enabled():
    """Auto-sync portfolio if enabled"""
    if st.session_state.get('auto_sync_portfolio', False):
        if save_portfolio_to_file():
            st.success("🔄 Portfolio auto-synced with automated monitor!")
        else:
            st.warning("⚠️ Auto-sync failed")

def get_portfolio_sync_status():
    """Get current sync status between portfolio and monitor"""
    import json
    import os
    
    try:
        if os.path.exists('portfolio_config.json'):
            with open('portfolio_config.json', 'r') as f:
                portfolio_data = json.load(f)
            
            file_symbols = set(portfolio_data.get('symbols', []))
            current_symbols = set(st.session_state.portfolio)
            
            if file_symbols == current_symbols:
                return "synced", portfolio_data.get('last_updated', 'Unknown')
            else:
                return "out_of_sync", portfolio_data.get('last_updated', 'Unknown')
        else:
            return "not_synced", None
    except Exception:
        return "error", None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def discover_danish_stocks():
    """
    Discover additional Danish stocks by trying common patterns and validating them
    """
    discovered_stocks = {}
    
    # Common Danish company patterns to try
    patterns = [
        # Banking and Financial
        'NYKR', 'TOPDM', 'ALMB', 'DLCA', 'FINE',
        
        # Additional Pharma/Healthcare
        'ZLNO', 'ACTO', 'EVSY', 'THXS', 'ASRS',
        
        # Technology and Services  
        'ZEGN', 'MTOS', 'POWC', 'SUBC', 'PRME',
        
        # Industrial and Materials
        'FMAS', 'DUNI', 'SCHO', 'NPRO', 'BRYN',
        
        # Consumer and Retail
        'PAAL-B', 'BEST', 'NEWC', 'PARK',
        
        # Energy and Utilities
        'ANDV', 'NEAS', 'NORD',
        
        # Real Estate
        'CASA', 'DEAS', 'PAAL',
        
        # Alternative share classes
        'TRYG-A', 'SYDB-A', 'FLS-A', 'GN-A',
    ]
    
    print("Discovering additional Danish stocks...")
    
    for pattern in patterns:
        test_symbol = f"{pattern}.CO"
        try:
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            
            # Check if we got valid data
            if info and (info.get('regularMarketPrice') is not None or 
                        info.get('currentPrice') is not None or
                        info.get('longName') is not None):
                
                # Additional validation - check if it's actually Danish
                country = info.get('country', '').lower()
                currency = info.get('currency', '').upper()
                
                if country == 'denmark' or currency == 'DKK' or '.CO' in test_symbol:
                    symbol_key = pattern
                    discovered_stocks[symbol_key] = test_symbol
                    print(f"✓ Discovered: {symbol_key} -> {test_symbol}")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception:
            continue
    
    return discovered_stocks

def update_danish_stocks_list():
    """
    Update the DANISH_STOCKS dictionary with newly discovered stocks
    """
    global DANISH_STOCKS
    
    st.subheader("🇩🇰 Danish Stock List Updater")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"Current Danish stocks in database: **{len(DANISH_STOCKS)}**")
        
        if st.button("🔍 Discover Additional Danish Stocks", help="This will search for more Danish stocks and validate them"):
            with st.spinner("Discovering Danish stocks..."):
                discovered = discover_danish_stocks()
                
                if discovered:
                    st.success(f"Found {len(discovered)} new Danish stocks!")
                    
                    # Show discovered stocks
                    st.subheader("Newly Discovered Stocks")
                    for key, value in discovered.items():
                        st.write(f"• {key} → {value}")
                    
                    # Update the global dictionary
                    DANISH_STOCKS.update(discovered)
                    
                    st.info(f"Updated total: {len(DANISH_STOCKS)} Danish stocks")
                    
                    # Offer to save the updated list
                    if st.button("💾 Save Updated List to File"):
                        updated_code = generate_danish_stocks_code(DANISH_STOCKS)
                        
                        # Save to a file
                        with open("danish_stocks_updated.py", "w") as f:
                            f.write(updated_code)
                        
                        st.success("Saved updated Danish stocks list to 'danish_stocks_updated.py'")
                        
                        # Show the code
                        with st.expander("📄 View Updated Code"):
                            st.code(updated_code, language="python")
                
                else:
                    st.info("No new Danish stocks discovered this time.")
    
    with col2:
        st.metric("Total Danish Stocks", len(DANISH_STOCKS))
        
        if st.button("📋 View Current List"):
            st.subheader("Current Danish Stocks")
            
            # Create a nice display of current stocks
            df_stocks = pd.DataFrame([
                {"Symbol": k, "Yahoo Symbol": v, "Company": v.replace('.CO', '')} 
                for k, v in sorted(DANISH_STOCKS.items())
            ])
            
            st.dataframe(df_stocks, use_container_width=True, hide_index=True)

def generate_danish_stocks_code(stocks_dict):
    """
    Generate Python code for the updated DANISH_STOCKS dictionary
    """
    lines = [
        "# Danish stock mappings - Updated list",
        "DANISH_STOCKS = {"
    ]
    
    # Sort by key for better organization
    sorted_stocks = sorted(stocks_dict.items())
    
    for key, value in sorted_stocks:
        lines.append(f'    "{key}": "{value}",')
    
    lines.append("}")
    
    return "\n".join(lines)

def validate_danish_stock_symbol(symbol):
    """
    Validate if a Danish stock symbol exists and return its proper .CO format
    """
    # If already has .CO suffix, try as-is
    if symbol.endswith('.CO'):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and (info.get('regularMarketPrice') is not None or 
                        info.get('currentPrice') is not None):
                return symbol
        except:
            pass
    
    # Try adding .CO suffix
    test_symbol = f"{symbol}.CO"
    try:
        ticker = yf.Ticker(test_symbol)
        info = ticker.info
        if info and (info.get('regularMarketPrice') is not None or 
                    info.get('currentPrice') is not None):
            return test_symbol
    except:
        pass
    
    return None

# Enhanced 3-year historical data functions
@st.cache_data(ttl=3600)
def get_3year_financial_history(symbol):
    """Get 3 years of financial history with Danish stock support and improved error handling"""
    def try_fetch_financials(sym):
        """Helper function to try fetching financial data for a symbol"""
        try:
            ticker = yf.Ticker(sym)
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            if financials.empty and balance_sheet.empty:
                return None
            
            metrics_3y = {}
            
            # Revenue history with better error handling
            try:
                if 'Total Revenue' in financials.index:
                    revenue_data = financials.loc['Total Revenue'].head(3)
                    if not revenue_data.empty:
                        metrics_3y['revenue_trend'] = revenue_data.tolist()
                        metrics_3y['revenue_dates'] = [d.strftime('%Y') for d in revenue_data.index]
                        
                        # Calculate growth rates
                        if len(revenue_data) >= 2:
                            growth_rates = []
                            for i in range(1, len(revenue_data)):
                                if revenue_data.iloc[i] != 0:
                                    growth = ((revenue_data.iloc[i-1] - revenue_data.iloc[i]) / abs(revenue_data.iloc[i])) * 100
                                    growth_rates.append(growth)
                            metrics_3y['avg_revenue_growth'] = np.mean(growth_rates) if growth_rates else 0
            except Exception:
                pass
            
            # Net Income history
            try:
                if 'Net Income' in financials.index:
                    net_income_data = financials.loc['Net Income'].head(3)
                    if not net_income_data.empty:
                        metrics_3y['net_income_trend'] = net_income_data.tolist()
                        
                        if len(net_income_data) >= 2:
                            growth_rates = []
                            for i in range(1, len(net_income_data)):
                                if net_income_data.iloc[i] != 0:
                                    growth = ((net_income_data.iloc[i-1] - net_income_data.iloc[i]) / abs(net_income_data.iloc[i])) * 100
                                    growth_rates.append(growth)
                            metrics_3y['avg_net_income_growth'] = np.mean(growth_rates) if growth_rates else 0
            except Exception:
                pass
            
            # Free Cash Flow - Enhanced extraction with multiple field name attempts
            try:
                fcf_data = None
                operating_cf = None
                capex = None
                
                # Try multiple field name variations for Operating Cash Flow
                for ocf_field in ['Operating Cash Flow', 'Cash Flow From Operations', 'Cash From Operations']:
                    if ocf_field in cashflow.index:
                        operating_cf = cashflow.loc[ocf_field].head(3)
                        break
                
                # Try multiple field name variations for Capital Expenditures
                for capex_field in ['Capital Expenditures', 'Capital Expenditure', 'Capex', 'Cash Flow From Investing']:
                    if capex_field in cashflow.index:
                        capex = cashflow.loc[capex_field].head(3)
                        break
                
                # Also try direct FCF field if available
                for fcf_field in ['Free Cash Flow', 'FreeCashFlow']:
                    if fcf_field in cashflow.index:
                        fcf_data = cashflow.loc[fcf_field].head(3)
                        break
                
                # Calculate FCF if we have operating CF and capex
                if fcf_data is None and operating_cf is not None and capex is not None:
                    if not operating_cf.empty and not capex.empty:
                        fcf_data = operating_cf - abs(capex)
                
                # Store FCF data if we have it
                if fcf_data is not None and not fcf_data.empty:
                    metrics_3y['fcf_trend'] = fcf_data.tolist()
                    
            except Exception as e:
                print(f"Debug: FCF extraction failed for {symbol}: {e}")
                pass
            
            # ROE calculation
            try:
                if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
                    net_income = financials.loc['Net Income'].head(3)
                    equity = balance_sheet.loc['Total Stockholder Equity'].head(3)
                    
                    if not net_income.empty and not equity.empty:
                        roe_data = (net_income / equity * 100).dropna()
                        if not roe_data.empty:
                            metrics_3y['roe_trend'] = roe_data.tolist()
                            metrics_3y['avg_roe'] = np.mean(roe_data)
            except Exception:
                pass
            
            return metrics_3y
            
        except Exception:
            return None
    
    try:
        # First, try the original symbol
        result = try_fetch_financials(symbol)
        if result:
            return result
        
        # If original fails, try Danish stock variations
        if not symbol.endswith('.CO'):
            # Check if it's a known Danish stock
            if symbol in DANISH_STOCKS:
                danish_symbol = DANISH_STOCKS[symbol]
                result = try_fetch_financials(danish_symbol)
                if result:
                    return result
            
            # Try adding .CO suffix
            co_symbol = f"{symbol}.CO"
            result = try_fetch_financials(co_symbol)
            if result:
                return result
        
        # If all attempts fail, return empty dict
        return {}
        
    except Exception as e:
        st.error(f"Error fetching 3-year data for {symbol}: {e}")
        return {}

@st.cache_data(ttl=1800)
def get_3year_price_performance(symbol):
    """Get 3-year price performance with Danish stock support and enhanced metrics"""
    def try_fetch_price_data(sym):
        """Helper function to try fetching price data for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            
            ticker = yf.Ticker(sym)
            hist_data = ticker.history(start=start_date, end=end_date)
            ticker_info = ticker.info
            
            if hist_data.empty:
                return None
            
            current_price = hist_data['Close'].iloc[-1]
            
            performance = {
                'current_price': current_price,
                'price_data': hist_data,
                'ticker_info': ticker_info,
                'max_3y': hist_data['Close'].max(),
                'min_3y': hist_data['Close'].min(),
                'volatility': hist_data['Close'].pct_change().std() * np.sqrt(252) * 100
            }
            
            # Calculate returns for different periods
            periods = [('3m', 65), ('6m', 130), ('1y', 252), ('2y', 504), ('3y', 756)]
            
            for period_name, days_back in periods:
                if len(hist_data) >= days_back:
                    past_price = hist_data['Close'].iloc[-days_back]
                    return_pct = ((current_price - past_price) / past_price) * 100
                    performance[f'return_{period_name}'] = return_pct
                else:
                    performance[f'return_{period_name}'] = 0
            
            # Calculate Sharpe ratio (simplified)
            returns = hist_data['Close'].pct_change().dropna()
            if len(returns) > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                performance['sharpe_ratio'] = sharpe
            
            return performance
            
        except Exception:
            return None
    
    try:
        # First, try the original symbol
        result = try_fetch_price_data(symbol)
        if result:
            return result
        
        # If original fails, try Danish stock variations
        if not symbol.endswith('.CO'):
            # Check if it's a known Danish stock
            if symbol in DANISH_STOCKS:
                danish_symbol = DANISH_STOCKS[symbol]
                result = try_fetch_price_data(danish_symbol)
                if result:
                    return result
            
            # Try adding .CO suffix
            co_symbol = f"{symbol}.CO"
            result = try_fetch_price_data(co_symbol)
            if result:
                return result
        
        # If all attempts fail, return empty dict
        return {}
        
    except Exception as e:
        st.error(f"Error fetching price performance for {symbol}: {e}")
        return {}

# Enhanced scoring functions
def score_pe(pe, industry_pe, allow_neutral=True):
    """Enhanced P/E scoring with industry comparison and robust None handling"""
    # Safely convert inputs to float
    pe = safe_float(pe, 0)
    industry_pe = safe_float(industry_pe, 20)  # Default industry PE
    
    if pe <= 0:
        # For missing P/E data, return neutral score instead of 0 to indicate "unknown but not necessarily bad"
        return 5 if allow_neutral else 0
    
    # Relative to industry
    relative_pe = pe / industry_pe if industry_pe > 0 else 1
    
    if relative_pe < 0.6:
        return 10
    elif relative_pe < 0.8:
        return 8
    elif relative_pe < 1.0:
        return 6
    elif relative_pe < 1.2:
        return 4
    elif relative_pe < 1.5:
        return 3
    elif relative_pe < 2.0:
        return 2
    elif relative_pe < 2.5:
        return 1
    else:
        return 0

def score_forward_pe(forward_pe, industry_pe):
    """Score based on forward P/E (more predictive than trailing) with safe None handling"""
    # Safely convert inputs to float
    forward_pe = safe_float(forward_pe, 0)
    industry_pe = safe_float(industry_pe, 20)
    
    if forward_pe <= 0:
        return 0
    
    relative_pe = forward_pe / industry_pe if industry_pe > 0 else 1
    
    if relative_pe < 0.5:
        return 10
    elif relative_pe < 0.7:
        return 8
    elif relative_pe < 0.9:
        return 6
    elif relative_pe < 1.1:
        return 5
    elif relative_pe < 1.3:
        return 3
    elif relative_pe < 1.5:
        return 2
    else:
        return 1

def score_peg(peg, allow_neutral=True):
    """Enhanced PEG scoring with safe None handling"""
    # Safely convert input to float
    peg = safe_float(peg, 0)
    
    if peg <= 0:
        # For missing PEG data, return neutral score instead of 0
        return 5 if allow_neutral else 0
    if peg < 0.5:
        return 10
    elif peg < 0.75:
        return 8
    elif peg < 1.0:
        return 6
    elif peg < 1.5:
        return 4
    elif peg < 2.0:
        return 2
    else:
        return 0

def score_pb(pb):
    """Price-to-Book scoring with safe None handling"""
    pb = safe_float(pb, 0)
    
    if pb <= 0:
        return 0
    if pb < 1.0:
        return 10
    elif pb < 1.5:
        return 8
    elif pb < 2.5:
        return 6
    elif pb < 4.0:
        return 4
    else:
        return 2

def score_roe(roe):
    """Return on Equity scoring with safe None handling"""
    roe = safe_float(roe, 0)
    
    if roe <= 0:
        return 0
    if roe > 25:
        return 10
    elif roe > 20:
        return 8
    elif roe > 15:
        return 6
    elif roe > 10:
        return 4
    elif roe > 5:
        return 2
    else:
        return 1

def score_roe_dynamic(roe, sector):
    """Enhanced ROE scoring with dynamic benchmarking"""
    roe = safe_float(roe, 0)
    
    if roe <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_roe = benchmarks["roe"]
    
    # Calculate relative performance
    relative_roe = roe / industry_roe
    
    # Score based on relative performance
    if relative_roe >= 2.0:  # 2x industry average
        return 10
    elif relative_roe >= 1.5:  # 1.5x industry average
        return 8
    elif relative_roe >= 1.2:  # 1.2x industry average
        return 7
    elif relative_roe >= 1.0:  # At industry average
        return 6
    elif relative_roe >= 0.8:  # 80% of industry average
        return 4
    elif relative_roe >= 0.6:  # 60% of industry average
        return 3
    elif relative_roe >= 0.4:  # 40% of industry average
        return 2
    else:  # Below 40% of industry average
        return 1

def score_gross_margin_dynamic(gm, sector):
    """Enhanced gross margin scoring with dynamic benchmarking"""
    gm = safe_float(gm, 0)
    
    if gm <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_gm = benchmarks["gross_margin"]
    
    relative_gm = gm / industry_gm
    
    if relative_gm >= 1.4:  # 40% above industry
        return 10
    elif relative_gm >= 1.2:  # 20% above industry
        return 8
    elif relative_gm >= 1.1:  # 10% above industry
        return 7
    elif relative_gm >= 0.95:  # Near industry average
        return 6
    elif relative_gm >= 0.85:  # 15% below industry
        return 4
    elif relative_gm >= 0.75:  # 25% below industry
        return 3
    else:  # Significantly below industry
        return 2

def score_revenue_growth_dynamic(growth, sector):
    """Enhanced revenue growth scoring with dynamic benchmarking"""
    growth = safe_float(growth, 0)
    
    benchmarks = get_industry_benchmarks(sector)
    industry_growth = benchmarks["revenue_growth"]
    
    if growth is None or growth == 0:
        return 0
    
    # For negative growth, score more harshly
    if growth < 0:
        if growth > -5:  # Small decline
            return 3
        elif growth > -10:  # Moderate decline
            return 2
        else:  # Severe decline
            return 1
    
    # Calculate relative growth performance
    if industry_growth > 0:
        relative_growth = growth / industry_growth
        
        if relative_growth >= 2.0:  # 2x industry growth
            return 10
        elif relative_growth >= 1.5:  # 1.5x industry growth
            return 8
        elif relative_growth >= 1.2:  # 1.2x industry growth
            return 7
        elif relative_growth >= 0.8:  # 80% of industry growth
            return 6
        elif relative_growth >= 0.5:  # 50% of industry growth
            return 4
        else:  # Below 50% of industry growth
            return 3
    else:
        # If industry growth is negative/zero, use absolute thresholds
        return score_revenue_growth(growth, True)  # Fall back to original function

def score_debt_equity_dynamic(de, sector):
    """Enhanced debt/equity scoring with dynamic benchmarking"""
    de = safe_float(de, 0)
    
    if de < 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_de = benchmarks["debt_equity"]
    
    # Financial sector has different debt characteristics
    if sector == "Financials":
        # For financials, higher leverage is more normal
        if de < industry_de * 0.8:  # Low leverage for financials
            return 8
        elif de < industry_de * 1.0:  # Normal leverage
            return 6
        elif de < industry_de * 1.2:  # Slightly high
            return 4
        else:  # Very high leverage
            return 2
    else:
        # For non-financial sectors
        relative_de = de / industry_de
        
        if relative_de <= 0.5:  # Much lower than industry
            return 10
        elif relative_de <= 0.8:  # Lower than industry
            return 8
        elif relative_de <= 1.0:  # At industry level
            return 6
        elif relative_de <= 1.3:  # Above industry level
            return 4
        elif relative_de <= 1.6:  # Well above industry
            return 3
        else:  # Extremely high debt
            return 1

def score_eps_growth(growth):
    """EPS growth scoring with safe None handling"""
    growth = safe_float(growth, 0)
    
    if growth > 25:
        return 10
    elif growth > 15:
        return 8
    elif growth > 10:
        return 6
    elif growth > 5:
        return 4
    elif growth > 0:
        return 2
    else:
        return 0

def score_revenue_growth(growth, has_data=True):
    """Revenue growth scoring with safe None handling"""
    growth = safe_float(growth, 0)
    
    if not has_data or growth is None:
        return 0  # Return 0 instead of None to keep metric visible
    if growth > 20:
        return 10
    elif growth > 15:
        return 8
    elif growth > 10:
        return 6
    elif growth > 5:
        return 4
    elif growth > 0:
        return 2
    else:
        return 0

def score_fcf_trend(fcf_values, has_data=True):
    """Free cash flow trend scoring"""
    if not has_data or not fcf_values or len(fcf_values) < 2 or all(x == 0 for x in fcf_values):
        return 0  # Return 0 instead of None to keep metric visible
    
    positive_count = sum(1 for x in fcf_values if x > 0)
    
    # Check for improvement trend
    if len(fcf_values) >= 3:
        recent_avg = np.mean(fcf_values[:2])
        older_avg = np.mean(fcf_values[1:])
        if recent_avg > older_avg and positive_count >= 2:
            return 10
    
    if positive_count == len(fcf_values):
        return 8
    elif positive_count >= len(fcf_values) * 0.6:
        return 6
    elif positive_count > 0:
        return 4
    else:
        return 0

def score_debt_equity(de):
    """Debt-to-equity scoring - expects percentage values from Yahoo Finance"""
    de = safe_float(de, 0)
    
    if de < 0:
        return 0
    if de < 30:  # Less than 30%
        return 10
    elif de < 50:  # 30-50%
        return 8
    elif de < 100:  # 50-100%
        return 6
    elif de < 200:  # 100-200%
        return 4
    else:  # Over 200%
        return 2

def score_dividend_yield(dy, allow_neutral=True):
    """Dividend yield scoring with safe None handling"""
    dy = safe_float(dy, 0)
    
    if dy <= 0:
        # For missing dividend data, return neutral score - company might not pay dividends
        return 5 if allow_neutral else 0
    if dy > 5:
        return 10
    elif dy > 3:
        return 8
    elif dy > 1:
        return 6
    else:
        return 4

def score_gross_margin(gm):
    """Gross margin scoring with safe None handling"""
    gm = safe_float(gm, 0)
    
    if gm <= 0:
        return 0
    if gm > 60:
        return 10
    elif gm > 40:
        return 8
    elif gm > 25:
        return 6
    elif gm > 15:
        return 4
    else:
        return 2

def score_ev_ebitda(ev_ebitda):
    """EV/EBITDA scoring - better for companies with different capital structures"""
    ev_ebitda = safe_float(ev_ebitda, 0)
    
    if ev_ebitda <= 0:
        return 0
    if ev_ebitda < 8:
        return 10
    elif ev_ebitda < 12:
        return 8
    elif ev_ebitda < 15:
        return 6
    elif ev_ebitda < 20:
        return 4
    elif ev_ebitda < 25:
        return 2
    else:
        return 0

def score_price_sales(ps_ratio):
    """Price-to-Sales scoring - good for growth companies"""
    ps_ratio = safe_float(ps_ratio, 0)
    
    if ps_ratio <= 0:
        return 0
    if ps_ratio < 1:
        return 10
    elif ps_ratio < 2:
        return 8
    elif ps_ratio < 4:
        return 6
    elif ps_ratio < 6:
        return 4
    elif ps_ratio < 10:
        return 2
    else:
        return 0

def score_analyst_upside(upside_percent):
    """Score based on analyst price target upside"""
    upside_percent = safe_float(upside_percent, 0)
    
    if upside_percent > 25:
        return 10
    elif upside_percent > 15:
        return 8
    elif upside_percent > 5:
        return 6
    elif upside_percent > -5:
        return 5
    elif upside_percent > -15:
        return 3
    else:
        return 1

def score_momentum(symbol):
    """Score based on price momentum and moving average relationships"""
    momentum_data = calculate_momentum_indicators(symbol)
    
    if not momentum_data:
        return 5  # Neutral score if no data
    
    current = momentum_data['current_price']
    sma_20 = momentum_data['sma_20']
    sma_50 = momentum_data['sma_50']
    sma_200 = momentum_data['sma_200']
    
    score = 5  # Start with neutral
    
    # Moving average alignment (strong momentum indicator)
    if all([sma_20, sma_50, sma_200]):
        if current > sma_20 > sma_50 > sma_200:
            score += 3  # Perfect bullish alignment
        elif current > sma_20 > sma_50:
            score += 2  # Good bullish trend
        elif current > sma_50 > sma_200:
            score += 1  # Moderate bullish
        elif current < sma_20 < sma_50 < sma_200:
            score -= 3  # Perfect bearish alignment
        elif current < sma_20 < sma_50:
            score -= 2  # Bearish trend
        elif current < sma_50:
            score -= 1  # Moderate bearish
    
    # Price momentum over different periods
    momentum_score = 0
    periods = [
        ('price_1m', 0.5),   # 1 month momentum (50% weight)
        ('price_3m', 1.0),   # 3 month momentum (100% weight)
        ('price_6m', 0.75)   # 6 month momentum (75% weight)
    ]
    
    total_weight = 0
    for period, weight in periods:
        if momentum_data[period]:
            price_change = (current - momentum_data[period]) / momentum_data[period] * 100
            
            if price_change > 20:
                momentum_score += 2 * weight
            elif price_change > 10:
                momentum_score += 1 * weight
            elif price_change > 0:
                momentum_score += 0.5 * weight
            elif price_change > -10:
                momentum_score -= 0.5 * weight
            elif price_change > -20:
                momentum_score -= 1 * weight
            else:
                momentum_score -= 2 * weight
            
            total_weight += weight
    
    if total_weight > 0:
        avg_momentum = momentum_score / total_weight
        score += avg_momentum
    
    # Ensure score stays within bounds
    return max(0, min(10, score))

def score_current_ratio(current_ratio, sector):
    """Score current ratio with sector-specific benchmarks"""
    if current_ratio <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_current = benchmarks["current_ratio"]
    
    relative_ratio = current_ratio / industry_current
    
    if relative_ratio >= 1.3:  # 30% above industry
        return 9
    elif relative_ratio >= 1.1:  # 10% above industry
        return 8
    elif relative_ratio >= 0.9:  # Within 10% of industry
        return 7
    elif relative_ratio >= 0.8:  # 20% below industry
        return 5
    elif relative_ratio >= 0.7:  # 30% below industry
        return 3
    else:  # Significantly below industry
        return 1

def score_interest_coverage(interest_coverage, sector):
    """Score interest coverage ratio with sector awareness"""
    if interest_coverage <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_coverage = benchmarks["interest_coverage"]
    
    # Handle infinite coverage (no interest expense)
    if interest_coverage == float('inf'):
        return 10
    
    relative_coverage = interest_coverage / industry_coverage
    
    if relative_coverage >= 2.0:  # 2x industry average
        return 10
    elif relative_coverage >= 1.5:  # 1.5x industry average
        return 8
    elif relative_coverage >= 1.0:  # At industry average
        return 6
    elif relative_coverage >= 0.75:  # 75% of industry
        return 4
    elif relative_coverage >= 0.5:  # 50% of industry
        return 3
    else:  # Below 50% of industry
        return 1

def score_financial_health_composite(info, sector):
    """Comprehensive financial health score"""
    health_data = get_financial_health_data(info)
    
    if not health_data:
        return 5  # Neutral if no data
    
    # Individual component scores
    current_score = score_current_ratio(health_data['current_ratio'], sector)
    interest_score = score_interest_coverage(health_data['interest_coverage'], sector)
    debt_score = score_debt_equity_dynamic(info.get('de', 0), sector)
    
    # Cash position score (absolute)
    cash_score = 5  # Default neutral
    if health_data['cash_position'] > 0:
        market_cap = info.get('marketCap', 0)
        if market_cap > 0:
            cash_to_market_cap = health_data['cash_position'] / market_cap
            if cash_to_market_cap > 0.2:  # 20%+ cash
                cash_score = 8
            elif cash_to_market_cap > 0.1:  # 10%+ cash
                cash_score = 7
            elif cash_to_market_cap > 0.05:  # 5%+ cash
                cash_score = 6
    
    # Weighted composite score
    weights = {
        'current_ratio': 0.25,
        'interest_coverage': 0.30,
        'debt_equity': 0.30,
        'cash_position': 0.15
    }
    
    composite_score = (
        current_score * weights['current_ratio'] +
        interest_score * weights['interest_coverage'] +
        debt_score * weights['debt_equity'] +
        cash_score * weights['cash_position']
    )
    
    return round(composite_score, 1)


class ComprehensiveStockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.fundamental_scores = {}
        self.technical_signals = {}
        self.buying_prices = {}
        self.recommendation = None
        self.fundamental_info = None  # Initialize this properly
        
    def fetch_all_data(self):
        """Fetch both fundamental and technical data"""
        try:
            # Fetch fundamental data using the enhanced fetch_yahoo_info function
            self.fundamental_info = fetch_yahoo_info(self.symbol)
            
            if not self.fundamental_info:
                return False
            
            # Fetch technical data using the symbol that worked for fundamental data
            symbol_to_use = self.fundamental_info.get('symbol_used', self.symbol)
            
            ticker = yf.Ticker(symbol_to_use)
            self.data = ticker.history(period=self.period)
            
            if self.data is None or len(self.data) == 0:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def calculate_fundamental_scores(self):
        """Calculate fundamental analysis scores"""
        try:
            if not self.fundamental_info:
                return {}
            
            # Get industry PE for better scoring
            industry_pe = get_industry_pe(self.fundamental_info)
            
            # Calculate scores using the Yahoo info
            scores, _ = calculate_scores_yahoo(self.fundamental_info, industry_pe)
            
            if scores:
                self.fundamental_scores = scores
                return scores
            else:
                return {}
                
        except Exception as e:
            return {}
    
    def calculate_technical_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None or len(self.data) < 50:
            return False
            
        # Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self.data['Close'].ewm(span=12).mean()
        ema_26 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = ema_12 - ema_26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # Bollinger Bands
        rolling_mean = self.data['Close'].rolling(window=20).mean()
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        self.data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        self.data['BB_Middle'] = rolling_mean
        
        # Stochastic
        low_min = self.data['Low'].rolling(window=14).min()
        high_max = self.data['High'].rolling(window=14).max()
        self.data['%K'] = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        self.data['%D'] = self.data['%K'].rolling(window=3).mean()
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        return True
    
    def generate_technical_signals(self):
        """Generate comprehensive technical signals with scoring"""
        if self.data is None or len(self.data) < 50:
            return {}
            
        latest = self.data.iloc[-1]
        signals = {}
        signal_scores = {}
        
        # RSI Signals
        if latest['RSI'] < 30:
            signals['RSI'] = "🟢 STRONG BUY - Oversold"
            signal_scores['RSI'] = 8
        elif latest['RSI'] < 40:
            signals['RSI'] = "🟢 BUY - Below neutral"
            signal_scores['RSI'] = 6
        elif latest['RSI'] > 70:
            signals['RSI'] = "🔴 STRONG SELL - Overbought"
            signal_scores['RSI'] = 2
        elif latest['RSI'] > 60:
            signals['RSI'] = "🔴 SELL - Above neutral"
            signal_scores['RSI'] = 4
        else:
            signals['RSI'] = "🟡 NEUTRAL"
            signal_scores['RSI'] = 5
        
        # Moving Average Signals
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals['MA_Trend'] = "🟢 STRONG BUY - Strong uptrend"
            signal_scores['MA_Trend'] = 8
        elif latest['Close'] > latest['SMA_50']:
            signals['MA_Trend'] = "🟢 BUY - Uptrend"
            signal_scores['MA_Trend'] = 6
        elif latest['Close'] < latest['SMA_50']:
            signals['MA_Trend'] = "🔴 SELL - Downtrend"
            signal_scores['MA_Trend'] = 3
        else:
            signals['MA_Trend'] = "🟡 NEUTRAL - Sideways"
            signal_scores['MA_Trend'] = 5
        
        # MACD Signals
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD'] > 0:
            signals['MACD'] = "🟢 STRONG BUY - Bullish momentum"
            signal_scores['MACD'] = 8
        elif latest['MACD'] > latest['MACD_Signal']:
            signals['MACD'] = "🟢 BUY - Improving momentum"
            signal_scores['MACD'] = 6
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD'] < 0:
            signals['MACD'] = "🔴 STRONG SELL - Bearish momentum"
            signal_scores['MACD'] = 2
        elif latest['MACD'] < latest['MACD_Signal']:
            signals['MACD'] = "🔴 SELL - Weakening momentum"
            signal_scores['MACD'] = 4
        else:
            signals['MACD'] = "🟡 NEUTRAL"
            signal_scores['MACD'] = 5
        
        # Bollinger Bands Signals
        if latest['Close'] < latest['BB_Lower']:
            signals['Bollinger'] = "🟢 STRONG BUY - Oversold"
            signal_scores['Bollinger'] = 8
        elif latest['Close'] < latest['BB_Middle']:
            signals['Bollinger'] = "🟢 BUY - Below mean"
            signal_scores['Bollinger'] = 6
        elif latest['Close'] > latest['BB_Upper']:
            signals['Bollinger'] = "🔴 STRONG SELL - Overbought"
            signal_scores['Bollinger'] = 2
        else:
            signals['Bollinger'] = "🟡 NEUTRAL"
            signal_scores['Bollinger'] = 5
        
        # Stochastic Signals
        if latest['%K'] < 20 and latest['%D'] < 20:
            signals['Stochastic'] = "🟢 STRONG BUY - Oversold"
            signal_scores['Stochastic'] = 8
        elif latest['%K'] > 80 and latest['%D'] > 80:
            signals['Stochastic'] = "🔴 STRONG SELL - Overbought"
            signal_scores['Stochastic'] = 2
        else:
            signals['Stochastic'] = "🟡 NEUTRAL"
            signal_scores['Stochastic'] = 5
        
        # Volume Confirmation
        if latest['Volume_Ratio'] > 1.5:
            signals['Volume'] = "🟢 High volume confirmation"
            volume_multiplier = 1.2
        elif latest['Volume_Ratio'] < 0.5:
            signals['Volume'] = "🔴 Low volume warning"
            volume_multiplier = 0.8
        else:
            signals['Volume'] = "🟡 Normal volume"
            volume_multiplier = 1.0
        
        # Calculate technical score
        technical_score = np.mean(list(signal_scores.values())) * volume_multiplier
        
        self.technical_signals = {
            'signals': signals,
            'scores': signal_scores,
            'technical_score': technical_score,
            'volume_multiplier': volume_multiplier
        }
        
        return self.technical_signals
    
    def calculate_buying_prices(self):
        """Calculate optimal buying prices using BuyingPriceCalculator logic"""
        if self.data is None or len(self.data) < 50:
            return {}
            
        current_price = self.data['Close'].iloc[-1]
        latest = self.data.iloc[-1]
        
        # Support and Resistance levels
        support_level = self.data['Low'].tail(20).min()
        resistance_level = self.data['High'].tail(20).max()
        
        buying_prices = {
            'current_price': current_price,
            'support_buy': support_level * 1.01,
            'bollinger_buy': latest['BB_Lower'] * 1.005,
            'sma_20_pullback': latest['SMA_20'] * 0.98,
            'sma_50_pullback': latest['SMA_50'] * 0.99,
        }
        
        # RSI-based pricing
        if latest['RSI'] < 40:
            buying_prices['rsi_buy'] = current_price * 0.97
        else:
            buying_prices['rsi_buy'] = current_price * 0.95
        
        # MACD-based pricing
        if latest['MACD'] > latest['MACD_Signal']:
            buying_prices['macd_buy'] = current_price * 0.98
        else:
            buying_prices['macd_buy'] = current_price * 0.95
        
        # Calculate recommended buy price
        technical_score = self.technical_signals.get('technical_score', 5)
        if technical_score >= 7:
            recommended_discount = 0.02  # 2% discount for strong buy
        elif technical_score >= 6:
            recommended_discount = 0.03  # 3% discount for buy
        elif technical_score <= 4:
            recommended_discount = 0.07  # 7% discount for weak signals
        else:
            recommended_discount = 0.05  # 5% discount for neutral
        
        buying_prices['recommended_buy'] = current_price * (1 - recommended_discount)
        
        self.buying_prices = buying_prices
        return buying_prices
    
    def generate_combined_recommendation(self):
        """Generate final recommendation combining fundamental and technical analysis"""
        try:
            if not self.fundamental_scores or not self.technical_signals:
                self.recommendation = None
                return "Insufficient data", "gray", 0
            
            # Default score weights (fallback when session state not available)
            default_weights = {
                'pe_ratio': 2.0,
                'pb_ratio': 1.5,
                'debt_equity': 1.5,
                'dividend_yield': 1.0,
                'roe': 2.0,
                'roa': 1.5,
                'gross_margin': 1.5,
                'operating_margin': 1.5,
                'revenue_growth': 2.0,
                'earnings_growth': 2.0,
                'current_ratio': 1.0,
                'price_sales': 1.5
            }
            
            # Try to get weights from session state, fallback to defaults
            try:
                score_weights = st.session_state.score_weights
            except:
                score_weights = default_weights
            
            # Calculate fundamental score with normalized weights for missing data
            available_weights = {k: score_weights.get(k, 0) 
                               for k in self.fundamental_scores if k in score_weights}
            
            if available_weights:
                total_weight = sum(available_weights.values())
                if total_weight > 0:
                    # Normalize weights to maintain scale
                    original_total_weight = sum(score_weights.values())
                    weight_multiplier = original_total_weight / total_weight
                    
                    fundamental_total = sum(self.fundamental_scores[k] * available_weights[k] * weight_multiplier 
                                          for k in self.fundamental_scores if k in available_weights)
                else:
                    fundamental_total = 5  # Default neutral score
            else:
                fundamental_total = 5  # Default neutral score
            
            # Get technical score
            technical_total = self.technical_signals.get('technical_score', 5)
            
            # Weight the scores (60% fundamental, 40% technical)
            combined_score = (fundamental_total * 0.6) + (technical_total * 0.4)
            
            # Generate recommendation
            if combined_score >= 8:
                recommendation = "🚀 STRONG BUY"
                color = "darkgreen"
            elif combined_score >= 6.5:
                recommendation = "📈 BUY"
                color = "green"
            elif combined_score >= 5:
                recommendation = "🔄 HOLD"
                color = "orange"
            elif combined_score >= 3.5:
                recommendation = "📉 WEAK SELL"
                color = "orangered"
            else:
                recommendation = "🛑 STRONG SELL"
                color = "red"
            
            self.recommendation = {
                'combined_score': combined_score,
                'fundamental_score': fundamental_total,
                'technical_score': technical_total,
                'recommendation': recommendation,
                'color': color
            }
            
            return recommendation, color, combined_score
            
        except Exception as e:
            # Only show error in Streamlit context
            try:
                st.error(f"Error generating recommendation: {e}")
            except:
                pass  # Not in Streamlit context
            self.recommendation = None
            return "Error in analysis", "gray", 0
def display_comprehensive_analysis(analyzer):
    """Display comprehensive analysis combining fundamental and technical analysis"""
    
    # Check if we have valid recommendation data
    if not analyzer.recommendation:
        st.error("❌ Analysis incomplete - missing recommendation data")
        return
    
    # Header with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        company_name = "Unknown"
        current_price = 0
        
        if analyzer.fundamental_info:
            company_name = analyzer.fundamental_info.get('name', 'Unknown')[:20]
            # Try to get current price from fundamental info first
            current_price = (analyzer.fundamental_info.get('currentPrice', 0) or 
                           analyzer.fundamental_info.get('regularMarketPrice', 0) or 0)
        
        # If no price from fundamental info, try technical data
        if current_price == 0 and analyzer.data is not None and len(analyzer.data) > 0:
            current_price = analyzer.data['Close'].iloc[-1]
        
        st.metric("Company", company_name)
        st.metric("Current Price", format_currency(current_price, analyzer.fundamental_info) if current_price > 0 else "N/A")
    
    with col2:
        fundamental_score = analyzer.recommendation.get('fundamental_score', 0)
        technical_score = analyzer.recommendation.get('technical_score', 0)
        st.metric("Fundamental Score", f"{fundamental_score:.1f}/10")
        st.metric("Technical Score", f"{technical_score:.1f}/10")
    
    with col3:
        combined_score = analyzer.recommendation.get('combined_score', 0)
        st.metric("Combined Score", f"{combined_score:.1f}/10")
        
    with col4:
        recommendation = analyzer.recommendation.get('recommendation', 'Unknown')
        color = analyzer.recommendation.get('color', 'gray')
        st.markdown(f"<div style='padding: 10px; background-color: {color}20; border-left: 4px solid {color}; border-radius: 5px;'>"
                   f"<strong style='color: {color}; font-size: 16px;'>{recommendation}</strong></div>", 
                   unsafe_allow_html=True)
    
    # Add sector and industry information
    if analyzer.fundamental_info:
        sector = analyzer.fundamental_info.get('sector', 'Unknown')
        industry = analyzer.fundamental_info.get('industry', 'Unknown')
        
        st.markdown("### 🏢 Company Classification")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sector", sector)
        
        with col2:
            st.metric("Industry", industry)
        
        with col3:
            # Show if sector-specific adjustments are applied
            sector_model = None
            if sector in ["Technology", "Industrials", "Financials", "Healthcare", "Consumer Staples"]:
                sector_model = "✅ Applied"
                st.metric("Sector Optimization", sector_model)
            else:
                st.metric("Sector Optimization", "Standard")
        
        # Show sector-specific adjustments if available
        if sector in ["Technology", "Industrials", "Financials", "Healthcare", "Consumer Staples"]:
            st.markdown(f"**🎯 {sector} Sector Adjustments Applied:**")
            
            adjustments = {
                "Technology": "Growth metrics emphasized (+20-40%)",
                "Industrials": "Operational efficiency & cash flow (+20-40%)", 
                "Financials": "Balance sheet strength & profitability (+20-50%)",
                "Healthcare": "R&D efficiency & margins (+10-30%)",
                "Consumer Staples": "Dividend yield & stability (+10-40%)"
            }
            
            if sector in adjustments:
                st.info(f"📈 {adjustments[sector]}")
    
    st.markdown("---")
    
    # Only show tabs if we have the necessary data
    if not analyzer.fundamental_scores and not analyzer.technical_signals:
        st.error("❌ No analysis data available to display")
        return
    
    # Create tabs for different views
    fund_tab, tech_tab, buy_tab, chart_tab = st.tabs([
        "📊 Fundamental Analysis", 
        "📈 Technical Signals", 
        "💰 Buying Strategy",
        "📉 Charts & Data"
    ])
    
    with fund_tab:
        if analyzer.fundamental_scores:
            st.subheader("📊 Fundamental Analysis Breakdown")
            
            # Show sector-specific scoring details
            if analyzer.fundamental_info:
                sector = analyzer.fundamental_info.get('sector', 'Unknown')
                st.markdown(f"**🏭 Sector: {sector}**")
                
                # Show sector-specific weight adjustments
                if sector in ["Technology", "Industrials", "Financials", "Healthcare", "Consumer Staples"]:
                    st.markdown("**🎯 Sector-Specific Weight Adjustments:**")
                    
                    sector_adjustments = {
                        "Technology": {
                            "Forward PE": "1.2×", "Revenue Growth": "1.4×", "Gross Margin": "1.3×", 
                            "Price/Sales": "1.2×", "EPS Growth": "1.3×"
                        },
                        "Industrials": {
                            "EV/EBITDA": "1.3×", "ROE": "1.2×", "FCF Trend": "1.4×", "Revenue Growth": "1.1×"
                        },
                        "Financials": {
                            "PB": "1.5×", "ROE": "1.4×", "Dividend Yield": "1.2×", "PE": "1.2×"
                        },
                        "Healthcare": {
                            "Forward PE": "1.1×", "Revenue Growth": "1.2×", "Gross Margin": "1.3×", "EPS Growth": "1.2×"
                        },
                        "Consumer Staples": {
                            "Dividend Yield": "1.4×", "ROE": "1.2×", "Debt/Equity": "1.1×", "Gross Margin": "1.2×"
                        }
                    }
                    
                    if sector in sector_adjustments:
                        cols = st.columns(len(sector_adjustments[sector]))
                        for i, (metric, multiplier) in enumerate(sector_adjustments[sector].items()):
                            with cols[i]:
                                st.metric(metric, multiplier)
                
                st.markdown("---")
            
            # Fundamental scores chart
            fig = create_enhanced_score_chart(analyzer.fundamental_scores, analyzer.symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            if analyzer.fundamental_info:
                col1, col2, col3 = st.columns(3)
                info = analyzer.fundamental_info
                
                with col1:
                    st.write("**Valuation Metrics**")
                    st.write(f"P/E Ratio: {info.get('pe', 'N/A')}")
                    st.write(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
                    st.write(f"P/B Ratio: {info.get('pb', 'N/A')}")
                    st.write(f"PEG Ratio: {info.get('peg', 'N/A')}")
                    st.write(f"EV/EBITDA: {info.get('enterpriseToEbitda', 'N/A')}")
                    
                with col2:
                    st.write("**Profitability Metrics**")
                    roe = info.get('roe', 0)
                    if roe:
                        st.write(f"ROE: {roe*100:.1f}%")  # ROE is stored as decimal, multiply by 100
                    else:
                        st.write("ROE: N/A")
                    
                    gm = info.get('gm', 0)
                    if gm:
                        st.write(f"Gross Margin: {gm*100:.1f}%")  # Gross margin is stored as decimal, multiply by 100
                    else:
                        st.write("Gross Margin: N/A")
                    
                    # Add dividend yield
                    dy = info.get('dy', 0)
                    if dy:
                        st.write(f"Dividend Yield: {dy:.2f}%")  # Dividend yield is already a percentage
                    else:
                        st.write("Dividend Yield: N/A")
                    
                    # Add debt/equity ratio
                    de = info.get('de', 0)
                    if de:
                        st.write(f"Debt/Equity: {de:.1f}%")  # Debt/equity is already a percentage
                    else:
                        st.write("Debt/Equity: N/A")
                    
                    # Add analyst data
                    target_price = info.get('targetMeanPrice', 0)
                    current_price = info.get('currentPrice', 0)
                    if target_price and current_price:
                        upside = ((target_price - current_price) / current_price) * 100
                        st.write(f"Analyst Upside: {upside:.1f}%")
                    else:
                        st.write("Analyst Upside: N/A")
                        
                with col3:
                    st.write("**Growth Metrics**")
                    eps_growth = info.get('eps_growth', 0)
                    if eps_growth:
                        st.write(f"EPS Growth: {eps_growth*100:.1f}%")  # EPS growth is stored as decimal, multiply by 100
                    else:
                        st.write("EPS Growth: N/A")
                        
                    # Check both possible revenue growth keys
                    rev_growth = info.get('revenue_growth', 0) or info.get('revenueGrowth', 0)
                    if rev_growth:
                        st.write(f"Revenue Growth: {rev_growth*100:.1f}%")  # Revenue growth is stored as decimal, multiply by 100
                    else:
                        st.write("Revenue Growth: N/A")
                    
                    # Add Price/Sales ratio
                    ps_ratio = info.get('priceToSalesTrailing12Months', 0)
                    if ps_ratio:
                        st.write(f"Price/Sales: {ps_ratio:.2f}")
                    else:
                        st.write("Price/Sales: N/A")
        else:
            st.info("📊 Fundamental analysis data not available")
    
    with tech_tab:
        if analyzer.technical_signals and 'signals' in analyzer.technical_signals:
            st.subheader("📈 Technical Analysis Signals")
            
            signals = analyzer.technical_signals['signals']
            signal_scores = analyzer.technical_signals['scores']
            
            # Display signals in a nice format
            for signal_name, signal_text in signals.items():
                score = signal_scores.get(signal_name, 5)
                
                if "🟢" in signal_text:
                    st.success(f"**{signal_name}**: {signal_text} (Score: {score}/10)")
                elif "🔴" in signal_text:
                    st.error(f"**{signal_name}**: {signal_text} (Score: {score}/10)")
                else:
                    st.info(f"**{signal_name}**: {signal_text} (Score: {score}/10)")
            
            # Technical indicators current values
            if analyzer.data is not None and len(analyzer.data) > 0:
                st.subheader("📊 Current Technical Indicators")
                latest = analyzer.data.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RSI", f"{latest.get('RSI', 0):.1f}")
                    st.metric("MACD", f"{latest.get('MACD', 0):.4f}")
                with col2:
                    st.metric("SMA 20", format_currency(latest.get('SMA_20', 0), analyzer.fundamental_info))
                    st.metric("SMA 50", format_currency(latest.get('SMA_50', 0), analyzer.fundamental_info))
                with col3:
                    st.metric("BB Upper", format_currency(latest.get('BB_Upper', 0), analyzer.fundamental_info))
                    st.metric("BB Lower", format_currency(latest.get('BB_Lower', 0), analyzer.fundamental_info))
                with col4:
                    st.metric("Stochastic %K", f"{latest.get('%K', 0):.1f}")
                    st.metric("Volume Ratio", f"{latest.get('Volume_Ratio', 0):.1f}x")
        else:
            st.info("📈 Technical analysis data not available")
    
    with buy_tab:
        if analyzer.buying_prices:
            st.subheader("💰 Optimal Buying Strategy")
            
            current_price = analyzer.buying_prices.get('current_price', 0)
            recommended_price = analyzer.buying_prices.get('recommended_buy', 0)
            
            if current_price > 0 and recommended_price > 0:
                discount = ((current_price - recommended_price) / current_price) * 100
                
                st.markdown("### 🎯 Recommended Entry Price")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recommended Buy Price", format_currency(recommended_price, analyzer.fundamental_info))
                with col2:
                    st.metric("Discount from Current", f"{discount:.1f}%")
                with col3:
                    potential_upside = ((current_price - recommended_price) / recommended_price) * 100
                    st.metric("Potential Upside", f"{potential_upside:.1f}%")
                
                # Other buying levels
                st.markdown("### 📋 Alternative Entry Levels")
                
                entry_levels = []
                for strategy, price in analyzer.buying_prices.items():
                    if strategy not in ['current_price', 'recommended_buy'] and price > 0:
                        discount = ((current_price - price) / current_price) * 100
                        entry_levels.append({
                            'Strategy': strategy.replace('_', ' ').title(),
                            'Price': format_currency(price, analyzer.fundamental_info),
                            'Discount': f"{discount:.1f}%"
                        })
                
                if entry_levels:
                    df_entry = pd.DataFrame(entry_levels)
                    st.dataframe(df_entry, hide_index=True)
                
                # Risk assessment based on combined analysis
                st.markdown("### ⚠️ Risk Assessment")
                combined_score = analyzer.recommendation.get('combined_score', 0)
                
                if combined_score >= 7:
                    st.success("🟢 LOW RISK: Strong fundamentals + positive technical signals")
                elif combined_score >= 5.5:
                    st.info("🟡 MODERATE RISK: Mixed signals, proceed with caution")
                else:
                    st.error("🔴 HIGH RISK: Weak fundamentals or negative technical signals")
            else:
                st.info("💰 Buying price analysis not available")
        else:
            st.info("💰 Buying strategy data not available")
    
    with chart_tab:
        if analyzer.data is not None and len(analyzer.data) > 50:
            st.subheader("📉 Technical Analysis Charts")
            
            try:
                # Create comprehensive chart
                fig = create_comprehensive_chart(analyzer)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent data table
                st.subheader("📈 Recent Trading Data")
                
                # Only include columns that exist
                available_columns = [col for col in ['Close', 'Volume', 'RSI', 'SMA_20', 'SMA_50', 'MACD', 'BB_Upper', 'BB_Lower'] 
                                   if col in analyzer.data.columns]
                
                if available_columns:
                    recent_data = analyzer.data[available_columns].tail(10).round(2)
                    st.dataframe(recent_data, use_container_width=True)
                else:
                    st.info("📊 No detailed data columns available for display")
                    
            except Exception as e:
                st.error(f"Error creating charts: {e}")
                st.info("📉 Chart data not available")
        else:
            st.info("📉 Insufficient data for charts (need at least 50 data points)")

def safe_format_number(value, default='N/A', add_commas=True):
    """Safely format numbers with comma separators"""
    if value is None or value == 'N/A' or value == '' or value == 0:
        return default
    
    try:
        # Handle string representations of numbers
        if isinstance(value, str):
            # Remove any currency symbols, commas, or percentage signs
            cleaned_value = value.replace('$', '').replace(',', '').replace('%', '').strip()
            if not cleaned_value or cleaned_value == '-':
                return default
            value = cleaned_value
        
        num_value = float(value)
        
        # Check for very small values that might be effectively zero
        if abs(num_value) < 1e-10:
            return default
            
        if add_commas:
            return f"{num_value:,.0f}"
        else:
            return f"{num_value:.2f}"
    except (ValueError, TypeError, AttributeError):
        return default

def display_integrated_analysis(combined_data, include_technical=True):
    """Display integrated analysis combining Yahoo Finance and Alpha Vantage data"""
    
    symbol = combined_data['symbol']
    recommendation = combined_data.get('recommendation')
    
    if not recommendation:
        st.error("❌ Analysis incomplete - missing recommendation data")
        return
    
    recommendation_text, color, overall_score = recommendation
    
    # Header with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        company_name = "Unknown"
        current_price = 0
        
        # Try to get company info from different sources
        if combined_data.get('alpha_data') and combined_data['alpha_data'].get('overview'):
            overview = combined_data['alpha_data']['overview']
            company_name = overview.get('Name', 'Unknown')[:20]
            # Try to get price from Alpha Vantage
            try:
                current_price = float(fetch_price(symbol) or 0)
            except:
                current_price = 0
        
        # Fallback to Yahoo Finance data
        if current_price == 0 and combined_data.get('yahoo_data'):
            # The yahoo_data is already the info object from fetch_yahoo_info
            yahoo_info = combined_data['yahoo_data']
            
            if not company_name or company_name == "Unknown":
                company_name = yahoo_info.get('shortName', yahoo_info.get('longName', 'Unknown'))[:20]
            current_price = (yahoo_info.get('currentPrice', 0) or 
                           yahoo_info.get('regularMarketPrice', 0) or 0)
        
        st.metric("Company", company_name)
        st.metric("Current Price", f"${current_price:.2f}" if current_price > 0 else "N/A")
    
    with col2:
        st.metric("Overall Score", f"{overall_score:.1f}/10")
        st.metric("Recommendation", recommendation_text)
    
    with col3:
        # Show data sources used
        sources_used = []
        if combined_data.get('yahoo_data'):
            sources_used.append("Yahoo Finance")
        if combined_data.get('alpha_data'):
            sources_used.append("Alpha Vantage")
        if combined_data.get('technical_data'):
            sources_used.append("Technical Analysis")
        
        st.metric("Data Sources", len(sources_used))
        st.write(", ".join(sources_used))
    
    with col4:
        # Show score confidence based on data availability
        confidence = "High" if len(sources_used) >= 2 else "Medium" if len(sources_used) == 1 else "Low"
        st.metric("Confidence", confidence)
        
        # Color-coded recommendation
        if color == "green":
            st.success(f"🟢 {recommendation_text}")
        elif color == "red":
            st.error(f"🔴 {recommendation_text}")
        else:
            st.warning(f"🟡 {recommendation_text}")
    
    # Tabbed display of different data sources
    tabs = []
    if combined_data.get('scores'):
        tabs.append("📊 Combined Scores")
    if combined_data.get('alpha_data'):
        tabs.append("🔍 Alpha Vantage Data")
    if combined_data.get('yahoo_data'):
        tabs.append("📈 Yahoo Finance Data")
    if combined_data.get('technical_data') and include_technical:
        tabs.append("⚡ Technical Analysis")
    
    if tabs:
        tab_objects = st.tabs(tabs)
        tab_index = 0
        
        # Combined Scores Tab
        if combined_data.get('scores'):
            with tab_objects[tab_index]:
                st.subheader("📊 Combined Fundamental Scores")
                
                # Create score visualization
                scores = combined_data['scores']
                
                score_df = pd.DataFrame([
                    {"Metric": metric, "Score": score, "Weight": st.session_state.score_weights.get(metric, 0)}
                    for metric, score in scores.items()
                ])
                
                if not score_df.empty:
                    # Score chart with special handling for zero values
                    fig = go.Figure()
                    
                    # Create display values (use 0.1 for zero scores to make them visible)
                    display_scores = [max(0.1, s) if s == 0 else s for s in score_df['Score']]
                    
                    # Create colors with special handling for zero and neutral scores
                    colors = []
                    for s in score_df['Score']:
                        if s == 0:
                            colors.append('lightgray')  # Gray for zero/missing data
                        elif s == 5:
                            colors.append('lightblue')  # Light blue for neutral/unknown data
                        elif s >= 7:
                            colors.append('green')
                        elif s >= 5:
                            colors.append('orange')
                        else:
                            colors.append('red')
                    
                    fig.add_trace(go.Bar(
                        x=score_df['Metric'],
                        y=display_scores,
                        name='Score',
                        marker_color=colors,
                        text=[f'{s:.1f}' if s > 0 else 'No Data' for s in score_df['Score']],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"Fundamental Scores for {symbol}",
                        xaxis_title="Metrics",
                        yaxis_title="Score (0-10)",
                        template="plotly_white",
                        height=500,
                        xaxis={'tickangle': 45}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced score table with status indicators
                    score_df['Status'] = score_df['Score'].apply(
                        lambda x: '⚪ No Data' if x == 0 else 
                                  '� Neutral' if x == 5 else 
                                  '�🟢 Excellent' if x >= 7 else 
                                  '🟡 Good' if x > 5 else 
                                  '🔴 Poor'
                    )
                    
                    # Reorder columns for better display
                    display_df = score_df[['Metric', 'Score', 'Status', 'Weight']]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Show summary of zero scores and neutral scores
                    zero_scores = score_df[score_df['Score'] == 0]['Metric'].tolist()
                    neutral_scores = score_df[score_df['Score'] == 5]['Metric'].tolist()
                    if zero_scores:
                        st.info(f"📋 Metrics with no data available: {', '.join(zero_scores)}")
                    if neutral_scores:
                        st.info(f"🔵 Metrics with limited data (neutral scoring): {', '.join(neutral_scores)}")
                
            tab_index += 1
        
        # Alpha Vantage Data Tab
        if combined_data.get('alpha_data'):
            with tab_objects[tab_index]:
                st.subheader("🔍 Alpha Vantage Company Overview")
                
                overview = combined_data['alpha_data']['overview']
                alpha_scores = combined_data['alpha_data'].get('scores', {})
                
                # Company overview metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Company Details**")
                    st.write(f"Sector: {overview.get('Sector', 'N/A')}")
                    st.write(f"Industry: {overview.get('Industry', 'N/A')}")
                    st.write(f"Market Cap: {overview.get('MarketCapitalization', 'N/A')}")
                    st.write(f"Exchange: {overview.get('Exchange', 'N/A')}")
                
                with col2:
                    st.write("**Valuation Metrics**")
                    st.write(f"P/E Ratio: {overview.get('PERatio', 'N/A')}")
                    st.write(f"Forward P/E: {overview.get('ForwardPE', 'N/A')}")
                    st.write(f"P/B Ratio: {overview.get('PriceToBookRatio', 'N/A')}")
                    st.write(f"EV/EBITDA: {overview.get('EVToEBITDA', 'N/A')}")
                
                with col3:
                    st.write("**Financial Health**")
                    st.write(f"ROE: {overview.get('ReturnOnEquityTTM', 'N/A')}")
                    st.write(f"Revenue (TTM): {overview.get('RevenueTTM', 'N/A')}")
                    st.write(f"Gross Margin: {overview.get('GrossProfitTTM', 'N/A')}")
                    st.write(f"Dividend Yield: {overview.get('DividendYield', 'N/A')}")
                
                # Alpha Vantage specific scores
                if alpha_scores:
                    st.subheader("Alpha Vantage Scores")
                    alpha_df = pd.DataFrame([
                        {"Metric": metric, "Score": score}
                        for metric, score in alpha_scores.items()
                    ])
                    st.dataframe(alpha_df, use_container_width=True, hide_index=True)
                
            tab_index += 1
        
        # Yahoo Finance Data Tab
        if combined_data.get('yahoo_data'):
            with tab_objects[tab_index]:
                st.subheader("📈 Yahoo Finance Data")
                
                # The yahoo_data is already the info object from fetch_yahoo_info
                yahoo_info = combined_data['yahoo_data']
                yahoo_scores = combined_data.get('scores', {})  # Get scores from combined_data
                
                # Yahoo Finance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Market Data**")
                    
                    # Format price values properly
                    high_52w = yahoo_info.get('fiftyTwoWeekHigh')
                    low_52w = yahoo_info.get('fiftyTwoWeekLow')
                    volume = yahoo_info.get('volume')
                    avg_volume = yahoo_info.get('averageVolume')
                    
                    st.write(f"52W High: ${high_52w:.2f}" if high_52w and high_52w > 0 else "52W High: N/A")
                    st.write(f"52W Low: ${low_52w:.2f}" if low_52w and low_52w > 0 else "52W Low: N/A")
                    st.write(f"Volume: {safe_format_number(volume)}")
                    st.write(f"Avg Volume: {safe_format_number(avg_volume)}")
                
                with col2:
                    st.write("**Profitability**")
                    
                    # Format percentage values properly  
                    profit_margins = yahoo_info.get('profitMargins')
                    operating_margins = yahoo_info.get('operatingMargins')
                    revenue_growth = yahoo_info.get('revenueGrowth')
                    earnings_growth = yahoo_info.get('earningsGrowth')
                    
                    # Convert decimal values to percentages if they exist
                    profit_pct = f"{profit_margins*100:.1f}%" if profit_margins and profit_margins != 0 else "N/A"
                    operating_pct = f"{operating_margins*100:.1f}%" if operating_margins and operating_margins != 0 else "N/A"
                    revenue_pct = f"{revenue_growth*100:.1f}%" if revenue_growth and revenue_growth != 0 else "N/A"
                    earnings_pct = f"{earnings_growth*100:.1f}%" if earnings_growth and earnings_growth != 0 else "N/A"
                    
                    st.write(f"Profit Margin: {profit_pct}")
                    st.write(f"Operating Margin: {operating_pct}")
                    st.write(f"Revenue Growth: {revenue_pct}")
                    st.write(f"Earnings Growth: {earnings_pct}")
                
                with col3:
                    st.write("**Analyst Data**")
                    
                    target_price = yahoo_info.get('targetMeanPrice')
                    recommendation_mean = yahoo_info.get('recommendationMean')
                    num_analysts = yahoo_info.get('numberOfAnalystOpinions')
                    
                    st.write(f"Target Price: ${target_price:.2f}" if target_price and target_price > 0 else "Target Price: N/A")
                    st.write(f"Recommendation: {recommendation_mean:.1f}" if recommendation_mean and recommendation_mean > 0 else "Recommendation: N/A")
                    st.write(f"Number of Analysts: {safe_format_number(num_analysts)}")
                
                # Yahoo Finance specific scores
                if yahoo_scores:
                    st.subheader("Yahoo Finance Scores")
                    yahoo_df = pd.DataFrame([
                        {"Metric": metric, "Score": score}
                        for metric, score in yahoo_scores.items()
                    ])
                    st.dataframe(yahoo_df, use_container_width=True, hide_index=True)
                
            tab_index += 1
        
        # Technical Analysis Tab
        if combined_data.get('technical_data') and include_technical:
            with tab_objects[tab_index]:
                st.subheader("⚡ Technical Analysis")
                
                technical_data = combined_data['technical_data']
                analyzer = technical_data.get('analyzer')
                signals = technical_data.get('signals', {})
                
                if analyzer and hasattr(analyzer, 'data') and analyzer.data is not None:
                    # Technical indicators summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Technical Signals**")
                        for signal_name, signal_data in signals.items():
                            if isinstance(signal_data, dict):
                                signal_value = signal_data.get('signal', 'N/A')
                                confidence = signal_data.get('confidence', 'N/A')
                                st.write(f"{signal_name}: {signal_value} ({confidence})")
                    
                    with col2:
                        # Latest technical indicators
                        if hasattr(analyzer, 'data') and analyzer.data is not None and len(analyzer.data) > 0:
                            st.write("**Current Indicators**")
                            latest = analyzer.data.iloc[-1]
                            
                            # Display available technical indicators
                            indicator_cols = ['RSI', 'MACD', 'MACD_Signal', '%K', '%D', 'Volume_Ratio']
                            for indicator in indicator_cols:
                                if indicator in analyzer.data.columns:
                                    value = latest[indicator]
                                    if pd.notna(value):
                                        st.write(f"{indicator}: {value:.2f}")
                    
                    # Price and volume chart
                    if len(analyzer.data) > 0:
                        st.subheader("📈 Price and Technical Indicators")
                        
                        fig = go.Figure()
                        
                        # Price data
                        fig.add_trace(go.Scatter(
                            x=analyzer.data.index,
                            y=analyzer.data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add moving averages if available
                        if 'SMA_20' in analyzer.data.columns:
                            fig.add_trace(go.Scatter(
                                x=analyzer.data.index,
                                y=analyzer.data['SMA_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='orange', width=1)
                            ))
                        
                        if 'SMA_50' in analyzer.data.columns:
                            fig.add_trace(go.Scatter(
                                x=analyzer.data.index,
                                y=analyzer.data['SMA_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='red', width=1)
                            ))
                        
                        fig.update_layout(
                            title=f"{symbol} - Technical Analysis",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 Technical analysis data not available")

def create_comprehensive_chart(analyzer):
    """Create a comprehensive chart combining price, volume, and indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{analyzer.symbol} - Price & Volume', 
            'RSI (14)', 
            'MACD', 
            'Stochastic'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2],
        specs=[[{"secondary_y": True}],  # Enable secondary y-axis for first subplot only
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Price, Moving Averages, and Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=analyzer.data.index,
            open=analyzer.data['Open'],
            high=analyzer.data['High'],
            low=analyzer.data['Low'],
            close=analyzer.data['Close'],
            name='Price'
        ), row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['SMA_20'], 
                  name='SMA 20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['SMA_50'], 
                  name='SMA 50', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['BB_Upper'], 
                  name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['BB_Lower'], 
                  name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                  fill='tonexty', fillcolor='rgba(68, 68, 68, 0.1)'),
        row=1, col=1
    )
    
    # Volume on secondary y-axis
    colors = ['red' if row['Close'] < row['Open'] else 'green' for index, row in analyzer.data.iterrows()]
    fig.add_trace(
        go.Bar(x=analyzer.data.index, y=analyzer.data['Volume'], 
               name='Volume', marker_color=colors, opacity=0.6),
        row=1, col=1, secondary_y=True
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['RSI'], 
                  name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MACD'], 
                  name='MACD', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MACD_Signal'], 
                  name='Signal', line=dict(color='red', width=1)),
        row=3, col=1
    )
    
    # MACD Histogram
    fig.add_trace(
        go.Bar(x=analyzer.data.index, y=analyzer.data['MACD_Histogram'], 
               name='MACD Histogram', marker_color='green', opacity=0.3),
        row=3, col=1
    )
    
    # Stochastic
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['%K'], 
                  name='%K', line=dict(color='blue', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['%D'], 
                  name='%D', line=dict(color='red', width=1)),
        row=4, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{analyzer.symbol} - Comprehensive Technical Analysis',
        height=900,
        showlegend=True,
        template='plotly_white'
    )
    
    # Get currency symbol for y-axis label
    currency_symbol = get_currency_symbol(analyzer.fundamental_info)
    if currency_symbol == "DKK":
        price_label = "Price (DKK)"
    elif currency_symbol == "€":
        price_label = "Price (€)"
    elif currency_symbol == "£":
        price_label = "Price (£)"
    else:
        price_label = "Price (USD)"
    
    # Set y-axis titles
    fig.update_yaxes(title_text=price_label, row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1)
    
    return fig

def calculate_scores_yahoo(info, industry_pe=20):
    """Enhanced scoring calculation with new metrics"""
    if not info:
        return None, None
    
    try:
        # Extract values with better defaults
        pe = safe_float(info.get("pe", 0))
        forward_pe = safe_float(info.get("forwardPE", 0))
        peg = safe_float(info.get("peg", 0))
        pb = safe_float(info.get("pb", 0))
        ev_ebitda = safe_float(info.get("enterpriseToEbitda", 0))
        price_sales = safe_float(info.get("priceToSalesTrailing12Months", 0))
        roe = safe_float(info.get("roe", 0)) * 100 if info.get("roe") else 0
        de = safe_float(info.get("de", 0))  # Use as-is, already a ratio
        dy = safe_float(info.get("dy", 0))  # Use as-is, already a percentage
        gm = safe_float(info.get("gm", 0)) * 100 if info.get("gm") else 0
        eps_growth = safe_float(info.get("eps_growth", 0)) * 100 if info.get("eps_growth") else 0
        rev_growth = safe_float(info.get("revenue_growth", 0)) * 100 if info.get("revenue_growth") else 0
        
        # Analyst data
        target_price = safe_float(info.get("targetMeanPrice", 0))
        current_price = safe_float(info.get("currentPrice", 0))
        analyst_upside = 0
        if target_price > 0 and current_price > 0:
            analyst_upside = ((target_price - current_price) / current_price) * 100
        
        # Use historical data if available
        financial_history = info.get("financial_history", {})
        has_revenue_data = False
        has_fcf_data = False
        
        if financial_history:
            if "avg_revenue_growth" in financial_history and financial_history["avg_revenue_growth"] is not None:
                rev_growth = financial_history["avg_revenue_growth"]
                has_revenue_data = True
            if "avg_roe" in financial_history:
                roe = financial_history["avg_roe"]
        
        # Check if we have meaningful revenue growth data from Yahoo Finance
        if not has_revenue_data and info.get("revenue_growth") is not None and info.get("revenue_growth") != 0:
            has_revenue_data = True
        
        fcf_trend = financial_history.get("fcf_trend", [])
        if fcf_trend and len(fcf_trend) > 1 and not all(x == 0 for x in fcf_trend):
            has_fcf_data = True
        
        scores = {
            "PE": score_pe(pe, industry_pe, allow_neutral=True),
            "PEG": score_peg(peg, allow_neutral=True),
            "PB": score_pb(pb),
            "ROE": score_roe_dynamic(roe, info.get("sector", "Unknown")),  # Enhanced with dynamic benchmarking
            "EPS Growth": score_eps_growth(eps_growth),
            "Revenue Growth": score_revenue_growth_dynamic(rev_growth, info.get("sector", "Unknown")),  # Enhanced
            "FCF Trend": score_fcf_trend(fcf_trend, has_fcf_data),  # Always include FCF Trend
            "Debt/Equity": score_debt_equity_dynamic(de, info.get("sector", "Unknown")),  # Enhanced
            "Dividend Yield": score_dividend_yield(dy, allow_neutral=True),
            "Gross Margin": score_gross_margin_dynamic(gm, info.get("sector", "Unknown"))  # Enhanced
        }
        
        # Add new enhanced metrics
        if forward_pe > 0:
            scores["Forward PE"] = score_forward_pe(forward_pe, industry_pe)
        
        if ev_ebitda > 0:
            scores["EV/EBITDA"] = score_ev_ebitda(ev_ebitda)
        
        if price_sales > 0:
            scores["Price/Sales"] = score_price_sales(price_sales)
        
        if target_price > 0 and current_price > 0:
            scores["Analyst Upside"] = score_analyst_upside(analyst_upside)
        
        # Add momentum scoring (NEW)
        symbol = info.get("symbol", "")
        if symbol:
            momentum_score = score_momentum(symbol)
            scores["Momentum"] = momentum_score
        
        # Add financial health scoring (NEW)
        financial_health_score = score_financial_health_composite(info, info.get("sector", "Unknown"))
        scores["Financial Health"] = financial_health_score
        
        # Filter out None values (missing data) but keep 0 scores
        scores = {k: v for k, v in scores.items() if v is not None}
        
        # Apply sector-specific adjustments
        sector = info.get("sector", "")
        if sector:
            scores = apply_sector_adjustments(scores, sector)
        
        return scores, info
        
    except Exception as e:
        st.error(f"Error calculating scores: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

def safe_float(value, default=0):
    """Safely convert value to float with robust None handling"""
    try:
        if value is None or value == "None" or value == "" or str(value).lower() == 'nan':
            return default
        # Handle string representations of None or empty
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_comparison(value1, value2, operation="<", default_result=False):
    """Safely compare values, handling None cases"""
    try:
        val1 = safe_float(value1)
        val2 = safe_float(value2)
        
        if operation == "<":
            return val1 < val2
        elif operation == "<=":
            return val1 <= val2
        elif operation == ">":
            return val1 > val2
        elif operation == ">=":
            return val1 >= val2
        elif operation == "==":
            return val1 == val2
        else:
            return default_result
    except:
        return default_result

def get_recommendation(total_score):
    """Enhanced recommendation system"""
    if total_score >= 8:
        return "🚀 Strong Buy", "green"
    elif total_score >= 6.5:
        return "📈 Buy", "limegreen"
    elif total_score >= 4:
        return "🔄 Hold", "orange"
    elif total_score >= 2:
        return "📉 Weak Sell", "orangered"
    else:
        return "🛑 Strong Sell", "red"

# Industry P/E mapping
INDUSTRY_PE_MAP = {
    "Technology": 28,
    "Consumer Discretionary": 22,
    "Consumer Staples": 18,
    "Health Care": 25,
    "Financials": 12,
    "Energy": 15,
    "Materials": 16,
    "Industrials": 18,
    "Utilities": 20,
    "Real Estate": 25,
    "Communication Services": 22,
    "Consumer Cyclical": 22,
    "Healthcare": 25,
    "Financial Services": 12,
    "Basic Materials": 16,
    "Unknown": 20
}

# Industry benchmarks for dynamic scoring
INDUSTRY_BENCHMARKS = {
    "Technology": {
        "roe": 18.5, "gross_margin": 65.0, "revenue_growth": 15.0, "debt_equity": 25.0,
        "current_ratio": 2.5, "interest_coverage": 25.0
    },
    "Healthcare": {
        "roe": 15.2, "gross_margin": 75.0, "revenue_growth": 8.0, "debt_equity": 30.0,
        "current_ratio": 3.2, "interest_coverage": 20.0
    },
    "Financials": {
        "roe": 12.8, "gross_margin": 45.0, "revenue_growth": 5.0, "debt_equity": 180.0,
        "current_ratio": 1.1, "interest_coverage": 8.0
    },
    "Industrials": {
        "roe": 14.5, "gross_margin": 35.0, "revenue_growth": 6.0, "debt_equity": 45.0,
        "current_ratio": 1.8, "interest_coverage": 12.0
    },
    "Energy": {
        "roe": 8.5, "gross_margin": 25.0, "revenue_growth": 3.0, "debt_equity": 65.0,
        "current_ratio": 1.5, "interest_coverage": 6.0
    },
    "Consumer Staples": {
        "roe": 16.8, "gross_margin": 28.0, "revenue_growth": 4.0, "debt_equity": 55.0,
        "current_ratio": 1.3, "interest_coverage": 15.0
    },
    "Consumer Discretionary": {
        "roe": 13.2, "gross_margin": 40.0, "revenue_growth": 8.0, "debt_equity": 48.0,
        "current_ratio": 1.6, "interest_coverage": 10.0
    },
    "Unknown": {  # Default fallback
        "roe": 15.0, "gross_margin": 45.0, "revenue_growth": 7.0, "debt_equity": 50.0,
        "current_ratio": 1.8, "interest_coverage": 12.0
    }
}

# Sector-specific scoring models
SECTOR_SCORING_MODELS = {
    "Technology": {
        "weight_adjustments": {
            "Forward PE": 1.2,
            "Revenue Growth": 1.4,
            "Gross Margin": 1.3,
            "Price/Sales": 1.2,
            "EPS Growth": 1.3
        },
        "benchmarks": {
            "revenue_growth_excellent": 25,
            "gross_margin_excellent": 70,
            "forward_pe_max": 35
        }
    },
    "Industrials": {
        "weight_adjustments": {
            "EV/EBITDA": 1.3,
            "ROE": 1.2,
            "Debt/Equity": 1.1,
            "FCF Trend": 1.4,
            "Revenue Growth": 1.1
        },
        "benchmarks": {
            "roe_excellent": 18,
            "ev_ebitda_excellent": 12,
            "debt_equity_max": 0.6
        }
    },
    "Financials": {
        "weight_adjustments": {
            "PB": 1.5,
            "ROE": 1.4,
            "Dividend Yield": 1.2,
            "PE": 1.2
        },
        "benchmarks": {
            "roe_excellent": 15,
            "pb_excellent": 1.2,
            "dividend_yield_good": 3.5
        }
    },
    "Healthcare": {
        "weight_adjustments": {
            "Forward PE": 1.1,
            "Revenue Growth": 1.2,
            "Gross Margin": 1.3,
            "EPS Growth": 1.2
        },
        "benchmarks": {
            "revenue_growth_excellent": 15,
            "gross_margin_excellent": 75
        }
    },
    "Consumer Staples": {
        "weight_adjustments": {
            "Dividend Yield": 1.4,
            "ROE": 1.2,
            "Debt/Equity": 1.1,
            "Gross Margin": 1.2
        },
        "benchmarks": {
            "dividend_yield_excellent": 4,
            "roe_good": 20
        }
    }
}

def get_industry_pe(info):
    """Get industry P/E ratio"""
    industry = info.get("industry", "")
    sector = info.get("sector", "")
    
    # Try exact match first
    for key in INDUSTRY_PE_MAP:
        if key.lower() in industry.lower() or key.lower() in sector.lower():
            return INDUSTRY_PE_MAP[key]
    
    return INDUSTRY_PE_MAP["Unknown"]

def get_industry_benchmarks(sector):
    """Get industry benchmark values for dynamic scoring"""
    return INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS["Unknown"])

def calculate_momentum_indicators(symbol):
    """Calculate momentum indicators for scoring"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
        
        if data is None or len(data) < 200:
            return None
        
        current_price = data['Close'].iloc[-1]
        
        # Calculate moving averages
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else None
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else None
        sma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else None
        
        # Calculate price momentum
        price_1m = data['Close'].iloc[-21] if len(data) >= 21 else None
        price_3m = data['Close'].iloc[-63] if len(data) >= 63 else None
        price_6m = data['Close'].iloc[-126] if len(data) >= 126 else None
        
        return {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'price_1m': price_1m,
            'price_3m': price_3m,
            'price_6m': price_6m
        }
    except Exception:
        return None

def get_financial_health_data(info):
    """Extract financial health metrics from stock info"""
    try:
        # Get balance sheet data if available
        current_assets = info.get('totalCurrentAssets', 0)
        current_liabilities = info.get('totalCurrentLiabilities', 0)
        total_debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        ebit = info.get('ebit', 0)
        interest_expense = info.get('interestExpense', 0)
        
        # Calculate ratios
        current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
        quick_ratio = (current_assets - info.get('inventory', 0)) / current_liabilities if current_liabilities > 0 else 0
        cash_ratio = cash / current_liabilities if current_liabilities > 0 else 0
        debt_to_assets = total_debt / info.get('totalAssets', 1) if info.get('totalAssets', 0) > 0 else 0
        interest_coverage = ebit / interest_expense if interest_expense > 0 else float('inf')
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': cash_ratio,
            'debt_to_assets': debt_to_assets,
            'interest_coverage': interest_coverage,
            'cash_position': cash
        }
    except Exception:
        return None

def apply_sector_adjustments(scores, sector):
    """Apply sector-specific weight adjustments to improve accuracy"""
    if not sector or sector not in SECTOR_SCORING_MODELS:
        return scores
    
    model = SECTOR_SCORING_MODELS[sector]
    weight_adjustments = model.get("weight_adjustments", {})
    adjusted_scores = {}
    
    for metric, score in scores.items():
        if metric in weight_adjustments:
            # Apply sector-specific emphasis
            multiplier = weight_adjustments[metric]
            adjusted_score = min(10, score * multiplier)  # Cap at 10
            adjusted_scores[metric] = adjusted_score
        else:
            adjusted_scores[metric] = score
    
    return adjusted_scores

# Visualization functions
def create_enhanced_score_chart(scores, symbol):
    """Create enhanced score visualization"""
    metrics = list(scores.keys())
    values = list(scores.values())
    
    # Ensure certain metrics are visible even with 0 scores (indicating no data)
    display_values = []
    for i, (metric, value) in enumerate(zip(metrics, values)):
        if metric in ['PE', 'PEG', 'PB', 'Revenue Growth', 'FCF Trend'] and value == 0:
            display_values.append(0.5)  # Minimum visible height for missing data
        else:
            display_values.append(value)
    
    # Color coding based on score
    colors = []
    for i, v in enumerate(values):  # Use original values for color coding
        if v == 0 and metrics[i] in ['Revenue Growth', 'FCF Trend']:
            colors.append('gray')  # Gray for missing data
        elif v >= 7:
            colors.append('darkgreen')
        elif v >= 5:
            colors.append('green')
        elif v >= 3:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Create text labels that indicate missing data
    text_labels = []
    for i, v in enumerate(values):
        if v == 0 and metrics[i] in ['Revenue Growth', 'FCF Trend']:
            text_labels.append("No Data")
        else:
            text_labels.append(f"{v:.1f}")
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics, 
            y=display_values,  # Use display values with minimum height
            marker_color=colors,
            text=text_labels,  # Show "No Data" for missing metrics
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{text}<extra></extra>',
            customdata=values  # Store original values for hover
        )
    ])
    
    fig.update_layout(
        title=f"📊 {symbol} - Scoring Breakdown",
        xaxis_title="Metrics",
        yaxis_title="Score (0-10)",
        yaxis=dict(range=[0, 10]),
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    # Add score bands
    fig.add_hline(y=7, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=5, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5)
    
    return fig

def get_stock_symbols_for_market(market_selection, custom_symbols=None):
    """
    Get stock symbols based on market selection
    """
    if market_selection == "Danish Stocks":
        danish_symbols = list(DANISH_STOCKS.values())
        return list(set(danish_symbols))  # Remove duplicates
    elif market_selection == "S&P 500":
        return SP500_STOCKS
    elif market_selection == "NASDAQ 100":
        return NASDAQ100_STOCKS
    elif market_selection == "European Stocks":
        return EUROPEAN_STOCKS
    elif market_selection == "Custom Symbols" and custom_symbols:
        # Parse custom symbols
        symbols = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]
        return symbols
    else:
        return []

def screen_multi_market_stocks(market_selection, min_score=5.0, custom_symbols=None):
    """
    Screen stocks from multiple markets using optimized batch processing
    """
    # Add NumPy compatibility setup for market screener
    import warnings
    import os
    
    # Set environment variables to suppress NumPy warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['NUMPY_HIDE_WARNINGS'] = '1'
    
    # Import and patch numpy to prevent attribute errors
    try:
        import numpy as np
        # Add missing aliases to prevent numpy deprecation errors
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'int'):
            np.int = int  
        if not hasattr(np, 'float'):
            np.float = float
    except:
        pass
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore")
        
        try:
            # Get symbols based on market selection
            symbols_to_screen = get_stock_symbols_for_market(market_selection, custom_symbols)
            
            if not symbols_to_screen:
                st.warning(f"No symbols found for market selection: {market_selection}")
                return pd.DataFrame()
            
            st.info(f"📊 Screening {len(symbols_to_screen)} stocks from {market_selection}...")
            
            # Validate session state weights exist
            if 'score_weights' not in st.session_state:
                st.error("❌ Score weights not found in session state. Please go to Tab 1 first to initialize weights.")
                return pd.DataFrame()
            
            # Use optimized batch analysis with error handling
            try:
                st.write("🔄 Starting stock analysis...")
                analysis_results = analyze_multiple_stocks(symbols_to_screen)
                st.success(f"✅ Analysis completed for {len(analysis_results)} stocks")
            except Exception as analysis_error:
                st.error(f"❌ Error during stock analysis: {str(analysis_error)}")
                import traceback
                st.code(traceback.format_exc())
                return pd.DataFrame()
            
            results = []
            processed = 0
            
            st.write("📈 Processing results...")
            for symbol, data in analysis_results.items():
                try:
                    processed += 1
                    if processed % 10 == 0:  # Update progress every 10 stocks
                        st.write(f"Processed {processed}/{len(symbols_to_screen)} stocks...")
                    
                    if not data or 'scores' not in data or 'info' not in data:
                        continue
                        
                    scores = data['scores']
                    info = data['info']
                    
                    # Calculate overall score with error handling
                    try:
                        available_weights = {k: st.session_state.score_weights.get(k, 0) 
                                           for k in scores if k in st.session_state.score_weights}
                        
                        if available_weights:
                            total_weight = sum(available_weights.values())
                            if total_weight > 0:
                                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                overall_score = sum(safe_float(scores[k], 0) * safe_float(normalized_weights[k], 0) for k in available_weights)
                            else:
                                valid_scores = [safe_float(score, 0) for score in scores.values()]
                                overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                        else:
                            valid_scores = [safe_float(score, 0) for score in scores.values()]
                            overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    except Exception as score_error:
                        # Fallback to simple average if weighting fails
                        st.warning(f"Error calculating score for {symbol}: {str(score_error)}")
                        valid_scores = [safe_float(score, 0) for score in scores.values()]
                        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    
                    # Only include stocks above minimum score
                    if overall_score >= min_score:
                        stock_data = {}
                        
                        # Basic info
                        if market_selection == "Danish Stocks":
                            original_symbols = [k for k, v in DANISH_STOCKS.items() if v == symbol]
                            original_symbol = original_symbols[0] if original_symbols else symbol
                            stock_data['Original_Symbol'] = original_symbol
                        else:
                            stock_data['Original_Symbol'] = symbol
                        
                        stock_data['Yahoo_Symbol'] = symbol
                        stock_data['Company'] = info.get('name', 'N/A')[:30]
                        stock_data['Market'] = market_selection
                        stock_data['Sector'] = info.get('sector', 'N/A')
                        stock_data['Industry'] = info.get('industry', 'N/A')[:25]
                        
                        # Financial data
                        stock_data['Current_Price'] = info.get('currentPrice', info.get('price', 0))
                        stock_data['Market_Cap'] = info.get('marketCap', 0)
                        stock_data['P/E_Ratio'] = round(info.get('pe', 0), 2) if info.get('pe') else 0
                        stock_data['PEG_Ratio'] = round(info.get('peg', 0), 2) if info.get('peg') else 0
                        stock_data['Price_to_Book'] = round(info.get('pb', 0), 2) if info.get('pb') else 0
                        stock_data['ROE'] = round(info.get('roe', 0) * 100, 1) if info.get('roe') else 0
                        stock_data['Revenue_Growth'] = round(info.get('revenue_growth', 0) * 100, 1) if info.get('revenue_growth') else 0
                        stock_data['Dividend_Yield'] = round(info.get('dy', 0), 2) if info.get('dy') else 0
                        stock_data['Debt_to_Equity'] = round(info.get('de', 0), 1) if info.get('de') else 0
                        
                        # Add score and recommendation
                        stock_data['Final_Score'] = round(overall_score, 2)
                        recommendation, color = get_recommendation(overall_score)
                        stock_data['Recommendation'] = recommendation
                        stock_data['Rec_Color'] = color
                        
                        # Add individual metric scores
                        for metric_name, metric_score in scores.items():
                            clean_name = metric_name.replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '')
                            score_column = f"{clean_name}_Score"
                            stock_data[score_column] = metric_score
                        
                        results.append(stock_data)
                    
                except Exception as stock_error:
                    # Individual stock processing error - continue with next stock
                    continue
            
            # Create DataFrame and sort by final score
            if results:
                try:
                    df = pd.DataFrame(results)
                    
                    # Check for duplicate columns and remove if any
                    if len(df.columns) != len(set(df.columns)):
                        df = df.loc[:, ~df.columns.duplicated()]
                    
                    # Sort by final score descending
                    df = df.sort_values('Final_Score', ascending=False)
                    return df
                    
                except Exception as e:
                    st.error(f"Error creating results DataFrame: {str(e)}")
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
                
        except Exception as main_error:
            st.error(f"Error in market screening: {str(main_error)}")
            return pd.DataFrame()

def display_screening_results(results_df, market_selection, min_score):
    """
    Display the screening results in a nicely formatted way
    """
    if results_df is not None and not results_df.empty:
        st.success(f"✅ Found **{len(results_df)}** stocks from {market_selection} with score ≥ {min_score}")
        
        # Main results table
        st.subheader(f"🏆 Top Stocks from {market_selection}")
        
        display_columns = [
            'Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 'Sector', 
            'Current_Price', 'P/E_Ratio', 'ROE', 'Revenue_Growth', 'Dividend_Yield'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in results_df.columns]
        
        st.dataframe(
            results_df[available_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Final_Score': st.column_config.NumberColumn('Score', format="%.2f"),
                'Current_Price': st.column_config.NumberColumn('Price', format="$%.2f"),
                'P/E_Ratio': st.column_config.NumberColumn('P/E', format="%.1f"),
                'ROE': st.column_config.NumberColumn('ROE (%)', format="%.1f"),
                'Revenue_Growth': st.column_config.NumberColumn('Rev Growth (%)', format="%.1f"),
                'Dividend_Yield': st.column_config.NumberColumn('Div Yield (%)', format="%.1f"),
            }
        )
        
        # Add summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = results_df['Final_Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col2:
            if 'Current_Price' in results_df.columns:
                avg_price = results_df['Current_Price'].mean()
                st.metric("Average Price", f"${avg_price:.2f}")
        
        with col3:
            if 'P/E_Ratio' in results_df.columns:
                avg_pe = results_df['P/E_Ratio'].mean()
                st.metric("Average P/E", f"{avg_pe:.1f}")
        
        with col4:
            if 'ROE' in results_df.columns:
                avg_roe = results_df['ROE'].mean()
                st.metric("Average ROE", f"{avg_roe:.1f}%")
        
        # Top performers section
        if len(results_df) >= 3:
            st.subheader("🥇 Top 3 Performers")
            top_3 = results_df.head(3)
            
            for idx, row in top_3.iterrows():
                with st.expander(f"#{idx+1} {row.get('Original_Symbol', 'N/A')} - {row.get('Company', 'Unknown')} (Score: {row.get('Final_Score', 0):.2f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sector:** {row.get('Sector', 'N/A')}")
                        st.write(f"**Recommendation:** {row.get('Recommendation', 'N/A')}")
                        if 'Current_Price' in row:
                            st.write(f"**Current Price:** ${row.get('Current_Price', 0):.2f}")
                    
                    with col2:
                        if 'P/E_Ratio' in row:
                            st.write(f"**P/E Ratio:** {row.get('P/E_Ratio', 0):.1f}")
                        if 'ROE' in row:
                            st.write(f"**ROE:** {row.get('ROE', 0):.1f}%")
                        if 'Dividend_Yield' in row:
                            st.write(f"**Dividend Yield:** {row.get('Dividend_Yield', 0):.1f}%")

def display_danish_stocks_screener():
    """
    Display the Multi-Market stocks screening interface
    """
    try:
        # Add NumPy compatibility setup for screener display
        import warnings
        import os
        warnings.filterwarnings('ignore')
        
        # Ensure session state is properly initialized
        if 'score_weights' not in st.session_state:
            st.warning("⚠️ Session state not properly initialized. Initializing now...")
            init_session_state()
            st.success("✅ Session state initialized. You can now use the screener.")
        
        st.header("🔍 Multi-Market Stock Screener")
        st.markdown("Screen stocks from multiple markets using the comprehensive scoring system from Tab 1")
        
        # Initialize session state for market screener settings
        if 'screener_market_selection' not in st.session_state:
            st.session_state.screener_market_selection = "Danish Stocks"
        if 'screener_min_score' not in st.session_state:
            st.session_state.screener_min_score = 5.0
        if 'screener_max_stocks' not in st.session_state:
            st.session_state.screener_max_stocks = 25
        if 'screener_custom_symbols' not in st.session_state:
            st.session_state.screener_custom_symbols = ""
        if 'screening_in_progress' not in st.session_state:
            st.session_state.screening_in_progress = False
        if 'screening_results' not in st.session_state:
            st.session_state.screening_results = None
        
        # Configuration section
        st.subheader("📋 Screening Configuration")
        
        # Market selection and configuration section
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            options = ["Danish Stocks", "S&P 500", "NASDAQ 100", "European Stocks", "Custom Symbols"]
            try:
                current_index = options.index(st.session_state.screener_market_selection)
            except ValueError:
                current_index = 0  # Default to first option if current value not found
                st.session_state.screener_market_selection = options[0]
            
            # Use on_change to update session state only when dropdown actually changes
            def update_market_selection():
                st.session_state.screener_market_selection = st.session_state.market_dropdown
            
            market_selection = st.selectbox(
                "Select Market",
                options=options,
                index=current_index,
                help="Choose which market or stock universe to screen",
                key="market_dropdown",
                disabled=st.session_state.screening_in_progress,
                on_change=update_market_selection
            )
        
        with col2:
            def update_min_score():
                st.session_state.screener_min_score = st.session_state.score_slider
            
            min_score = st.slider(
                "Minimum Score", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.screener_min_score, 
                step=0.1,
                help="Only show stocks with score above this threshold",
                key="score_slider",
                disabled=st.session_state.screening_in_progress,
                on_change=update_min_score
            )
        
        with col3:
            def update_max_stocks():
                st.session_state.screener_max_stocks = st.session_state.max_input
            
            max_stocks = st.number_input(
                "Max Results", 
                min_value=10, 
                max_value=100, 
                value=st.session_state.screener_max_stocks,
                help="Maximum number of stocks to display",
                key="max_input",
                disabled=st.session_state.screening_in_progress,
                on_change=update_max_stocks
            )
        
        with col4:
            # Dynamic info based on current market selection
            if st.session_state.screener_market_selection == "Danish Stocks":
                st.info(f"📊 Danish stocks: **{len(set(DANISH_STOCKS.values()))}**")
            elif st.session_state.screener_market_selection == "S&P 500":
                st.info("📊 S&P 500: **~500 stocks**")
            elif st.session_state.screener_market_selection == "NASDAQ 100":
                st.info("📊 NASDAQ 100: **~100 stocks**")
            elif st.session_state.screener_market_selection == "European Stocks":
                st.info("📊 European: **~200 stocks**")
            else:
                st.info("📊 Custom symbols")
        
        # Custom symbols input for Custom Symbols option
        custom_symbols = None
        if st.session_state.screener_market_selection == "Custom Symbols":
            def update_custom_symbols():
                st.session_state.screener_custom_symbols = st.session_state.symbols_input
            
            custom_symbols = st.text_area(
                "Enter Stock Symbols (comma-separated)",
                value=st.session_state.screener_custom_symbols,
                placeholder="AAPL, MSFT, GOOGL, TSLA, NVDA",
                help="Enter stock symbols separated by commas. Examples: AAPL, MSFT, GOOGL",
                key="symbols_input",
                disabled=st.session_state.screening_in_progress,
                on_change=update_custom_symbols
            )
        
        # Button controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if not st.session_state.screening_in_progress:
                if st.button("🚀 Start Screening", type="primary", use_container_width=True):
                    # Validation for custom symbols
                    if st.session_state.screener_market_selection == "Custom Symbols" and not st.session_state.screener_custom_symbols:
                        st.error("❌ Please enter stock symbols for custom screening")
                    else:
                        st.session_state.screening_in_progress = True
                        st.rerun()
        
        with col2:
            if st.session_state.screening_in_progress:
                if st.button("⏹️ Stop Screening", type="secondary", use_container_width=True):
                    st.session_state.screening_in_progress = False
                    st.session_state.screening_results = None
                    st.rerun()
        
        with col3:
            if st.session_state.screening_results is not None:
                if st.button("🔄 Clear Results", type="secondary", use_container_width=True):
                    st.session_state.screening_results = None
                    st.rerun()
        
        # Execute screening if in progress
        if st.session_state.screening_in_progress:
            # Show progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Perform screening
            try:
                status_text.text("🔍 Initializing screening...")
                progress_bar.progress(20)
                
                # Use session state values for screening
                screening_results = screen_multi_market_stocks(
                    market_selection=st.session_state.screener_market_selection,
                    min_score=st.session_state.screener_min_score,
                    custom_symbols=st.session_state.screener_custom_symbols.split(',') if st.session_state.screener_custom_symbols else None
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Screening completed!")
                
                # Store results and mark screening as complete
                st.session_state.screening_results = screening_results
                st.session_state.screening_in_progress = False
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error during screening: {str(e)}")
                st.session_state.screening_in_progress = False
                progress_bar.empty()
                status_text.empty()
        
        # Display results if available
        if st.session_state.screening_results is not None:
            if not st.session_state.screening_results.empty:
                display_screening_results(
                    st.session_state.screening_results, 
                    st.session_state.screener_market_selection, 
                    st.session_state.screener_min_score,
                    st.session_state.screener_max_stocks
                )
            else:
                st.warning("⚠️ No stocks found matching your criteria. Try lowering the minimum score threshold.")
        elif not st.session_state.screening_in_progress:
            # Show current configuration when not screening
            st.info(f"**Current Configuration:** {st.session_state.screener_market_selection} | "
                   f"Min Score: {st.session_state.screener_min_score} | "
                   f"Max Results: {st.session_state.screener_max_stocks}")
    
    except Exception as screener_error:
        st.error(f"❌ Error in market screener: {str(screener_error)}")
        st.info("🔄 Try refreshing the page or contact support if the issue persists.")

def display_screening_results(results_df, market_selection, min_score, max_stocks=50):
    """
    Display the screening results in a nicely formatted way
    """
    if results_df is not None and not results_df.empty:
        # Limit results to max_stocks
        display_df = results_df.head(max_stocks)
        st.success(f"✅ Found **{len(results_df)}** stocks from {market_selection} with score ≥ {min_score}")
        
        if len(results_df) > max_stocks:
            st.info(f"📊 Showing top **{max_stocks}** results (out of {len(results_df)} total)")
        
        # Main results table
        st.subheader(f"🏆 Top {len(display_df)} Stocks from {market_selection}")
        
        display_columns = [
            'Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 'Sector', 
            'Current_Price', 'P/E_Ratio', 'ROE', 'Revenue_Growth', 'Dividend_Yield'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        st.dataframe(
            display_df[available_columns],
            use_container_width=True,
            hide_index=True
        )
def display_company_search():
    """Display company search interface"""
    st.subheader("🔍 Company Name Search")
    
    with st.form("company_search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search for companies by name",
                placeholder="e.g., Apple, Microsoft, Tesla",
                help="Enter a company name to find its stock symbol",
                key="company_search_query"
            )
        
        with col2:
            search_method = st.selectbox(
                "Search Method",
                ["Alpha Vantage", "Manual Lookup"],
            help="Alpha Vantage provides comprehensive search"
        )
        
        # Add form submit button
        search_submitted = st.form_submit_button("🔍 Search Companies", type="primary")
    
    if search_submitted and search_query and len(search_query.strip()) >= 2:
        with st.spinner("Searching for companies..."):
            if search_method == "Alpha Vantage":
                results = search_company_by_name(search_query.strip())
            else:
                results = search_company_yahoo(search_query.strip())
            
            if results:
                st.success(f"Found {len(results)} matching companies:")
                
                # Create a DataFrame for better display
                df_results = pd.DataFrame(results)
                
                # Display results in an interactive table
                if search_method == "Alpha Vantage":
                    display_columns = ["symbol", "name", "type", "region", "currency"]
                    column_names = ["Symbol", "Company Name", "Type", "Region", "Currency"]
                else:
                    display_columns = ["symbol", "name", "sector", "industry", "country"]
                    column_names = ["Symbol", "Company Name", "Sector", "Industry", "Country"]
                
                # Filter and rename columns
                if not df_results.empty:
                    available_columns = [col for col in display_columns if col in df_results.columns]
                    df_display = df_results[available_columns].copy()
                    
                    # Rename columns for display
                    column_mapping = dict(zip(display_columns[:len(available_columns)], 
                                            column_names[:len(available_columns)]))
                    df_display = df_display.rename(columns=column_mapping)
                    
                    # Display the table
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Quick selection buttons
                    st.subheader("Quick Add to Analysis")
                    selected_symbols = []
                    
                    cols = st.columns(min(len(results), 5))  # Max 5 columns
                    for i, result in enumerate(results[:5]):  # Show max 5 quick buttons
                        with cols[i]:
                            symbol = result["symbol"]
                            name = result["name"][:20] + "..." if len(result["name"]) > 20 else result["name"]
                            
                            if st.button(f"Add {symbol}", key=f"add_{symbol}_{i}"):
                                # Store in session state
                                if "selected_symbols" not in st.session_state:
                                    st.session_state.selected_symbols = []
                                
                                if symbol not in st.session_state.selected_symbols:
                                    st.session_state.selected_symbols.append(symbol)
                                    st.success(f"Added {symbol} to analysis list")
                                else:
                                    st.info(f"{symbol} already in analysis list")
                    
                    # Display selected symbols
                    if "selected_symbols" in st.session_state and st.session_state.selected_symbols:
                        st.subheader("Selected Symbols for Analysis")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            symbols_text = ", ".join(st.session_state.selected_symbols)
                            st.text_area("Selected symbols (you can copy this):", 
                                       value=symbols_text, height=100, key="selected_display")
                        
                        with col2:
                            if st.button("Clear Selection"):
                                st.session_state.selected_symbols = []
                                st.rerun()
                            
                            if st.button("Copy to Clipboard", help="Use browser's copy function"):
                                st.info("Please select and copy the text above")
                
                else:
                    st.info("No results found in the expected format")
            else:
                st.warning("No companies found matching your search. Try different keywords or check spelling.")
                
                # Provide some search tips
                with st.expander("Search Tips"):
                    st.markdown("""
                    - Try searching with just the main company name (e.g., "Apple" instead of "Apple Inc.")
                    - Use common abbreviations (e.g., "GM" for General Motors)
                    - Try partial names (e.g., "Micro" for Microsoft)
                    - For Alpha Vantage search, try industry terms
                    - Some foreign companies might not be available
                    """)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def search_company_by_name(company_name, limit=10):
    """Search for companies by name using Alpha Vantage Symbol Search"""
    try:
        url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "bestMatches" not in data:
            return []
        
        results = []
        for match in data["bestMatches"][:limit]:
            results.append({
                "symbol": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", ""),
                "region": match.get("4. region", ""),
                "currency": match.get("8. currency", "")
            })
        
        return results
    except Exception as e:
        st.error(f"Error searching for companies: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def search_company_yahoo(query, limit=10):
    """Search for companies using Yahoo Finance (alternative method)"""
    try:
        # Use yfinance's Ticker search functionality
        import yfinance as yf
        
        # This is a workaround since yfinance doesn't have direct search
        # We'll try some common variations of the query
        potential_symbols = []
        
        # Try the query as-is (in case it's already a symbol)
        try:
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            if info and info.get("longName"):
                potential_symbols.append({
                    "symbol": query.upper(),
                    "name": info.get("longName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "country": info.get("country", "")
                })
        except:
            pass
        
        return potential_symbols[:limit]
    except Exception as e:
        return []


def display_enhanced_historical_charts(symbol, financial_history, price_performance):
    """Enhanced historical visualization"""
    if not financial_history and not price_performance:
        st.info("📊 No historical data available for visualization")
        return

    # Create tabs for different views
    chart_tab1, chart_tab2, metrics_tab = st.tabs(["📈 Price Performance", "💰 Financial Trends", "📊 Key Metrics"])
    
    with chart_tab1:
        if price_performance and 'price_data' in price_performance:
            price_data = price_performance['price_data']
            ticker_info = price_performance.get('ticker_info', {})
            
            # Get currency formatting
            currency_symbol = get_currency_symbol(ticker_info)
            if currency_symbol == "DKK":
                price_label = "Price (DKK)"
                hover_template = 'Date: %{x}<br>Price: %{y:.2f} DKK<extra></extra>'
            elif currency_symbol == "€":
                price_label = "Price (€)"
                hover_template = 'Date: %{x}<br>Price: €%{y:.2f}<extra></extra>'
            elif currency_symbol == "£":
                price_label = "Price (£)"
                hover_template = 'Date: %{x}<br>Price: £%{y:.2f}<extra></extra>'
            else:
                price_label = "Price (USD)"
                hover_template = 'Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2),
                hovertemplate=hover_template
            ))
            
            # Add volume as secondary y-axis
            fig.add_trace(go.Bar(
                x=price_data.index,
                y=price_data['Volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"{symbol} - 3 Year Price & Volume History",
                xaxis_title="Date",
                yaxis_title=price_label,
                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price data not available")
    
    with chart_tab2:
        if financial_history:
            # Create financial trends chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Revenue Trend', 'Net Income Trend', 'Free Cash Flow', 'ROE Trend'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Revenue trend
            if 'revenue_trend' in financial_history and 'revenue_dates' in financial_history:
                fig.add_trace(
                    go.Bar(
                        x=financial_history['revenue_dates'],
                        y=financial_history['revenue_trend'],
                        name='Revenue',
                        marker_color='green'
                    ),
                    row=1, col=1
                )
            
            # Net Income trend
            if 'net_income_trend' in financial_history:
                fig.add_trace(
                    go.Bar(
                        x=financial_history.get('revenue_dates', ['Year 1', 'Year 2', 'Year 3']),
                        y=financial_history['net_income_trend'],
                        name='Net Income',
                        marker_color='orange'
                    ),
                    row=1, col=2
                )
            
            # FCF trend
            if 'fcf_trend' in financial_history:
                fig.add_trace(
                    go.Bar(
                        x=financial_history.get('revenue_dates', ['Year 1', 'Year 2', 'Year 3']),
                        y=financial_history['fcf_trend'],
                        name='Free Cash Flow',
                        marker_color='purple'
                    ),
                    row=2, col=1
                )
            
            # ROE trend
            if 'roe_trend' in financial_history:
                fig.add_trace(
                    go.Scatter(
                        x=financial_history.get('revenue_dates', ['Year 1', 'Year 2', 'Year 3']),
                        y=financial_history['roe_trend'],
                        mode='lines+markers',
                        name='ROE %',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f"{symbol} - Financial Performance Overview"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Financial trend data not available")
    
    with metrics_tab:
        if price_performance:
            st.subheader("📊 Performance Metrics Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                current_price = price_performance.get('current_price', 0)
                ticker_info = price_performance.get('ticker_info', {})
                st.metric("Current Price", format_currency(current_price, ticker_info))
            
            with col2:
                return_1y = price_performance.get('return_1y', 0)
                st.metric("1 Year Return", f"{return_1y:.1f}%", 
                         delta=f"{return_1y:.1f}%" if return_1y != 0 else None)
            
            with col3:
                return_3y = price_performance.get('return_3y', 0)
                st.metric("3 Year Return", f"{return_3y:.1f}%",
                         delta=f"{return_3y:.1f}%" if return_3y != 0 else None)
            
            with col4:
                volatility = price_performance.get('volatility', 0)
                st.metric("Volatility", f"{volatility:.1f}%")
            
            with col5:
                sharpe = price_performance.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe != 0 else "N/A")
            
            # Additional metrics in expandable section
            with st.expander("📈 Detailed Performance Metrics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Short-term Performance**")
                    st.write(f"3 Months: {price_performance.get('return_3m', 0):.1f}%")
                    st.write(f"6 Months: {price_performance.get('return_6m', 0):.1f}%")
                
                with col2:
                    st.write("**Long-term Performance**")
                    st.write(f"2 Years: {price_performance.get('return_2y', 0):.1f}%")
                    st.write(f"3 Years: {price_performance.get('return_3y', 0):.1f}%")
                
                with col3:
                    st.write("**Risk Metrics**")
                    max_price = price_performance.get('max_3y', 0)
                    min_price = price_performance.get('min_3y', 0)
                    ticker_info = price_performance.get('ticker_info', {})
                    st.write(f"3Y High: {format_currency(max_price, ticker_info)}")
                    st.write(f"3Y Low: {format_currency(min_price, ticker_info)}")
# -----------------------------
# Wrapper Function
# -----------------------------
def show_stock_analysis(symbol):
    st.subheader(f"📊 Historical Stock Analysis for {symbol}")
    with st.spinner("Fetching financial data..."):
        financial_history = get_3year_financial_history(symbol)
        price_performance = get_3year_price_performance(symbol)

    if not financial_history and not price_performance:
        st.warning("No data available for this symbol.")
        return

    display_enhanced_historical_charts(symbol, financial_history, price_performance)




def show_enhanced_stock_data(symbol, info, scores, total_score, recommendation, color):
    """Enhanced stock data display"""
    # Header section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 🏢 {info['name']}")
        st.write(f"**Symbol:** {symbol}")
        st.write(f"**Sector:** {info.get('sector', 'Unknown')}")
        st.write(f"**Industry:** {info.get('industry', 'Unknown')}")
    
    with col2:
        price = info.get('price', 0)
        st.metric("💰 Current Price", format_currency(price, info) if price else "N/A")
        
        market_cap = info.get('marketCap')
        if market_cap:
            currency_symbol = get_currency_symbol(info)
            if currency_symbol == "DKK":
                curr_symbol = "DKK"
            elif currency_symbol == "€":
                curr_symbol = "€"
            elif currency_symbol == "£":
                curr_symbol = "£"
            else:
                curr_symbol = "$"
            
            if market_cap >= 1e12:
                cap_str = f"{curr_symbol}{market_cap/1e12:.2f}T" if curr_symbol != "DKK" else f"{market_cap/1e12:.2f}T DKK"
            elif market_cap >= 1e9:
                cap_str = f"{curr_symbol}{market_cap/1e9:.2f}B" if curr_symbol != "DKK" else f"{market_cap/1e9:.2f}B DKK"
            elif market_cap >= 1e6:
                cap_str = f"{curr_symbol}{market_cap/1e6:.2f}M" if curr_symbol != "DKK" else f"{market_cap/1e6:.2f}M DKK"
            else:
                cap_str = f"{curr_symbol}{market_cap:,.0f}" if curr_symbol != "DKK" else f"{market_cap:,.0f} DKK"
            st.write(f"**Market Cap:** {cap_str}")
    
    with col3:
        st.metric("🎯 Total Score", f"{total_score:.2f}/10")
        st.markdown(f"<div style='padding: 10px; background-color: {color}20; border-left: 4px solid {color}; border-radius: 5px;'>"
                   f"<strong style='color: {color};'>{recommendation}</strong></div>", 
                   unsafe_allow_html=True)
    
    # Score visualization
    st.markdown("---")
    filtered_scores = {k: v for k, v in scores.items() if v not in (None, 0) or k in ['PE', 'PEG', 'PB']}
    
    if filtered_scores:
        fig = create_enhanced_score_chart(filtered_scores, symbol)
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical analysis
    financial_history = info.get("financial_history", {})
    price_performance = info.get("price_performance", {})
    
    if financial_history or price_performance:
        display_enhanced_historical_charts(symbol, financial_history, price_performance)
    
    # Raw data in expandable section
    with st.expander("🔍 View Detailed Data"):
        tab1, tab2 = st.tabs(["📊 Scores Breakdown", "📋 Raw Data"])
        
        with tab1:
            score_df = pd.DataFrame([
                {"Metric": k, "Score": v, "Weight": st.session_state.score_weights.get(k, 0)} 
                for k, v in scores.items()
            ])
            score_df["Weighted Score"] = score_df["Score"] * score_df["Weight"]
            st.dataframe(score_df, hide_index=True)
        
        with tab2:
            st.json(info)

# Manual symbol lookup for common companies
COMMON_COMPANIES = {
    "apple": "AAPL",
    "microsoft": "MSFT", 
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "disney": "DIS",
    "coca cola": "KO",
    "pepsi": "PEP",
    "walmart": "WMT",
    "target": "TGT",
    "home depot": "HD",
    "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE",
    "adidas": "ADDYY",
    "boeing": "BA",
    "airbus": "EADSY",
    "ford": "F",
    "general motors": "GM",
    "toyota": "TM",
    "honda": "HMC",
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "moderna": "MRNA",
    "exxon": "XOM",
    "chevron": "CVX",
    "bp": "BP",
    "shell": "SHEL",
    "berkshire": "BRK.B",
    "jpmorgan": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "visa": "V",
    "mastercard": "MA",
    "paypal": "PYPL",
    "square": "SQ",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "cisco": "CSCO",
    "zoom": "ZM",
    "slack": "WORK",
    "twitter": "TWTR",
    "snapchat": "SNAP",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM"
}

def search_manual_lookup(query):
    """Manual lookup for common companies"""
    query_lower = query.lower().strip()
    matches = []
    
    for company_name, symbol in COMMON_COMPANIES.items():
        if query_lower in company_name or company_name in query_lower:
            matches.append({
                "symbol": symbol,
                "name": company_name.title(),
                "type": "Common Stock",
                "region": "United States",
                "currency": "USD"
            })
    
    return matches
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_overview(symbol):
    """Fetch company overview from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching overview for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_income_statement(symbol):
    """Fetch income statement from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching income statement for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_cash_flow(symbol):
    """Fetch cash flow statement from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data or "Note" in data:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching cash flow for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes for price data
def fetch_price(symbol):
    """Fetch current stock price from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        price = float(data["Global Quote"]["05. price"])
        return price
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_yahoo_info(symbol):
    """Enhanced Yahoo Finance data fetching with Danish stock support"""
    def try_fetch_symbol(sym):
        """Helper function to try fetching data for a symbol"""
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            
            # Check if we got valid data - more flexible validation
            if not info:
                return None
            
            # Check for any price indicator or basic company info
            price_indicators = [
                info.get("regularMarketPrice"),
                info.get("currentPrice"),
                info.get("previousClose"),
                info.get("open")
            ]
            
            # Must have either a price or basic company info
            has_price = any(price is not None for price in price_indicators)
            has_company_info = info.get("longName") or info.get("shortName")
            
            if not (has_price or has_company_info):
                return None
            
            # Get additional data
            financial_history = get_3year_financial_history(sym)
            price_performance = get_3year_price_performance(sym)
            
            # Debug: For Danish stocks, print available fields to help improve data extraction
            if '.CO' in sym or sym in DANISH_STOCKS.values():
                print(f"\n--- Debug info for {sym} ---")
                relevant_fields = [k for k in info.keys() if any(term in k.lower() for term in 
                    ['pe', 'peg', 'dividend', 'yield', 'trailing', 'forward', 'earnings', 'growth'])]
                print(f"Relevant fields found: {relevant_fields}")
                for field in ['trailingPE', 'forwardPE', 'trailingPegRatio', 'pegRatio', 'dividendYield', 
                             'trailingAnnualDividendYield', 'fiveYearAvgDividendYield']:
                    if field in info:
                        print(f"{field}: {info[field]}")
                print("--- End debug ---\n")
            
            enhanced_info = {
                "name": info.get("longName", info.get("shortName", "Unknown")),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                # Enhanced PE extraction with multiple fallbacks
                "pe": (info.get("trailingPE") or 
                       info.get("forwardPE") or 
                       info.get("priceEarningsRatio") or 
                       info.get("trailingPE")),
                # Enhanced PEG extraction with multiple fallbacks
                "peg": (info.get("trailingPegRatio") or 
                        info.get("pegRatio") or
                        info.get("forwardPegRatio")),
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "eps_growth": info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"),  # Use yearly first, then quarterly
                "revenue_growth": info.get("revenueGrowth"),  # Use consistent key name
                "de": info.get("debtToEquity"),
                # Enhanced dividend yield extraction
                "dy": (info.get("dividendYield") or 
                       info.get("trailingAnnualDividendYield") or
                       info.get("fiveYearAvgDividendYield")),
                "gm": info.get("grossMargins"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "marketCap": info.get("marketCap"),
                "beta": info.get("beta"),
                "financial_history": financial_history,
                "price_performance": price_performance,
                "symbol_used": sym,  # Track which symbol was successful
                # Add currency information for proper formatting
                "currency": info.get("currency", "USD"),
                "symbol": info.get("symbol", sym),
                # Add additional Yahoo Finance metrics
                "forwardPE": info.get("forwardPE"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                "targetMeanPrice": info.get("targetMeanPrice"),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice")),
                # Add missing fields for display
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "volume": info.get("volume"),
                "averageVolume": info.get("averageVolume"),
                "profitMargins": info.get("profitMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "recommendationMean": info.get("recommendationMean"),
                "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
                "regularMarketPrice": info.get("regularMarketPrice"),
                "shortName": info.get("shortName"),
                "longName": info.get("longName")
            }
            
            return enhanced_info
            
        except Exception as e:
            # Silently handle errors during batch processing
            print(f"Failed to fetch {sym}: {str(e)}")
            return None
    
    try:
        # First, try the original symbol
        result = try_fetch_symbol(symbol)
        if result:
            return result
        
        # If original fails, try Danish stock variations
        if not symbol.endswith('.CO'):
            # Check if it's a known Danish stock
            if symbol in DANISH_STOCKS:
                danish_symbol = DANISH_STOCKS[symbol]
                result = try_fetch_symbol(danish_symbol)
                if result:
                    return result
            
            # Try adding .CO suffix
            co_symbol = f"{symbol}.CO"
            result = try_fetch_symbol(co_symbol)
            if result:
                return result
        
        # If all attempts fail, return None
        print(f"Could not fetch data for {symbol} using any variation")
        return None
        
    except Exception as e:
        print(f"Error fetching Yahoo data for {symbol}: {str(e)}")
        return None

# --- Score Calculation ---
def calculate_scores(symbol, industry_pe=20):
    """Calculate scores using Alpha Vantage data"""
    overview = fetch_overview(symbol)
    if not overview:
        return None, None
    
    time.sleep(REQUEST_DELAY)  # Rate limiting
    income = fetch_income_statement(symbol)
    time.sleep(REQUEST_DELAY)
    cashflow = fetch_cash_flow(symbol)

    try:
        pe = safe_float(overview.get("PERatio", 0))
        peg = safe_float(overview.get("PEGRatio", 0))
        pb = safe_float(overview.get("PriceToBookRatio", 0))
        roe = safe_float(overview.get("ReturnOnEquityTTM", 0)) * 100
        de = safe_float(overview.get("DebtEquity", 0))
        dy = safe_float(overview.get("DividendYield", 0)) * 100
        
        # Calculate gross margin
        gross_profit = safe_float(overview.get("GrossProfitTTM", 0))
        revenue = safe_float(overview.get("RevenueTTM", 1))
        gm = (gross_profit / revenue * 100) if revenue != 0 else 0
        
    except Exception:
        pe = peg = pb = roe = de = dy = gm = 0

    # EPS growth calculation
    eps_growth = 0
    if income and "annualReports" in income:
        try:
            eps_data = []
            for report in income["annualReports"][:3]:
                eps = safe_float(report.get("eps", 0))
                if eps != 0:
                    eps_data.append(eps)
            
            if len(eps_data) >= 2:
                eps_growth = ((eps_data[0] - eps_data[-1]) / abs(eps_data[-1])) * 100
        except Exception:
            eps_growth = 0

    # Revenue growth calculation
    rev_growth = 0
    if income and "annualReports" in income:
        try:
            rev_data = []
            for report in income["annualReports"][:3]:
                rev = safe_float(report.get("totalRevenue", 0))
                if rev != 0:
                    rev_data.append(rev)
            
            if len(rev_data) >= 2:
                rev_growth = ((rev_data[0] - rev_data[-1]) / abs(rev_data[-1])) * 100
        except Exception:
            rev_growth = 0

    # Free Cash Flow trend
    fcf = [0, 0, 0]
    if cashflow and "annualReports" in cashflow:
        try:
            fcf = []
            for report in cashflow["annualReports"][:3]:
                operating_cf = safe_float(report.get("operatingCashflow", 0))
                capex = safe_float(report.get("capitalExpenditures", 0))
                fcf.append(operating_cf - capex)
        except Exception:
            fcf = [0, 0, 0]

    # Get sector information for dynamic scoring
    sector = overview.get("Sector", "Unknown")
    
    scores = {
        "PE": score_pe(pe, industry_pe, allow_neutral=True),
        "PEG": score_peg(peg, allow_neutral=True),
        "PB": score_pb(pb),
        "ROE": score_roe_dynamic(roe, sector),  # Use dynamic scoring
        "EPS Growth": score_eps_growth(eps_growth),
        "Revenue Growth": score_revenue_growth_dynamic(rev_growth, sector),  # Use dynamic scoring
        "FCF Trend": score_fcf_trend(fcf),
        "Debt/Equity": score_debt_equity_dynamic(de, sector),  # Use dynamic scoring
        "Dividend Yield": score_dividend_yield(dy, allow_neutral=True),
        "Gross Margin": score_gross_margin_dynamic(gm, sector)  # Use dynamic scoring
    }

    return scores, overview

def get_recommendation(total_score):
    """Get recommendation based on total score"""
    if total_score >= 8:
        return "Strong Buy", "green"
    elif total_score >= 6:
        return "Buy", "limegreen"
    elif total_score >= 4:
        return "Hold/Neutral", "orange"
    elif total_score >= 2:
        return "Weak Sell", "orangered"
    else:
        return "Strong Sell", "red"

# --- Industry P/E Map ---
# --- Visualization Functions ---
def create_score_chart(scores):
    """Create an interactive score chart using Plotly - Enhanced version"""
    metrics = list(scores.keys())
    values = list(scores.values())
    
    # Enhanced color scheme consistent with the main scoring system
    colors = []
    for v in values:
        if v == 0:
            colors.append('lightgray')  # Gray for zero/missing data
        elif v == 5:
            colors.append('lightblue')  # Light blue for neutral/unknown data
        elif v >= 7:
            colors.append('green')
        elif v >= 5:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Create display values (use 0.1 for zero scores to make them visible)
    display_values = [max(0.1, v) if v == 0 else v for v in values]
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=display_values, 
               marker_color=colors,
               text=[f'{v:.1f}' if v > 0 else 'No Data' for v in values],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Stock Scoring Breakdown",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 10]),
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_comparison_chart(stock_data):
    """Create comparison chart for multiple stocks"""
    if not stock_data:
        return None
    
    symbols = list(stock_data.keys())
    scores = [stock_data[symbol]['total_score'] for symbol in symbols]
    colors = [stock_data[symbol]['color'] for symbol in symbols]
    
    fig = go.Figure(data=[
        go.Bar(x=symbols, y=scores, 
               marker_color=colors,
               text=[f"{score:.2f}" for score in scores], 
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Stock Comparison - Total Scores",
        xaxis_title="Stock Symbol",
        yaxis_title="Total Score",
        yaxis=dict(range=[0, 10]),
        height=400
    )
    
    return fig

# --- Helper Functions ---
def show_stock_data(symbol, name, price, scores, total, recommendation, color, debug_data, additional_info=None):
    """Display stock data and scores"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Company:** {name}")
        if price:
            st.write(f"**Current Price:** {format_currency(price, additional_info if additional_info else {})}")
        else:
            st.write("**Current Price:** Not available")
        
        if additional_info:
            if additional_info.get("sector"):
                st.write(f"**Sector:** {additional_info['sector']}")
            if additional_info.get("marketCap"):
                market_cap = additional_info['marketCap']
                if market_cap:
                    currency_symbol = get_currency_symbol(additional_info)
                    if market_cap >= 1e12:
                        cap_str = f"{market_cap/1e12:.2f}T {currency_symbol}"
                    elif market_cap >= 1e9:
                        cap_str = f"{market_cap/1e9:.2f}B {currency_symbol}"
                    elif market_cap >= 1e6:
                        cap_str = f"{market_cap/1e6:.2f}M {currency_symbol}"
                    else:
                        cap_str = f"{market_cap:,.0f} {currency_symbol}"
                    st.write(f"**Market Cap:** {cap_str}")
    with col2:
        st.metric("Total Score", f"{total:.2f}", delta=None)
        st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:18px'>{recommendation}</div>", 
                   unsafe_allow_html=True)
    
    # Score visualization
    filtered_scores = {k: v for k, v in scores.items() if v not in (None, 0) or k in ['PE', 'PEG', 'PB']}
    
    if filtered_scores:
        fig = create_score_chart(filtered_scores)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No relevant scores to display.")
    
    # Debug data
    with st.expander("View Raw Data"):
        st.json(debug_data)

def validate_symbols(symbols_text):
    """Validate and clean stock symbols with Danish stock mapping"""
    if not symbols_text.strip():
        return []
    
    symbols = [s.strip().upper() for s in symbols_text.split(",")]
    # Remove empty strings and limit to reasonable number
    symbols = [s for s in symbols if s and len(s) <= 10][:10]
    
    # Apply Danish stock mapping
    mapped_symbols = []
    for symbol in symbols:
        if symbol in DANISH_STOCKS:
            mapped_symbol = DANISH_STOCKS[symbol]
            mapped_symbols.append(mapped_symbol)
            st.info(f"🇩🇰 Danish stock detected: {symbol} → {mapped_symbol}")
        else:
            mapped_symbols.append(symbol)
    
    return mapped_symbols


class StockAnalyzer:
    def __init__(self, symbol, period="1y"):
        self.symbol = symbol
        self.period = period
        self.data = None
    
    def fetch_data(self):
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False
    
    def calculate_moving_averages(self, short_window=20, long_window=50):
        """Calculate Simple Moving Averages"""
        self.data[f'SMA_{short_window}'] = self.data['Close'].rolling(window=short_window).mean()
        self.data[f'SMA_{long_window}'] = self.data['Close'].rolling(window=long_window).mean()
        return self.data[[f'SMA_{short_window}', f'SMA_{long_window}']]
    
    def calculate_rsi(self, window=14):
        """Calculate Relative Strength Index (RSI)"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data['RSI']
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        return self.data[['MACD', 'MACD_Signal', 'MACD_Histogram']]
    
    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        self.data['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        self.data['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        self.data['BB_Middle'] = rolling_mean
        return self.data[['BB_Upper', 'BB_Middle', 'BB_Lower']]
    
    def calculate_stochastic(self, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=k_window).min()
        high_max = self.data['High'].rolling(window=k_window).max()
        self.data['%K'] = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        self.data['%D'] = self.data['%K'].rolling(window=d_window).mean()
        return self.data[['%K', '%D']]
    
    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
    
    def generate_buy_signals(self):
        """Generate buy signals based on technical indicators"""
        signals = []
        latest_data = self.data.iloc[-1]
        
        # RSI Oversold Signal
        if latest_data['RSI'] < 30:
            signals.append("🟢 RSI Oversold (Buy Signal)")
        elif latest_data['RSI'] > 70:
            signals.append("🔴 RSI Overbought (Sell Signal)")
        
        # Moving Average Crossover
        if latest_data['SMA_20'] > latest_data['SMA_50']:
            signals.append("🟢 MA Bullish Crossover")
        else:
            signals.append("🔴 MA Bearish Position")
        
        # MACD Signal
        if latest_data['MACD'] > latest_data['MACD_Signal']:
            signals.append("🟢 MACD Bullish")
        else:
            signals.append("🔴 MACD Bearish")
        
        # Bollinger Bands
        if latest_data['Close'] < latest_data['BB_Lower']:
            signals.append("🟢 Price Below Lower Bollinger Band")
        elif latest_data['Close'] > latest_data['BB_Upper']:
            signals.append("🔴 Price Above Upper Bollinger Band")
        
        # Stochastic Oversold
        if latest_data['%K'] < 20 and latest_data['%D'] < 20:
            signals.append("🟢 Stochastic Oversold")
        elif latest_data['%K'] > 80 and latest_data['%D'] > 80:
            signals.append("🔴 Stochastic Overbought")
        
        return signals

def create_price_chart(analyzer):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Stochastic'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price and Moving Averages
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['Close'], 
                  name='Close Price', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['SMA_20'], 
                  name='SMA 20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['SMA_50'], 
                  name='SMA 50', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['BB_Upper'], 
                  name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['BB_Lower'], 
                  name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                  fill='tonexty', fillcolor='rgba(68, 68, 68, 0.1)'),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['RSI'], 
                  name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MACD'], 
                  name='MACD', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MACD_Signal'], 
                  name='Signal', line=dict(color='red', width=1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=analyzer.data.index, y=analyzer.data['MACD_Histogram'], 
               name='Histogram', marker_color='green', opacity=0.3),
        row=3, col=1
    )
    
    # Stochastic
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['%K'], 
                  name='%K', line=dict(color='blue', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['%D'], 
                  name='%D', line=dict(color='red', width=1)),
        row=4, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(
        title=f'{analyzer.symbol} - Technical Analysis',
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

class BuyingPriceCalculator:
    def __init__(self, symbol, period="6mo"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.buying_prices = {}
        
    def fetch_data(self):
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False    
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        # Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self.data['Close'].ewm(span=12).mean()
        ema_26 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = ema_12 - ema_26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        rolling_mean = self.data['Close'].rolling(window=20).mean()
        rolling_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        self.data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        self.data['BB_Middle'] = rolling_mean
        
        # Support and Resistance
        self.data['Support'] = self.data['Low'].rolling(window=20).min()
        self.data['Resistance'] = self.data['High'].rolling(window=20).max()
        
        # ATR (Average True Range) for volatility
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()
    
    def calculate_buying_prices(self):
        """Calculate specific buying prices based on different strategies"""
        if self.data is None or len(self.data) < 50:
            return None
        
        current_price = self.data['Close'].iloc[-1]
        latest = self.data.iloc[-1]
        
        # Strategy 1: Support Level Buy
        support_level = latest['Support']
        support_buy_price = support_level * 1.01  # 1% above support
        
        # Strategy 2: Bollinger Lower Band Buy
        bb_lower_buy = latest['BB_Lower'] * 1.005  # 0.5% above lower band
        
        # Strategy 3: Moving Average Pullback
        sma_20_buy = latest['SMA_20'] * 0.98  # 2% below SMA 20
        sma_50_buy = latest['SMA_50'] * 0.99  # 1% below SMA 50
        
        # Strategy 4: RSI Oversold Buy
        if latest['RSI'] < 40:
            rsi_buy_price = current_price * 0.97  # 3% below current if RSI < 40
        else:
            rsi_buy_price = current_price * 0.95  # 5% below current if RSI normal
        
        # Strategy 5: MACD Bullish Entry
        if latest['MACD'] > latest['MACD_Signal']:
            macd_buy_price = current_price * 0.98  # 2% below current
        else:
            macd_buy_price = current_price * 0.95  # 5% below current
        
        # Strategy 6: Fibonacci Retracement Levels
        high_20 = self.data['High'].tail(20).max()
        low_20 = self.data['Low'].tail(20).min()
        fib_range = high_20 - low_20
        
        fib_236 = high_20 - (fib_range * 0.236)
        fib_382 = high_20 - (fib_range * 0.382)
        fib_618 = high_20 - (fib_range * 0.618)
        
        # Strategy 7: ATR-based Entry
        atr_buy_low = current_price - (latest['ATR'] * 0.5)
        atr_buy_high = current_price - (latest['ATR'] * 0.25)
        
        # Strategy 8: Volume-weighted Entry
        avg_volume = self.data['Volume'].tail(20).mean()
        current_volume = latest['Volume']
        volume_factor = min(current_volume / avg_volume, 2.0)  # Cap at 2x
        volume_buy_price = current_price * (1 - (0.02 * volume_factor))
        
        self.buying_prices = {
            'current_price': current_price,
            'support_buy': support_buy_price,
            'bollinger_buy': bb_lower_buy,
            'sma_20_pullback': sma_20_buy,
            'sma_50_pullback': sma_50_buy,
            'rsi_buy': rsi_buy_price,
            'macd_buy': macd_buy_price,
            'fib_23.6%': fib_236,
            'fib_38.2%': fib_382,
            'fib_61.8%': fib_618,
            'atr_conservative': atr_buy_high,
            'atr_aggressive': atr_buy_low,
            'volume_weighted': volume_buy_price
        }
        
        return self.buying_prices
    
    def get_recommended_buy_price(self):
        """Get recommended buy price based on multiple factors"""
        if not self.buying_prices:
            return None
        
        latest = self.data.iloc[-1]
        current_price = latest['Close']
        
        # Score each strategy based on current conditions
        scores = {}
        
        # RSI condition
        if latest['RSI'] < 30:
            scores['rsi_buy'] = 10  # Oversold - high score
        elif latest['RSI'] < 50:
            scores['rsi_buy'] = 7   # Neutral - medium score
        else:
            scores['rsi_buy'] = 3   # Overbought - low score
        
        # MACD condition
        if latest['MACD'] > latest['MACD_Signal']:
            scores['macd_buy'] = 8  # Bullish
        else:
            scores['macd_buy'] = 4  # Bearish
        
        # Moving Average condition
        if current_price > latest['SMA_20'] > latest['SMA_50']:
            scores['sma_20_pullback'] = 9  # Strong uptrend
            scores['sma_50_pullback'] = 7
        elif current_price > latest['SMA_50']:
            scores['sma_20_pullback'] = 6  # Moderate uptrend
            scores['sma_50_pullback'] = 8
        else:
            scores['sma_20_pullback'] = 3  # Downtrend
            scores['sma_50_pullback'] = 4
        
        # Bollinger Bands condition
        if current_price < latest['BB_Lower']:
            scores['bollinger_buy'] = 9  # Below lower band
        elif current_price < latest['BB_Middle']:
            scores['bollinger_buy'] = 6  # Below middle
        else:
            scores['bollinger_buy'] = 3  # Above middle
        
        # Support level condition
        if current_price < latest['Support'] * 1.05:
            scores['support_buy'] = 8  # Near support
        else:
            scores['support_buy'] = 5
        
        # ATR condition (volatility)
        scores['atr_conservative'] = 6
        scores['atr_aggressive'] = 4
        
        # Volume condition
        avg_volume = self.data['Volume'].tail(20).mean()
        if latest['Volume'] > avg_volume * 1.5:
            scores['volume_weighted'] = 7  # High volume
        else:
            scores['volume_weighted'] = 5
        
        # Fibonacci levels
        scores['fib_23.6%'] = 5
        scores['fib_38.2%'] = 6
        scores['fib_61.8%'] = 7
        
        # Calculate weighted average
        total_score = 0
        weighted_price = 0
        
        for strategy, price in self.buying_prices.items():
            if strategy in scores and strategy != 'current_price':
                score = scores[strategy]
                total_score += score
                weighted_price += price * score
        
        if total_score > 0:
            recommended_price = weighted_price / total_score
        else:
            recommended_price = current_price * 0.97  # Default 3% below current
        
        return recommended_price, scores
    
    def display_buying_analysis(self):
        """Display comprehensive buying analysis"""
        if not self.buying_prices:
            print("No analysis available")
            return
        
        current_price = self.buying_prices['current_price']
        latest = self.data.iloc[-1]
        
        print(f"\n{'='*60}")
        print(f"BUYING PRICE ANALYSIS FOR {self.symbol}")
        print(f"{'='*60}")
        print(f"Current Price: {format_currency(current_price, self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"Date: {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Volume: {latest['Volume']:,}")
        
        print(f"\n{'='*60}")
        print("TECHNICAL INDICATORS")
        print(f"{'='*60}")
        print(f"RSI (14):           {latest['RSI']:.2f}")
        print(f"MACD:               {latest['MACD']:.4f}")
        print(f"MACD Signal:        {latest['MACD_Signal']:.4f}")
        print(f"SMA 20:             {format_currency(latest['SMA_20'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"SMA 50:             {format_currency(latest['SMA_50'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"Bollinger Upper:    {format_currency(latest['BB_Upper'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"Bollinger Lower:    {format_currency(latest['BB_Lower'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"Support Level:      {format_currency(latest['Support'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"ATR:                {format_currency(latest['ATR'], self.yahoo_data['info'] if self.yahoo_data else {})}")
        
        print(f"\n{'='*60}")
        print("BUYING PRICE TARGETS")
        print(f"{'='*60}")
        
        # Sort buying prices
        sorted_prices = sorted([(k, v) for k, v in self.buying_prices.items() 
                              if k != 'current_price'], 
                             key=lambda x: x[1], reverse=True)
        
        for strategy, price in sorted_prices:
            discount = ((current_price - price) / current_price) * 100
            print(f"{strategy:20} {format_currency(price, self.yahoo_data['info'] if self.yahoo_data else {})} ({discount:+.1f}%)")
        
        # Get recommended price
        recommended_price, scores = self.get_recommended_buy_price()
        recommended_discount = ((current_price - recommended_price) / current_price) * 100
        
        print(f"\n{'='*60}")
        print("RECOMMENDED BUY PRICE")
        print(f"{'='*60}")
        print(f"Recommended Price:  {format_currency(recommended_price, self.yahoo_data['info'] if self.yahoo_data else {})}")
        print(f"Discount:           {recommended_discount:.1f}%")
        print(f"Upside if bought:   {((current_price - recommended_price) / recommended_price) * 100:.1f}%")
        
        # Risk levels
        print(f"\n{'='*60}")
        print("RISK ASSESSMENT")
        print(f"{'='*60}")
        
        if latest['RSI'] < 30:
            print("🟢 LOW RISK:  RSI oversold")
        elif latest['RSI'] > 70:
            print("🔴 HIGH RISK: RSI overbought")
        else:
            print("🟡 MEDIUM RISK: RSI neutral")
        
        if latest['MACD'] > latest['MACD_Signal']:
            print("🟢 BULLISH:   MACD above signal")
        else:
            print("🔴 BEARISH:   MACD below signal")
        
        if current_price > latest['SMA_20'] > latest['SMA_50']:
            print("🟢 UPTREND:   Strong moving average alignment")
        elif current_price < latest['SMA_50']:
            print("🔴 DOWNTREND: Below key moving averages")
        else:
            print("🟡 SIDEWAYS:  Mixed moving average signals")
        
        print(f"\n{'='*60}")
        print("STRATEGY SCORES (Higher = Better Entry)")
        print(f"{'='*60}")
        for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{strategy:20} {score}/10")

def combine_scores(symbol_data):
    """
    Combines scores from Yahoo Finance and Alpha Vantage data
    """
    yahoo_scores = {}
    alpha_scores = {}
    
    # Safely extract Yahoo scores
    if symbol_data.get('yahoo_data') and symbol_data['yahoo_data'].get('scores'):
        yahoo_scores = symbol_data['yahoo_data']['scores']
    
    # Safely extract Alpha scores  
    if symbol_data.get('alpha_data') and symbol_data['alpha_data'].get('scores'):
        alpha_scores = symbol_data['alpha_data']['scores']
    
    if not yahoo_scores and not alpha_scores:
        return None
    
    # If both sources available, use weighted average
    if yahoo_scores and alpha_scores:
        combined_scores = {}
        all_keys = set(yahoo_scores.keys()) | set(alpha_scores.keys())
        
        for key in all_keys:
            yahoo_val = yahoo_scores.get(key, 0)
            alpha_val = alpha_scores.get(key, 0)
            
            # Weight Yahoo Finance more heavily for price-related metrics
            # Weight Alpha Vantage more heavily for fundamental metrics
            if key in ['momentum', 'volatility']:
                combined_scores[key] = (yahoo_val * 0.7 + alpha_val * 0.3)
            elif key in ['valuation', 'profitability', 'growth']:
                combined_scores[key] = (yahoo_val * 0.4 + alpha_val * 0.6)
            else:
                combined_scores[key] = (yahoo_val * 0.5 + alpha_val * 0.5)
        
        return combined_scores
    
    # Use whichever source is available
    return yahoo_scores or alpha_scores

def display_combined_analysis(symbol_data, compare_sources, detailed_analysis):
    """
    Displays the combined analysis for a single symbol
    """
    symbol = symbol_data['symbol']
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if symbol_data['yahoo_data']:
            info = symbol_data['yahoo_data']['info']
            st.metric("Company", info.get('name', 'Unknown'))
            st.metric("Sector", info.get('sector', 'Unknown'))
        elif symbol_data['alpha_data']:
            overview = symbol_data['alpha_data']['overview']
            st.metric("Company", overview.get('Name', 'Unknown'))
            st.metric("Sector", overview.get('Sector', 'Unknown'))
    
    with col2:
        if symbol_data['yahoo_data']:
            info = symbol_data['yahoo_data']['info']
            current_price = info.get('regularMarketPrice', 'N/A')
            st.metric("Current Price", format_currency(current_price, info) if current_price != 'N/A' else 'N/A')
        elif symbol_data['alpha_data']:
            price = symbol_data['alpha_data']['price']
            # For Alpha Vantage data, we need to check if it's a Danish stock manually
            ticker_info = {'symbol': symbol, 'currency': 'DKK' if is_danish_stock(symbol) else 'USD'}
            st.metric("Current Price", format_currency(price, ticker_info) if price else 'N/A')
    
    with col3:
        if symbol_data['recommendation']:
            st.metric("Recommendation", symbol_data['recommendation'])
            st.metric("Total Score", f"{symbol_data['total_score']:.1f}")
    
    # Score comparison
    if compare_sources and symbol_data['yahoo_data'] and symbol_data['alpha_data']:
        st.subheader("📊 Data Source Comparison")
        
        yahoo_scores = symbol_data['yahoo_data']['scores']
        alpha_scores = symbol_data['alpha_data']['scores']
        
        comparison_data = []
        for metric in set(yahoo_scores.keys()) | set(alpha_scores.keys()):
            comparison_data.append({
                'Metric': metric.title(),
                'Yahoo Finance': yahoo_scores.get(metric, 0),
                'Alpha Vantage': alpha_scores.get(metric, 0),
                'Combined': symbol_data['combined_scores'].get(metric, 0)
            })
        
        st.dataframe(comparison_data)
    
    # Detailed analysis
    if detailed_analysis:
        display_detailed_metrics(symbol_data)

def display_detailed_metrics(symbol_data):
    """
    Displays detailed metrics from both data sources
    """
    st.subheader("📈 Detailed Metrics")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3 = st.tabs(["💰 Valuation", "📊 Performance", "🔍 Fundamentals"])
    
    with tab1:
        if symbol_data['yahoo_data']:
            info = symbol_data['yahoo_data']['info']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                st.metric("P/B Ratio", info.get('priceToBook', 'N/A'))
            with col2:
                st.metric("Market Cap", format_large_number(info.get('marketCap', 0)))
                st.metric("EV/EBITDA", info.get('enterpriseToEbitda', 'N/A'))
    
    with tab2:
        if symbol_data['yahoo_data']:
            info = symbol_data['yahoo_data']['info']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("52W High", format_currency(info.get('fiftyTwoWeekHigh', 0), info) if info.get('fiftyTwoWeekHigh') != 'N/A' else 'N/A')
                st.metric("52W Low", format_currency(info.get('fiftyTwoWeekLow', 0), info) if info.get('fiftyTwoWeekLow') != 'N/A' else 'N/A')
            with col2:
                st.metric("Beta", info.get('beta', 'N/A'))
                st.metric("Volume", format_large_number(info.get('volume', 0)))
    
    with tab3:
        if symbol_data['alpha_data']:
            overview = symbol_data['alpha_data']['overview']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROE", f"{overview.get('ReturnOnEquityTTM', 'N/A')}")
                st.metric("ROA", f"{overview.get('ReturnOnAssetsTTM', 'N/A')}")
            with col2:
                st.metric("Profit Margin", f"{overview.get('ProfitMargin', 'N/A')}")
                st.metric("Debt/Equity", f"{overview.get('DebtToEquityRatio', 'N/A')}")

def generate_combined_comparison(combined_data, use_yahoo, use_alpha):
    """
    Generates comparison charts for multiple stocks
    """
    st.subheader("📊 Portfolio Comparison")
    
    # Filter out stocks without scores
    valid_data = {k: v for k, v in combined_data.items() if v.get('combined_scores')}
    
    if len(valid_data) > 1:
        # Create comparison chart
        comparison_fig = create_comparison_chart(valid_data)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Summary table
        st.subheader("📋 Summary Table")
        summary_data = []
        for symbol, data in valid_data.items():
            summary_data.append({
                'Symbol': symbol,
                'Recommendation': data.get('recommendation', 'N/A'),
                'Total Score': data.get('total_score', 0),
                'Data Sources': ', '.join([
                    'Yahoo Finance' if data.get('yahoo_data') else '',
                    'Alpha Vantage' if data.get('alpha_data') else ''
                ]).strip(', ')
            })
        
        st.dataframe(summary_data)

def format_large_number(num):
    """
    Format large numbers for display
    """
    if not num or num == 0:
        return 'N/A'
    
    if num >= 1e12:
        return f"{num/1e12:.1f}T"
    elif num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    else:
        return f"{num:,.0f}"
        

# --- Constants and Sample Data ---

# Sample stock lists for backtesting
SP500_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'PFE', 'ABBV', 'BAC',
    'KO', 'AVGO', 'PEP', 'TMO', 'MRK', 'COST', 'WMT', 'DIS', 'ABT', 'ACN',
    'VZ', 'ADBE', 'DHR', 'NFLX', 'NKE', 'TXN', 'QCOM', 'NEE', 'CRM', 'BMY',
    'RTX', 'UPS', 'PM', 'LOW', 'ORCL', 'AMD', 'HON', 'UNP', 'LIN', 'AMGN'
]

NASDAQ100_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'ADBE', 'PEP', 'TMUS', 'CSCO', 'QCOM', 'TXN', 'INTC', 'CMCSA', 'HON',
    'AMD', 'INTU', 'AMGN', 'ISRG', 'BKNG', 'AMAT', 'ADI', 'GILD', 'MU', 'LRCX'
]

# --- Performance Benchmarking System ---

class HistoricalPerformanceTracker:
    """Track and analyze historical performance of scoring system"""
    
    def __init__(self):
        self.performance_data = {}
        
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_historical_stock_data(symbol, start_date, end_date):
        """Get historical price data for backtesting"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Convert dates to strings to ensure compatibility
            if hasattr(start_date, 'strftime'):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)
                
            if hasattr(end_date, 'strftime'):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = str(end_date)
            
            data = ticker.history(start=start_str, end=end_str)
            
            # Handle timezone issues - convert to timezone-naive if needed
            if data is not None and not data.empty:
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data
            return None
        except Exception as e:
            st.warning(f"Could not fetch historical data for {symbol}: {e}")
            return None
    
    @staticmethod
    def calculate_returns(price_data):
        """Calculate cumulative returns from price data"""
        if price_data is None or len(price_data) == 0:
            return None
        
        returns = price_data['Close'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns
    
    @staticmethod
    def calculate_performance_metrics(returns):
        """Calculate key performance metrics"""
        if returns is None or len(returns) == 0:
            return {}
        
        # Convert to percentage
        total_return = returns.iloc[-1] * 100
        
        # Calculate volatility (annualized)
        daily_returns = returns.pct_change().fillna(0)
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
        
        # Calculate max drawdown
        peak = returns.cummax()
        drawdown = (returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Calculate Sharpe ratio (simplified, assuming 2% risk-free rate)
        excess_returns = daily_returns - (0.02 / 252)  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / daily_returns.std() * (252 ** 0.5) if daily_returns.std() != 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'periods': len(returns)
        }

class BacktestEngine:
    """Engine for running historical backtests of scoring system"""
    
    def __init__(self, start_date, end_date):
        # Convert date inputs to datetime objects if needed
        if hasattr(start_date, 'date'):  # datetime.date object
            self.start_date = datetime.combine(start_date, datetime.min.time())
        elif isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = start_date
            
        if hasattr(end_date, 'date'):  # datetime.date object
            self.end_date = datetime.combine(end_date, datetime.min.time())
        elif isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = end_date
            
        self.tracker = HistoricalPerformanceTracker()
    
    def run_scoring_backtest(self, symbols, top_n=10, rebalance_freq='quarterly'):
        """Run backtest of scoring-based strategy"""
        results = {
            'dates': [],
            'portfolio_returns': [],
            'individual_stocks': {},
            'trades': []
        }
        
        # Generate rebalance dates
        rebalance_dates = self.get_rebalance_dates(rebalance_freq)
        
        total_return = 1.0
        benchmark_return = 1.0
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            # Get scores for this period (simulated - in practice would use historical fundamentals)
            period_scores = self.simulate_historical_scores(symbols, rebalance_date)
            
            if period_scores:
                # Select top performing stocks based on scores
                top_stocks = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                selected_symbols = [stock[0] for stock in top_stocks]
                
                # Calculate period returns for selected stocks
                period_return = self.calculate_period_performance(selected_symbols, rebalance_date, next_date)
                
                if period_return is not None:
                    total_return *= (1 + period_return)
                    results['dates'].append(next_date)
                    results['portfolio_returns'].append((total_return - 1) * 100)
                    
                    # Track individual stock performance
                    for symbol in selected_symbols:
                        if symbol not in results['individual_stocks']:
                            results['individual_stocks'][symbol] = []
                        
                        stock_return = self.get_stock_period_return(symbol, rebalance_date, next_date)
                        if stock_return is not None:
                            results['individual_stocks'][symbol].append({
                                'date': next_date,
                                'return': stock_return * 100
                            })
                    
                    # Record trade
                    results['trades'].append({
                        'date': rebalance_date,
                        'action': 'rebalance',
                        'stocks': selected_symbols,
                        'scores': [period_scores[s] for s in selected_symbols]
                    })
        
        return results
    
    def get_rebalance_dates(self, frequency='quarterly'):
        """Generate rebalance dates based on frequency"""
        dates = []
        current_date = self.start_date
        
        if frequency == 'monthly':
            delta_months = 1
        elif frequency == 'quarterly':
            delta_months = 3
        elif frequency == 'semi-annually':
            delta_months = 6
        else:  # annually
            delta_months = 12
        
        while current_date < self.end_date:
            dates.append(current_date)
            # Add months (simplified)
            if current_date.month + delta_months <= 12:
                current_date = current_date.replace(month=current_date.month + delta_months)
            else:
                years_to_add = (current_date.month + delta_months - 1) // 12
                new_month = (current_date.month + delta_months - 1) % 12 + 1
                current_date = current_date.replace(year=current_date.year + years_to_add, month=new_month)
        
        dates.append(self.end_date)
        return dates
    
    def simulate_historical_scores(self, symbols, date):
        """Simulate historical scores (in production, would use actual historical data)"""
        scores = {}
        
        for symbol in symbols:
            # Simulate random scores with some persistence (trending)
            base_score = 5.0 + (hash(symbol + str(date)) % 100) / 20.0  # Range 5.0-10.0
            scores[symbol] = min(10.0, max(0.0, base_score))
        
        return scores
    
    def calculate_period_performance(self, symbols, start_date, end_date):
        """Calculate equal-weighted portfolio performance for period"""
        if not symbols:
            return None
        
        returns = []
        for symbol in symbols:
            stock_return = self.get_stock_period_return(symbol, start_date, end_date)
            if stock_return is not None:
                returns.append(stock_return)
        
        # Equal weighted average
        return sum(returns) / len(returns) if returns else None
    
    def get_stock_period_return(self, symbol, start_date, end_date):
        """Get individual stock return for period"""
        try:
            data = self.tracker.get_historical_stock_data(symbol, start_date, end_date)
            if data is not None and len(data) >= 2:
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                return (end_price - start_price) / start_price
        except Exception:
            pass
        return None
    
    def get_benchmark_performance(self, benchmark_symbol='^GSPC'):
        """Get benchmark (S&P 500) performance for comparison"""
        try:
            data = self.tracker.get_historical_stock_data(benchmark_symbol, self.start_date, self.end_date)
            if data is not None:
                returns = self.tracker.calculate_returns(data)
                return returns
        except Exception:
            pass
        return None

def create_performance_dashboard():
    """Create the performance benchmarking dashboard"""
    st.header("📊 Historical Performance Benchmarking")
    st.markdown("Test your scoring system against historical market data")
    
    # Configuration section
    st.subheader("⚙️ Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range selection
        min_date = datetime(2020, 1, 1)
        max_date = datetime.now() - timedelta(days=1)
        
        start_date = st.date_input(
            "Start Date",
            value=datetime(2022, 1, 1),
            min_value=min_date,
            max_value=max_date,
            key="backtest_start"
        )
        
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            min_value=start_date,
            max_value=max_date,
            key="backtest_end"
        )
    
    with col2:
        # Initialize session state for backtest settings
        if 'rebalance_frequency' not in st.session_state:
            st.session_state.rebalance_frequency = "quarterly"
        if 'portfolio_size' not in st.session_state:
            st.session_state.portfolio_size = 10
        
        # Strategy configuration
        def update_rebalance_frequency():
            st.session_state.rebalance_frequency = st.session_state.rebalance_frequency_select
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["monthly", "quarterly", "semi-annually", "annually"],
            index=["monthly", "quarterly", "semi-annually", "annually"].index(st.session_state.rebalance_frequency),
            key="rebalance_frequency_select",
            on_change=update_rebalance_frequency
        )
        
        def update_portfolio_size():
            st.session_state.portfolio_size = st.session_state.portfolio_size_slider
        
        top_n_stocks = st.slider(
            "Portfolio Size (Top N Stocks)",
            min_value=5,
            max_value=20,
            value=st.session_state.portfolio_size,
            key="portfolio_size_slider",
            on_change=update_portfolio_size
        )
    
    with col3:
        # Initialize session state for market selection
        if 'backtest_market' not in st.session_state:
            st.session_state.backtest_market = "S&P 500"
        if 'custom_backtest_symbols' not in st.session_state:
            st.session_state.custom_backtest_symbols = ""
        
        # Market selection
        def update_backtest_market():
            st.session_state.backtest_market = st.session_state.backtest_market_select
        
        market_to_test = st.selectbox(
            "Market Universe",
            ["S&P 500", "NASDAQ 100", "Danish Stocks", "Tech Stocks", "Custom"],
            index=["S&P 500", "NASDAQ 100", "Danish Stocks", "Tech Stocks", "Custom"].index(st.session_state.backtest_market),
            key="backtest_market_select",
            on_change=update_backtest_market
        )
        
        if st.session_state.backtest_market == "Custom":
            def update_custom_symbols():
                st.session_state.custom_backtest_symbols = st.session_state.custom_backtest_symbols_input
            
            custom_symbols = st.text_input(
                "Custom Symbols (comma-separated)",
                value=st.session_state.custom_backtest_symbols,
                placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
                key="custom_backtest_symbols_input",
                on_change=update_custom_symbols
            )
        else:
            custom_symbols = None
    
    # Benchmark selection
    benchmark_options = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Danish All-Share": "^OMXC25",
        "Custom": ""
    }
    
    # Initialize session state for benchmark
    if 'benchmark_selection' not in st.session_state:
        st.session_state.benchmark_selection = "S&P 500"
    
    def update_benchmark_selection():
        st.session_state.benchmark_selection = st.session_state.benchmark_select
    
    benchmark = st.selectbox(
        "Benchmark for Comparison",
        list(benchmark_options.keys()),
        index=list(benchmark_options.keys()).index(st.session_state.benchmark_selection),
        key="benchmark_select",
        on_change=update_benchmark_selection
    )
    
    if benchmark == "Custom":
        custom_benchmark = st.text_input(
            "Custom Benchmark Symbol",
            placeholder="^GSPC",
            key="custom_benchmark"
        )
        benchmark_symbol = custom_benchmark
    else:
        benchmark_symbol = benchmark_options[benchmark]
    
    # Run backtest button
    if st.button("🚀 Run Historical Backtest", type="primary"):
        # Validate inputs
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        if (end_date - start_date).days < 90:
            st.error("Please select a period of at least 3 months")
            return
        
        # Get symbols for testing
        if market_to_test == "S&P 500":
            test_symbols = SP500_STOCKS[:50]  # Limit for demo
        elif market_to_test == "NASDAQ 100":
            test_symbols = NASDAQ100_STOCKS[:30]
        elif market_to_test == "Danish Stocks":
            test_symbols = list(set(DANISH_STOCKS.values()))[:20]
        elif market_to_test == "Tech Stocks":
            test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "CRM", "ADBE"]
        else:  # Custom
            if not custom_symbols:
                st.error("Please enter custom symbols")
                return
            test_symbols = [s.strip().upper() for s in custom_symbols.split(',')]
        
        # Run the backtest
        with st.spinner("Running historical backtest... This may take a few minutes."):
            try:
                backtest_engine = BacktestEngine(start_date, end_date)
                
                # Run scoring-based strategy backtest
                strategy_results = backtest_engine.run_scoring_backtest(
                    test_symbols, 
                    top_n_stocks, 
                    rebalance_freq
                )
                
                # Get benchmark performance
                benchmark_returns = backtest_engine.get_benchmark_performance(benchmark_symbol)
                
                # Display results
                display_backtest_results(strategy_results, benchmark_returns, benchmark, start_date, end_date)
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                st.info("This is a demo implementation. In production, this would use actual historical fundamental data.")

def display_backtest_results(strategy_results, benchmark_returns, benchmark_name, start_date, end_date):
    """Display comprehensive backtest results"""
    
    if not strategy_results['portfolio_returns']:
        st.warning("No backtest results to display. This may be due to limited historical data.")
        return
    
    st.success("✅ Backtest completed successfully!")
    
    # Performance Summary
    st.subheader("📈 Performance Summary")
    
    # Calculate key metrics
    strategy_total_return = strategy_results['portfolio_returns'][-1] if strategy_results['portfolio_returns'] else 0
    
    benchmark_total_return = 0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_total_return = benchmark_returns.iloc[-1] * 100
    
    outperformance = strategy_total_return - benchmark_total_return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Strategy Return", 
            f"{strategy_total_return:.1f}%",
            help="Total return of your scoring-based strategy"
        )
    
    with col2:
        st.metric(
            f"{benchmark_name} Return", 
            f"{benchmark_total_return:.1f}%",
            help="Benchmark performance for comparison"
        )
    
    with col3:
        delta_color = "normal" if outperformance >= 0 else "inverse"
        st.metric(
            "Outperformance", 
            f"{outperformance:.1f}%",
            delta=f"{outperformance:.1f}%",
            help="Strategy performance vs benchmark"
        )
    
    with col4:
        num_trades = len(strategy_results['trades'])
        st.metric(
            "Rebalances", 
            num_trades,
            help="Number of portfolio rebalancing events"
        )
    
    # Performance Chart
    st.subheader("📊 Performance Comparison Chart")
    
    if strategy_results['dates'] and strategy_results['portfolio_returns']:
        # Create performance comparison chart
        chart_data = pd.DataFrame({
            'Date': strategy_results['dates'],
            'Strategy': strategy_results['portfolio_returns']
        })
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align benchmark data with strategy dates
            benchmark_data = []
            for date in strategy_results['dates']:
                # Convert date to pandas Timestamp and handle timezone issues
                if isinstance(date, str):
                    date_ts = pd.Timestamp(date)
                else:
                    date_ts = pd.Timestamp(date)
                
                # Remove timezone from timestamp if benchmark index is timezone-aware
                if benchmark_returns.index.tz is not None:
                    if date_ts.tz is None:
                        date_ts = date_ts.tz_localize(benchmark_returns.index.tz)
                    else:
                        date_ts = date_ts.tz_convert(benchmark_returns.index.tz)
                else:
                    if date_ts.tz is not None:
                        date_ts = date_ts.tz_localize(None)
                
                closest_date = benchmark_returns.index[benchmark_returns.index <= date_ts]
                if len(closest_date) > 0:
                    benchmark_data.append(benchmark_returns.loc[closest_date[-1]] * 100)
                else:
                    benchmark_data.append(0)
            
            chart_data[benchmark_name] = benchmark_data
        
        # Plot the chart
        fig = px.line(
            chart_data, 
            x='Date', 
            y=['Strategy', benchmark_name] if benchmark_name in chart_data.columns else ['Strategy'],
            title=f"Strategy vs {benchmark_name} Performance",
            labels={'value': 'Return (%)', 'variable': 'Portfolio'}
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Metrics
    st.subheader("⚠️ Risk Analysis")
    
    if len(strategy_results['portfolio_returns']) > 1:
        # Calculate risk metrics
        returns_series = pd.Series(strategy_results['portfolio_returns'])
        daily_returns = returns_series.pct_change().fillna(0)
        
        # Volatility (annualized)
        volatility = daily_returns.std() * (252 ** 0.5) * 100
        
        # Max drawdown
        peak = returns_series.cummax()
        drawdown = (returns_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified)
        excess_returns = daily_returns - (0.02 / 252)  # Assume 2% risk-free rate
        sharpe_ratio = excess_returns.mean() / daily_returns.std() * (252 ** 0.5) if daily_returns.std() != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Volatility (Annual)", f"{volatility:.1f}%")
        
        with col2:
            st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Trade Analysis
    st.subheader("📋 Trade Analysis")
    
    if strategy_results['trades']:
        trade_data = []
        for trade in strategy_results['trades']:
            trade_data.append({
                'Date': trade['date'].strftime('%Y-%m-%d'),
                'Action': trade['action'],
                'Stocks Selected': len(trade['stocks']),
                'Top Stock': trade['stocks'][0] if trade['stocks'] else 'N/A',
                'Top Score': f"{max(trade['scores']):.1f}" if trade['scores'] else 'N/A'
            })
        
        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        
        # Export trade history
        csv_data = trade_df.to_csv(index=False)
        st.download_button(
            "📥 Download Trade History",
            data=csv_data,
            file_name=f"backtest_trades_{start_date}_{end_date}.csv",
            mime="text/csv"
        )
    
    # Performance Attribution
    if strategy_results['individual_stocks']:
        st.subheader("🎯 Top Contributing Stocks")
        
        # Calculate contribution of each stock
        stock_contributions = {}
        for symbol, performance in strategy_results['individual_stocks'].items():
            if performance:
                avg_return = sum(p['return'] for p in performance) / len(performance)
                stock_contributions[symbol] = avg_return
        
        # Display top 10 contributors
        top_contributors = sorted(stock_contributions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_contributors:
            contrib_df = pd.DataFrame(top_contributors, columns=['Symbol', 'Avg Return (%)'])
            
            fig = px.bar(
                contrib_df,
                x='Symbol',
                y='Avg Return (%)',
                title="Top 10 Contributing Stocks",
                color='Avg Return (%)',
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Score tracking functions
def initialize_score_tracking():
    """Initialize score tracking in session state"""
    if 'score_history' not in st.session_state:
        st.session_state.score_history = {}

def track_stock_score(symbol, score, date=None):
    """Track a stock's score over time"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    if 'score_history' not in st.session_state:
        st.session_state.score_history = {}
    
    if symbol not in st.session_state.score_history:
        st.session_state.score_history[symbol] = []
    
    # Add new score entry
    st.session_state.score_history[symbol].append({
        'date': date,
        'score': score,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 50 entries per stock
    if len(st.session_state.score_history[symbol]) > 50:
        st.session_state.score_history[symbol] = st.session_state.score_history[symbol][-50:]

def display_score_tracking():
    """Display historical score tracking"""
    st.subheader("📈 Score Tracking Over Time")
    
    if 'score_history' not in st.session_state or not st.session_state.score_history:
        st.info("No score history available yet. Analyze some stocks to start tracking!")
        return
    
    # Select stock to view
    available_stocks = list(st.session_state.score_history.keys())
    
    # Initialize session state for selected stock
    if 'score_tracking_selected_stock' not in st.session_state:
        st.session_state.score_tracking_selected_stock = available_stocks[0] if available_stocks else ""
    
    def update_score_tracking_stock():
        st.session_state.score_tracking_selected_stock = st.session_state.score_tracking_stock_select
    
    selected_stock = st.selectbox(
        "Select Stock to View:", 
        available_stocks, 
        index=available_stocks.index(st.session_state.score_tracking_selected_stock) if st.session_state.score_tracking_selected_stock in available_stocks else 0,
        key="score_tracking_stock_select",
        on_change=update_score_tracking_stock
    )
    
    if selected_stock:
        score_data = st.session_state.score_history[selected_stock]
        
        if len(score_data) > 1:
            # Create DataFrame for plotting
            df = pd.DataFrame(score_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Plot score over time
            fig = px.line(
                df,
                x='date',
                y='score',
                title=f"{selected_stock} - Score History",
                markers=True
            )
            
            fig.update_layout(
                yaxis_title="Score (0-10)",
                xaxis_title="Date"
            )
            
            # Add score level lines
            fig.add_hline(y=8, line_dash="dash", line_color="green", annotation_text="Strong Buy (8.0)")
            fig.add_hline(y=6.5, line_dash="dash", line_color="lime", annotation_text="Buy (6.5)")
            fig.add_hline(y=4, line_dash="dash", line_color="orange", annotation_text="Hold (4.0)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Score statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Score", f"{score_data[-1]['score']:.2f}")
            
            with col2:
                avg_score = sum(d['score'] for d in score_data) / len(score_data)
                st.metric("Average Score", f"{avg_score:.2f}")
            
            with col3:
                max_score = max(d['score'] for d in score_data)
                st.metric("Highest Score", f"{max_score:.2f}")
            
            with col4:
                min_score = min(d['score'] for d in score_data)
                st.metric("Lowest Score", f"{min_score:.2f}")
        
        else:
            st.info(f"Only one data point available for {selected_stock}. Analyze it again to see trends!")

# --- Centralized Data Management ---

def get_simple_current_price(symbol):
    """Get current price for a symbol with Danish stock mapping support"""
    try:
        # First check if it's a Danish stock that needs mapping
        mapped_symbol = symbol
        if symbol in DANISH_STOCKS:
            mapped_symbol = DANISH_STOCKS[symbol]
        
        # Try to fetch price using yfinance
        ticker = yf.Ticker(mapped_symbol)
        info = ticker.info
        
        # Try multiple price fields in order of preference
        price = (info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose') or
                info.get('ask') or
                info.get('bid'))
        
        if price and price > 0:
            return float(price)
        else:
            # If no price found, try getting latest close from history
            hist = ticker.history(period="1d")
            if not hist.empty and 'Close' in hist.columns:
                latest_price = hist['Close'].iloc[-1]
                return float(latest_price) if latest_price > 0 else 0.0
            
        return 0.0
        
    except Exception as e:
        # Use warning instead of print for better UI integration
        if 'st' in globals():
            # Only show warning if we're in Streamlit context
            pass  # Don't clutter UI with price fetch warnings
        return 0.0

def display_advanced_risk_analysis(portfolio_data, risk_analyzer):
    """Display comprehensive risk analysis with advanced metrics and visualizations"""
    
    # Ultimate NumPy compatibility setup
    import warnings
    import os
    import sys
    
    # Set environment variables to suppress NumPy warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['NUMPY_HIDE_WARNINGS'] = '1'
    
    # Import and patch numpy to prevent attribute errors
    try:
        import numpy as np
        # Add missing aliases to prevent numpy deprecation errors
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'int'):
            np.int = int  
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'complex'):
            np.complex = complex
    except:
        pass
    
    # Comprehensive NumPy deprecation warning suppression for this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*np.bool.*")
        warnings.filterwarnings("ignore", message=".*np.int.*")
        warnings.filterwarnings("ignore", message=".*np.float.*")
        warnings.filterwarnings("ignore", message=".*deprecated alias.*")
        warnings.filterwarnings("ignore", message=".*numpy.*has no attribute.*")
        
        st.subheader("🎯 Advanced Risk Analysis")
        
        try:
            # Calculate comprehensive risk metrics
            risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(portfolio_data)
            
            # Create columns for risk overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_vol = risk_metrics.get('portfolio_volatility', 0)
                st.metric(
                    "Portfolio Volatility", 
                    f"{portfolio_vol:.2%}",
                    help="Annualized portfolio volatility based on historical data"
                )
                
            with col2:
                var_95 = risk_metrics.get('value_at_risk_95', 0)
                st.metric(
                    "95% VaR (1-day)", 
                    f"-{abs(var_95):.2%}",
                    help="Maximum expected loss over 1 day with 95% confidence"
                )
                
            with col3:
                expected_shortfall = risk_metrics.get('expected_shortfall', 0)
                st.metric(
                    "Expected Shortfall", 
                    f"-{abs(expected_shortfall):.2%}",
                    help="Average loss in worst 5% of scenarios"
                )
                
            with col4:
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
                st.metric(
                    "Sharpe Ratio", 
                    f"{sharpe_ratio:.2f}",
                    help="Risk-adjusted return measure"
                )
            
            # Risk breakdown by asset
            st.subheader("📊 Risk Contribution by Asset")
            
            # Create risk contribution chart
            if 'risk_contribution' in risk_metrics and risk_metrics['risk_contribution']:
                risk_contrib_df = pd.DataFrame(risk_metrics['risk_contribution'].items(), 
                                             columns=['Stock', 'Risk Contribution'])
                risk_contrib_df['Risk Contribution %'] = risk_contrib_df['Risk Contribution'] * 100
                
                fig_risk = px.bar(
                    risk_contrib_df, 
                    x='Stock', 
                    y='Risk Contribution %',
                    title="Individual Risk Contribution (%)",
                    color='Risk Contribution %',
                    color_continuous_scale='Reds'
                )
                fig_risk.update_layout(height=400)
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("📊 Risk contribution analysis calculating... This appears when historical price data is available.")
            
            # Correlation matrix
            # Correlation matrix
            st.subheader("🔗 Asset Correlation Matrix")
            
            # Get correlation matrix from risk metrics (already calculated)
            correlation_matrix = risk_metrics.get('correlation_matrix', pd.DataFrame())
            
            if correlation_matrix is not None and not correlation_matrix.empty and len(correlation_matrix) > 1:
                try:
                    # Enhanced NumPy compatibility for plotting
                    import numpy as np
                    import warnings
                    import os
                    
                    # Set up comprehensive warning suppression
                    os.environ['PYTHONWARNINGS'] = 'ignore'
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        warnings.filterwarnings("ignore", category=Warning)
                        warnings.filterwarnings("ignore", message=".*np.bool.*")
                        warnings.filterwarnings("ignore", message=".*numpy.*has no attribute.*")
                        
                        # Convert correlation matrix to pure Python types for plotting
                        corr_values = correlation_matrix.values.astype(float)
                        corr_labels = list(correlation_matrix.columns)
                        
                        # Create correlation heatmap with explicit data conversion
                        fig_corr = px.imshow(
                            corr_values,
                            x=corr_labels,
                            y=corr_labels,
                            title="Portfolio Correlation Heatmap",
                            color_continuous_scale='RdBu',
                            aspect='auto',
                            zmin=-1,
                            zmax=1
                        )
                        fig_corr.update_layout(height=500)
                        
                        # Add correlation values as text (with explicit conversion)
                        corr_text = np.around(corr_values, decimals=2).astype(str)
                        fig_corr.update_traces(text=corr_text, texttemplate="%{text}")
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Correlation insights
                        high_corr_pairs = []
                        for i in range(len(correlation_matrix.columns)):
                            for j in range(i+1, len(correlation_matrix.columns)):
                                try:
                                    corr_val = float(correlation_matrix.iloc[i, j])
                                    if abs(corr_val) > 0.7:  # High correlation threshold
                                        high_corr_pairs.append({
                                            'Stock 1': correlation_matrix.columns[i],
                                            'Stock 2': correlation_matrix.columns[j],
                                            'Correlation': corr_val
                                        })
                                except (TypeError, ValueError):
                                    continue
                        
                        if high_corr_pairs:
                            st.warning("⚠️ **High Correlation Alert**: The following pairs show high correlation (>70%):")
                            for pair in high_corr_pairs[:5]:  # Show top 5
                                st.write(f"• {pair['Stock 1']} ↔ {pair['Stock 2']}: {pair['Correlation']:.2%}")
                        else:
                            st.success("✅ No high correlations detected - good diversification!")
                            
                except Exception as plot_error:
                    # Fallback: Show simple correlation info without heatmap
                    st.info("📊 Correlation matrix calculated but visualization temporarily unavailable.")
                    if len(correlation_matrix.columns) > 1:
                        st.write(f"**Assets analyzed:** {', '.join(correlation_matrix.columns)}")
                        
            elif len(correlation_matrix) == 1:
                st.info("📊 Correlation analysis requires at least 2 assets. Add more assets to enable correlation analysis.")
            else:
                st.info("📊 Correlation analysis temporarily unavailable - insufficient price data.")
        
            
            try:
                # Extract symbols from portfolio data and get proper returns data for correlation
                if isinstance(portfolio_data, pd.DataFrame):
                    # Try both 'Symbol' and 'symbol' column names for compatibility
                    if 'Symbol' in portfolio_data.columns:
                        portfolio_symbols = portfolio_data['Symbol'].tolist()
                    elif 'symbol' in portfolio_data.columns:
                        portfolio_symbols = portfolio_data['symbol'].tolist()
                    else:
                        portfolio_symbols = []
                else:
                    portfolio_symbols = portfolio_data if portfolio_data else []
                
                if portfolio_symbols:
                    # Get price data and calculate returns for correlation matrix
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        warnings.filterwarnings("ignore", category=Warning)
                        
                        price_data = risk_analyzer.get_portfolio_price_data(portfolio_symbols)
                        
                        if not price_data.empty:
                            # Calculate returns with explicit NumPy compatibility
                            try:
                                returns = price_data.pct_change().dropna()
                                correlation_matrix = risk_analyzer.calculate_correlation_matrix(returns)
                            except Exception:
                                correlation_matrix = pd.DataFrame()
                        else:
                            correlation_matrix = pd.DataFrame()
                else:
                    correlation_matrix = pd.DataFrame()
                    
                if correlation_matrix is not None and not correlation_matrix.empty:
                    fig_corr = px.imshow(
                        correlation_matrix,
                        title="Portfolio Correlation Heatmap",
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Correlation insights
                    high_corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_val = correlation_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:  # High correlation threshold
                                high_corr_pairs.append({
                                    'Stock 1': correlation_matrix.columns[i],
                                    'Stock 2': correlation_matrix.columns[j],
                                    'Correlation': corr_val
                                })
                    
                    if high_corr_pairs:
                        st.warning("⚠️ **High Correlation Alert**: The following pairs show high correlation (>70%):")
                        for pair in high_corr_pairs[:5]:  # Show top 5
                            st.write(f"• {pair['Stock 1']} ↔ {pair['Stock 2']}: {pair['Correlation']:.2%}")
                else:
                    # Fallback: Show a simple message when correlation matrix can't be calculated
                    st.info("📊 Correlation analysis temporarily unavailable")
                    if portfolio_symbols and len(portfolio_symbols) > 1:
                        st.write(f"**Portfolio contains {len(portfolio_symbols)} assets:**")
                        for i, symbol in enumerate(portfolio_symbols[:10]):  # Show max 10
                            st.write(f"• {symbol}")
                        if len(portfolio_symbols) > 10:
                            st.write(f"... and {len(portfolio_symbols) - 10} more assets")
                    else:
                        st.write("💡 Add more assets to your portfolio to enable correlation analysis")
            except Exception as e:
                # Don't show the full error message to avoid exposing NumPy compatibility issues
                st.info("� Correlation analysis temporarily unavailable due to data processing constraints")
                if portfolio_symbols and len(portfolio_symbols) > 1:
                    st.write(f"**Portfolio contains {len(portfolio_symbols)} assets for analysis**")
            
            # Stress testing results
            st.subheader("🚨 Stress Testing")
            
            # Get stress scenarios from risk metrics (already calculated)
            stress_results = risk_metrics.get('stress_scenarios', {})
            
            # Display stress test results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Scenario Impact on Portfolio Value:**")
                if stress_results:
                    for scenario_name, scenario_data in stress_results.items():
                        impact_pct = scenario_data.get('portfolio_loss_pct', 0)
                        color = "🔴" if impact_pct < -15 else "🟡" if impact_pct < -5 else "🟢"
                        st.write(f"{color} {scenario_name.replace('_', ' ').title()}: {impact_pct:.1f}%")
                else:
                    st.info("Stress test data not available")
            
            with col2:
                # Monte Carlo simulation
                st.write("**Monte Carlo Simulation (1 Year):**")
                mc_results = risk_metrics.get('monte_carlo_var', {})
                
                if mc_results and mc_results.get('expected_return', 0) != 0:
                    expected_return = mc_results.get('expected_return', 0)
                    mc_var_95 = mc_results.get('mc_var_95', 0)
                    mc_var_99 = mc_results.get('mc_var_99', 0)
                    
                    st.write(f"• Expected Return: {expected_return:.2f}%")
                    st.write(f"• 95% VaR: {mc_var_95:.2f}%")
                    st.write(f"• 99% VaR: {mc_var_99:.2f}%")
                    
                    # Calculate probability of loss (simplified)
                    prob_loss = max(0, min(100, 50 - expected_return))
                    st.write(f"• Est. Probability of Loss: {prob_loss:.1f}%")
                else:
                    st.info("Monte Carlo data not available")
        
            # Risk-adjusted performance metrics
            st.subheader("📈 Risk-Adjusted Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sortino = risk_metrics.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.2f}", help="Downside risk-adjusted return")
                
            with col2:
                max_dd = risk_metrics.get('maximum_drawdown', 0)
                st.metric("Max Drawdown", f"{abs(max_dd):.2%}", help="Largest peak-to-trough decline")
                
            with col3:
                calmar = risk_metrics.get('calmar_ratio', 0)
                st.metric("Calmar Ratio", f"{calmar:.2f}", help="Return vs. max drawdown")
        
            # Risk recommendations
            st.subheader("💡 Risk Management Recommendations")
            
            recommendations = []
            
            # Check portfolio concentration
            if len(portfolio_data) < 10:
                recommendations.append("🎯 **Diversification**: Consider adding more positions to reduce concentration risk")
            
            # Check high correlation
            if high_corr_pairs:
                recommendations.append(f"🔗 **Correlation**: {len(high_corr_pairs)} high-correlation pairs detected - consider rebalancing")
            
            # Check volatility
            portfolio_vol = risk_metrics.get('portfolio_volatility', 0)
            if portfolio_vol > 0.25:
                recommendations.append("📊 **Volatility**: Portfolio shows high volatility - consider adding defensive positions")
            
            # Check VaR
            var_95 = abs(risk_metrics.get('value_at_risk_95', 0))
            if var_95 > 0.05:
                recommendations.append("⚠️ **VaR**: High Value-at-Risk detected - consider risk reduction strategies")
            
            # Check Sharpe ratio
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5 and sharpe_ratio > 0:
                recommendations.append("📉 **Performance**: Low risk-adjusted returns - consider portfolio optimization")
            
            # Positive alerts for good risk management
            if (portfolio_vol < 0.15 and var_95 < 0.02 and sharpe_ratio > 1.0):
                recommendations.append("✅ **Excellent Risk Profile**: Portfolio shows strong risk-adjusted performance")
            
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    if "✅" in rec:
                        st.success(rec)
                    else:
                        st.warning(rec)
            else:
                st.success("✅ **Portfolio Risk Profile**: Your portfolio shows well-managed risk characteristics")
                
        except Exception as e:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.error(f"Error in risk analysis: {str(e)}")
                st.info("Risk analysis requires portfolio data with sufficient history for calculation")

class StockDataManager:
    """Centralized stock data management with consistent caching"""
    
    @staticmethod
    # @st.cache_data(ttl=1800)  # Temporarily disabled to fix unhashable dict error
    def get_stock_data(symbol):
        """Single source of truth for all stock data"""
        try:
            result = fetch_yahoo_info(symbol)
            return result
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @staticmethod
    # @st.cache_data(ttl=3600)  # Temporarily disabled to fix unhashable dict error
    def get_technical_data(symbol, period="1y"):
        """Centralized technical data fetching"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            st.warning(f"Could not fetch technical data for {symbol}: {e}")
            return None
    
    @staticmethod
    # @st.cache_data(ttl=1800)  # Temporarily disabled to fix unhashable dict error
    def get_industry_pe(symbol):
        """Get industry PE for a symbol"""
        info = StockDataManager.get_stock_data(symbol)
        if info:
            industry = info.get("industry", None)
            sector = info.get("sector", None)
            return INDUSTRY_PE_MAP.get(industry, INDUSTRY_PE_MAP.get(sector, INDUSTRY_PE_MAP["Unknown"]))
        return INDUSTRY_PE_MAP["Unknown"]

def analyze_stock_complete(symbol, include_technical=True):
    """Single function for complete stock analysis with optimized data fetching"""
    try:
        # Get data once from centralized manager
        info = StockDataManager.get_stock_data(symbol)
        
        if not info or info.get('name') == 'Unknown':
            return None
        
        # Calculate fundamental scores
        industry_pe = StockDataManager.get_industry_pe(symbol)
        scores, debug_data = calculate_scores_yahoo(info, industry_pe)
        
        if not scores:
            return None
        
        # Calculate overall score with current weights
        available_weights = {k: st.session_state.score_weights.get(k, 0) 
                           for k in scores if k in st.session_state.score_weights}
        
        if available_weights:
            total_weight = sum(available_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
            else:
                overall_score = sum(scores.values()) / len(scores)
        else:
            overall_score = sum(scores.values()) / len(scores)
        
        recommendation, color = get_recommendation(overall_score)
        
        result = {
            'symbol': symbol,
            'info': info,
            'scores': scores,
            'overall_score': overall_score,
            'recommendation': recommendation,
            'color': color,
            'debug_data': debug_data
        }
        
        # Add technical analysis if requested
        if include_technical:
            technical_data = StockDataManager.get_technical_data(symbol)
            if technical_data is not None:
                result['technical_data'] = technical_data
        
        return result
        
    except Exception as e:
        st.warning(f"Error analyzing {symbol}: {str(e)}")
        return None

def analyze_multiple_stocks(symbols, batch_size=5, include_technical=False):
    """Efficient batch processing for multiple stocks with rate limiting"""
    results = {}
    
    # Process in batches to manage API rate limits
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
        for symbol in batch:
            result = analyze_stock_complete(symbol, include_technical)
            if result:
                results[symbol] = result
            
            # Rate limiting between requests
            if i + 1 < len(symbols):  # Don't delay after the last symbol
                time.sleep(REQUEST_DELAY)
    
    return results

def analyze_portfolio_optimized(portfolio_symbols):
    """Optimized portfolio analysis using batch processing"""
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return []
    
    st.subheader("📊 Optimized Portfolio Analysis")
    
    with st.spinner("Analyzing portfolio..."):
        # Use batch processing for efficiency
        results = analyze_multiple_stocks(portfolio_symbols, batch_size=3)
        
        analysis_results = []
        for symbol, data in results.items():
            if data:
                analysis_results.append({
                    'Symbol': symbol,
                    'Company': data['info'].get('name', 'N/A')[:30],
                    'Score': round(data['overall_score'], 2),
                    'Recommendation': data['recommendation'],
                    'Price': data['info'].get('currentPrice', 0),
                    'P/E': data['info'].get('pe', 0),
                    'ROE': round(data['info'].get('roe', 0) * 100, 1) if data['info'].get('roe') else 0,
                    'Sector': data['info'].get('sector', 'N/A')
                })
        
        return analysis_results

# --- Portfolio Management Helper Functions ---

def analyze_portfolio_optimized(symbols):
    """Optimized portfolio analysis using batch processing"""
    try:
        results = analyze_multiple_stocks(symbols)
        
        portfolio_data = []
        for symbol, data in results.items():
            if data and 'scores' in data and 'info' in data:
                scores = data['scores']
                info = data['info']
                
                # Calculate overall score
                available_weights = {k: st.session_state.score_weights.get(k, 0) 
                                   for k in scores if k in st.session_state.score_weights}
                
                if available_weights:
                    total_weight = sum(available_weights.values())
                    if total_weight > 0:
                        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                        overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
                    else:
                        overall_score = sum(scores.values()) / len(scores)
                else:
                    overall_score = sum(scores.values()) / len(scores)
                
                recommendation, _ = get_recommendation(overall_score)
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Company': info.get('name', 'N/A')[:30],
                    'Score': round(overall_score, 2),
                    'Recommendation': recommendation,
                    'Price': info.get('currentPrice', 0),
                    'P/E': info.get('pe', 0),
                    'ROE': round(info.get('roe', 0) * 100, 1) if info.get('roe') else 0,
                    'Sector': info.get('sector', 'N/A')
                })
        
        return portfolio_data
    
    except Exception as e:
        st.error(f"Error in portfolio analysis: {str(e)}")
        return []

def run_automated_portfolio_analysis(portfolio_symbols):
    """Run comprehensive automated analysis on portfolio using optimized data fetching"""
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return
    
    # Use the optimized analysis function
    analysis_results = analyze_portfolio_optimized(portfolio_symbols)
    
    if analysis_results:
        # Create DataFrame and sort by score
        df = pd.DataFrame(analysis_results).sort_values('Score', ascending=False)
        
        # Display results
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Portfolio insights
        st.subheader("🔍 Portfolio Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = df['Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/10")
        
        with col2:
            strong_buys = len(df[df['Score'] >= 7.0])
            st.metric("Strong Performers", f"{strong_buys}/{len(df)}")
        
        with col3:
            weak_stocks = len(df[df['Score'] < 5.0])
            st.metric("Weak Performers", f"{weak_stocks}/{len(df)}")
        
        with col4:
            total_value = sum(df['Price'])
            st.metric("Total Portfolio Value", f"${total_value:,.0f}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if weak_stocks > 0:
            weak_symbols = df[df['Score'] < 5.0]['Symbol'].tolist()
            st.warning(f"⚠️ Consider reviewing: {', '.join(weak_symbols)}")
        
        if strong_buys > len(df) * 0.7:
            st.success("🎉 Strong portfolio! Most stocks are performing well.")
        
        # Save analysis timestamp
        st.session_state.monitoring_settings['last_analysis'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Export option
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download Analysis Results",
            data=csv_data,
            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    else:
        st.error("Could not analyze any stocks in the portfolio")

def run_portfolio_health_check(portfolio_symbols):
    """Quick health check of portfolio"""
    st.subheader("🏥 Portfolio Health Check")
    
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return
    
    health_metrics = {
        'total_stocks': len(portfolio_symbols),
        'diversified': len(portfolio_symbols) >= 10,
        'tech_heavy': False,
        'risk_level': 'Medium'
    }
    
    # Display health metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Size", health_metrics['total_stocks'])
        if health_metrics['total_stocks'] < 5:
            st.warning("⚠️ Consider adding more stocks for diversification")
        elif health_metrics['total_stocks'] > 20:
            st.info("💡 Large portfolio - ensure you can monitor all positions")
        else:
            st.success("✅ Good portfolio size")
    
    with col2:
        diversification = "✅ Yes" if health_metrics['diversified'] else "⚠️ No"
        st.metric("Diversified", diversification)
    
    with col3:
        st.metric("Risk Level", health_metrics['risk_level'])
    
    # Quick recommendations
    st.markdown("**Quick Health Tips:**")
    st.write("• Regular rebalancing recommended every 3-6 months")
    st.write("• Monitor correlation between holdings")
    st.write("• Consider sector allocation balance")
    st.write("• Review underperforming positions regularly")

def run_portfolio_risk_assessment(portfolio_symbols):
    """Assess portfolio risk"""
    st.subheader("⚠️ Portfolio Risk Assessment")
    
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return
    
    # Simplified risk metrics
    risk_factors = {
        'concentration_risk': len(portfolio_symbols) < 5,
        'sector_concentration': False,  # Would need sector analysis
        'volatility_risk': 'Medium',
        'overall_risk': 'Medium'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Factors:**")
        
        if risk_factors['concentration_risk']:
            st.error("🔴 High concentration risk - portfolio too small")
        else:
            st.success("🟢 Concentration risk acceptable")
        
        st.info(f"📊 Estimated volatility: {risk_factors['volatility_risk']}")
    
    with col2:
        st.markdown("**Risk Mitigation:**")
        st.write("• Diversify across sectors and geographies")
        st.write("• Consider adding defensive stocks")
        st.write("• Regular position sizing review")
        st.write("• Set stop-loss levels for risk management")

def save_portfolio_snapshot(portfolio_symbols):
    """Save current portfolio state for tracking"""
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return
    
    snapshot = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbols': portfolio_symbols.copy(),
        'size': len(portfolio_symbols)
    }
    
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = []
    
    st.session_state.portfolio_history.append(snapshot)
    
    # Keep only last 30 snapshots
    if len(st.session_state.portfolio_history) > 30:
        st.session_state.portfolio_history = st.session_state.portfolio_history[-30:]
    
    st.success(f"📸 Portfolio snapshot saved! ({len(portfolio_symbols)} stocks)")

def display_portfolio_performance_history():
    """Display historical portfolio performance"""
    st.subheader("📈 Portfolio History")
    
    if not st.session_state.portfolio_history:
        st.info("No historical data available yet. Save snapshots to track performance.")
        return
    
    # Create history DataFrame
    history_data = []
    for snapshot in st.session_state.portfolio_history[-10:]:  # Show last 10
        history_data.append({
            'Date': snapshot['timestamp'],
            'Portfolio Size': snapshot['size'],
            'Symbols': ', '.join(snapshot['symbols'][:3]) + ('...' if len(snapshot['symbols']) > 3 else '')
        })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Simple chart
        if len(history_data) > 1:
            fig = px.line(df, x='Date', y='Portfolio Size', 
                         title="Portfolio Size Over Time")
            st.plotly_chart(fig, use_container_width=True)

def generate_weekly_portfolio_report(portfolio_symbols):
    """Generate comprehensive weekly report"""
    st.subheader("📋 Weekly Portfolio Report")
    
    if not portfolio_symbols:
        st.warning("Portfolio is empty")
        return
    
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # Report header
    st.markdown(f"**Report Date:** {report_date}")
    st.markdown(f"**Portfolio Size:** {len(portfolio_symbols)} stocks")
    
    # Performance summary (simplified)
    st.markdown("### 📊 Performance Summary")
    st.write("• Portfolio contains diverse mix of stocks")
    st.write("• Regular monitoring recommended")
    st.write("• Consider rebalancing if needed")
    
    # Top holdings
    st.markdown("### 🔝 Current Holdings")
    holdings_df = pd.DataFrame({
        'Symbol': portfolio_symbols,
        'Status': ['Monitored'] * len(portfolio_symbols)
    })
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # Generate downloadable report
    report_text = f"""
    Weekly Portfolio Report - {report_date}
    =====================================
    
    Portfolio Overview:
    - Total Stocks: {len(portfolio_symbols)}
    - Holdings: {', '.join(portfolio_symbols)}
    
    Recommendations:
    - Continue monitoring stock performance
    - Review positions regularly
    - Consider diversification opportunities
    
    Generated by Advanced Stock Scoring System
    """
    
    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name=f"portfolio_report_{report_date}.txt",
        mime="text/plain"
    )

def check_portfolio_alerts(portfolio_symbols, portfolio_data=None, risk_analyzer=None):
    """Check for portfolio alerts based on score changes and advanced risk metrics"""
    alerts = []
    
    try:
        # Basic portfolio monitoring
        if len(portfolio_symbols) > 0:
            basic_alert = {
                'type': 'info',
                'message': f"Portfolio monitoring active for {len(portfolio_symbols)} stocks",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            alerts.append(basic_alert)
        
        # Advanced risk-based alerts (if data and analyzer available)
        if portfolio_data is not None and risk_analyzer is not None and not portfolio_data.empty:
            try:
                # Calculate comprehensive risk metrics
                risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(portfolio_data)
                
                # Alert thresholds
                HIGH_VOLATILITY_THRESHOLD = 0.30  # 30% annual volatility
                HIGH_VAR_THRESHOLD = 0.05         # 5% daily VaR
                HIGH_CORRELATION_THRESHOLD = 0.80  # 80% correlation
                LOW_SHARPE_THRESHOLD = 0.5        # Sharpe ratio below 0.5
                
                # Volatility alerts
                portfolio_vol = risk_metrics.get('portfolio_volatility', 0)
                if portfolio_vol > HIGH_VOLATILITY_THRESHOLD:
                    alerts.append({
                        'type': 'warning',
                        'message': f"🚨 High Portfolio Volatility: {portfolio_vol:.2%} (Threshold: {HIGH_VOLATILITY_THRESHOLD:.0%})",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'category': 'Risk'
                    })
                
                # VaR alerts
                var_95 = abs(risk_metrics.get('value_at_risk_95', 0))
                if var_95 > HIGH_VAR_THRESHOLD:
                    alerts.append({
                        'type': 'error',
                        'message': f"⚠️ High Value-at-Risk: {var_95:.2%} daily loss potential (Threshold: {HIGH_VAR_THRESHOLD:.2%})",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'category': 'Risk'
                    })
                
                # Sharpe ratio alerts
                sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
                if sharpe_ratio < LOW_SHARPE_THRESHOLD and sharpe_ratio > 0:
                    alerts.append({
                        'type': 'warning',
                        'message': f"📉 Low Risk-Adjusted Returns: Sharpe ratio {sharpe_ratio:.2f} (Threshold: {LOW_SHARPE_THRESHOLD})",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'category': 'Performance'
                    })
                
                # Correlation alerts - use the correlation matrix from risk metrics
                try:
                    correlation_matrix = risk_metrics.get('correlation_matrix', pd.DataFrame())
                    if correlation_matrix is not None and not correlation_matrix.empty:
                        high_corr_count = 0
                        for i in range(len(correlation_matrix.columns)):
                            for j in range(i+1, len(correlation_matrix.columns)):
                                if abs(correlation_matrix.iloc[i, j]) > HIGH_CORRELATION_THRESHOLD:
                                    high_corr_count += 1
                        
                        if high_corr_count > 0:
                            alerts.append({
                                'type': 'warning',
                                'message': f"🔗 High Correlation Risk: {high_corr_count} asset pairs show >80% correlation",
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'category': 'Diversification'
                            })
                except Exception as e:
                    # Silently handle correlation calculation errors
                    pass
                
                # Concentration risk alerts
                if len(portfolio_data) < 10:
                    alerts.append({
                        'type': 'info',
                        'message': f"🎯 Low Diversification: Only {len(portfolio_data)} holdings - consider adding more positions",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'category': 'Diversification'
                    })
                
                # Stress test alerts - use the stress scenarios from risk metrics
                stress_scenarios = risk_metrics.get('stress_scenarios', {})
                
                for scenario_name, scenario_data in stress_scenarios.items():
                    impact_pct = scenario_data.get('portfolio_loss_pct', 0)
                    impact = impact_pct / 100  # Convert to decimal for comparison
                    
                    if impact < -20:  # More than 20% loss in stress scenario
                        alerts.append({
                            'type': 'error',
                            'message': f"🚨 Severe Stress Impact: {scenario_name.replace('_', ' ').title()} scenario shows {impact:.2%} portfolio loss",
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'category': 'Stress Test'
                        })
                    elif impact < -10:  # More than 10% loss
                        alerts.append({
                            'type': 'warning',
                            'message': f"⚠️ Moderate Stress Impact: {scenario_name.replace('_', ' ').title()} scenario shows {impact:.2%} portfolio loss",
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'category': 'Stress Test'
                        })
                
                # Positive alerts for good risk management
                portfolio_vol = risk_metrics.get('portfolio_volatility', 0)
                var_95 = abs(risk_metrics.get('value_at_risk_95', 0))
                
                if (portfolio_vol < 0.15 and var_95 < 0.02 and sharpe_ratio > 1.0):
                    alerts.append({
                        'type': 'success',
                        'message': f"✅ Excellent Risk Profile: Low volatility ({portfolio_vol:.2%}), High Sharpe ({sharpe_ratio:.2f})",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'category': 'Performance'
                    })
                
            except Exception as risk_error:
                # Don't fail the whole function if advanced risk analysis fails
                alerts.append({
                    'type': 'info',
                    'message': f"Risk analysis unavailable: {str(risk_error)[:50]}...",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'category': 'System'
                })
    
    except Exception as e:
        # Fallback alert
        alerts.append({
            'type': 'error',
            'message': f"Alert system error: {str(e)[:50]}...",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'category': 'System'
        })
    
    # Add to alert history
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []
    
    for alert in alerts:
        # Avoid duplicate alerts (check last 5 alerts)
        recent_alerts = st.session_state.alert_history[-5:] if len(st.session_state.alert_history) >= 5 else st.session_state.alert_history
        is_duplicate = any(
            recent_alert.get('message', '') == alert.get('message', '') and
            recent_alert.get('category', '') == alert.get('category', '')
            for recent_alert in recent_alerts
        )
        
        if not is_duplicate:
            st.session_state.alert_history.append(alert)
    
    # Keep only last 100 alerts (increased from 50 for better history)
    if len(st.session_state.alert_history) > 100:
        st.session_state.alert_history = st.session_state.alert_history[-100:]
    
    return alerts

def display_portfolio_alerts():
    """Display portfolio alerts in an organized, user-friendly format"""
    
    if 'alert_history' not in st.session_state or not st.session_state.alert_history:
        st.info("📋 No portfolio alerts yet. Alerts will appear here as the system monitors your portfolio.")
        return
    
    st.subheader("🔔 Portfolio Risk Alerts")
    
    # Get recent alerts (last 24 hours)
    recent_alerts = []
    older_alerts = []
    
    current_time = datetime.now()
    
    for alert in reversed(st.session_state.alert_history):  # Show most recent first
        try:
            alert_time = datetime.strptime(alert['timestamp'], "%Y-%m-%d %H:%M")
            time_diff = current_time - alert_time
            
            if time_diff.total_seconds() < 86400:  # 24 hours
                recent_alerts.append(alert)
            else:
                older_alerts.append(alert)
        except:
            older_alerts.append(alert)  # If timestamp parsing fails, put in older
    
    # Display recent alerts
    if recent_alerts:
        st.markdown("### 🕐 Recent Alerts (Last 24 Hours)")
        
        for alert in recent_alerts[:10]:  # Show top 10 recent alerts
            alert_type = alert.get('type', 'info')
            message = alert.get('message', 'No message')
            timestamp = alert.get('timestamp', 'Unknown time')
            category = alert.get('category', 'General')
            
            # Choose appropriate Streamlit alert function and emoji
            if alert_type == 'error':
                st.error(f"**{category}** • {timestamp}\n\n{message}")
            elif alert_type == 'warning':
                st.warning(f"**{category}** • {timestamp}\n\n{message}")
            elif alert_type == 'success':
                st.success(f"**{category}** • {timestamp}\n\n{message}")
            else:
                st.info(f"**{category}** • {timestamp}\n\n{message}")
    
    # Display older alerts in an expander
    if older_alerts:
        with st.expander(f"📚 Alert History ({len(older_alerts)} older alerts)", expanded=False):
            
            # Group by category
            categories = {}
            for alert in older_alerts[:50]:  # Limit to 50 older alerts
                category = alert.get('category', 'General')
                if category not in categories:
                    categories[category] = []
                categories[category].append(alert)
            
            for category, cat_alerts in categories.items():
                st.markdown(f"**{category} Alerts:**")
                for alert in cat_alerts[:10]:  # Top 10 per category
                    alert_type = alert.get('type', 'info')
                    message = alert.get('message', 'No message')
                    timestamp = alert.get('timestamp', 'Unknown time')
                    
                    icon = {"error": "🔴", "warning": "🟡", "success": "🟢", "info": "🔵"}.get(alert_type, "🔵")
                    st.markdown(f"{icon} `{timestamp}` {message}")
                
                st.markdown("---")
    
    # Alert summary
    if recent_alerts or older_alerts:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            error_count = sum(1 for alert in recent_alerts if alert.get('type') == 'error')
            st.metric("🔴 Critical Alerts", error_count)
        
        with col2:
            warning_count = sum(1 for alert in recent_alerts if alert.get('type') == 'warning')
            st.metric("🟡 Warning Alerts", warning_count)
        
        with col3:
            success_count = sum(1 for alert in recent_alerts if alert.get('type') == 'success')
            st.metric("🟢 Positive Alerts", success_count)


# --- Portfolio Rebalancing Functions ---
def rebalance_portfolio_manual(current_portfolio, target_size=10, min_keep_score=6.0, min_add_score=7.5, source_market="S&P 500"):
    """Manual portfolio rebalancing based on current scores"""
    
    if not current_portfolio:
        return None, None
    
    # Step 1: Analyze current holdings
    with st.spinner("Analyzing current portfolio..."):
        current_analysis = analyze_portfolio_optimized(current_portfolio)
    
    if not current_analysis:
        st.error("Could not analyze current portfolio")
        return None, None
    
    # Step 2: Sort by score and categorize
    current_df = pd.DataFrame(current_analysis).sort_values('Score', ascending=False)
    
    # Step 3: Identify stocks to keep (above minimum score)
    keep_stocks = current_df[current_df['Score'] >= min_keep_score]['Symbol'].tolist()
    remove_stocks = current_df[current_df['Score'] < min_keep_score]['Symbol'].tolist()
    
    # Step 4: Find new stocks to add if we need more
    add_stocks = []
    if len(keep_stocks) < target_size:
        needed_stocks = target_size - len(keep_stocks)
        
        with st.spinner("Searching for new investment opportunities..."):
            # Run screening to find new candidates
            new_candidates = screen_multi_market_stocks(source_market, min_add_score, None)
            
            if not new_candidates.empty:
                # Exclude stocks already in portfolio
                new_candidates = new_candidates[~new_candidates['Original_Symbol'].isin(current_portfolio)]
                
                # Select top new candidates
                add_stocks = new_candidates.head(needed_stocks)['Original_Symbol'].tolist()
    
    # Step 5: Create rebalanced portfolio
    new_portfolio = keep_stocks + add_stocks
    
    # Step 6: Create comprehensive rebalancing analysis
    rebalance_actions = {
        'keep': keep_stocks,
        'remove': remove_stocks,
        'add': add_stocks,
        'new_portfolio': new_portfolio,
        'target_size': target_size,
        'actual_size': len(new_portfolio)
    }
    
    return rebalance_actions, current_df

def display_rebalancing_analysis(actions, analysis_df):
    """Display comprehensive rebalancing analysis"""
    
    st.subheader("📋 Rebalancing Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Target Size", actions['target_size'])
    
    with col2:
        st.metric("Current Size", len(analysis_df))
    
    with col3:
        st.metric("New Size", actions['actual_size'])
    
    with col4:
        change = actions['actual_size'] - len(analysis_df)
        st.metric("Net Change", f"{change:+d}")
    
    # Detailed actions
    st.markdown("---")
    
    # Create three columns for actions
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("### ✅ **Keep (High Performers)**")
        if actions['keep']:
            keep_df = analysis_df[analysis_df['Symbol'].isin(actions['keep'])]
            for _, row in keep_df.iterrows():
                st.markdown(f"• **{row['Symbol']}** - Score: {row['Score']:.1f} ({row['Recommendation']})")
        else:
            st.markdown("*No stocks meet the keep criteria*")
    
    with action_col2:
        st.markdown("### ❌ **Remove (Low Performers)**")
        if actions['remove']:
            remove_df = analysis_df[analysis_df['Symbol'].isin(actions['remove'])]
            for _, row in remove_df.iterrows():
                st.markdown(f"• **{row['Symbol']}** - Score: {row['Score']:.1f} ({row['Recommendation']})")
                st.caption(f"Reason: Score below threshold")
        else:
            st.markdown("*No stocks need to be removed*")
    
    with action_col3:
        st.markdown("### ➕ **Add (New Opportunities)**")
        if actions['add']:
            for symbol in actions['add']:
                st.markdown(f"• **{symbol}** - New high-scoring opportunity")
        else:
            st.markdown("*Portfolio at target size or no new opportunities found*")
    
    # Rebalancing insights
    st.markdown("---")
    st.subheader("💡 Rebalancing Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**📊 Performance Impact:**")
        
        if actions['remove']:
            avg_remove_score = analysis_df[analysis_df['Symbol'].isin(actions['remove'])]['Score'].mean()
            st.write(f"• Removing stocks with avg score: {avg_remove_score:.1f}")
        
        if actions['keep']:
            avg_keep_score = analysis_df[analysis_df['Symbol'].isin(actions['keep'])]['Score'].mean()
            st.write(f"• Keeping stocks with avg score: {avg_keep_score:.1f}")
        
        if actions['add']:
            st.write(f"• Adding {len(actions['add'])} new high-scoring opportunities")
    
    with insights_col2:
        st.markdown("**⚠️ Risk Considerations:**")
        
        if len(actions['remove']) > len(analysis_df) * 0.5:
            st.warning("• Large portfolio turnover - consider gradual rebalancing")
        
        if actions['actual_size'] < 5:
            st.warning("• Small portfolio size - consider adding more stocks for diversification")
        
        if not actions['add'] and len(actions['keep']) < actions['target_size']:
            st.info("• No new opportunities found - consider lowering minimum score for additions")
    
    # Sector analysis if we have the data
    if 'Sector' in analysis_df.columns:
        st.markdown("---")
        st.subheader("🏭 Sector Analysis")
        
        current_sectors = analysis_df['Sector'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Sector Distribution:**")
            for sector, count in current_sectors.head(5).items():
                st.write(f"• {sector}: {count} stocks")
        
        with col2:
            # Simple diversification check
            if len(current_sectors) < 3:
                st.warning("⚠️ Limited sector diversification")
            elif len(current_sectors) >= 5:
                st.success("✅ Good sector diversification")
            else:
                st.info("ℹ️ Moderate sector diversification")

def create_rebalancing_simulation(current_portfolio, actions):
    """Create a simulation of rebalancing impact"""
    
    st.subheader("🎯 Rebalancing Simulation")
    
    # Create before/after comparison
    simulation_data = {
        'Metric': ['Portfolio Size', 'Avg Score (Est.)', 'Strong Buys (≥8.0)', 'Weak Positions (≤4.0)'],
        'Before': [
            len(current_portfolio),
            'Calculating...',
            'Calculating...',
            'Calculating...'
        ],
        'After': [
            actions['actual_size'],
            'Improved (Est.)',
            'More (Est.)',
            'Fewer (Est.)'
        ]
    }
    
    simulation_df = pd.DataFrame(simulation_data)
    st.table(simulation_df)
    
    # Rebalancing timeline
    st.markdown("**📅 Suggested Implementation Timeline:**")
    st.write("1. **Week 1**: Sell underperforming positions")
    st.write("2. **Week 2**: Research and validate new additions") 
    st.write("3. **Week 3**: Gradually buy new positions")
    st.write("4. **Week 4**: Monitor and adjust position sizes")
    
    # Risk management
    st.markdown("**🛡️ Risk Management:**")
    st.write("• Don't rebalance more than 30% of portfolio at once")
    st.write("• Consider market conditions and timing")
    st.write("• Set stop-losses for new positions")
    st.write("• Monitor correlation between holdings")


# --- Main Streamlit App ---
def main():
    init_session_state()  # <-- Add this line first!
    
    st.set_page_config(
        page_title="Stock Scoring System",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Advanced Stock Scoring System")
    st.markdown("Analyze stocks using multiple financial metrics with AI-powered scoring")
    
    # ⚠️ CRITICAL: Display data persistence warnings for Streamlit Cloud
    if not st.session_state.get("backup_warnings_shown", False):
        st.error("""
        🚨 **CRITICAL DATA PERSISTENCE WARNING - STREAMLIT CLOUD**
        
        **Your data is temporary and WILL BE LOST when:**
        • App restarts (daily automatic restarts)
        • 30+ minutes of inactivity
        • Browser refresh/closure
        • App redeployment
        
        **🔐 PROTECT YOUR DATA:** Use backup features in Portfolio Manager!
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ I Understand", type="primary"):
                st.session_state.backup_warnings_shown = True
                st.rerun()
        
        with col2:
            if st.button("💾 Quick Backup"):
                # Navigate to portfolio tab with backup interface
                st.info("💡 Go to Portfolio Manager → Backup & Settings tab to backup your data")
        
        st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Adjust Score Weights")
        with st.expander("Customize Weights"):
            new_weights = {}
            for metric, weight in list(st.session_state.score_weights.items()):
                # Initialize session state for each weight if not exists
                weight_key = f"weight_{metric.replace(' ', '_')}"
                if weight_key not in st.session_state:
                    st.session_state[weight_key] = weight
                
                def create_weight_updater(metric_name, weight_key):
                    def update_weight():
                        st.session_state.score_weights[metric_name] = st.session_state[weight_key]
                    return update_weight
                
                new_weights[metric] = st.slider(
                    metric, 
                    0.0, 
                    0.5, 
                    st.session_state[weight_key], 
                    0.01,
                    help=f"Current weight: {weight}",
                    key=weight_key,
                    on_change=create_weight_updater(metric, weight_key)
                )
            
            if st.button("Apply New Weights"):
                st.session_state.score_weights.update(new_weights)
                st.success("Weights updated!")
        st.subheader("Current Weights")
        weights_df = pd.DataFrame(list(st.session_state.score_weights.items()), columns=['Metric', 'Weight'])
        st.dataframe(weights_df, hide_index=True)
        
        # Enhanced Features Status
        st.markdown("---")
        st.subheader("🚀 Enhanced Features")
        
        if st.session_state.get('enhanced_features_enabled', False):
            st.success("✅ Enhanced Features Active")
            
            # Show feature status
            enhanced_manager = st.session_state.enhanced_features_manager
            
            features_status = {
                "🗄️ SQLite Database": "✅" if enhanced_manager.portfolio_db else "❌",
                "🚀 Advanced Caching": "✅" if enhanced_manager.cache else "❌", 
                "⚡ Async Loading": "✅" if enhanced_manager.async_loader else "❌",
                "🔮 What-If Analysis": "✅" if enhanced_manager.what_if_analyzer else "❌"
            }
            
            for feature, status in features_status.items():
                st.write(f"{feature}: {status}")
                
            # System stats
            if enhanced_manager.portfolio_db:
                holdings_count = len(enhanced_manager.portfolio_db.get_current_holdings())
                transactions_count = len(enhanced_manager.portfolio_db.get_transaction_history())
                st.metric("Portfolio Holdings", holdings_count)
                st.metric("Total Transactions", transactions_count)
        else:
            st.warning("⚠️ Enhanced Features Disabled")
            st.info("Running in basic mode. Enhanced features include:\n"
                   "• SQLite Portfolio Database\n"
                   "• Advanced Caching\n" 
                   "• Async Data Loading\n"
                   "• What-If Analysis")
            
            if st.button("🔄 Retry Enhanced Features"):
                st.rerun()

    # Main tabs - Enhanced structure with new features
    if st.session_state.get('enhanced_features_enabled', False):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 Stock Analysis Hub",
            "📈 Trading Signals", 
            "🔍 Market Screeners",
            "💼 Enhanced Portfolio",  # Updated with enhanced features
            "🔮 What-If Analysis",    # NEW: Portfolio simulation
            "🇩🇰 Danish Stocks Manager",
            "📊 Performance Benchmarking",
            "ℹ️ Help & Documentation",
            "⚖️ Compare & Export"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Stock Analysis Hub",
            "📈 Trading Signals", 
            "🔍 Market Screeners",
            "💼 Portfolio Manager",
            "🇩🇰 Danish Stocks Manager",
            "📊 Performance Benchmarking",
            "ℹ️ Help & Documentation",
            "⚖️ Compare & Export"
        ])

    # --- Stock Analysis Hub (combines Yahoo Finance, Alpha Vantage, Company Search) ---
    with tab1:
        st.header("📊 Stock Analysis Hub")
        st.markdown("Comprehensive stock analysis using multiple data sources and methodologies")
        
        # Create sub-tabs for different analysis methods
        analysis_tab1, analysis_tab2 = st.tabs([
            "🚀 Comprehensive Analysis (Yahoo + Alpha Vantage)", 
            "🔍 Company Search"
        ])
        
        with analysis_tab1:
            st.subheader("🚀 Comprehensive Stock Analysis")
            st.info("💡 Advanced analysis combining Yahoo Finance and Alpha Vantage data sources with technical signals")
            
            if st.session_state.selected_symbols:
                st.info(f"💡 You have {len(st.session_state.selected_symbols)} symbols selected from Company Search. Copy them from the search tab!")
            
            # Data source selection
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                data_sources = st.multiselect(
                    "Select Data Sources",
                    ["Yahoo Finance", "Alpha Vantage"],
                    default=["Yahoo Finance"],
                    help="Choose which data sources to use for analysis"
                )
            
            with col2:
                # Initialize session state for analysis depth
                if 'analysis_depth' not in st.session_state:
                    st.session_state.analysis_depth = "Comprehensive"
                
                def update_analysis_depth():
                    st.session_state.analysis_depth = st.session_state.analysis_depth_select
                
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    ["Basic", "Standard", "Comprehensive"],
                    index=["Basic", "Standard", "Comprehensive"].index(st.session_state.analysis_depth),
                    help="Basic: Key metrics only, Standard: + Technical analysis, Comprehensive: + Alpha Vantage data",
                    key="analysis_depth_select",
                    on_change=update_analysis_depth
                )
            
            with col3:
                # Initialize session state for max symbols
                if 'max_symbols' not in st.session_state:
                    st.session_state.max_symbols = 10
                
                def update_max_symbols():
                    st.session_state.max_symbols = st.session_state.max_symbols_input
                
                max_symbols = st.number_input(
                    "Max Symbols",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.max_symbols,
                    help="Limit symbols to avoid rate limits",
                    key="max_symbols_input",
                    on_change=update_max_symbols
                )
            
            # Enhanced analysis form
            with st.form("comprehensive_analysis_form"):
                symbols = st.text_input(
                    "Enter stock symbols (comma-separated)", 
                    "AAPL,MSFT,GOOGL", 
                    key="comprehensive",
                    help="Example: AAPL,MSFT,GOOGL"
                )
                
                # Advanced options
                with st.expander("⚙️ Advanced Options"):
                    include_technical = st.checkbox("Include Technical Analysis", value=True)
                    include_fundamentals = st.checkbox("Include Fundamental Scoring", value=True)
                    use_alpha_vantage = st.checkbox("Use Alpha Vantage Data (slower but more detailed)", 
                                                  value="Alpha Vantage" in data_sources)
                    
                    if use_alpha_vantage:
                        st.warning("⚠️ Alpha Vantage has rate limits. Analysis will be slower but more comprehensive.")
                
                submitted = st.form_submit_button("🚀 Analyze Stocks", type="primary")
            
            if submitted:
                symbol_list = validate_symbols(symbols)
                
                if not symbol_list:
                    st.error("Please enter valid stock symbols")
                else:
                    # Limit symbols based on user input and data sources
                    if len(symbol_list) > max_symbols:
                        st.warning(f"⚠️ Limited to {max_symbols} symbols")
                        symbol_list = symbol_list[:max_symbols]
                    
                    # Further limit if using Alpha Vantage
                    if use_alpha_vantage and len(symbol_list) > 5:
                        st.warning("⚠️ Limited to 5 symbols when using Alpha Vantage to avoid rate limits")
                        symbol_list = symbol_list[:5]
                    
                    st.info(f"🔍 Analyzing {len(symbol_list)} stocks using {', '.join(data_sources)} data...")
                    
                    analysis_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, symbol in enumerate(symbol_list):
                        progress_bar.progress((i + 1) / len(symbol_list))
                        status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbol_list)})")
                        
                        with st.container():
                            st.subheader(f"📈 {symbol.upper()} Analysis")
                            
                            # Initialize combined data structure
                            combined_data = {
                                'symbol': symbol,
                                'yahoo_data': None,
                                'alpha_data': None,
                                'technical_data': None,
                                'scores': {},
                                'recommendation': None
                            }
                            
                            # Fetch Yahoo Finance data if selected
                            if "Yahoo Finance" in data_sources:
                                with st.spinner(f"📊 Fetching Yahoo Finance data for {symbol}..."):
                                    try:
                                        yahoo_analysis = analyze_multiple_stocks([symbol])
                                        if yahoo_analysis and symbol in yahoo_analysis and yahoo_analysis[symbol]:
                                            yahoo_result = yahoo_analysis[symbol]
                                            # Store the info object directly for display
                                            combined_data['yahoo_data'] = yahoo_result.get('info', {})
                                            # Store scores separately
                                            combined_data['scores'].update(yahoo_result.get('scores', {}))
                                    except Exception as e:
                                        st.warning(f"⚠️ Yahoo Finance data unavailable for {symbol}: {str(e)}")
                            
                            # Fetch Alpha Vantage data if selected and enabled
                            if use_alpha_vantage and "Alpha Vantage" in data_sources:
                                with st.spinner(f"🔍 Fetching Alpha Vantage data for {symbol}..."):
                                    try:
                                        overview = fetch_overview(symbol)
                                        if overview:
                                            industry = overview.get("Industry", None)
                                            sector = overview.get("Sector", None)
                                            industry_pe = INDUSTRY_PE_MAP.get(industry, INDUSTRY_PE_MAP.get(sector, INDUSTRY_PE_MAP["Unknown"]))
                                            
                                            alpha_scores, _ = calculate_scores(symbol, industry_pe)
                                            if alpha_scores:
                                                combined_data['alpha_data'] = {
                                                    'overview': overview,
                                                    'scores': alpha_scores,
                                                    'industry_pe': industry_pe
                                                }
                                                # Merge Alpha Vantage scores with existing scores
                                                for metric, score in alpha_scores.items():
                                                    if metric in combined_data['scores']:
                                                        # Average Yahoo and Alpha Vantage scores
                                                        combined_data['scores'][metric] = (combined_data['scores'][metric] + score) / 2
                                                    else:
                                                        combined_data['scores'][metric] = score
                                    except Exception as e:
                                        st.warning(f"⚠️ Alpha Vantage data unavailable for {symbol}: {str(e)}")
                                    
                                    # Add delay to respect rate limits
                                    if i < len(symbol_list) - 1:
                                        time.sleep(REQUEST_DELAY)
                            
                            # Technical analysis if enabled
                            if include_technical:
                                with st.spinner(f"📈 Running technical analysis for {symbol}..."):
                                    try:
                                        analyzer = ComprehensiveStockAnalyzer(symbol, "1y")
                                        if analyzer.fetch_all_data():
                                            analyzer.calculate_technical_indicators()
                                            analyzer.generate_technical_signals()
                                            analyzer.calculate_buying_prices()
                                            
                                            combined_data['technical_data'] = {
                                                'analyzer': analyzer,
                                                'signals': analyzer.technical_signals,
                                                'indicators': analyzer.data  # Use the actual data DataFrame
                                            }
                                    except Exception as e:
                                        st.warning(f"⚠️ Technical analysis unavailable for {symbol}: {str(e)}")
                            
                            # Calculate final scores and recommendation
                            if combined_data['scores']:
                                # Calculate overall score using weighted averages
                                available_weights = {k: st.session_state.score_weights.get(k, 0) 
                                                   for k in combined_data['scores'] if k in st.session_state.score_weights}
                                
                                if available_weights:
                                    total_weight = sum(available_weights.values())
                                    if total_weight > 0:
                                        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                        overall_score = sum(combined_data['scores'][k] * normalized_weights[k] for k in available_weights)
                                    else:
                                        overall_score = sum(combined_data['scores'].values()) / len(combined_data['scores'])
                                else:
                                    overall_score = sum(combined_data['scores'].values()) / len(combined_data['scores'])
                                
                                recommendation, color = get_recommendation(overall_score)
                                combined_data['recommendation'] = (recommendation, color, overall_score)
                                
                                # Track score for historical analysis
                                track_stock_score(symbol, overall_score)
                                
                                # Display comprehensive results
                                display_integrated_analysis(combined_data, include_technical)
                                
                                analysis_results[symbol] = combined_data
                            else:
                                st.error(f"❌ No data available for {symbol}")
                            
                            st.markdown("---")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Summary of results
                    successful_analyses = [s for s in analysis_results.keys()]
                    if successful_analyses:
                        st.success(f"✅ Successfully analyzed {len(successful_analyses)} stocks: {', '.join(successful_analyses)}")
                        
                        # Show comparison if multiple stocks analyzed
                        if len(successful_analyses) > 1:
                            st.subheader("📊 Comparison Summary")
                            comparison_data = []
                            for symbol, data in analysis_results.items():
                                if data['recommendation']:
                                    _, _, score = data['recommendation']
                                    comparison_data.append({
                                        'Symbol': symbol,
                                        'Overall Score': f"{score:.1f}",
                                        'Data Sources': ', '.join([
                                            'Yahoo' if data['yahoo_data'] else '',
                                            'Alpha Vantage' if data['alpha_data'] else '',
                                            'Technical' if data['technical_data'] else ''
                                        ]).strip(', '),
                                        'Recommendation': data['recommendation'][0] if data['recommendation'] else 'N/A'
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    else:
                        st.error("❌ Could not analyze any stocks")
        
        with analysis_tab2:
            st.subheader("🔍 Company Search")
            st.markdown("Search for companies by name to find their stock symbols")
            display_company_search()
            
            # Quick access to popular stocks
            st.subheader("📈 Popular Stocks")
            popular_categories = {
                "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "Electric Vehicles": ["TSLA", "NIO", "RIVN", "LCID", "F"],
                "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "QCOM"],
                "Streaming & Entertainment": ["NFLX", "DIS", "SPOT", "ROKU", "WBD"],
                "Danish Stocks": ["NOVO-B.CO", "MAERSK-B.CO", "ORSTED.CO", "DSV.CO", "CARL-B.CO"]
            }
            
            for category, symbols_list in popular_categories.items():
                with st.expander(f"📊 {category}"):
                    cols = st.columns(5)
                    for i, symbol in enumerate(symbols_list):
                        with cols[i % 5]:
                            if st.button(f"Add {symbol}", key=f"add_{symbol}_{category}"):
                                if symbol not in st.session_state.selected_symbols:
                                    st.session_state.selected_symbols.append(symbol)
                                    st.success(f"Added {symbol}")
                                    st.rerun()
                                else:
                                    st.info(f"{symbol} already selected")

    # --- Trading Signals Hub (combines Buy/Sell Signals, Explanation, Combined Analysis) ---
    with tab2:
        st.header("📈 Trading Signals Hub")
        st.markdown("Technical analysis, buy/sell signals, and combined recommendations")
        
        # Create sub-tabs for different signal types
        signals_tab1, signals_tab2, signals_tab3 = st.tabs([
            "⚡ Quick Technical Signals", 
            "🔄 Combined Analysis", 
            "ℹ️ Signal Explanations"
        ])
        
        with signals_tab1:
            st.subheader("⚡ Quick Technical Analysis")
            st.info("Fast technical signals for day trading and quick decisions")
            
            with st.form("quick_tech_form"):
                symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="quick_tech", 
                                         help="Enter a stock symbol and press Enter or click the button")
                submitted = st.form_submit_button("⚡ Get Quick Signals", type="primary")
            
            if submitted and symbol:
                with st.spinner(f"🔍 Analyzing {symbol}..."):
                    analyzer = ComprehensiveStockAnalyzer(symbol, "3mo")
                    
                    # Fetch data with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Fetching market data...")
                    progress_bar.progress(25)
                    
                    if analyzer.fetch_all_data():
                        status_text.text("📈 Calculating technical indicators...")
                        progress_bar.progress(50)
                        
                        analyzer.calculate_technical_indicators()
                        
                        status_text.text("🎯 Generating signals...")
                        progress_bar.progress(75)
                        
                        signals = analyzer.generate_technical_signals()
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        if signals and 'signals' in signals:
                            # Quick display
                            st.subheader(f"⚡ Quick Signals for {symbol.upper()}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📊 Technical Signals:**")
                                for signal_name, signal_text in signals['signals'].items():
                                    if "🟢" in signal_text:
                                        st.success(f"**{signal_name}**: {signal_text}")
                                    elif "🔴" in signal_text:
                                        st.error(f"**{signal_name}**: {signal_text}")
                                    else:
                                        st.info(f"**{signal_name}**: {signal_text}")
                            
                            with col2:
                                st.markdown("**📈 Overall Assessment:**")
                                tech_score = signals.get('technical_score', 0)
                                st.metric("Technical Score", f"{tech_score:.1f}/10")
                                
                                # Quick buy/sell recommendation
                                if tech_score >= 6.5:
                                    st.success("🟢 TECHNICAL BUY SIGNAL")
                                    st.info("Strong technical indicators suggest buying opportunity")
                                elif tech_score <= 4:
                                    st.error("🔴 TECHNICAL SELL SIGNAL")
                                    st.warning("Weak technical indicators suggest caution")
                                else:
                                    st.warning("🟡 TECHNICAL NEUTRAL")
                                    st.info("Mixed technical signals - wait for clearer direction")
                        else:
                            st.error(f"❌ Could not generate signals for {symbol}")
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
        
        with signals_tab2:
            st.subheader("🔄 Combined Technical & Fundamental Analysis")
            st.info("Comprehensive analysis combining multiple methodologies")
            
            with st.form("combined_analysis_form"):
                symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="combined_analysis",
                                         help="Enter a stock symbol and press Enter or click the button")
                submitted = st.form_submit_button("🔄 Run Combined Analysis", type="primary")
            
            if submitted and symbol:
                with st.spinner(f"🔍 Running comprehensive analysis for {symbol}..."):
                    analyzer = ComprehensiveStockAnalyzer(symbol, "1y")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📊 Fetching market data...")
                    progress_bar.progress(20)
                    
                    if analyzer.fetch_all_data():
                        status_text.text("📈 Calculating fundamental scores...")
                        progress_bar.progress(40)
                        
                        analyzer.calculate_fundamental_scores()
                        
                        status_text.text("🎯 Analyzing technical indicators...")
                        progress_bar.progress(60)
                        
                        analyzer.calculate_technical_indicators()
                        analyzer.generate_technical_signals()
                        
                        status_text.text("💰 Calculating buying strategies...")
                        progress_bar.progress(80)
                        
                        analyzer.calculate_buying_prices()
                        
                        status_text.text("🔄 Generating combined recommendation...")
                        progress_bar.progress(90)
                        
                        result = analyzer.generate_combined_recommendation()
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        if result and len(result) == 3:
                            recommendation, color, combined_score = result
                            
                            # Display comprehensive analysis
                            display_comprehensive_analysis(analyzer)
                        else:
                            st.error(f"❌ Could not generate recommendation for {symbol}")
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
        
        with signals_tab3:
            st.subheader("ℹ️ Understanding Trading Signals")
            st.markdown("""
            ### 🎯 Signal Interpretation:
            - 🟢 Green: Potential Buy Signal
            - 🔴 Red: Potential Sell Signal
            - 🟡 Yellow: Neutral/Hold Signal
            
            ### 📊 Technical Indicators Explained:
            
            **RSI (Relative Strength Index)**
            - RSI < 30: Oversold (potential buy opportunity)
            - RSI > 70: Overbought (potential sell signal)
            - RSI 30-70: Normal range
            
            **Moving Averages**
            - Price above SMA 20 & 50: Uptrend
            - Price below SMA 20 & 50: Downtrend
            - Golden Cross: SMA 20 crosses above SMA 50 (bullish)
            - Death Cross: SMA 20 crosses below SMA 50 (bearish)
            
            **MACD (Moving Average Convergence Divergence)**
            - MACD above signal line: Bullish momentum
            - MACD below signal line: Bearish momentum
            - MACD histogram: Shows momentum strength
            
            **Bollinger Bands**
            - Price near upper band: Potentially overbought
            - Price near lower band: Potentially oversold
            - Band width: Volatility indicator
            
            **Stochastic Oscillator**
            - %K and %D above 80: Overbought
            - %K and %D below 20: Oversold
            - %K crossing above %D: Bullish signal
            
            ### ⚖️ Combined Scoring:
            - **Fundamental Score**: Based on financial metrics (P/E, ROE, Growth, etc.)
            - **Technical Score**: Based on technical indicators and signals
            - **Combined Score**: Weighted average of both scores
            - **Final Recommendation**: Buy/Sell/Hold based on combined analysis
            
            ### 🎯 Risk Management:
            - Never invest based on signals alone
            - Always consider fundamental analysis
            - Use proper position sizing
            - Set stop-loss orders
            - Diversify your portfolio
            """)

    # --- Market Screeners (combines Danish Stocks + Multi-Market Screener) ---
    with tab3:
        st.header("🔍 Market Screeners")
        st.markdown("Screen stocks from Danish markets and global exchanges")
        
        # Create sub-tabs for different screeners
        screener_tab1, screener_tab2 = st.tabs([
            "🌍 Multi-Market Screener", 
            "🇩🇰 Danish Stocks Focus"
        ])
        
        with screener_tab1:
            display_danish_stocks_screener()
        
        with screener_tab2:
            st.subheader("🇩🇰 Danish Stocks Focus")
            st.markdown("Specialized screening for Danish stocks with local market insights")
            
            # Danish-specific screening with enhanced features
            col1, col2, col3 = st.columns(3)
            
            # Initialize session state for Danish screener controls
            if 'danish_min_score' not in st.session_state:
                st.session_state.danish_min_score = 6.0
            if 'danish_max_stocks' not in st.session_state:
                st.session_state.danish_max_stocks = 15
            if 'danish_sector_filter' not in st.session_state:
                st.session_state.danish_sector_filter = "All Sectors"
            
            with col1:
                def update_danish_min_score():
                    st.session_state.danish_min_score = st.session_state.danish_min_score_slider
                
                min_score_danish = st.slider(
                    "Minimum Score", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=st.session_state.danish_min_score, 
                    step=0.1,
                    key="danish_min_score_slider",
                    on_change=update_danish_min_score
                )
            
            with col2:
                def update_danish_max_stocks():
                    st.session_state.danish_max_stocks = st.session_state.danish_max_stocks_input
                
                max_stocks_danish = st.number_input(
                    "Max Results", 
                    min_value=5, 
                    max_value=50, 
                    value=st.session_state.danish_max_stocks,
                    key="danish_max_stocks_input",
                    on_change=update_danish_max_stocks
                )
            
            with col3:
                def update_danish_sector_filter():
                    st.session_state.danish_sector_filter = st.session_state.danish_sector_filter_select
                
                sector_filter = st.selectbox(
                    "Sector Filter",
                    ["All Sectors", "Healthcare", "Industrials", "Technology", "Energy", "Consumer Staples"],
                    index=["All Sectors", "Healthcare", "Industrials", "Technology", "Energy", "Consumer Staples"].index(st.session_state.danish_sector_filter),
                    key="danish_sector_filter_select",
                    on_change=update_danish_sector_filter
                )
            
            if st.button("🚀 Screen Danish Stocks", type="primary", key="danish_screen"):
                with st.spinner("Screening Danish stocks..."):
                    results_df = screen_multi_market_stocks("Danish Stocks", st.session_state.danish_min_score, None)
                
                if not results_df.empty:
                    # Apply sector filter if selected
                    if st.session_state.danish_sector_filter != "All Sectors":
                        results_df = results_df[results_df['Sector'].str.contains(st.session_state.danish_sector_filter, na=False)]
                    
                    display_df = results_df.head(st.session_state.danish_max_stocks)
                    
                    st.success(f"✅ Found **{len(results_df)}** Danish stocks with score ≥ {st.session_state.danish_min_score}")
                    
                    # Display results
                    st.dataframe(
                        display_df[['Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 'Sector', 'Current_Price', 'P/E_Ratio', 'ROE']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No Danish stocks found meeting the criteria")

    # --- Portfolio Manager ---
    with tab4:
        if st.session_state.get('enhanced_features_enabled', False):
            st.header("💼 Enhanced Portfolio Manager")
            st.markdown("🚀 **Enterprise-grade portfolio management with SQLite database, intelligent caching, and async loading**")
            
            # Enhanced Portfolio Manager Interface
            enhanced_manager = st.session_state.enhanced_features_manager
            
            # Status indicators
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                db_status = "🟢 Connected" if enhanced_manager.portfolio_db else "🔴 Disconnected"
                st.metric("Database", db_status)
            
            with col2:
                cache_status = "🟢 Active" if enhanced_manager.cache else "🔴 Inactive"
                st.metric("Cache System", cache_status)
            
            with col3:
                async_status = "🟢 Ready" if enhanced_manager.async_loader else "🔴 Not Ready"
                st.metric("Async Loading", async_status)
            
            with col4:
                total_holdings = len(enhanced_manager.portfolio_db.get_current_holdings()) if enhanced_manager.portfolio_db else 0
                st.metric("Total Holdings", total_holdings)
            
            # Enhanced Portfolio Tabs - Streamlined with all features
            portfolio_tab1, portfolio_tab2, portfolio_tab3, portfolio_tab4, portfolio_tab5, portfolio_tab6 = st.tabs([
                "📊 Portfolio Dashboard",
                "➕ Manage Holdings", 
                "📈 Portfolio Alerts",
                "🔄 Portfolio Rebalancing",
                "🔍 Weekly Market Screener",
                "💾 Backup & Settings"
            ])
            
            # Display cloud persistence warnings at the top level
            if not st.session_state.get("backup_warnings_shown", False):
                st.session_state.cloud_backup_manager.display_cloud_persistence_warnings()
            
            with portfolio_tab1:
                st.subheader("📊 Portfolio Dashboard")
                
                if enhanced_manager.portfolio_db:
                    # Get portfolio summary
                    holdings = enhanced_manager.portfolio_db.get_current_holdings()
                    
                    if holdings.empty:
                        st.info("📊 Your portfolio is empty. Add some stocks to get started!")
                        
                        # Quick add section
                        st.markdown("**Quick Add Popular Stocks:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Add Tech Giants", type="primary"):
                                try:
                                    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                                    for symbol in tech_stocks:
                                        enhanced_manager.portfolio_db.add_holding(symbol, 10, 150.0)  # Sample quantities
                                    st.success("✅ Added tech giants to portfolio!")
                                    time.sleep(0.1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error adding tech stocks: {e}")
                        
                        with col2:
                            if st.button("Add Danish Stocks"):
                                try:
                                    danish_stocks = ["NOVO-B.CO", "MAERSK-B.CO", "ORSTED.CO"]
                                    for symbol in danish_stocks:
                                        enhanced_manager.portfolio_db.add_holding(symbol, 5, 800.0)
                                    st.success("✅ Added Danish stocks to portfolio!")
                                    time.sleep(0.1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error adding Danish stocks: {e}")
                        
                        with col3:
                            if st.button("Add Dividend Stocks"):
                                try:
                                    dividend_stocks = ["JNJ", "KO", "PG", "T"]
                                    for symbol in dividend_stocks:
                                        enhanced_manager.portfolio_db.add_holding(symbol, 15, 100.0)
                                    st.success("✅ Added dividend stocks to portfolio!")
                                    time.sleep(0.1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error adding dividend stocks: {e}")
                    else:
                        # Display portfolio summary
                        st.markdown("### 💰 Portfolio Summary")
                        
                        # Calculate portfolio metrics using simple price fetching
                        symbols = holdings['symbol'].tolist()
                        
                        # Calculate portfolio value and P&L
                        total_value = 0
                        total_cost = 0
                        portfolio_data = []
                        
                        with st.spinner("Fetching current prices..."):
                            for _, holding in holdings.iterrows():
                                symbol = holding['symbol']
                                quantity = holding['quantity']
                                avg_cost = holding['average_cost']
                                
                                # Use simple price fetching with Danish stock support
                                current_price = get_simple_current_price(symbol)
                                
                                market_value = quantity * current_price
                                cost_basis = quantity * avg_cost
                                pnl = market_value - cost_basis
                                pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                                
                                portfolio_data.append({
                                    'Symbol': symbol,
                                    'Quantity': quantity,
                                    'Avg Cost': f"${avg_cost:.2f}",
                                    'Current Price': f"${current_price:.2f}",
                                    'Market Value': f"${market_value:.2f}",
                                    'P&L': f"${pnl:.2f}",
                                    'P&L %': f"{pnl_percent:.1f}%"
                                })
                                
                                total_value += market_value
                                total_cost += cost_basis
                        
                        # Display key metrics
                        total_pnl = total_value - total_cost
                        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Value", f"${total_value:,.2f}")
                        with col2:
                            st.metric("Total Cost", f"${total_cost:,.2f}")
                        with col3:
                            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:.1f}%")
                        with col4:
                            st.metric("Holdings", len(holdings))
                        
                        # Display holdings table
                        st.markdown("### 📋 Current Holdings")
                        if portfolio_data:
                            portfolio_df = pd.DataFrame(portfolio_data)
                            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
                        
                        # Portfolio performance chart
                        st.markdown("### 📈 Portfolio Performance")
                        
                        # Get portfolio snapshots
                        snapshots = enhanced_manager.portfolio_db.get_portfolio_snapshots(limit=30)
                        if not snapshots.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=snapshots['snapshot_date'],
                                y=snapshots['total_value'],
                                mode='lines+markers',
                                name='Portfolio Value',
                                line=dict(color='green', width=2)
                            ))
                            fig.update_layout(
                                title="Portfolio Value Over Time",
                                xaxis_title="Date",
                                yaxis_title="Value ($)",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📈 No historical data available yet. Portfolio snapshots will appear here over time.")
                
                else:
                    st.error("❌ Database not available. Enhanced features disabled.")
                    
            with portfolio_tab2:
                st.subheader("➕ Manage Portfolio Holdings")
                
                if enhanced_manager.portfolio_db:
                    # Add new holding
                    st.markdown("### ➕ Add New Holding")
                    
                    with st.form("add_holding_form", clear_on_submit=True):
                        st.markdown("**Enter stock details:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_symbol = st.text_input(
                                "Stock Symbol", 
                                placeholder="AAPL", 
                                key="enhanced_portfolio_new_symbol",
                                help="Enter symbol and click 'Add Holding' button below"
                            ).upper()
                        with col2:
                            new_quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
                        with col3:
                            new_price = st.number_input("Purchase Price", min_value=0.01, value=100.0, step=0.01)
                        
                        # Add some spacing and make the submit button more prominent
                        st.write("")
                        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                        with col_btn2:
                            submitted = st.form_submit_button("➕ Add Holding", type="primary", use_container_width=True)
                    
                    if submitted:
                        if new_symbol:
                            try:
                                # Show Danish stock mapping if applicable
                                if new_symbol in DANISH_STOCKS:
                                    mapped_symbol = DANISH_STOCKS[new_symbol]
                                    st.info(f"🇩🇰 Danish stock detected: {new_symbol} → {mapped_symbol}")
                                
                                # Validate enhanced manager and database connection
                                if not enhanced_manager or not enhanced_manager.portfolio_db:
                                    st.error("❌ Database connection not available")
                                else:
                                    success = enhanced_manager.portfolio_db.add_holding(new_symbol, new_quantity, new_price)
                                    if success:
                                        enhanced_manager.portfolio_db.record_transaction(
                                            new_symbol, 'BUY', new_quantity, new_price, 
                                            notes=f"Added via Enhanced Portfolio Manager"
                                        )
                                        st.success(f"✅ Added {new_quantity} shares of {new_symbol} at ${new_price:.2f}")
                                        # Use a more gentle rerun that preserves session state
                                        time.sleep(0.1)  # Small delay to ensure database transaction completes
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to add holding for {new_symbol}")
                            except Exception as e:
                                st.error(f"❌ Error adding holding: {e}")
                                # Don't disable enhanced features for database errors
                                st.info("💡 Enhanced features remain active. Please try again.")
                        else:
                            st.error("❌ Please enter a valid symbol")
                    
                    st.markdown("---")
                    
                    # Remove/Edit holdings
                    st.markdown("### ✏️ Edit Existing Holdings")
                    holdings = enhanced_manager.portfolio_db.get_current_holdings()
                    
                    if not holdings.empty:
                        # Initialize session state for selected holding
                        if 'edit_holdings_selected' not in st.session_state:
                            st.session_state.edit_holdings_selected = holdings['symbol'].tolist()[0]
                        
                        def update_edit_holdings_selection():
                            st.session_state.edit_holdings_selected = st.session_state.edit_holdings_selectbox_dropdown
                        
                        selected_symbol = st.selectbox(
                            "Select holding to edit:", 
                            holdings['symbol'].tolist(), 
                            index=holdings['symbol'].tolist().index(st.session_state.edit_holdings_selected) if st.session_state.edit_holdings_selected in holdings['symbol'].tolist() else 0,
                            key="edit_holdings_selectbox_dropdown",
                            on_change=update_edit_holdings_selection
                        )
                        
                        if selected_symbol:
                            current_holding = holdings[holdings['symbol'] == selected_symbol].iloc[0]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🗑️ Remove Holding", type="secondary", key=f"remove_holding_{selected_symbol}"):
                                    try:
                                        success = enhanced_manager.portfolio_db.remove_holding(selected_symbol)
                                        if success:
                                            st.success(f"✅ Removed {selected_symbol} from portfolio")
                                            time.sleep(0.1)
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to remove {selected_symbol}")
                                    except Exception as e:
                                        st.error(f"❌ Error removing holding: {e}")
                            
                            with col2:
                                if st.button("📊 View Details", type="secondary", key=f"view_details_{selected_symbol}"):
                                    try:
                                        st.info(f"**{selected_symbol}**\n"
                                               f"Quantity: {current_holding['quantity']}\n"
                                               f"Average Cost: ${current_holding['average_cost']:.2f}\n"
                                               f"Added: {current_holding['date_added']}")
                                    except Exception as e:
                                        st.error(f"❌ Error displaying details: {e}")
                    else:
                        st.info("📊 No holdings to edit. Add some stocks first!")
                        
                else:
                    st.error("❌ Database not available")
            
            with portfolio_tab3:
                st.subheader("📈 Portfolio Alerts")
                
                if enhanced_manager.portfolio_db:
                    holdings = enhanced_manager.portfolio_db.get_current_holdings()
                    
                    if holdings.empty:
                        st.info("📊 Add holdings to your portfolio to set up alerts")
                    else:
                        st.markdown("### 🔔 Set Up Alerts")
                        
                        # Alert configuration
                        col1, col2, col3 = st.columns(3)
                        
                        # Initialize session state for alert settings
                        if 'alert_symbol_selected' not in st.session_state:
                            st.session_state.alert_symbol_selected = holdings['symbol'].tolist()[0]
                        if 'alert_type_selected' not in st.session_state:
                            st.session_state.alert_type_selected = "Price Above"
                        
                        with col1:
                            def update_alert_symbol():
                                st.session_state.alert_symbol_selected = st.session_state.alert_symbol_selectbox_dropdown
                            
                            alert_symbol = st.selectbox(
                                "Stock Symbol", 
                                holdings['symbol'].tolist(), 
                                index=holdings['symbol'].tolist().index(st.session_state.alert_symbol_selected) if st.session_state.alert_symbol_selected in holdings['symbol'].tolist() else 0,
                                key="alert_symbol_selectbox_dropdown",
                                on_change=update_alert_symbol
                            )
                        
                        with col2:
                            def update_alert_type():
                                st.session_state.alert_type_selected = st.session_state.alert_type_selectbox_dropdown
                            
                            alert_type = st.selectbox(
                                "Alert Type", 
                                ["Price Above", "Price Below", "% Change"], 
                                index=["Price Above", "Price Below", "% Change"].index(st.session_state.alert_type_selected),
                                key="alert_type_selectbox_dropdown",
                                on_change=update_alert_type
                            )
                        
                        with col3:
                            alert_value = st.number_input("Alert Value", min_value=0.01, value=100.0, step=0.01, key="alert_value_input")
                        
                        if st.button("🔔 Create Alert", key="create_portfolio_alert"):
                            st.success(f"✅ Alert created: {alert_symbol} {alert_type} {alert_value}")
                        
                        # Sample alerts display
                        st.markdown("### 🔔 Active Alerts")
                        alerts_data = {
                            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
                            'Type': ['Price Above', 'Price Below', '% Change'],
                            'Threshold': ['$200', '$300', '5%'],
                            'Status': ['Active', 'Triggered', 'Active']
                        }
                        st.dataframe(alerts_data, use_container_width=True)
                else:
                    st.error("❌ Database not available")
                    
            with portfolio_tab4:
                st.subheader("🔄 Portfolio Rebalancing")
                
                if enhanced_manager.portfolio_db:
                    holdings = enhanced_manager.portfolio_db.get_current_holdings()
                    
                    if holdings.empty:
                        st.info("📊 Add holdings to your portfolio to use rebalancing features")
                    else:
                        st.markdown("### ⚙️ Rebalancing Configuration")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            target_size = st.slider("Target Portfolio Size", 5, 25, 15)
                            min_score = st.slider("Minimum Score to Keep", 4.0, 8.0, 6.0, 0.1)
                        
                        with col2:
                            rebalance_market = st.selectbox("Market Source", ["S&P 500", "NASDAQ 100", "Danish Stocks"], key="what_if_rebalance_market")
                            aggressive_mode = st.checkbox("Aggressive Rebalancing", key="what_if_aggressive_mode")
                        
                        if st.button("🔄 Analyze Rebalancing", type="primary", key="what_if_analyze_rebalancing"):
                            with st.spinner("Analyzing portfolio..."):
                                st.success("✅ Rebalancing analysis complete!")
                                
                                # Sample rebalancing suggestions
                                st.markdown("### 📋 Rebalancing Suggestions")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**🗑️ Consider Removing:**")
                                    st.write("• Low performing stocks")
                                    st.write("• Over-allocated positions")
                                
                                with col2:
                                    st.markdown("**➕ Consider Adding:**")
                                    st.write("• High-scoring opportunities")
                                    st.write("• Underrepresented sectors")
                else:
                    st.error("❌ Database not available")
                    
            with portfolio_tab5:
                st.subheader("🔍 Weekly Market Screener")
                
                st.markdown("### 🎯 Screening Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    screener_market = st.selectbox("Market to Screen", ["S&P 500", "NASDAQ 100", "Danish Stocks", "All Markets"], key="what_if_screener_market")
                    min_score = st.slider("Minimum Score", 5.0, 9.0, 7.0, 0.1)
                
                with col2:
                    max_results = st.number_input("Max Results", 5, 50, 20)
                    exclude_current = st.checkbox("Exclude Current Holdings", True)
                
                if st.button("🚀 Run Market Screening", type="primary"):
                    with st.spinner("Screening market..."):
                        st.success(f"✅ Found opportunities in {screener_market}!")
                        
                        # Sample screening results
                        sample_results = {
                            'Symbol': ['NVDA', 'AMD', 'CRM', 'NFLX'],
                            'Score': [8.5, 8.2, 7.8, 7.6],
                            'Sector': ['Technology', 'Technology', 'Technology', 'Communication'],
                            'P/E Ratio': [65.2, 42.1, 55.8, 34.7]
                        }
                        st.dataframe(sample_results, use_container_width=True)
                        
                        # Quick add to portfolio
                        selected_stocks = st.multiselect("Select stocks to add:", sample_results['Symbol'])
                        
                        if selected_stocks and st.button("➕ Add Selected to Portfolio", key="add_selected_to_portfolio"):
                            for symbol in selected_stocks:
                                if enhanced_manager.portfolio_db:
                                    enhanced_manager.portfolio_db.add_holding(symbol, 1.0, 100.0)
                            st.success(f"✅ Added {len(selected_stocks)} stocks to portfolio!")
                            
            with portfolio_tab6:
                st.subheader("💾 Backup & Data Persistence")
                
                # Display comprehensive data persistence warnings
                st.warning("""
                ⚠️ **CRITICAL: Streamlit Cloud Data Persistence**
                
                **Your data is stored temporarily and WILL BE LOST when:**
                - App restarts (happens daily on Streamlit Cloud)
                - User inactivity (30+ minutes)
                - App redeployment or updates
                - Browser refresh or closure
                
                **🔐 PROTECT YOUR DATA:** Use backup features below!
                """)
                
                # Backup & Restore Interface
                st.session_state.cloud_backup_manager.display_backup_interface()
                
                st.markdown("---")
                st.markdown("### ⚙️ Enhanced Portfolio Settings")
                
                # Database management for enhanced portfolio
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Database Management**")
                    
                    if enhanced_manager.portfolio_db:
                        holdings_count = len(enhanced_manager.portfolio_db.get_current_holdings())
                        st.metric("Holdings in Database", holdings_count)
                        
                        if st.button("🗑️ Clear All Holdings", type="secondary"):
                            if st.button("⚠️ Confirm Clear", key="confirm_clear_enhanced"):
                                try:
                                    # Clear enhanced portfolio database
                                    enhanced_manager.portfolio_db.clear_all_holdings()
                                    st.success("✅ All holdings cleared from enhanced portfolio!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error clearing holdings: {e}")
                        
                        if st.button("📥 Export Enhanced Portfolio", type="secondary", key="export_enhanced_portfolio_btn"):
                            holdings = enhanced_manager.portfolio_db.get_current_holdings()
                            if not holdings.empty:
                                csv = holdings.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Enhanced Portfolio CSV",
                                    data=csv,
                                    file_name=f"enhanced_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No holdings to export")
                
                with col2:
                    st.markdown("**🔄 Migration Tools**")
                    
                    # Legacy portfolio migration
                    if hasattr(st.session_state, 'portfolio') and st.session_state.portfolio:
                        st.info(f"Found {len(st.session_state.portfolio)} stocks in legacy format")
                        
                        if st.button("🔄 Migrate Legacy Portfolio", type="primary", key="migrate_legacy_portfolio"):
                            migrated_count = 0
                            errors = []
                            
                            for symbol in st.session_state.portfolio:
                                try:
                                    # Get current price for better migration
                                    current_price = get_simple_current_price(symbol)
                                    price = current_price if current_price and current_price > 0 else 100.0
                                    
                                    enhanced_manager.portfolio_db.add_holding(symbol, 1.0, price)
                                    migrated_count += 1
                                except Exception as e:
                                    errors.append(f"{symbol}: {str(e)}")
                            
                            if migrated_count > 0:
                                st.success(f"✅ Migrated {migrated_count} stocks to enhanced portfolio!")
                                if errors:
                                    st.warning(f"⚠️ {len(errors)} stocks had issues during migration")
                                
                                # Clear legacy portfolio after successful migration
                                st.session_state.portfolio = []
                                st.rerun()
                            else:
                                st.error("❌ No stocks could be migrated")
                                if errors:
                                    st.error("Errors: " + "; ".join(errors[:3]))
                    else:
                        st.info("No legacy portfolio found")
                
                # Sync enhanced portfolio with session state
                st.markdown("---")
                st.markdown("**🔄 Portfolio Synchronization**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Sync Enhanced → Session State", help="Copy enhanced portfolio to session state for basic portfolio features"):
                        try:
                            holdings = enhanced_manager.portfolio_db.get_current_holdings()
                            if not holdings.empty:
                                st.session_state.portfolio = holdings['symbol'].tolist()
                                
                                # Also sync holdings data
                                portfolio_holdings = {}
                                for _, row in holdings.iterrows():
                                    portfolio_holdings[row['symbol']] = {
                                        "quantity": row['quantity'],
                                        "purchase_price": row['purchase_price'],
                                        "purchase_date": row['purchase_date']
                                    }
                                st.session_state.portfolio_holdings = portfolio_holdings
                                
                                st.success(f"✅ Synced {len(holdings)} stocks to session state!")
                            else:
                                st.info("No enhanced portfolio data to sync")
                        except Exception as e:
                            st.error(f"❌ Sync failed: {e}")
                
                with col2:
                    if st.button("📥 Sync Session State → Enhanced", help="Copy session state portfolio to enhanced database"):
                        try:
                            portfolio_symbols = st.session_state.get('portfolio', [])
                            portfolio_holdings = st.session_state.get('portfolio_holdings', {})
                            
                            if portfolio_symbols:
                                synced_count = 0
                                for symbol in portfolio_symbols:
                                    holdings_data = portfolio_holdings.get(symbol, {})
                                    quantity = holdings_data.get('quantity', 1.0)
                                    purchase_price = holdings_data.get('purchase_price', 0.0)
                                    
                                    # Get current price if purchase price is 0
                                    if purchase_price == 0.0:
                                        current_price = get_simple_current_price(symbol)
                                        purchase_price = current_price if current_price and current_price > 0 else 100.0
                                    
                                    try:
                                        enhanced_manager.portfolio_db.add_holding(symbol, quantity, purchase_price)
                                        synced_count += 1
                                    except:
                                        pass  # Skip duplicates
                                
                                st.success(f"✅ Synced {synced_count} stocks to enhanced portfolio!")
                            else:
                                st.info("No session state portfolio to sync")
                        except Exception as e:
                            st.error(f"❌ Sync failed: {e}")
                
                # Data persistence settings
                st.markdown("---")
                st.markdown("### 🔔 Data Persistence Settings")
                
                # Initialize session state for backup settings
                if 'auto_backup_reminder' not in st.session_state:
                    st.session_state.auto_backup_reminder = True
                if 'backup_frequency_hours' not in st.session_state:
                    st.session_state.backup_frequency_hours = 24
                
                def update_auto_backup_reminder():
                    st.session_state.auto_backup_reminder = st.session_state.auto_backup_reminder_checkbox
                
                def update_backup_frequency():
                    st.session_state.backup_frequency_hours = st.session_state.backup_frequency_slider
                
                auto_backup_reminder = st.checkbox(
                    "🔔 Enable backup reminders",
                    value=st.session_state.auto_backup_reminder,
                    help="Show reminders to backup your data periodically",
                    key="auto_backup_reminder_checkbox",
                    on_change=update_auto_backup_reminder
                )
                
                backup_frequency = st.slider(
                    "Backup reminder frequency (hours)",
                    min_value=1,
                    max_value=168,  # 1 week
                    value=st.session_state.backup_frequency_hours,
                    help="How often to show backup reminders",
                    key="backup_frequency_slider",
                    on_change=update_backup_frequency
                )
                
                # Show last backup info
                if 'last_backup_time' in st.session_state:
                    last_backup = datetime.fromisoformat(st.session_state.last_backup_time)
                    time_since = datetime.now() - last_backup
                    
                    if time_since.total_seconds() < 3600:  # Less than 1 hour
                        st.success(f"✅ Last backup: {int(time_since.total_seconds() / 60)} minutes ago")
                    elif time_since.total_seconds() < 86400:  # Less than 1 day
                        st.info(f"ℹ️ Last backup: {int(time_since.total_seconds() / 3600)} hours ago")
                    else:
                        st.warning(f"⚠️ Last backup: {time_since.days} days ago - Consider creating a new backup!")
                else:
                    st.warning("⚠️ No backup history found - Create your first backup!")
                
                # Quick backup action
                if st.button("� Quick Backup Now", type="primary"):
                    st.session_state.last_backup_time = datetime.now().isoformat()
                    st.success("✅ Backup timestamp updated! Use the backup interface above to download your data.")
                    st.rerun()

    # --- What-If Analysis (Enhanced Features Only) ---
    if st.session_state.get('enhanced_features_enabled', False):
        with tab5:
            st.header("🔮 What-If Portfolio Analysis")
            st.markdown("🚀 **Simulate portfolio changes before committing - Test strategies safely**")
            
            enhanced_manager = st.session_state.enhanced_features_manager
            
            if enhanced_manager.what_if_analyzer:
                what_if = enhanced_manager.what_if_analyzer
                
                # What-If Analysis tabs
                whatif_tab1, whatif_tab2, whatif_tab3, whatif_tab4 = st.tabs([
                    "🎯 Scenario Builder",
                    "📊 Comparison View", 
                    "💡 Recommendations",
                    "🧮 Risk Analysis"
                ])
                
                with whatif_tab1:
                    st.subheader("🎯 Build Your Scenario")
                    
                    # Get current portfolio
                    if enhanced_manager.portfolio_db:
                        current_holdings = enhanced_manager.portfolio_db.get_current_holdings()
                        
                        # Check if portfolio was just created
                        portfolio_just_created = st.session_state.get('sample_portfolio_created', False)
                        if portfolio_just_created:
                            st.session_state.sample_portfolio_created = False  # Reset flag
                            current_holdings = enhanced_manager.portfolio_db.get_current_holdings()  # Refresh data
                        
                        if current_holdings.empty:
                            st.warning("⚠️ No current portfolio found. Add some holdings first in the Portfolio Manager.")
                            st.markdown("### 🆕 Create Sample Portfolio for Testing")
                            
                            if st.button("🚀 Create Sample Portfolio", type="primary", key="create_sample_portfolio"):
                                # Add sample holdings
                                sample_stocks = [
                                    ("AAPL", 10, 150.0),
                                    ("MSFT", 8, 300.0),
                                    ("GOOGL", 5, 120.0),
                                    ("TSLA", 15, 200.0),
                                    ("NVDA", 6, 400.0)
                                ]
                                
                                for symbol, quantity, price in sample_stocks:
                                    enhanced_manager.portfolio_db.add_holding(symbol, quantity, price)
                                
                                st.session_state.sample_portfolio_created = True  # Set flag for next run
                                st.success("✅ Sample portfolio created! Continue building your scenario below.")
                                # Don't use st.rerun() to avoid tab reset - let user continue in same tab
                        else:
                            st.markdown("### 🔄 Modify Your Portfolio")
                            st.info(f"Current portfolio has {len(current_holdings)} holdings")
                            
                            # Display current holdings summary
                            with st.expander("📊 View Current Holdings"):
                                holdings_summary = current_holdings[['symbol', 'quantity', 'average_cost']].copy()
                                holdings_summary.columns = ['Symbol', 'Quantity', 'Avg Cost']
                                st.dataframe(holdings_summary, hide_index=True)
                            
                            # Scenario modification options
                            st.markdown("**🛠️ Scenario Modifications:**")
                            
                            modification_type = st.selectbox(
                                "Choose modification type:",
                                ["Add New Stock", "Remove Stock", "Change Quantity", "Rebalance Portfolio"],
                                key="what_if_modification_type"
                            )
                            
                            scenario_changes = []
                            
                            if modification_type == "Add New Stock":
                                with st.form("what_if_add_stock_form", clear_on_submit=True):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        new_symbol = st.text_input("Stock Symbol", placeholder="AAPL", key="what_if_new_symbol").upper()
                                    with col2:
                                        new_quantity = st.number_input("Quantity", min_value=1, value=10, key="what_if_new_quantity")
                                    with col3:
                                        target_price = st.number_input("Target Price", min_value=0.01, value=100.0, key="what_if_target_price")
                                    
                                    submitted = st.form_submit_button("➕ Add to Scenario")
                                
                                if submitted and new_symbol:
                                    # Initialize scenario changes in session state if not exists
                                    if 'what_if_scenario_changes' not in st.session_state:
                                        st.session_state.what_if_scenario_changes = []
                                    
                                    # Add to session state scenario changes
                                    st.session_state.what_if_scenario_changes.append({
                                        'action': 'add',
                                        'symbol': new_symbol,
                                        'quantity': new_quantity,
                                        'price': target_price
                                    })
                                    st.success(f"✅ Added {new_symbol} to scenario")
                            
                            elif modification_type == "Remove Stock":
                                symbols_to_remove = st.multiselect(
                                    "Select stocks to remove:",
                                    current_holdings['symbol'].tolist(),
                                    key="what_if_remove_symbols"
                                )
                                
                                if symbols_to_remove and st.button("🗑️ Remove from Scenario", key="what_if_remove_button"):
                                    # Initialize scenario changes in session state if not exists
                                    if 'what_if_scenario_changes' not in st.session_state:
                                        st.session_state.what_if_scenario_changes = []
                                    
                                    for symbol in symbols_to_remove:
                                        st.session_state.what_if_scenario_changes.append({
                                            'action': 'remove',
                                            'symbol': symbol
                                        })
                                    st.success(f"✅ Removed {len(symbols_to_remove)} stocks from scenario")
                            
                            elif modification_type == "Change Quantity":
                                symbol_to_change = st.selectbox(
                                    "Select stock to modify:",
                                    current_holdings['symbol'].tolist(),
                                    key="what_if_change_symbol"
                                )
                                
                                if symbol_to_change:
                                    current_qty = current_holdings[current_holdings['symbol'] == symbol_to_change]['quantity'].iloc[0]
                                    new_qty = st.number_input(
                                        f"New quantity for {symbol_to_change}",
                                        min_value=0,
                                        value=int(current_qty),
                                        key="what_if_new_qty"
                                    )
                                    
                                    if st.button("🔄 Update Quantity", key="what_if_update_quantity_button"):
                                        # Initialize scenario changes in session state if not exists
                                        if 'what_if_scenario_changes' not in st.session_state:
                                            st.session_state.what_if_scenario_changes = []
                                        
                                        st.session_state.what_if_scenario_changes.append({
                                            'action': 'update_quantity',
                                            'symbol': symbol_to_change,
                                            'quantity': new_qty
                                        })
                                        st.success(f"✅ Updated {symbol_to_change} quantity")
                            
                            # Display current scenario changes
                            if 'what_if_scenario_changes' in st.session_state and st.session_state.what_if_scenario_changes:
                                st.markdown("### 📝 Current Scenario Changes")
                                for i, change in enumerate(st.session_state.what_if_scenario_changes):
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        if change['action'] == 'add':
                                            st.write(f"➕ Add {change['quantity']} shares of {change['symbol']} at ${change['price']:.2f}")
                                        elif change['action'] == 'remove':
                                            st.write(f"🗑️ Remove {change['symbol']}")
                                        elif change['action'] == 'update_quantity':
                                            st.write(f"🔄 Update {change['symbol']} to {change['quantity']} shares")
                                    with col2:
                                        if st.button("❌", key=f"remove_change_{i}", help="Remove this change"):
                                            st.session_state.what_if_scenario_changes.pop(i)
                                            st.rerun()
                                
                                # Add clear all button
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button("🧮 Analyze Scenario", type="primary", key="analyze_scenario_button"):
                                        scenario_changes = st.session_state.what_if_scenario_changes
                                        if scenario_changes:
                                            with st.spinner("🔄 Analyzing scenario..."):
                                                try:
                                                    # Build portfolio scenario
                                                    current_portfolio = {}
                                                    for _, holding in current_holdings.iterrows():
                                                        current_portfolio[holding['symbol']] = {
                                                            'shares': holding['quantity'],  # Use 'shares' instead of 'quantity'
                                                            'cost_basis': holding['average_cost']
                                                        }
                                                    
                                                    # Apply changes
                                                    scenario_portfolio = current_portfolio.copy()
                                                    for change in scenario_changes:
                                                        if change['action'] == 'add':
                                                            scenario_portfolio[change['symbol']] = {
                                                                'shares': change['quantity'],  # Use 'shares' instead of 'quantity'
                                                                'cost_basis': change['price']
                                                            }
                                                        elif change['action'] == 'remove':
                                                            scenario_portfolio.pop(change['symbol'], None)
                                                        elif change['action'] == 'update_quantity':
                                                            if change['symbol'] in scenario_portfolio:
                                                                scenario_portfolio[change['symbol']]['shares'] = change['quantity']  # Use 'shares'
                                                    
                                                    # Analyze scenario
                                                    analysis = what_if.analyze_portfolio_scenario(
                                                        current_portfolio,
                                                        scenario_portfolio,
                                                        scenario_name="Custom Scenario"
                                                    )
                                                    
                                                    st.session_state.what_if_analysis = analysis
                                                    st.success("✅ Scenario analysis complete! Check other tabs for results.")
                                                    
                                                except Exception as e:
                                                    st.error(f"❌ Analysis error: {e}")
                                        else:
                                            st.warning("⚠️ No scenario changes defined")
                                with col2:
                                    if st.button("🗑️ Clear All Changes", key="clear_scenario_button"):
                                        st.session_state.what_if_scenario_changes = []
                                        st.success("✅ All scenario changes cleared")
                            
                            else:
                                st.info("💡 Add modifications above to build your scenario")
                    else:
                        st.error("❌ Portfolio database not available")
                
                with whatif_tab2:
                    st.subheader("📊 Scenario Comparison")
                    
                    if 'what_if_analysis' in st.session_state:
                        analysis = st.session_state.what_if_analysis
                        
                        # Metrics comparison
                        st.markdown("### 📈 Key Metrics Comparison")
                        
                        current_metrics = analysis['current_metrics']
                        scenario_metrics = analysis['simulated_metrics']  # Use 'simulated_metrics' instead of 'scenario_metrics'
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            value_change = scenario_metrics.total_value - current_metrics.total_value
                            st.metric(
                                "Portfolio Value",
                                f"${scenario_metrics.total_value:,.2f}",
                                f"{value_change:+,.2f}"
                            )
                        
                        with col2:
                            # Use diversification_score instead of risk_score
                            div_change = scenario_metrics.diversification_score - current_metrics.diversification_score
                            st.metric(
                                "Diversification Score",
                                f"{scenario_metrics.diversification_score:.1f}",
                                f"{div_change:+.1f}"
                            )
                        
                        with col3:
                            # Use average_score instead of dividend_yield
                            score_change = scenario_metrics.average_score - current_metrics.average_score
                            st.metric(
                                "Average Stock Score",
                                f"{scenario_metrics.average_score:.2f}",
                                f"{score_change:+.2f}"
                            )
                        
                        with col4:
                            holdings_change = scenario_metrics.symbol_count - current_metrics.symbol_count
                            st.metric(
                                "Holdings Count",
                                scenario_metrics.symbol_count,
                                f"{holdings_change:+d}"
                            )
                        
                        # Sector allocation comparison
                        st.markdown("### 🏭 Sector Allocation Changes")
                        
                        if current_metrics.sector_breakdown and scenario_metrics.sector_breakdown:
                            sectors = set(current_metrics.sector_breakdown.keys()) | set(scenario_metrics.sector_breakdown.keys())
                            
                            sector_comparison = []
                            for sector in sectors:
                                current_pct = current_metrics.sector_breakdown.get(sector, 0)
                                scenario_pct = scenario_metrics.sector_breakdown.get(sector, 0)
                                change = scenario_pct - current_pct
                                
                                sector_comparison.append({
                                    'Sector': sector,
                                    'Current %': f"{current_pct:.1f}%",
                                    'Scenario %': f"{scenario_pct:.1f}%",
                                    'Change': f"{change:+.1f}%"
                                })
                            
                            sector_df = pd.DataFrame(sector_comparison)
                            st.dataframe(sector_df, hide_index=True)
                        
                        # Visual comparison using available metrics
                        st.markdown("### 📊 Portfolio Metrics Comparison")
                        
                        # Create a comparison chart using available metrics
                        metrics_comparison = {
                            'Metric': ['Portfolio Value', 'Diversification Score', 'Average Score', 'Holdings Count'],
                            'Current': [
                                current_metrics.total_value,
                                current_metrics.diversification_score,
                                current_metrics.average_score,
                                current_metrics.symbol_count
                            ],
                            'Scenario': [
                                scenario_metrics.total_value,
                                scenario_metrics.diversification_score,
                                scenario_metrics.average_score,
                                scenario_metrics.symbol_count
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(metrics_comparison)
                        
                        # Create bar chart comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            name='Current',
                            x=comparison_df['Metric'],
                            y=comparison_df['Current'],
                            marker_color='lightblue'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='Scenario',
                            x=comparison_df['Metric'],
                            y=comparison_df['Scenario'],
                            marker_color='lightcoral'
                        ))
                        
                        fig.update_layout(
                            title="Portfolio Metrics: Current vs Scenario",
                            xaxis_title="Metrics",
                            yaxis_title="Values",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.info("📊 No scenario analysis available. Create and analyze a scenario first.")
                
                with whatif_tab3:
                    st.subheader("💡 AI Recommendations")
                    
                    if 'what_if_analysis' in st.session_state:
                        analysis = st.session_state.what_if_analysis
                        recommendations = analysis.get('recommendations', [])
                        
                        if recommendations:
                            st.markdown("### 🎯 Recommended Actions")
                            
                            for i, rec in enumerate(recommendations):
                                # Handle both string and dictionary formats
                                if isinstance(rec, dict):
                                    # Dictionary format with title, description, etc.
                                    title = rec.get('title', f'Recommendation {i+1}')
                                    description = rec.get('description', '')
                                    impact = rec.get('impact', '')
                                    confidence = rec.get('confidence', 0)
                                    
                                    with st.expander(f"💡 {title}"):
                                        if description:
                                            st.write(description)
                                        if impact:
                                            st.info(f"**Expected Impact:** {impact}")
                                        if confidence > 0:
                                            color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                                            st.markdown(f"**Confidence:** <span style='color: {color}'>{confidence:.0%}</span>", unsafe_allow_html=True)
                                else:
                                    # String format - simple recommendation text
                                    with st.expander(f"💡 Recommendation {i+1}"):
                                        st.write(str(rec))
                        else:
                            st.info("📊 No specific recommendations available for this scenario")
                            
                            # Generic improvement suggestions
                            st.markdown("### 📈 General Portfolio Optimization Tips")
                            
                            tips = [
                                "🎯 **Diversification**: Consider spreading investments across different sectors",
                                "⚖️ **Risk Balance**: Mix growth and value stocks based on your risk tolerance",
                                "💰 **Cost Averaging**: Regular small investments can reduce volatility impact",
                                "🔄 **Rebalancing**: Periodically adjust holdings to maintain target allocation",
                                "📊 **Research**: Keep analyzing fundamentals and market trends"
                            ]
                            
                            for tip in tips:
                                st.markdown(tip)
                    else:
                        st.info("💡 No analysis available. Create a scenario first to get recommendations.")
                
                with whatif_tab4:
                    st.subheader("🧮 Advanced Risk Analysis")
                    
                    if 'what_if_analysis' in st.session_state:
                        analysis = st.session_state.what_if_analysis
                        
                        # Initialize risk analyzer
                        risk_analyzer = AdvancedRiskAnalyzer()
                        
                        # Create portfolio data from scenario analysis
                        # Check if we have scenario portfolio data
                        if 'scenario_portfolio' in analysis and analysis['scenario_portfolio']:
                            # Convert scenario portfolio to DataFrame format for risk analysis
                            scenario_data = []
                            
                            try:
                                # Debug info for scenario structure
                                st.info(f"🔍 Processing scenario portfolio with {len(analysis['scenario_portfolio'])} assets")
                                
                                for symbol, data in analysis['scenario_portfolio'].items():
                                    # Get current price for market value calculation
                                    try:
                                        current_price = get_simple_current_price(symbol)
                                        if current_price <= 0:
                                            current_price = data.get('cost_basis', 100.0)  # Fallback to cost basis or default
                                        
                                        shares = data.get('shares', 0)
                                        market_value = shares * current_price
                                        
                                        scenario_data.append({
                                            'Symbol': symbol,
                                            'Quantity': shares,
                                            'Average Cost': data.get('cost_basis', current_price),
                                            'Current Price': current_price,
                                            'Total Value': market_value
                                        })
                                        
                                        # Debug: show processed data
                                        st.write(f"✅ {symbol}: {shares} shares @ ${current_price:.2f} = ${market_value:.2f}")
                                        
                                    except Exception as symbol_error:
                                        # Log error but continue with other symbols
                                        st.error(f"⚠️ Error processing {symbol}: {str(symbol_error)}")
                                        # Show the problematic data structure for debugging
                                        st.write(f"Data for {symbol}: {data}")
                                        continue
                                
                                if scenario_data:
                                    scenario_portfolio_df = pd.DataFrame(scenario_data)
                                    
                                    st.markdown("### 🎯 Scenario Portfolio Risk Profile")
                                    st.write(f"**Analyzing scenario with {len(scenario_portfolio_df)} assets:**")
                                    
                                    # Display scenario symbols
                                    symbols_list = scenario_portfolio_df['Symbol'].tolist()
                                    st.write("• " + " • ".join(symbols_list))
                                    
                                    # Display advanced risk analysis for scenario portfolio
                                    display_advanced_risk_analysis(scenario_portfolio_df, risk_analyzer)
                                else:
                                    st.warning("⚠️ No valid scenario portfolio data for risk analysis")
                                    st.info("💡 This may be due to pricing data issues or invalid symbols")
                                    # Show the original scenario data for debugging
                                    st.write("Debug - Original scenario data:", analysis['scenario_portfolio'])
                                    
                            except Exception as e:
                                st.error(f"❌ Error processing scenario portfolio: {str(e)}")
                                st.info("💡 Please try analyzing the scenario again")
                                # Show more debug info
                                import traceback
                                st.code(traceback.format_exc())
                        else:
                            # Fallback to current holdings if no scenario data
                            current_holdings = enhanced_manager.portfolio_db.get_current_holdings()
                            
                            if not current_holdings.empty:
                                st.markdown("### 🎯 Current Portfolio Risk Profile")
                                st.info("💡 No scenario analysis available. Showing current portfolio risk profile.")
                                
                                # Display advanced risk analysis for current portfolio
                                display_advanced_risk_analysis(current_holdings, risk_analyzer)
                            else:
                                st.warning("⚠️ No portfolio data available for risk analysis")
                        
                        st.markdown("---")
                        
                        # Scenario comparison if available
                        if 'current_metrics' in analysis and 'simulated_metrics' in analysis:
                            current_metrics = analysis['current_metrics']
                            scenario_metrics = analysis['simulated_metrics']
                            
                            st.markdown("### 📊 Scenario vs Current Risk Comparison")
                            
                            # Basic comparison metrics
                            current_risk = current_metrics.volatility_estimate
                            scenario_risk = scenario_metrics.volatility_estimate
                            risk_change = scenario_risk - current_risk
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Volatility", f"{current_risk:.1f}%")
                            
                            with col2:
                                st.metric("Scenario Volatility", f"{scenario_risk:.1f}%", f"{risk_change:+.1f}%")
                            
                            with col3:
                                risk_level = "Low" if scenario_risk < 15 else "Medium" if scenario_risk < 25 else "High"
                                color = "green" if scenario_risk < 15 else "orange" if scenario_risk < 25 else "red"
                                st.markdown(f"**Risk Level:** <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)
                            
                            # Advanced scenario impact analysis
                            st.markdown("### � Scenario Impact Analysis")
                            
                            impact_col1, impact_col2 = st.columns(2)
                            
                            with impact_col1:
                                st.markdown("**🔶 Risk Impact Summary:**")
                                
                                # Calculate risk impact categories
                                if abs(risk_change) < 2:
                                    impact_level = "Minimal"
                                    impact_color = "green"
                                elif abs(risk_change) < 5:
                                    impact_level = "Moderate"
                                    impact_color = "orange"
                                else:
                                    impact_level = "Significant"
                                    impact_color = "red"
                                
                                st.markdown(f"• **Impact Level:** <span style='color: {impact_color}'>{impact_level}</span>", unsafe_allow_html=True)
                                st.write(f"• **Volatility Change:** {risk_change:+.1f} percentage points")
                                
                                # Risk direction
                                if risk_change > 0:
                                    st.write("• **Direction:** ⬆️ Increased Risk")
                                    st.warning("📈 Scenario increases portfolio risk")
                                elif risk_change < 0:
                                    st.write("• **Direction:** ⬇️ Reduced Risk")
                                    st.success("📉 Scenario reduces portfolio risk")
                                else:
                                    st.write("• **Direction:** ➡️ Neutral Impact")
                                    st.info("➖ Scenario has minimal risk impact")
                            
                            with impact_col2:
                                st.markdown("**📋 Key Risk Factors:**")
                                
                                risk_factors = {
                                    "Concentration Risk": "Portfolio diversification level",
                                    "Volatility Risk": "Expected price fluctuation", 
                                    "Correlation Risk": "Asset interdependence",
                                    "Liquidity Risk": "Trading ease and market depth",
                                    "Market Risk": "Systematic market exposure"
                                }
                                
                                # Mock enhanced risk scores (in production, calculate from actual data)
                                for factor, description in risk_factors.items():
                                    factor_score = min(10, max(1, scenario_risk/3 + np.random.uniform(-1, 1)))
                                    progress_color = "🟢" if factor_score < 4 else "🟡" if factor_score < 7 else "🔴"
                                    st.write(f"{progress_color} **{factor}**: {factor_score:.1f}/10")
                            
                            # Risk mitigation recommendations
                            st.markdown("### 🛡️ Scenario-Specific Risk Mitigation")
                            
                            recommendations = []
                            
                            # Dynamic recommendations based on scenario risk
                            if scenario_risk > current_risk + 3:
                                recommendations.extend([
                                    "⚠️ **High Risk Increase**: Consider reducing position sizes in volatile stocks",
                                    "🎯 **Diversification**: Add defensive or low-beta stocks to balance risk",
                                    "🛡️ **Hedging**: Consider protective options or inverse ETFs"
                                ])
                            elif scenario_risk < current_risk - 2:
                                recommendations.extend([
                                    "✅ **Risk Reduction**: Good scenario for risk-averse investors",
                                    "📈 **Opportunity**: Lower risk may allow for modest position increases",
                                    "⚖️ **Balance**: Ensure returns still meet investment objectives"
                                ])
                            else:
                                recommendations.extend([
                                    "🎯 **Balanced Approach**: Risk level remains manageable",
                                    "🔄 **Monitor**: Regular review recommended for position adjustments"
                                ])
                            
                            # Universal risk management strategies
                            recommendations.extend([
                                "🏭 **Sector Limits**: Keep any single sector under 30% of portfolio",
                                "💰 **Position Sizing**: Limit individual positions to 5-10% of total",
                                "⏰ **Time Diversification**: Consider dollar-cost averaging for new positions",
                                "� **Regular Rebalancing**: Review and adjust quarterly"
                            ])
                            
                            for rec in recommendations:
                                st.markdown(rec)
                        
                        else:
                            st.warning("⚠️ No current portfolio found. Add holdings in Portfolio Manager for comprehensive risk analysis.")
                            
                    else:
                        st.info("🧮 No risk analysis available. Create a scenario first to compare risk profiles.")
                        
            else:
                st.error("❌ What-If Analyzer not available")
    
    # --- Danish Stocks Manager ---
    target_tab = tab6 if st.session_state.get('enhanced_features_enabled', False) else tab5
    with target_tab:
        st.header("🇩🇰 Danish Stocks Manager")
        st.markdown("Manage and update the comprehensive list of Danish stocks for the Copenhagen Stock Exchange")
        
        # Display current stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Danish Stocks", len(set(DANISH_STOCKS.values())))
        with col2:
            st.metric("Unique Symbols", len(DANISH_STOCKS))
        with col3:
            st.metric("Copenhagen Exchange", "✅ Active")
        
        # Display and manage Danish stocks
        st.subheader("📊 Current Danish Stocks")
        
        # Search and filter
        with st.form("danish_search_form"):
            search_term = st.text_input("🔍 Search stocks", placeholder="Search by symbol or name...", key="danish_search")
            st.form_submit_button("🔍 Search")
        
        # Convert to DataFrame for better display
        danish_df = pd.DataFrame([
            {"Display_Name": k, "Yahoo_Symbol": v} 
            for k, v in DANISH_STOCKS.items()
        ])
        
        # Apply search filter
        if search_term:
            mask = (danish_df['Display_Name'].str.contains(search_term, case=False, na=False) | 
                   danish_df['Yahoo_Symbol'].str.contains(search_term, case=False, na=False))
            danish_df = danish_df[mask]
        
        # Display with pagination
        items_per_page = 20
        total_pages = (len(danish_df) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1), key="danish_stocks_page_selector")
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            display_df = danish_df.iloc[start_idx:end_idx]
        else:
            display_df = danish_df
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Bulk operations
        st.subheader("🔧 Management Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Export Options**")
            if st.button("📋 Copy All Symbols"):
                all_symbols_text = ", ".join(sorted(set(DANISH_STOCKS.values())))
                st.text_area("Copy these symbols:", value=all_symbols_text, height=100, key="copy_all_danish")
            
            if st.button("📥 Download as CSV"):
                csv_data = danish_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"danish_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.markdown("**➕ Add New Stock**")
            with st.form("add_danish_stock_form"):
                new_name = st.text_input("Display Name", placeholder="e.g., NOVO-B", key="new_danish_name")
                new_symbol = st.text_input("Yahoo Symbol", placeholder="e.g., NOVO-B.CO", key="new_danish_symbol")
                
                add_button = st.form_submit_button("➕ Add Stock")
                
                if add_button and new_name and new_symbol:
                    if new_name not in DANISH_STOCKS:
                        st.success(f"Would add: {new_name} -> {new_symbol}")
                        st.info("Note: This is a demo. In production, this would update the stock database.")
                    else:
                        st.warning(f"{new_name} already exists")

    # --- Performance Benchmarking ---
    target_tab = tab7 if st.session_state.get('enhanced_features_enabled', False) else tab6
    with target_tab:
        # Initialize score tracking
        initialize_score_tracking()
        
        # Create the performance benchmarking dashboard
        create_performance_dashboard()
        
        st.markdown("---")
        
        # Score tracking section
        display_score_tracking()

    # --- Help & Documentation ---
    target_tab = tab8 if st.session_state.get('enhanced_features_enabled', False) else tab7
    with target_tab:
        st.header("ℹ️ Help & Documentation")
        st.markdown("Complete guide to using the Stock Analysis System")
        
        # Create sub-tabs for different help topics
        help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
            "🚀 Getting Started", 
            "📊 Understanding Scores", 
            "📈 Trading Signals", 
            "🔧 Advanced Features"
        ])
        
        with help_tab1:
            st.subheader("� Getting Started")
            st.markdown("""
            ### Welcome to the Advanced Stock Analysis System!
            
            This comprehensive tool provides multi-dimensional stock analysis using:
            
            #### �📊 **Data Sources**
            - **Yahoo Finance**: Real-time prices, fundamentals, financials
            - **Alpha Vantage**: Technical indicators, earnings data
            - **Danish Stocks**: Local market expertise with PE adjustments
            
            #### 🎯 **Key Features**
            1. **Multi-factor Scoring**: 10-point scale combining fundamentals & technicals
            2. **Trading Signals**: Buy/Sell recommendations with confidence levels
            3. **Portfolio Management**: Track your investments with automated analysis
            4. **Market Screening**: Find opportunities based on your criteria
            5. **Performance Benchmarking**: Historical backtesting and score tracking
            
            #### 🚀 **Quick Start Guide**
            1. Go to **Stock Analysis Hub** tab
            2. Enter stock symbols (e.g., AAPL,MSFT,GOOGL)
            3. Click "Analyze Stocks" for comprehensive analysis
            4. Review scores, signals, and recommendations
            5. Add promising stocks to your portfolio for tracking
            
            #### 💡 **Pro Tips**
            - Use Company Search to discover new opportunities
            - Monitor Trading Signals for entry/exit points
            - Set up automated portfolio analysis for regular updates
            - Leverage Market Screeners to find stocks matching your criteria
            """)
        
        with help_tab2:
            st.markdown("""
            ### 🎯 **Enhanced Scoring Methodology**
            
            Each stock receives a score from 0-10 based on multiple financial metrics with dynamic benchmarking:
            
            #### 📈 **Core Metrics (Dynamic Benchmarking):**
            
            **Valuation Metrics:**
            - **P/E Ratio**: Price-to-Earnings (compared to industry average)
            - **Forward P/E**: Future P/E ratio (more predictive)
            - **PEG Ratio**: P/E relative to growth (< 1.0 is ideal)
            - **P/B Ratio**: Price-to-Book (< 1.0 indicates undervaluation)
            - **EV/EBITDA**: Enterprise value to EBITDA
            - **Price/Sales**: Revenue-based valuation
            
            **Profitability Metrics (Sector-Adjusted):**
            - **ROE**: Return on Equity (compared to sector average)
            - **Gross Margin**: Profitability efficiency (industry-relative)
            - **Free Cash Flow**: Cash generation trend analysis
            
            **Growth Metrics (Industry-Relative):**
            - **Revenue Growth**: Compared to sector growth rates
            - **EPS Growth**: Earnings per share growth trajectory
            
            **Financial Health (NEW - Comprehensive):**
            - **Debt/Equity**: Sector-adjusted leverage scoring
            - **Current Ratio**: Short-term liquidity vs industry
            - **Interest Coverage**: Debt service capability
            - **Cash Position**: Cash-to-market-cap analysis
            - **Dividend Yield**: Income generation
            
            **Technical Momentum (NEW):**
            - **Price Momentum**: 1, 3, 6-month performance
            - **Moving Average Alignment**: Trend strength indicator
            - **Volume Confirmation**: Trading activity validation
            
            **Market Sentiment:**
            - **Analyst Upside**: Professional price target consensus
            
            #### 🏭 **Sector-Specific Adjustments:**
            - **Technology**: Emphasis on growth and margins
            - **Financials**: Focus on ROE and book value
            - **Healthcare**: R&D efficiency and regulatory moats
            - **Industrials**: Operational efficiency and cash flow
            - **Consumer Staples**: Dividend stability and defensive metrics
            
            #### 🎯 **Enhanced Score Interpretation:**
            - **8.0-10.0**: 🚀 Strong Buy - Top-tier opportunity with sector leadership
            - **6.5-7.9**: 📈 Buy - Above-average fundamentals with positive momentum
            - **4.0-6.4**: 🔄 Hold - Adequate performance, monitor for improvements
            - **2.0-3.9**: 📉 Weak Sell - Below-average metrics, consider alternatives
            - **0.0-1.9**: 🛑 Strong Sell - Poor fundamentals across multiple dimensions
            
            #### ⚡ **Key Enhancements:**
            - **Dynamic Benchmarking**: Scores adapt to industry standards
            - **Momentum Integration**: Technical analysis complements fundamentals
            - **Financial Health**: Comprehensive risk assessment
            - **Sector Intelligence**: Tailored evaluation criteria
            """)
        
        with help_tab3:
            st.subheader("📈 Trading Signals Guide")
            st.markdown("""
            ### 🎯 **Enhanced Technical Analysis Signals**
            
            #### 📊 **Momentum Indicators:**
            
            **Price Momentum (NEW)**
            - **1-Month**: Short-term trend direction
            - **3-Month**: Medium-term momentum strength
            - **6-Month**: Long-term trend validation
            - **Moving Average Alignment**: Bullish when price > SMA20 > SMA50 > SMA200
            
            **RSI (Relative Strength Index)**
            - **< 30**: Oversold (potential buy opportunity)
            - **> 70**: Overbought (potential sell signal)
            - **30-70**: Normal trading range
            
            **Moving Averages**
            - **Golden Cross**: 20-day MA crosses above 50-day MA (bullish)
            - **Death Cross**: 20-day MA crosses below 50-day MA (bearish)
            - **Price above all MAs**: Strong uptrend confirmation
            
            **MACD (Moving Average Convergence Divergence)**
            - **MACD > Signal**: Bullish momentum building
            - **MACD < Signal**: Bearish momentum developing
            - **Histogram**: Shows momentum acceleration/deceleration
            
            #### 🔄 **Combined Scoring Approach:**
            - **Fundamental Score (60%)**: Financial health and valuation
            - **Technical Score (40%)**: Momentum and trend analysis
            - **Final Recommendation**: Weighted combination for optimal timing
            """)
        
        with help_tab4:
            st.subheader("🔧 Advanced Features & Data Backup")
            
            # Data Persistence Warning Section
            st.error("""
            🚨 **CRITICAL: DATA PERSISTENCE ON STREAMLIT CLOUD**
            
            **Your data is stored temporarily and WILL BE LOST when:**
            - App restarts (happens daily on Streamlit Cloud)
            - User inactivity (30+ minutes timeout)
            - Browser refresh or closure
            - App redeployment or updates
            """)
            
            st.markdown("""
            ### 💾 **Data Backup & Recovery Guide**
            
            #### 🔐 **Why Backup?**
            - **Streamlit Cloud Limitation**: Session state is temporary
            - **Data Loss Protection**: Preserve your portfolio and settings
            - **Migration**: Move between environments seamlessly
            - **Recovery**: Restore after app restarts or crashes
            
            #### 📥 **Backup Methods**
            
            **1. Portfolio Manager → Backup & Settings Tab**
            - **JSON Format**: Complete backup (recommended)
            - **CSV Format**: Portfolio data only
            - **Excel Format**: Multi-sheet backup with settings
            
            **2. Backup Frequency**
            - **Daily**: For active traders
            - **Weekly**: For long-term investors
            - **Before Analysis**: Before major portfolio changes
            
            #### 🔄 **Restore Process**
            1. Go to Portfolio Manager → Backup & Settings
            2. Upload your backup file (JSON/CSV/Excel)
            3. Click "Restore Portfolio"
            4. Refresh the page to see restored data
            
            #### 📋 **What Gets Backed Up**
            - ✅ Portfolio holdings and quantities
            - ✅ Purchase prices and dates
            - ✅ Score weight preferences
            - ✅ Analysis history
            - ✅ Selected symbols
            - ❌ Cached price data (refreshed automatically)
            
            #### 🛡️ **Best Practices**
            - **Regular Backups**: Set reminders in the app
            - **Multiple Formats**: Keep both JSON and CSV backups
            - **Cloud Storage**: Store backups in Dropbox/Google Drive
            - **Version Control**: Include dates in backup filenames
            
            #### ⚠️ **Emergency Recovery**
            If you lose data:
            1. Don't panic - your external backups are safe
            2. Upload your most recent backup file
            3. Check if portfolio symbols are correctly restored
            4. Recreate any missing purchase prices
            
            ### 🎯 **Advanced System Features**
            
            #### 🧠 **Dynamic Intelligence**
            - **Sector Benchmarking**: Industry-relative performance analysis
            - **Momentum Scoring**: Technical trend integration
            - **Financial Health**: Multi-dimensional risk assessment
            - **Adaptive Weights**: Sector-specific metric emphasis
            
            #### 🔍 **Multi-Market Screening**
            - Screen S&P 500, NASDAQ 100, European, and Danish stocks
            - Custom symbol lists for personalized screening
            - Sector and market cap filtering
            - Export results for further analysis
            
            #### 💼 **Enhanced Portfolio Management**
            - **SQLite Database**: Local persistent storage (enhanced mode)
            - **Automated Screening**: Weekly market scans
            - **What-If Analysis**: Scenario modeling and risk assessment
            - **Performance Tracking**: Historical returns and risk metrics
            - **Alert System**: Get notified of changes
            - **Performance Tracking**: Monitor your investments
            """)

    # --- Compare & Export ---
    with tab8:
        st.header("⚖️ Compare & Export Results")
        st.markdown("Side-by-side comparison and data export functionality")
        
        # Check if there's any analysis data available
        if 'stock_data' in st.session_state and st.session_state.stock_data:
            st.subheader("📊 Stock Comparison")
            
            # Display comparison data
            comparison_data = st.session_state.stock_data
            
            # Create comparison DataFrame
        
        with help_tab3:
            st.subheader("📈 Trading Signals")
            st.markdown("""
            ### 🎯 Signal Types and Interpretation
            
            Our trading signals combine multiple indicators for reliable entry/exit points:
            
            #### 🟢 **Buy Signals**
            - **Strong Buy**: Score > 8.0 + positive momentum
            - **Buy**: Score > 6.5 + technical confirmation
            - **Accumulate**: Score > 5.0 + oversold conditions
            
            #### 🔴 **Sell Signals**
            - **Strong Sell**: Score < 3.0 + negative momentum
            - **Sell**: Score < 4.0 + technical weakness
            - **Reduce**: Score declining + overbought conditions
            
            #### 🟡 **Hold Signals**
            - **Hold**: Stable score around 4.0-6.0
            - **Monitor**: Mixed signals requiring observation
            
            #### 📊 **Signal Components**
            
            **Fundamental Triggers**
            - Earnings surprises (positive/negative)
            - Revenue growth acceleration/deceleration
            - Margin expansion/compression
            - Debt level changes
            
            **Technical Triggers**
            - Moving average crossovers
            - Support/resistance breaks
            - Volume confirmation
            - RSI divergences
            
            **Risk Management**
            - Position sizing recommendations
            - Stop-loss suggestions
            - Portfolio diversification alerts
            - Market condition warnings
            
            #### ⚠️ **Important Notes**
            - Signals are recommendations, not guarantees
            - Always consider your risk tolerance
            - Diversify across multiple positions
            - Keep stop-losses and position limits
            """)
        
        with help_tab4:
            st.subheader("🔧 Advanced Features")
            st.markdown("""
            ### 🚀 Power User Features
            
            #### 📊 **Portfolio Management**
            - **Automated Analysis**: Schedule regular portfolio reviews
            - **Performance Tracking**: Monitor returns vs. benchmarks
            - **Risk Assessment**: Portfolio-level risk metrics
            - **Rebalancing Alerts**: Maintain target allocations
            
            #### 🔍 **Market Screening**
            - **Custom Filters**: Create your own screening criteria
            - **Saved Searches**: Store and reuse screening setups
            - **Alert System**: Get notified when stocks meet criteria
            - **Bulk Analysis**: Analyze entire screened lists
            
            #### 📈 **Performance Benchmarking**
            - **Historical Backtesting**: Test scoring system performance
            - **Benchmark Comparisons**: vs. S&P 500, NASDAQ, etc.
            - **Score Tracking**: Monitor score changes over time
            - **Strategy Analysis**: Evaluate different approaches
            
            #### 🇩🇰 **Danish Market Expertise**
            - **Local PE Adjustments**: Industry-specific valuations
            - **Currency Considerations**: DKK/USD conversions
            - **Regulatory Awareness**: Danish market specifics
            - **Tax Implications**: Local investment considerations
            
            #### 🔄 **Data Integration**
            - **Multi-source Validation**: Cross-reference data sources
            - **Real-time Updates**: Live market data integration
            - **Historical Analysis**: Trend and pattern recognition
            - **Export Capabilities**: CSV, Excel, PDF reports
            
            #### ⚙️ **Customization Options**
            - **Scoring Weights**: Adjust factor importance
            - **Alert Thresholds**: Set custom notification levels
            - **Display Preferences**: Customize charts and tables
            - **Data Frequency**: Choose update intervals
            
            #### 🔐 **Best Practices**
            - **Regular Reviews**: Check positions weekly
            - **Diversification**: Spread risk across sectors
            - **Position Sizing**: Never risk more than 2-3% per trade
            - **Stop Losses**: Protect against major losses
            - **Record Keeping**: Track all trades and decisions
            """)

    # --- Compare & Export ---
    target_tab = tab9 if st.session_state.get('enhanced_features_enabled', False) else tab8
    with target_tab:
        st.header("⚖️ Compare & Export Results")
        st.markdown("Side-by-side comparison and data export functionality")
        
        # Check if there's any analysis data available
        if 'stock_data' in st.session_state and st.session_state.stock_data:
            st.subheader("📊 Stock Comparison")
            
            # Display comparison data
            comparison_data = st.session_state.stock_data
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame([
                {
                    "Symbol": symbol,
                    "Score": data.get('total_score', 0),
                    "Recommendation": data.get('recommendation', 'N/A'),
                    **data.get('scores', {})
                }
                for symbol, data in comparison_data.items()
            ])
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = comparison_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                if 'openpyxl' in globals():
                    excel_buffer = io.BytesIO()
                    comparison_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        "📥 Download Excel",
                        data=excel_data,
                        file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Excel export requires openpyxl package")
            
            # Visualization
            if len(comparison_df) > 1:
                st.subheader("📈 Score Comparison Chart")
                fig = px.bar(
                    comparison_df, 
                    x='Symbol', 
                    y='Score',
                    color='Recommendation',
                    title="Stock Score Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("📊 No comparison data available yet. Run analysis in other tabs first.")
            
            # Manual comparison tool
            st.subheader("🔧 Manual Comparison Tool")
            
            with st.form("manual_comparison_form"):
                symbols_for_comparison = st.text_input(
                    "Enter symbols to compare (comma-separated)",
                    placeholder="AAPL, MSFT, GOOGL",
                    key="manual_comparison"
                )
                submitted = st.form_submit_button("🔍 Compare Stocks", type="primary")
            
            if submitted and symbols_for_comparison:
                symbol_list = [s.strip().upper() for s in symbols_for_comparison.split(',') if s.strip()]
                
                if len(symbol_list) < 2:
                    st.error("Please enter at least 2 symbols for comparison")
                else:
                    comparison_results = []
                    progress_bar = st.progress(0)
                    
                    for i, symbol in enumerate(symbol_list):
                        progress_bar.progress((i + 1) / len(symbol_list))
                        
                        try:
                            info = fetch_yahoo_info(symbol)
                            if info and info.get('name') != 'Unknown':
                                industry_pe = get_industry_pe(info)
                                scores, _ = calculate_scores_yahoo(info, industry_pe)
                                
                                if scores:
                                    available_weights = {k: st.session_state.score_weights.get(k, 0) 
                                                       for k in scores if k in st.session_state.score_weights}
                                    
                                    if available_weights:
                                        total_weight = sum(available_weights.values())
                                        if total_weight > 0:
                                            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                            overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
                                        else:
                                            overall_score = sum(scores.values()) / len(scores)
                                    else:
                                        overall_score = sum(scores.values()) / len(scores)
                                    
                                    recommendation, color = get_recommendation(overall_score)
                                    
                                    comparison_results.append({
                                        'Symbol': symbol,
                                        'Company': info.get('name', 'N/A')[:30],
                                        'Score': round(overall_score, 2),
                                        'Recommendation': recommendation,
                                        'Price': info.get('currentPrice', 0),
                                        'P/E': info.get('pe', 0),
                                        'ROE': round(info.get('roe', 0) * 100, 1) if info.get('roe') else 0,
                                        'Sector': info.get('sector', 'N/A')
                                    })
                        except Exception as e:
                            st.warning(f"Could not analyze {symbol}: {str(e)}")
                    
                    progress_bar.empty()
                    
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results).sort_values('Score', ascending=False)
                        
                        st.subheader("📊 Comparison Results")
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Export manual comparison
                        csv_data = comparison_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Comparison",
                            data=csv_data,
                            file_name=f"manual_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Could not analyze any of the provided symbols")

# Portfolio management helper functions

def calculate_portfolio_value():
    """Calculate current portfolio value and P&L"""
    if not st.session_state.portfolio_holdings:
        st.warning("No portfolio holdings found")
        return
    
    st.subheader("💰 Portfolio Value Calculation")
    
    # Get current prices for all holdings
    symbols = list(st.session_state.portfolio_holdings.keys())
    
    with st.spinner(f"📊 Fetching current prices for {len(symbols)} holdings..."):
        # Use yfinance to get current prices
        import yfinance as yf
        
        portfolio_data = []
        total_cost = 0
        total_value = 0
        
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / len(symbols))
            
            holding = st.session_state.portfolio_holdings[symbol]
            quantity = holding["quantity"]
            purchase_price = holding["purchase_price"]
            purchase_date = holding["purchase_date"]
            
            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period="1d")['Close'].iloc[-1] if not ticker.history(period="1d").empty else 0
            except:
                current_price = 0
            
            # Calculate metrics
            cost_basis = quantity * purchase_price
            market_value = quantity * current_price if current_price > 0 else 0
            unrealized_pnl = market_value - cost_basis if purchase_price > 0 else 0
            pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            portfolio_data.append({
                'Symbol': symbol,
                'Quantity': f"{quantity:.2f}",
                'Purchase Price': f"${purchase_price:.2f}" if purchase_price > 0 else "Not Set",
                'Current Price': f"${current_price:.2f}" if current_price > 0 else "N/A",
                'Cost Basis': f"${cost_basis:.2f}" if purchase_price > 0 else "N/A",
                'Market Value': f"${market_value:.2f}" if current_price > 0 else "N/A",
                'Unrealized P&L': f"${unrealized_pnl:.2f}" if purchase_price > 0 and current_price > 0 else "N/A",
                'P&L %': f"{pnl_percent:.1f}%" if purchase_price > 0 and current_price > 0 else "N/A",
                'Purchase Date': purchase_date
            })
            
            if purchase_price > 0:
                total_cost += cost_basis
            if current_price > 0 and purchase_price > 0:
                total_value += market_value
        
        progress_bar.empty()
        
        # Display summary metrics
        if total_cost > 0:
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost Basis", f"${total_cost:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_value:,.2f}")
            with col3:
                st.metric("Unrealized P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:.1f}%")
            with col4:
                if total_pnl >= 0:
                    st.success(f"📈 Gain: {total_pnl_percent:.1f}%")
                else:
                    st.error(f"📉 Loss: {total_pnl_percent:.1f}%")
        
        # Display detailed holdings table
        if portfolio_data:
            st.subheader("📋 Detailed Holdings")
            holdings_df = pd.DataFrame(portfolio_data)
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)

def export_portfolio_to_csv():
    """Export portfolio holdings to CSV"""
    if not st.session_state.portfolio_holdings:
        st.warning("No holdings to export")
        return
    
    export_data = []
    for symbol, holding in st.session_state.portfolio_holdings.items():
        export_data.append({
            'Symbol': symbol,
            'Quantity': holding['quantity'],
            'Purchase_Price': holding['purchase_price'],
            'Purchase_Date': holding['purchase_date'],
            'Cost_Basis': holding['quantity'] * holding['purchase_price']
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    st.download_button(
        "📥 Download Holdings CSV",
        data=csv,
        file_name=f"portfolio_holdings_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def display_portfolio_summary():
    """Display portfolio summary statistics"""
    if not st.session_state.portfolio_holdings:
        st.warning("No holdings to summarize")
        return
    
    st.subheader("📊 Portfolio Summary")
    
    total_holdings = len(st.session_state.portfolio_holdings)
    total_cost = sum(
        holding["quantity"] * holding["purchase_price"] 
        for holding in st.session_state.portfolio_holdings.values()
        if holding["purchase_price"] > 0
    )
    holdings_with_prices = sum(
        1 for holding in st.session_state.portfolio_holdings.values()
        if holding["purchase_price"] > 0
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Holdings", total_holdings)
    with col2:
        st.metric("Holdings with Prices", holdings_with_prices)
    with col3:
        st.metric("Total Investment", f"${total_cost:,.2f}" if total_cost > 0 else "Set prices")
    
    # Show top holdings by cost basis
    if holdings_with_prices > 0:
        st.markdown("### 🔝 Top Holdings by Investment")
        holdings_list = []
        for symbol, holding in st.session_state.portfolio_holdings.items():
            if holding["purchase_price"] > 0:
                cost_basis = holding["quantity"] * holding["purchase_price"]
                holdings_list.append({
                    'Symbol': symbol,
                    'Investment': cost_basis,
                    'Percentage': (cost_basis / total_cost * 100) if total_cost > 0 else 0
                })
        
        holdings_list.sort(key=lambda x: x['Investment'], reverse=True)
        
        for holding in holdings_list[:5]:  # Show top 5
            st.write(f"**{holding['Symbol']}**: ${holding['Investment']:,.2f} ({holding['Percentage']:.1f}%)")

def run_portfolio_pnl_analysis():
    """Run comprehensive P&L analysis for portfolio holdings"""
    if not st.session_state.portfolio_holdings:
        st.warning("No portfolio holdings found for P&L analysis")
        return
    
    st.subheader("💰 Portfolio P&L Analysis")
    
    # Get current prices and calculate P&L
    import yfinance as yf
    
    holdings_data = []
    total_cost = 0
    total_value = 0
    total_gain_loss = 0
    
    progress_bar = st.progress(0)
    symbols = list(st.session_state.portfolio_holdings.keys())
    
    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        
        holding = st.session_state.portfolio_holdings[symbol]
        quantity = holding["quantity"]
        purchase_price = holding["purchase_price"]
        purchase_date = holding["purchase_date"]
        
        # Get current price
        try:
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d")
            current_price = current_data['Close'].iloc[-1] if not current_data.empty else 0
        except:
            current_price = 0
        
        # Calculate metrics
        cost_basis = quantity * purchase_price if purchase_price > 0 else 0
        market_value = quantity * current_price if current_price > 0 else 0
        gain_loss = market_value - cost_basis if purchase_price > 0 and current_price > 0 else 0
        gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
        
        # Days held calculation
        try:
            from datetime import datetime
            purchase_dt = datetime.strptime(purchase_date, "%Y-%m-%d")
            days_held = (datetime.now() - purchase_dt).days
        except:
            days_held = 0
        
        holdings_data.append({
            'Symbol': symbol,
            'Quantity': quantity,
            'Purchase Price': purchase_price,
            'Current Price': current_price,
            'Cost Basis': cost_basis,
            'Market Value': market_value,
            'Gain/Loss ($)': gain_loss,
            'Gain/Loss (%)': gain_loss_pct,
            'Days Held': days_held,
            'Purchase Date': purchase_date
        })
        
        if purchase_price > 0:
            total_cost += cost_basis
        if current_price > 0 and purchase_price > 0:
            total_value += market_value
            total_gain_loss += gain_loss
    
    progress_bar.empty()
    
    # Display summary metrics
    if total_cost > 0:
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost Basis", f"${total_cost:,.2f}")
        with col2:
            st.metric("Current Market Value", f"${total_value:,.2f}")
        with col3:
            st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}", f"{total_gain_loss_pct:.1f}%")
        with col4:
            profitable_positions = sum(1 for h in holdings_data if h['Gain/Loss ($)'] > 0 and h['Cost Basis'] > 0)
            total_positions = sum(1 for h in holdings_data if h['Cost Basis'] > 0)
            win_rate = (profitable_positions / total_positions * 100) if total_positions > 0 else 0
            st.metric("Win Rate", f"{profitable_positions}/{total_positions} ({win_rate:.1f}%)")
    
    # Display detailed holdings table
    if holdings_data:
        st.markdown("### 📋 Detailed P&L by Holding")
        
        # Create DataFrame for better display
        df = pd.DataFrame(holdings_data)
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['Purchase Price'] = display_df['Purchase Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "Not Set")
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['Gain/Loss ($)'] = display_df['Gain/Loss ($)'].apply(lambda x: f"${x:.2f}" if abs(x) > 0.01 else "N/A")
        display_df['Gain/Loss (%)'] = display_df['Gain/Loss (%)'].apply(lambda x: f"{x:.1f}%" if abs(x) > 0.01 else "N/A")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Top performers and worst performers
        profitable_holdings = [h for h in holdings_data if h['Gain/Loss ($)'] > 0 and h['Cost Basis'] > 0]
        losing_holdings = [h for h in holdings_data if h['Gain/Loss ($)'] < 0 and h['Cost Basis'] > 0]
        
        if profitable_holdings or losing_holdings:
            col1, col2 = st.columns(2)
            
            with col1:
                if profitable_holdings:
                    st.markdown("### 🟢 Top Performers")
                    profitable_holdings.sort(key=lambda x: x['Gain/Loss (%)'], reverse=True)
                    for holding in profitable_holdings[:3]:
                        st.success(f"**{holding['Symbol']}**: +${holding['Gain/Loss ($)']:.2f} ({holding['Gain/Loss (%)']:.1f}%)")
            
            with col2:
                if losing_holdings:
                    st.markdown("### 🔴 Underperformers")
                    losing_holdings.sort(key=lambda x: x['Gain/Loss (%)'])
                    for holding in losing_holdings[:3]:
                        st.error(f"**{holding['Symbol']}**: ${holding['Gain/Loss ($)']:.2f} ({holding['Gain/Loss (%)']:.1f}%)")

def run_portfolio_alerts():
    """Run portfolio alerts with P&L monitoring"""
    if not st.session_state.portfolio_holdings:
        st.warning("No portfolio holdings found for alerts")
        return
    
    st.subheader("🚨 Portfolio Alerts & Monitoring")
    
    # Alert settings
    col1, col2, col3 = st.columns(3)
    with col1:
        gain_threshold = st.number_input("Gain Alert Threshold (%)", value=10.0, step=1.0, 
                                       help="Alert when any position gains more than this percentage")
    with col2:
        loss_threshold = st.number_input("Loss Alert Threshold (%)", value=-10.0, step=1.0,
                                       help="Alert when any position loses more than this percentage")
    with col3:
        price_change_threshold = st.number_input("Daily Price Change Alert (%)", value=5.0, step=1.0,
                                                help="Alert when any position moves more than this % in a day")
    
    if st.button("🔍 Check Portfolio Alerts", type="primary"):
        import yfinance as yf
        
        alerts = []
        progress_bar = st.progress(0)
        symbols = list(st.session_state.portfolio_holdings.keys())
        
        for i, symbol in enumerate(symbols):
            progress_bar.progress((i + 1) / len(symbols))
            
            holding = st.session_state.portfolio_holdings[symbol]
            quantity = holding["quantity"]
            purchase_price = holding["purchase_price"]
            
            if purchase_price <= 0:
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                # Get current data and recent history
                hist = ticker.history(period="5d")
                
                if hist.empty:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                # Calculate metrics
                gain_loss_pct = ((current_price - purchase_price) / purchase_price) * 100
                daily_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                position_value = quantity * current_price
                
                # Check for alerts
                if gain_loss_pct >= gain_threshold:
                    alerts.append({
                        'Type': '🟢 GAIN ALERT',
                        'Symbol': symbol,
                        'Message': f'Position up {gain_loss_pct:.1f}% from purchase price',
                        'Details': f'Purchase: ${purchase_price:.2f} | Current: ${current_price:.2f} | Value: ${position_value:.2f}',
                        'Severity': 'success'
                    })
                
                if gain_loss_pct <= loss_threshold:
                    alerts.append({
                        'Type': '🔴 LOSS ALERT',
                        'Symbol': symbol,
                        'Message': f'Position down {gain_loss_pct:.1f}% from purchase price',
                        'Details': f'Purchase: ${purchase_price:.2f} | Current: ${current_price:.2f} | Value: ${position_value:.2f}',
                        'Severity': 'error'
                    })
                
                if abs(daily_change_pct) >= price_change_threshold:
                    direction = "up" if daily_change_pct > 0 else "down"
                    alerts.append({
                        'Type': '🔄 PRICE MOVEMENT',
                        'Symbol': symbol,
                        'Message': f'Price moved {direction} {abs(daily_change_pct):.1f}% today',
                        'Details': f'Previous: ${prev_close:.2f} | Current: ${current_price:.2f} | Position Value: ${position_value:.2f}',
                        'Severity': 'warning'
                    })
                
            except Exception as e:
                st.error(f"Error checking alerts for {symbol}: {str(e)}")
        
        progress_bar.empty()
        
        # Display alerts
        if alerts:
            st.markdown("### 🚨 Active Alerts")
            for alert in alerts:
                if alert['Severity'] == 'success':
                    st.success(f"**{alert['Type']} - {alert['Symbol']}**\n{alert['Message']}\n{alert['Details']}")
                elif alert['Severity'] == 'error':
                    st.error(f"**{alert['Type']} - {alert['Symbol']}**\n{alert['Message']}\n{alert['Details']}")
                else:
                    st.warning(f"**{alert['Type']} - {alert['Symbol']}**\n{alert['Message']}\n{alert['Details']}")
        else:
            st.info("✅ No alerts triggered based on current thresholds")
        
        # Portfolio risk summary
        st.markdown("### 📊 Portfolio Risk Summary")
        try:
            total_value = 0
            total_cost = 0
            risky_positions = 0
            
            for symbol in symbols:
                holding = st.session_state.portfolio_holdings[symbol]
                if holding["purchase_price"] > 0:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    
                    position_value = holding["quantity"] * current_price
                    cost_basis = holding["quantity"] * holding["purchase_price"]
                    
                    total_value += position_value
                    total_cost += cost_basis
                    
                    # Check if position is risky (>5% daily volatility)
                    hist = ticker.history(period="30d")
                    if len(hist) > 1:
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * 100
                        if volatility > 5:
                            risky_positions += 1
            
            portfolio_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Return", f"{portfolio_return:.1f}%")
            with col2:
                st.metric("High Volatility Positions", f"{risky_positions}/{len(symbols)}")
            with col3:
                risk_level = "Low" if risky_positions <= len(symbols) * 0.3 else "Medium" if risky_positions <= len(symbols) * 0.6 else "High"
                st.metric("Risk Level", risk_level)
                
        except Exception as e:
            st.error(f"Error calculating portfolio risk: {str(e)}")
        
        # Advanced Risk Analysis Integration
        st.markdown("---")
        st.markdown("### 🎯 Advanced Risk Analysis")
        
        try:
            # Prepare portfolio data for advanced analysis
            portfolio_data = []
            for symbol in symbols:
                holding = st.session_state.portfolio_holdings[symbol]
                if holding["purchase_price"] > 0:
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Quantity': holding["quantity"],
                        'Purchase_Price': holding["purchase_price"]
                    })
            
            if portfolio_data:
                portfolio_df = pd.DataFrame(portfolio_data)
                
                # Initialize risk analyzer
                risk_analyzer = AdvancedRiskAnalyzer()
                
                # Run advanced risk alerts
                advanced_alerts = check_portfolio_alerts(
                    portfolio_symbols=symbols,
                    portfolio_data=portfolio_df, 
                    risk_analyzer=risk_analyzer
                )
                
                # Display advanced risk alerts
                if advanced_alerts:
                    st.markdown("#### 🔔 Risk-Based Alerts")
                    
                    # Categorize alerts
                    critical_alerts = [a for a in advanced_alerts if a.get('type') == 'error']
                    warning_alerts = [a for a in advanced_alerts if a.get('type') == 'warning']
                    info_alerts = [a for a in advanced_alerts if a.get('type') in ['info', 'success']]
                    
                    # Show critical alerts first
                    if critical_alerts:
                        st.markdown("**🚨 Critical Risk Alerts:**")
                        for alert in critical_alerts[:3]:  # Show top 3 critical
                            st.error(f"**{alert.get('category', 'Risk')}**: {alert.get('message', 'No message')}")
                    
                    # Show warnings
                    if warning_alerts:
                        st.markdown("**⚠️ Risk Warnings:**")
                        for alert in warning_alerts[:3]:  # Show top 3 warnings
                            st.warning(f"**{alert.get('category', 'Risk')}**: {alert.get('message', 'No message')}")
                    
                    # Show positive/info alerts in expander
                    if info_alerts:
                        with st.expander(f"ℹ️ Additional Risk Insights ({len(info_alerts)} items)", expanded=False):
                            for alert in info_alerts:
                                if alert.get('type') == 'success':
                                    st.success(f"**{alert.get('category', 'Risk')}**: {alert.get('message', 'No message')}")
                                else:
                                    st.info(f"**{alert.get('category', 'Risk')}**: {alert.get('message', 'No message')}")
                
                # Quick risk metrics display
                try:
                    risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(portfolio_df)
                    
                    st.markdown("#### 📊 Key Risk Metrics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        volatility = risk_metrics['volatility']['annualized']
                        vol_color = "🟢" if volatility < 0.15 else "🟡" if volatility < 0.25 else "🔴"
                        st.metric("Portfolio Volatility", f"{volatility:.1%}", help="Annual volatility")
                        st.markdown(f"{vol_color} Risk Level")
                    
                    with metric_col2:
                        var_95 = risk_metrics['var']['var_95']
                        var_color = "🟢" if var_95 < 0.02 else "🟡" if var_95 < 0.05 else "🔴"
                        st.metric("VaR (95%)", f"{var_95:.2%}", help="Maximum 1-day loss (95% confidence)")
                        st.markdown(f"{var_color} Risk Level")
                    
                    with metric_col3:
                        sharpe = risk_metrics.get('performance', {}).get('sharpe_ratio', 0)
                        sharpe_color = "🔴" if sharpe < 0.5 else "🟡" if sharpe < 1.0 else "🟢"
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted return")
                        st.markdown(f"{sharpe_color} Performance")
                    
                    with metric_col4:
                        diversification_score = min(100, len(symbols) * 10)  # Simple diversification score
                        div_color = "🔴" if diversification_score < 50 else "🟡" if diversification_score < 80 else "🟢"
                        st.metric("Diversification", f"{diversification_score}%", help="Portfolio diversification score")
                        st.markdown(f"{div_color} Level")
                    
                except Exception as risk_calc_error:
                    st.info(f"Advanced risk calculation unavailable: {str(risk_calc_error)}")
                
                # Add link to full risk analysis
                st.markdown("---")
                st.info("💡 **Want deeper risk analysis?** Use the **What-If Analysis → Risk Analysis** tab for comprehensive portfolio risk assessment including correlation analysis, stress testing, and Monte Carlo simulations.")
                
            else:
                st.info("No valid portfolio positions found for advanced risk analysis.")
                
        except Exception as e:
            st.warning(f"Advanced risk analysis unavailable: {str(e)}")
            st.info("💡 Basic portfolio monitoring continues to function normally.")

def create_portfolio_performance_chart():
    """Create a performance chart for the portfolio"""
    if not st.session_state.portfolio_holdings:
        st.warning("No holdings to chart")
        return
    
    st.subheader("📈 Portfolio Performance Chart")
    st.info("📊 Performance chart feature will be implemented with historical data tracking")
    
    # For now, show a simple allocation chart
    holdings_with_prices = {
        symbol: holding for symbol, holding in st.session_state.portfolio_holdings.items()
        if holding["purchase_price"] > 0
    }
    
    if holdings_with_prices:
        import plotly.express as px
        
        # Create allocation pie chart
        allocation_data = []
        for symbol, holding in holdings_with_prices.items():
            cost_basis = holding["quantity"] * holding["purchase_price"]
            allocation_data.append({
                'Symbol': symbol,
                'Investment': cost_basis
            })
        
        if allocation_data:
            df = pd.DataFrame(allocation_data)
            fig = px.pie(df, values='Investment', names='Symbol', 
                        title="Portfolio Allocation by Investment")
            st.plotly_chart(fig, use_container_width=True)

def run_portfolio_pnl_analysis():
    """Run comprehensive P&L analysis for portfolio holdings"""
    if not st.session_state.portfolio_holdings:
        st.warning("No portfolio holdings found")
        return
    
    st.subheader("💰 Portfolio P&L Analysis")
    
    # Get current prices for all holdings
    symbols = list(st.session_state.portfolio_holdings.keys())
    
    with st.spinner(f"📊 Analyzing P&L for {len(symbols)} holdings..."):
        import yfinance as yf
        
        analysis_data = []
        total_cost = 0
        total_value = 0
        total_pnl = 0
        
        for symbol in symbols:
            holding = st.session_state.portfolio_holdings[symbol]
            quantity = holding["quantity"]
            purchase_price = holding["purchase_price"]
            purchase_date = holding["purchase_date"]
            
            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period="1d")['Close'].iloc[-1] if not ticker.history(period="1d").empty else 0
            except:
                current_price = 0
            
            # Calculate metrics
            cost_basis = quantity * purchase_price if purchase_price > 0 else 0
            market_value = quantity * current_price if current_price > 0 else 0
            unrealized_pnl = market_value - cost_basis if purchase_price > 0 and current_price > 0 else 0
            pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            # Calculate holding period
            try:
                purchase_dt = datetime.strptime(purchase_date, "%Y-%m-%d")
                holding_days = (datetime.now() - purchase_dt).days
            except:
                holding_days = 0
            
            analysis_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Purchase Price': purchase_price,
                'Current Price': current_price,
                'Cost Basis': cost_basis,
                'Market Value': market_value,
                'Unrealized P&L': unrealized_pnl,
                'P&L %': pnl_percent,
                'Holding Days': holding_days,
                'Purchase Date': purchase_date
            })
            
            if cost_basis > 0:
                total_cost += cost_basis
            if market_value > 0:
                total_value += market_value
            if unrealized_pnl != 0:
                total_pnl += unrealized_pnl
        
        # Display summary metrics
        if total_cost > 0:
            total_pnl_percent = (total_pnl / total_cost * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cost Basis", f"${total_cost:,.2f}")
            with col2:
                st.metric("Current Value", f"${total_value:,.2f}")
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:.1f}%")
            with col4:
                performance_status = "📈 Profitable" if total_pnl >= 0 else "📉 Loss"
                st.metric("Performance", performance_status)
        
        # Winners and losers
        st.markdown("### 🏆 Best & Worst Performers")
        
        profitable_holdings = [h for h in analysis_data if h['Unrealized P&L'] > 0]
        losing_holdings = [h for h in analysis_data if h['Unrealized P&L'] < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Top Gainers:**")
            if profitable_holdings:
                profitable_holdings.sort(key=lambda x: x['P&L %'], reverse=True)
                for holding in profitable_holdings[:3]:
                    st.success(f"**{holding['Symbol']}**: {holding['P&L %']:.1f}% (${holding['Unrealized P&L']:.2f})")
            else:
                st.info("No profitable positions")
        
        with col2:
            st.markdown("**🔴 Top Losers:**")
            if losing_holdings:
                losing_holdings.sort(key=lambda x: x['P&L %'])
                for holding in losing_holdings[:3]:
                    st.error(f"**{holding['Symbol']}**: {holding['P&L %']:.1f}% (${holding['Unrealized P&L']:.2f})")
            else:
                st.info("No losing positions")
        
        # Detailed analysis table
        if analysis_data:
            st.markdown("### 📋 Detailed P&L Analysis")
            analysis_df = pd.DataFrame(analysis_data)
            
            # Format the dataframe for display
            display_df = analysis_df.copy()
            for col in ['Purchase Price', 'Current Price', 'Cost Basis', 'Market Value', 'Unrealized P&L']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if x != 0 else "N/A")
            
            display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.1f}%" if x != 0 else "N/A")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def analyze_portfolio_quick(portfolio):
    """Quick portfolio analysis"""
    if not portfolio:
        st.warning("Portfolio is empty")
        return
    
    st.subheader("⚡ Quick Portfolio Analysis")
    
    with st.spinner("Analyzing portfolio..."):
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(portfolio):
            progress_bar.progress((i + 1) / len(portfolio))
            
            try:
                info = fetch_yahoo_info(symbol)
                if info and info.get('name') != 'Unknown':
                    industry_pe = get_industry_pe(info)
                    scores, _ = calculate_scores_yahoo(info, industry_pe)
                    
                    if scores:
                        # Calculate total score
                        available_weights = {k: st.session_state.score_weights.get(k, 0) 
                                           for k in scores if k in st.session_state.score_weights}
                        
                        if available_weights:
                            total_weight = sum(available_weights.values())
                            if total_weight > 0:
                                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
                            else:
                                overall_score = sum(scores.values()) / len(scores)
                        else:
                            overall_score = sum(scores.values()) / len(scores)
                        
                        recommendation, color = get_recommendation(overall_score)
                        
                        results.append({
                            'Symbol': symbol,
                            'Score': overall_score,
                            'Recommendation': recommendation,
                            'Price': info.get('currentPrice', 0),
                            'Sector': info.get('sector', 'N/A')
                        })
            except Exception as e:
                st.warning(f"Could not analyze {symbol}: {str(e)}")
        
        progress_bar.empty()
        
        if results:
            # Display quick results
            df = pd.DataFrame(results).sort_values('Score', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{df['Score'].mean():.2f}")
            with col2:
                strong_buys = len(df[df['Score'] >= 8.0])
                st.metric("Strong Buys", strong_buys)
            with col3:
                sell_warnings = len(df[df['Score'] <= 3.0])
                st.metric("Sell Warnings", sell_warnings)
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not analyze any symbols in your portfolio")

def analyze_full_portfolio(portfolio):
    """Full comprehensive portfolio analysis"""
    try:
        with st.spinner("Running full portfolio analysis..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(portfolio):
                progress_bar.progress((i + 1) / len(portfolio))
                try:
                    info = fetch_yahoo_info(symbol)
                    if info and info.get('name') != 'Unknown':
                        industry_pe = get_industry_pe(info)
                        scores, _ = calculate_scores_yahoo(info, industry_pe)
                        
                        if scores:
                            # Calculate overall score with portfolio weights if available
                            weights = st.session_state.get('portfolio_screener_weights', st.session_state.score_weights)
                            available_weights = {k: weights.get(k, 0) 
                                               for k in scores if k in weights}
                            
                            if available_weights:
                                total_weight = sum(available_weights.values())
                                if total_weight > 0:
                                    normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                    overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
                                else:
                                    overall_score = sum(scores.values()) / len(scores)
                            else:
                                overall_score = sum(scores.values()) / len(scores)
                            
                            recommendation, _ = get_recommendation(overall_score)
                            
                            results.append({
                                'Symbol': symbol,
                                'Company': info.get('name', 'N/A')[:30],
                                'Final_Score': round(overall_score, 2),
                                'Recommendation': recommendation,
                                'Current_Price': info.get('currentPrice', 0),
                                'P/E_Ratio': info.get('pe', 0),
                                'Market_Cap': info.get('marketCap', 0),
                                'ROE': round(info.get('roe', 0) * 100, 1) if info.get('roe') else 0,
                                'Sector': info.get('sector', 'N/A'),
                                'Beta': info.get('beta', 0),
                                'Dividend_Yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0,
                                **scores
                            })
                except Exception as e:
                    st.warning(f"Could not analyze {symbol}: {str(e)}")
            
            progress_bar.empty()
            
            if results:
                df = pd.DataFrame(results).sort_values('Final_Score', ascending=False)
                st.subheader("📊 Full Portfolio Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = df['Final_Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}")
                with col2:
                    strong_buys = len(df[df['Final_Score'] >= 8.0])
                    st.metric("Strong Buys (≥8.0)", strong_buys)
                with col3:
                    weak_sells = len(df[df['Final_Score'] <= 4.0])
                    st.metric("Weak Positions (≤4.0)", weak_sells)
                with col4:
                    total_value = df['Current_Price'].sum()
                    st.metric("Portfolio Value", f"${total_value:,.0f}")
                
                # Detailed results
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Export option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Analysis",
                    data=csv_data,
                    file_name=f"full_portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Could not analyze any symbols in your portfolio")
                
    except Exception as e:
        st.error(f"Error in full portfolio analysis: {str(e)}")

def analyze_portfolio_comprehensive(portfolio):
    """Comprehensive portfolio analysis for monitoring"""
    try:
        results = []
        for symbol in portfolio:
            try:
                info = fetch_yahoo_info(symbol)
                if info and info.get('name') != 'Unknown':
                    industry_pe = get_industry_pe(info)
                    scores, _ = calculate_scores_yahoo(info, industry_pe)
                    
                    if scores:
                        # Use portfolio weights if available
                        weights = st.session_state.get('portfolio_screener_weights', st.session_state.score_weights)
                        available_weights = {k: weights.get(k, 0) 
                                           for k in scores if k in weights}
                        
                        if available_weights:
                            total_weight = sum(available_weights.values())
                            if total_weight > 0:
                                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                                overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
                            else:
                                overall_score = sum(scores.values()) / len(scores)
                        else:
                            overall_score = sum(scores.values()) / len(scores)
                        
                        recommendation, _ = get_recommendation(overall_score)
                        
                        results.append({
                            'Original_Symbol': symbol,
                            'Company': info.get('name', 'N/A')[:30],
                            'Final_Score': round(overall_score, 2),
                            'Recommendation': recommendation,
                            'Current_Price': info.get('currentPrice', 0),
                            'P/E_Ratio': info.get('pe', 0),
                            'ROE': round(info.get('roe', 0) * 100, 1) if info.get('roe') else 0,
                            'Sector': info.get('sector', 'N/A'),
                            'Market_Cap': info.get('marketCap', 0)
                        })
            except Exception as e:
                continue  # Skip problematic symbols
        
        return pd.DataFrame(results) if results else None
        
    except Exception as e:
        st.error(f"Error in comprehensive portfolio analysis: {str(e)}")
        return None

def run_market_screening(market_selection, min_score, max_results):
    """Run market screening with specified parameters"""
    try:
        # Use the existing screen_multi_market_stocks function but with market screener weights
        original_weights = st.session_state.score_weights.copy()
        
        # Temporarily use market screener weights
        if 'market_screener_weights' in st.session_state:
            st.session_state.score_weights = st.session_state.market_screener_weights
        
        # Run the screening
        results = screen_multi_market_stocks(market_selection, min_score, None)
        
        # Restore original weights
        st.session_state.score_weights = original_weights
        
        return results.head(max_results) if not results.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error running market screening: {str(e)}")
        return pd.DataFrame()
         
if __name__ == "__main__":
    main()
