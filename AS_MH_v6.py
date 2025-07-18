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
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_KEY = "7J1AJVC9MAYLRRA7"
REQUEST_DELAY = 0.5

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

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "score_weights" not in st.session_state:
        st.session_state.score_weights = {
            "PE": 0.10,
            "Forward PE": 0.15,  # More weight on forward P/E
            "PEG": 0.10,
            "PB": 0.08,
            "EV/EBITDA": 0.12,  # New metric
            "ROE": 0.12,
            "EPS Growth": 0.15,
            "Revenue Growth": 0.10,
            "FCF Trend": 0.05,
            "Debt/Equity": 0.04,
            "Dividend Yield": 0.03,
            "Gross Margin": 0.03,
            "Price/Sales": 0.05,  # New metric
            "Analyst Upside": 0.08   # New metric
        }
    
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    if "selected_symbols" not in st.session_state:
        st.session_state.selected_symbols = []
    
    # Initialize portfolio if it doesn't exist
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

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
            
            # Free Cash Flow
            try:
                if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditures' in cashflow.index:
                    operating_cf = cashflow.loc['Operating Cash Flow'].head(3)
                    capex = cashflow.loc['Capital Expenditures'].head(3)
                    
                    if not operating_cf.empty and not capex.empty:
                        fcf_data = operating_cf - abs(capex)
                        metrics_3y['fcf_trend'] = fcf_data.tolist()
            except Exception:
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
def score_pe(pe, industry_pe):
    """Enhanced P/E scoring with industry comparison"""
    if pe <= 0:
        return 0
    
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
    """Score based on forward P/E (more predictive than trailing)"""
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

def score_peg(peg):
    """Enhanced PEG scoring"""
    if peg <= 0:
        return 0
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
    """Price-to-Book scoring"""
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
    """Return on Equity scoring"""
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
        return 0

def score_eps_growth(growth):
    """EPS growth scoring"""
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
    """Revenue growth scoring"""
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

def score_dividend_yield(dy):
    """Dividend yield scoring"""
    if dy <= 0:
        return 0
    if dy > 5:
        return 10
    elif dy > 3:
        return 8
    elif dy > 1:
        return 6
    else:
        return 4

def score_gross_margin(gm):
    """Gross margin scoring"""
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
            st.info(f"🔍 Fetching fundamental data for {self.symbol}...")
            self.fundamental_info = fetch_yahoo_info(self.symbol)
            
            if not self.fundamental_info:
                st.warning(f"⚠️ Could not fetch fundamental data for {self.symbol}")
                return False
            
            if self.fundamental_info.get('name') == 'Unknown':
                st.warning(f"⚠️ Limited fundamental data available for {self.symbol}")
            else:
                st.success(f"✅ Fundamental data loaded for {self.fundamental_info.get('name', self.symbol)}")
            
            # Fetch technical data using the symbol that worked for fundamental data
            st.info(f"📈 Fetching technical data for {self.symbol}...")
            
            # Try the symbol that worked for fundamental data
            symbol_to_use = self.fundamental_info.get('symbol_used', self.symbol)
            
            ticker = yf.Ticker(symbol_to_use)
            self.data = ticker.history(period=self.period)
            
            if self.data is None or len(self.data) == 0:
                st.error(f"❌ No technical data available for {symbol_to_use}")
                return False
            
            st.success(f"✅ Technical data loaded for {symbol_to_use} ({len(self.data)} data points)")
            return True
            
        except Exception as e:
            st.error(f"❌ Error fetching data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_fundamental_scores(self):
        """Calculate fundamental analysis scores"""
        try:
            if not self.fundamental_info:
                st.warning(f"⚠️ No fundamental data available for scoring {self.symbol}")
                return {}
            
            # Get industry PE for better scoring
            industry_pe = get_industry_pe(self.fundamental_info)
            
            # Calculate scores using the Yahoo info
            scores, _ = calculate_scores_yahoo(self.fundamental_info, industry_pe)
            
            if scores:
                self.fundamental_scores = scores
                st.success(f"✅ Fundamental scores calculated for {self.symbol}")
                return scores
            else:
                st.warning(f"⚠️ Could not calculate fundamental scores for {self.symbol}")
                return {}
                
        except Exception as e:
            st.error(f"❌ Error calculating fundamental scores for {self.symbol}: {str(e)}")
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
            
            # Calculate fundamental score with normalized weights for missing data
            available_weights = {k: st.session_state.score_weights.get(k, 0) 
                               for k in self.fundamental_scores if k in st.session_state.score_weights}
            
            if available_weights:
                total_weight = sum(available_weights.values())
                if total_weight > 0:
                    # Normalize weights to maintain scale
                    original_total_weight = sum(st.session_state.score_weights.values())
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
            st.error(f"Error generating recommendation: {e}")
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
        
        if analyzer.data is not None and len(analyzer.data) > 0:
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
            "PE": score_pe(pe, industry_pe),
            "PEG": score_peg(peg),
            "PB": score_pb(pb),
            "ROE": score_roe(roe),
            "EPS Growth": score_eps_growth(eps_growth),
            "Revenue Growth": score_revenue_growth(rev_growth, has_revenue_data),
            "FCF Trend": score_fcf_trend(fcf_trend, has_fcf_data),
            "Debt/Equity": score_debt_equity(de),
            "Dividend Yield": score_dividend_yield(dy),
            "Gross Margin": score_gross_margin(gm)
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
        
        # Filter out None values (missing data) but keep 0 scores
        scores = {k: v for k, v in scores.items() if v is not None}
        
        # Apply sector-specific adjustments
        sector = info.get("sector", "")
        if sector:
            scores = apply_sector_adjustments(scores, sector)
        
        return scores, info
        
    except Exception as e:
        st.error(f"Error calculating scores: {e}")
        return None, None

def safe_float(value, default=0):
    """Safely convert value to float"""
    try:
        if value is None or value == "None" or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

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
    # Get symbols based on market selection
    symbols_to_screen = get_stock_symbols_for_market(market_selection, custom_symbols)
    
    if not symbols_to_screen:
        return pd.DataFrame()
    
    st.info(f"Screening {len(symbols_to_screen)} stocks...")
    
    # Use optimized batch analysis
    analysis_results = analyze_multiple_stocks(symbols_to_screen)
    
    results = []
    processed = 0
    
    for symbol, data in analysis_results.items():
        processed += 1
        if processed % 10 == 0:  # Update progress every 10 stocks
            st.write(f"Processed {processed}/{len(symbols_to_screen)} stocks...")
        
        if not data or 'scores' not in data or 'info' not in data:
            continue
            
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

def display_danish_stocks_screener():
    """
    Display the Multi-Market stocks screening interface
    """
    st.header("🔍 Multi-Market Stock Screener")
    st.markdown("Screen stocks from multiple markets using the comprehensive scoring system from Tab 1")
    
    # Market selection and configuration section
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        market_selection = st.selectbox(
            "Select Market",
            options=["Danish Stocks", "S&P 500", "NASDAQ 100", "European Stocks", "Custom Symbols"],
            help="Choose which market or stock universe to screen"
        )
    
    with col2:
        min_score = st.slider(
            "Minimum Score", 
            min_value=0.0, 
            max_value=10.0, 
            value=5.0, 
            step=0.1,
            help="Only show stocks with score above this threshold"
        )
    
    with col3:
        max_stocks = st.number_input(
            "Max Results", 
            min_value=10, 
            max_value=100, 
            value=25,
            help="Maximum number of stocks to display"
        )
    
    with col4:
        # Dynamic info based on market selection
        if market_selection == "Danish Stocks":
            st.info(f"📊 Total Danish stocks available: **{len(set(DANISH_STOCKS.values()))}**")
        elif market_selection == "S&P 500":
            st.info("📊 S&P 500 companies: **~500 stocks**")
        elif market_selection == "NASDAQ 100":
            st.info("📊 NASDAQ 100 companies: **~100 stocks**")
        elif market_selection == "European Stocks":
            st.info("📊 Major European indices: **~200 stocks**")
        else:
            st.info("📊 Custom symbol screening available")
    
    # Custom symbols input for Custom Symbols option
    custom_symbols = None
    if market_selection == "Custom Symbols":
        custom_symbols = st.text_area(
            "Enter Stock Symbols (comma-separated)",
            placeholder="AAPL, MSFT, GOOGL, TSLA, NVDA",
            help="Enter stock symbols separated by commas. Examples: AAPL, MSFT, GOOGL"
        )
    
    # Run screening button
    if st.button("🚀 Start Screening", type="primary"):
        # Validation for custom symbols
        if market_selection == "Custom Symbols" and not custom_symbols:
            st.error("❌ Please enter stock symbols for custom screening")
            return
        
        st.markdown("---")
        
        # Aggressive cache clearing to avoid duplicate column issues
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Force Python to clear any cached modules
        import importlib
        import sys
        for module_name in list(sys.modules.keys()):
            if 'yfinance' in module_name or 'yahoo' in module_name:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
        
        screening_text = f"Screening {market_selection.lower()}... This may take a few minutes."
        with st.spinner(screening_text):
            results_df = screen_multi_market_stocks(market_selection, min_score, custom_symbols if market_selection == "Custom Symbols" else None)
        
        if not results_df.empty:
            # Limit results
            display_df = results_df.head(max_stocks)
            
            # Additional safety check for duplicate columns
            if len(display_df.columns) != len(set(display_df.columns)):
                st.error("❌ Duplicate columns detected in results. Removing duplicates...")
                display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                st.success(f"✅ Cleaned DataFrame now has {len(display_df.columns)} unique columns")
            
            st.success(f"✅ Found **{len(results_df)}** stocks from {market_selection} with score ≥ {min_score}")
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Average Score", f"{results_df['Final_Score'].mean():.2f}")
            with col2:
                st.metric("Highest Score", f"{results_df['Final_Score'].max():.2f}")
            with col3:
                st.metric("Above 7.0", len(results_df[results_df['Final_Score'] >= 7.0]))
            with col4:
                st.metric("Above 8.0", len(results_df[results_df['Final_Score'] >= 8.0]))
            with col5:
                # Count Strong Buy and Buy recommendations
                strong_buy_count = len(results_df[results_df['Recommendation'].str.contains('Strong Buy', na=False)])
                buy_count = len(results_df[results_df['Recommendation'].str.contains('📈 Buy', na=False)])
                st.metric("Buy Signals", strong_buy_count + buy_count)
            
            # Display results in tabs
            result_tab1, result_tab2, result_tab3 = st.tabs(["📊 Results Overview", "📈 Detailed Scores", "💾 Export Data"])
            
            with result_tab1:
                st.subheader(f"Top {len(display_df)} Stocks from {market_selection}")
                
                # Main results table - dynamic columns based on market
                if market_selection == "Danish Stocks":
                    display_columns = [
                        'Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 'Sector', 
                        'Current_Price', 'P/E_Ratio', 'ROE', 'Revenue_Growth', 'Dividend_Yield'
                    ]
                    price_label = 'Price (DKK)'
                else:
                    display_columns = [
                        'Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 'Market', 'Sector', 
                        'Current_Price', 'P/E_Ratio', 'ROE', 'Revenue_Growth', 'Dividend_Yield'
                    ]
                    price_label = 'Price (USD)'
                
                st.dataframe(
                    display_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Final_Score': st.column_config.NumberColumn('Score', format="%.2f"),
                        'Current_Price': st.column_config.NumberColumn(price_label, format="%.2f"),
                        'P/E_Ratio': st.column_config.NumberColumn('P/E', format="%.1f"),
                        'ROE': st.column_config.NumberColumn('ROE (%)', format="%.1f"),
                        'Revenue_Growth': st.column_config.NumberColumn('Rev Growth (%)', format="%.1f"),
                        'Dividend_Yield': st.column_config.NumberColumn('Div Yield (%)', format="%.2f"),
                    }
                )
                
                # Sector breakdown
                st.subheader("📊 Sector Distribution")
                sector_counts = display_df['Sector'].value_counts()
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    fig_pie = px.pie(
                        values=sector_counts.values, 
                        names=sector_counts.index,
                        title="Stocks by Sector"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Score distribution
                    fig_hist = px.histogram(
                        display_df, 
                        x='Final_Score', 
                        nbins=10,
                        title="Score Distribution"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col3:
                    # Recommendation distribution
                    rec_counts = display_df['Recommendation'].value_counts()
                    fig_rec = px.pie(
                        values=rec_counts.values,
                        names=rec_counts.index,
                        title="Recommendation Distribution"
                    )
                    st.plotly_chart(fig_rec, use_container_width=True)
            
            with result_tab2:
                st.subheader("📈 Detailed Scoring Breakdown")
                
                # Show individual metric scores
                score_columns = [col for col in display_df.columns if col.endswith('_Score') and col != 'Final_Score']
                if score_columns:
                    detailed_df = display_df[['Original_Symbol', 'Company', 'Final_Score'] + score_columns].copy()
                    
                    st.dataframe(
                        detailed_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            col: st.column_config.NumberColumn(col.replace('_Score', ''), format="%.1f")
                            for col in score_columns
                        }
                    )
                    
                    # Heatmap of scores - using scatter plot instead of imshow for NumPy compatibility
                    if len(detailed_df) > 1:
                        st.subheader("🎯 Scoring Heatmap")
                        
                        # Create a melted dataframe for the heatmap
                        heatmap_data = detailed_df[['Original_Symbol'] + score_columns].melt(
                            id_vars=['Original_Symbol'], 
                            var_name='Metric', 
                            value_name='Score'
                        )
                        
                        try:
                            # Use a scatter plot with size and color to represent scores
                            fig_heatmap = px.scatter(
                                heatmap_data,
                                x='Metric',
                                y='Original_Symbol',
                                color='Score',
                                size='Score',
                                title="Individual Metric Scores by Stock",
                                color_continuous_scale="RdYlGn",
                                size_max=20
                            )
                            fig_heatmap.update_layout(height=600)
                            fig_heatmap.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        except Exception as e:
                            st.info("📊 Score heatmap temporarily unavailable due to library compatibility")
                            # Show a simple table instead
                            st.dataframe(detailed_df[['Original_Symbol'] + score_columns], use_container_width=True)
                
                # Recommendation explanation
                st.markdown("---")
                st.subheader("🎯 Recommendation Guide")
                
                rec_explanation = {
                    "🚀 Strong Buy": "Score ≥ 8.0 - Exceptional value with strong fundamentals",
                    "📈 Buy": "Score ≥ 6.5 - Good investment opportunity with solid metrics", 
                    "🔄 Hold": "Score ≥ 4.0 - Adequate performance, consider for portfolio balance",
                    "📉 Weak Sell": "Score ≥ 2.0 - Below average performance, watch closely",
                    "🛑 Strong Sell": "Score < 2.0 - Poor fundamentals, consider avoiding"
                }
                
                for rec, explanation in rec_explanation.items():
                    st.markdown(f"**{rec}**: {explanation}")
            
            with result_tab3:
                st.subheader("💾 Export Screening Results")
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Results (CSV)",
                        data=csv_data,
                        file_name=f"{market_selection.lower().replace(' ', '_')}_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    symbols_list = ", ".join(display_df['Original_Symbol'].tolist())
                    st.text_area(
                        "📋 Top Stock Symbols (Copy)",
                        value=symbols_list,
                        height=100
                    )
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                selected_symbols = get_stock_symbols_for_market(market_selection, custom_symbols)
                summary_stats = {
                    'Total Stocks Screened': len(selected_symbols),
                    'Stocks Meeting Criteria': len(results_df),
                    'Average Score': f"{results_df['Final_Score'].mean():.2f}",
                    'Median Score': f"{results_df['Final_Score'].median():.2f}",
                    'Standard Deviation': f"{results_df['Final_Score'].std():.2f}",
                    'Highest Scoring Stock': f"{results_df.iloc[0]['Original_Symbol']} ({results_df.iloc[0]['Final_Score']:.2f})",
                    'Most Common Sector': results_df['Sector'].mode().iloc[0] if not results_df['Sector'].mode().empty else 'N/A'
                }
                
                stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        else:
            st.warning(f"❌ No stocks found with score ≥ {min_score} in {market_selection}")
            st.info("Try lowering the minimum score threshold or check if the scoring system is working correctly.")
    
    # Information section
    st.markdown("---")
    st.markdown("""
    ### ℹ️ How the Multi-Market Screening Works
    
    **📊 Scoring System**: Uses the same comprehensive scoring from Tab 1, including:
    - Financial ratios (P/E, PEG, P/B, ROE, etc.)
    - Growth metrics (Revenue, EPS growth)
    - Profitability indicators
    - Debt and dividend metrics
    - Sector-specific adjustments
    
    **� Market Options**: 
    - **Danish Stocks**: All major Danish stocks with .CO suffix handling
    - **S&P 500**: Top US large-cap stocks
    - **NASDAQ 100**: Technology-focused US stocks  
    - **European Stocks**: Major European companies
    - **Custom Symbols**: Enter your own comma-separated stock symbols
    
    **⚖️ Weight Adjustments**: Uses your custom weights from the sidebar configuration
    
    **🔄 Real-time Data**: Fetches live data from Yahoo Finance for accurate analysis
    """)

def display_company_search():
    """Display company search interface"""
    st.subheader("🔍 Company Name Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for companies by name",
            placeholder="e.g., Apple, Microsoft, Tesla",
            help="Enter a company name to find its stock symbol"
        )
    
    with col2:
        search_method = st.selectbox(
            "Search Method",
            ["Alpha Vantage", "Manual Lookup"],
            help="Alpha Vantage provides comprehensive search"
        )
    
    if search_query and len(search_query.strip()) >= 2:
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
            
            enhanced_info = {
                "name": info.get("longName", info.get("shortName", "Unknown")),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "pe": info.get("trailingPE"),
                "peg": info.get("trailingPegRatio"),  # Fixed: use correct Yahoo Finance key
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "eps_growth": info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"),  # Use yearly first, then quarterly
                "revenue_growth": info.get("revenueGrowth"),  # Use consistent key name
                "de": info.get("debtToEquity"),
                "dy": info.get("dividendYield"),
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
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice"))
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

    scores = {
        "PE": score_pe(pe, industry_pe),
        "PEG": score_peg(peg),
        "PB": score_pb(pb),
        "ROE": score_roe(roe),
        "EPS Growth": score_eps_growth(eps_growth),
        "Revenue Growth": score_revenue_growth(rev_growth),
        "FCF Trend": score_fcf_trend(fcf),
        "Debt/Equity": score_debt_equity(de),
        "Dividend Yield": score_dividend_yield(dy),
        "Gross Margin": score_gross_margin(gm)
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
    """Create an interactive score chart using Plotly"""
    metrics = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, 
               marker_color=['green' if v >= 5 else 'orange' if v >= 3 else 'red' for v in values],
               text=values, textposition='auto')
    ])
    
    fig.update_layout(
        title="Stock Scoring Breakdown",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 10]),
        height=400
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
        # Strategy configuration
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["monthly", "quarterly", "semi-annually", "annually"],
            index=1,
            key="rebalance_frequency"
        )
        
        top_n_stocks = st.slider(
            "Portfolio Size (Top N Stocks)",
            min_value=5,
            max_value=20,
            value=10,
            key="portfolio_size"
        )
    
    with col3:
        # Market selection
        market_to_test = st.selectbox(
            "Market Universe",
            ["S&P 500", "NASDAQ 100", "Danish Stocks", "Tech Stocks", "Custom"],
            key="backtest_market"
        )
        
        if market_to_test == "Custom":
            custom_symbols = st.text_input(
                "Custom Symbols (comma-separated)",
                placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
                key="custom_backtest_symbols"
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
    
    benchmark = st.selectbox(
        "Benchmark for Comparison",
        list(benchmark_options.keys()),
        key="benchmark_selection"
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
    selected_stock = st.selectbox("Select Stock to View:", available_stocks, key="score_tracking_stock")
    
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

class StockDataManager:
    """Centralized stock data management with consistent caching"""
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30 minutes
    def get_stock_data(symbol):
        """Single source of truth for all stock data"""
        return fetch_yahoo_info(symbol)
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1 hour
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
    @st.cache_data(ttl=1800)  # 30 minutes
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

def check_portfolio_alerts(portfolio_symbols):
    """Check for portfolio alerts based on score changes"""
    # This would typically compare current scores with historical scores
    # For now, it's a placeholder for the alert system
    alerts = []
    
    # Simulate some alerts for demonstration
    if len(portfolio_symbols) > 0:
        sample_alert = {
            'type': 'info',
            'message': f"Portfolio monitoring active for {len(portfolio_symbols)} stocks",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        alerts.append(sample_alert)
    
    # Add to alert history
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []
    
    for alert in alerts:
        st.session_state.alert_history.append(alert)
    
    # Keep only last 50 alerts
    if len(st.session_state.alert_history) > 50:
        st.session_state.alert_history = st.session_state.alert_history[-50:]
    
    return alerts


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

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Adjust Score Weights")
        with st.expander("Customize Weights"):
            new_weights = {}
            for metric, weight in list(st.session_state.score_weights.items()):
                new_weights[metric] = st.slider(
                    metric, 0.0, 0.5, weight, 0.01,
                    help=f"Current weight: {weight}"
                )
            if st.button("Apply New Weights"):
                st.session_state.score_weights.update(new_weights)
                st.success("Weights updated!")
        st.subheader("Current Weights")
        weights_df = pd.DataFrame(list(st.session_state.score_weights.items()), columns=['Metric', 'Weight'])
        st.dataframe(weights_df, hide_index=True)

    # Main tabs - Streamlined structure
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
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "🥇 Yahoo Finance Analysis", 
            "📊 Alpha Vantage Analysis", 
            "🔍 Company Search"
        ])
        
        with analysis_tab1:
            st.subheader("🚀 Comprehensive Stock Analysis")
            st.info("💡 Complete analysis combining fundamental scoring with technical signals and buying strategies")
            
            if st.session_state.selected_symbols:
                st.info(f"💡 You have {len(st.session_state.selected_symbols)} symbols selected from Company Search. Copy them from the search tab!")
            
            symbols = st.text_input(
                "Enter stock symbols (comma-separated)", 
                "AAPL,MSFT,GOOGL", 
                key="comprehensive",
                help="Example: AAPL,MSFT,GOOGL"
            )
            
            if st.button("🚀 Analyze Stocks", type="primary", key="yahoo_analyze"):
                symbol_list = validate_symbols(symbols)
                
                if not symbol_list:
                    st.error("Please enter valid stock symbols")
                else:
                    # Use optimized analysis
                    st.info(f"Analyzing {len(symbol_list)} stocks using optimized data fetching...")
                    
                    # Get comprehensive analysis for all stocks
                    analysis_results = analyze_multiple_stocks(symbol_list)
                    
                    successful_analyses = []
                    for symbol, data in analysis_results.items():
                        if data and 'scores' in data and 'info' in data:
                            try:
                                with st.container():
                                    st.subheader(f"📈 {symbol.upper()} Analysis")
                                    
                                    # Use the comprehensive data we already have
                                    info = data['info']
                                    scores = data['scores']
                                    
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
                                    
                                    # Track this score for historical analysis
                                    track_stock_score(symbol, overall_score)
                                    
                                    recommendation, color = get_recommendation(overall_score)
                                    
                                    # Display the comprehensive analysis using existing function
                                    # Create an analyzer object for backward compatibility
                                    analyzer = ComprehensiveStockAnalyzer(symbol, "1y")
                                    analyzer.info = info
                                    analyzer.scores = scores
                                    analyzer.final_score = overall_score
                                    analyzer.recommendation = recommendation
                                    analyzer.color = color
                                    
                                    display_comprehensive_analysis(analyzer)
                                    successful_analyses.append(symbol)
                                    
                                    st.markdown("---")
                                    
                            except Exception as e:
                                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                        else:
                            st.warning(f"⚠️ Could not fetch data for {symbol}")
                    
                    if successful_analyses:
                        st.success(f"✅ Successfully analyzed {len(successful_analyses)} stocks: {', '.join(successful_analyses)}")
                    else:
                        st.error("❌ Could not analyze any stocks")
                    
                    if successful_analyses:
                        st.success(f"✅ Successfully analyzed {len(successful_analyses)} stocks: {', '.join(successful_analyses)}")
                    else:
                        st.warning("⚠️ No stocks were successfully analyzed.")
        
        with analysis_tab2:
            st.subheader("📊 Alpha Vantage Analysis")
            st.warning("⚠️ Alpha Vantage has rate limits. Analysis may be slower.")
            
            symbols = st.text_input(
                "Enter stock symbols (comma-separated)", 
                "AAPL,MSFT,GOOGL", 
                key="av",
                help="Limited to avoid rate limits"
            )
            
            if st.button("🔍 Analyze Stocks", key="av_btn"):
                symbol_list = validate_symbols(symbols)
                if not symbol_list:
                    st.error("Please enter valid stock symbols")
                else:
                    if len(symbol_list) > 5:
                        st.warning("⚠️ Limited to 5 symbols to avoid rate limits")
                        symbol_list = symbol_list[:5]
                    
                    stock_data = {}
                    progress_bar = st.progress(0)
                    
                    for i, symbol in enumerate(symbol_list):
                        progress_bar.progress((i + 1) / len(symbol_list))
                        
                        with st.container():
                            st.subheader(f"🔍 {symbol}")
                            with st.spinner(f"Fetching data for {symbol}... (this may take a moment)"):
                                overview = fetch_overview(symbol)
                                
                                if not overview:
                                    st.error(f"❌ Symbol '{symbol}' not found or no data available")
                                    continue
                                
                                industry = overview.get("Industry", None)
                                sector = overview.get("Sector", None)
                                industry_pe = INDUSTRY_PE_MAP.get(industry, INDUSTRY_PE_MAP.get(sector, INDUSTRY_PE_MAP["Unknown"]))
                                st.write(f"Industry: {industry or sector or 'Unknown'} | Industry P/E: {industry_pe}")
                                
                                scores, _ = calculate_scores(symbol, industry_pe)
                                
                                if not scores or not overview:
                                    st.error(f"❌ Symbol '{symbol}' not found or no data available")
                                    continue
                                
                                company_name = overview.get("Name", "Unknown")
                                price = fetch_price(symbol)
                                total = sum(scores[k] * st.session_state.score_weights.get(k, 0) for k in scores)
                                recommendation, color = get_recommendation(total)
                                
                                stock_data[symbol] = {
                                    'total_score': total,
                                    'recommendation': recommendation,
                                    'color': color,
                                    'scores': scores,
                                    'overview': overview
                                }
                                
                                show_stock_data(symbol, company_name, price, scores, total, recommendation, color, overview)
                                st.divider()
                                
                                if i < len(symbol_list) - 1:
                                    time.sleep(REQUEST_DELAY)
                    
                    progress_bar.empty()
                    
                    if len(stock_data) > 1:
                        st.subheader("📊 Stock Comparison")
                        comparison_fig = create_comparison_chart(stock_data)
                        if comparison_fig:
                            st.plotly_chart(comparison_fig, use_container_width=True)
        
        with analysis_tab3:
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
            
            symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="quick_tech")
            
            if st.button("⚡ Get Quick Signals", type="primary"):
                analyzer = ComprehensiveStockAnalyzer(symbol, "3mo")
                
                if analyzer.fetch_all_data():
                    analyzer.calculate_technical_indicators()
                    signals = analyzer.generate_technical_signals()
                    
                    # Quick display
                    st.subheader(f"⚡ Quick Signals for {symbol.upper()}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for signal_name, signal_text in signals['signals'].items():
                            if "🟢" in signal_text:
                                st.success(f"**{signal_name}**: {signal_text}")
                            elif "🔴" in signal_text:
                                st.error(f"**{signal_name}**: {signal_text}")
                            else:
                                st.info(f"**{signal_name}**: {signal_text}")
                    
                    with col2:
                        st.metric("Technical Score", f"{signals['technical_score']:.1f}/10")
                        
                        # Quick buy/sell recommendation
                        if signals['technical_score'] >= 6.5:
                            st.success("🟢 TECHNICAL BUY SIGNAL")
                        elif signals['technical_score'] <= 4:
                            st.error("🔴 TECHNICAL SELL SIGNAL")
                        else:
                            st.warning("🟡 TECHNICAL NEUTRAL")
        
        with signals_tab2:
            st.subheader("🔄 Combined Technical & Fundamental Analysis")
            st.info("Comprehensive analysis combining multiple methodologies")
            
            symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="combined_analysis")
            
            if st.button("🔄 Run Combined Analysis", type="primary"):
                analyzer = ComprehensiveStockAnalyzer(symbol, "1y")
                
                if analyzer.fetch_all_data():
                    analyzer.calculate_fundamental_scores()
                    analyzer.calculate_technical_indicators()
                    analyzer.generate_technical_signals()
                    analyzer.calculate_buying_prices()
                    
                    result = analyzer.generate_combined_recommendation()
                    if result and len(result) == 3:
                        recommendation, color, combined_score = result
                        
                        # Display comprehensive analysis
                        display_comprehensive_analysis(analyzer)
                    else:
                        st.error(f"❌ Could not generate recommendation for {symbol}")
                else:
                    st.error(f"❌ Could not fetch data for {symbol}")
        
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
            
            with col1:
                min_score_danish = st.slider(
                    "Minimum Score", 
                    min_value=0.0, 
                    max_value=10.0, 
                    value=6.0, 
                    step=0.1,
                    key="danish_min_score"
                )
            
            with col2:
                max_stocks_danish = st.number_input(
                    "Max Results", 
                    min_value=5, 
                    max_value=50, 
                    value=15,
                    key="danish_max_stocks"
                )
            
            with col3:
                sector_filter = st.selectbox(
                    "Sector Filter",
                    ["All Sectors", "Healthcare", "Industrials", "Technology", "Energy", "Consumer Staples"],
                    key="danish_sector_filter"
                )
            
            if st.button("🚀 Screen Danish Stocks", type="primary", key="danish_screen"):
                with st.spinner("Screening Danish stocks..."):
                    results_df = screen_multi_market_stocks("Danish Stocks", min_score_danish, None)
                
                if not results_df.empty:
                    # Apply sector filter if selected
                    if sector_filter != "All Sectors":
                        results_df = results_df[results_df['Sector'].str.contains(sector_filter, na=False)]
                    
                    display_df = results_df.head(max_stocks_danish)
                    
                    st.success(f"✅ Found **{len(results_df)}** Danish stocks with score ≥ {min_score_danish}")
                    
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
        st.header("📊 Portfolio Manager & Weekly Screeners")
        st.markdown("Manage your stock portfolio and setup automated weekly screening alerts")
        
        # Initialize portfolio in session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        
        # Create sub-tabs for different portfolio functions
        portfolio_tab1, portfolio_tab2, portfolio_tab3, portfolio_tab4, portfolio_tab5 = st.tabs([
            "💼 My Portfolio", 
            "🔄 Portfolio Rebalancing",
            "🔍 Weekly Market Screener", 
            "📈 Portfolio Alerts", 
            "⚙️ Screener Settings"
        ])
        
        with portfolio_tab1:
            st.subheader("💼 My Stock Portfolio")
            
            # Portfolio input section
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Add stocks to portfolio
                st.markdown("**➕ Add Stocks to Portfolio**")
                
                # Quick add from popular stocks
                popular_stocks = {
                    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                    "Danish Stocks": ["NOVO-B.CO", "MAERSK-B.CO", "ORSTED.CO", "DSV.CO", "CARL-B.CO"],
                    "Growth Stocks": ["NVDA", "TSLA", "NFLX", "SHOP", "SQ"],
                    "Value Stocks": ["BRK-B", "JPM", "JNJ", "PG", "KO"]
                }
                
                selected_category = st.selectbox("Quick Add Category:", ["Manual Entry"] + list(popular_stocks.keys()), key="portfolio_quick_add_category")
                
                if selected_category == "Manual Entry":
                    new_symbols = st.text_input(
                        "Enter stock symbols (comma-separated):",
                        placeholder="AAPL, MSFT, GOOGL",
                        key="portfolio_manual_entry"
                    )
                    
                    if st.button("➕ Add to Portfolio") and new_symbols:
                        symbols_to_add = [s.strip().upper() for s in new_symbols.split(',') if s.strip()]
                        added_count = 0
                        
                        for symbol in symbols_to_add:
                            if symbol not in st.session_state.portfolio:
                                st.session_state.portfolio.append(symbol)
                                added_count += 1
                        
                        if added_count > 0:
                            st.success(f"Added {added_count} stocks to portfolio!")
                            auto_sync_if_enabled()  # Auto-sync if enabled
                            st.rerun()
                        else:
                            st.info("All symbols are already in your portfolio")
                
                else:
                    # Quick add from categories
                    st.write(f"**{selected_category} Stocks:**")
                    cols = st.columns(5)
                    for i, symbol in enumerate(popular_stocks[selected_category]):
                        with cols[i % 5]:
                            if st.button(f"Add {symbol}", key=f"add_portfolio_{symbol}"):
                                if symbol not in st.session_state.portfolio:
                                    st.session_state.portfolio.append(symbol)
                                    st.success(f"Added {symbol}!")
                                    auto_sync_if_enabled()  # Auto-sync if enabled
                                    st.rerun()
                                else:
                                    st.info(f"{symbol} already in portfolio")
            
            with col2:
                st.markdown("**📊 Portfolio Stats**")
                st.metric("Total Stocks", len(st.session_state.portfolio))
                
                # Portfolio sync status
                sync_status, last_updated = get_portfolio_sync_status()
                
                if sync_status == "synced":
                    st.success("🔄 Synced with Monitor")
                    if last_updated:
                        st.caption(f"Last sync: {last_updated[:16]}")
                elif sync_status == "out_of_sync":
                    st.warning("⚠️ Out of sync - Click sync!")
                    if last_updated:
                        st.caption(f"Last sync: {last_updated[:16]}")
                elif sync_status == "not_synced":
                    st.info("📄 Not synced yet")
                else:
                    st.error("❌ Sync status error")
                
                # Sync controls
                if st.session_state.portfolio:
                    st.markdown("---")
                    st.markdown("**🔄 Monitor Integration**")
                    
                    # Sync button
                    if st.button("🔄 Sync with Automated Monitor", type="primary"):
                        if save_portfolio_to_file():
                            st.success("✅ Portfolio synced with automated monitoring!")
                            st.info("Your portfolio will now be monitored automatically")
                            st.rerun()
                        else:
                            st.error("❌ Failed to sync portfolio")
                    
                    # Auto-sync toggle
                    auto_sync = st.checkbox(
                        "🔄 Auto-sync on changes",
                        value=st.session_state.get('auto_sync_portfolio', False),
                        help="Automatically sync portfolio when changes are made"
                    )
                    st.session_state['auto_sync_portfolio'] = auto_sync
                    
                    # Quick portfolio analysis
                    if st.button("📈 Quick Portfolio Analysis"):
                        analyze_portfolio_quick(st.session_state.portfolio)
                else:
                    # Quick portfolio analysis
                    if st.button("📈 Quick Portfolio Analysis"):
                        analyze_portfolio_quick(st.session_state.portfolio)
            
            # Display current portfolio
            if st.session_state.portfolio:
                st.markdown("---")
                st.subheader("📋 Current Portfolio")
                
                # Portfolio management
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Display portfolio as editable list
                    portfolio_df = pd.DataFrame({
                        'Symbol': st.session_state.portfolio,
                        'Remove': [False] * len(st.session_state.portfolio)
                    })
                    
                    edited_df = st.data_editor(
                        portfolio_df,
                        column_config={
                            "Symbol": st.column_config.TextColumn("Stock Symbol", disabled=True),
                            "Remove": st.column_config.CheckboxColumn("Remove?")
                        },
                        use_container_width=True,
                        hide_index=True,
                        key="portfolio_editor"
                    )
                    
                    # Remove selected stocks
                    if st.button("🗑️ Remove Selected"):
                        symbols_to_remove = edited_df[edited_df['Remove']]['Symbol'].tolist()
                        if symbols_to_remove:
                            for symbol in symbols_to_remove:
                                st.session_state.portfolio.remove(symbol)
                            st.success(f"Removed {len(symbols_to_remove)} stocks")
                            auto_sync_if_enabled()  # Auto-sync if enabled
                            st.rerun()
                
                with col2:
                    # Portfolio actions
                    st.markdown("**Actions:**")
                    
                    if st.button("📊 Full Analysis"):
                        analyze_full_portfolio(st.session_state.portfolio)
                    
                    if st.button("📋 Copy Symbols"):
                        symbols_text = ", ".join(st.session_state.portfolio)
                        st.text_area("Copy these symbols:", value=symbols_text, height=100, key="copy_portfolio")
                    
                    if st.button("🗑️ Clear All"):
                        st.session_state.portfolio = []
                        auto_sync_if_enabled()  # Auto-sync if enabled
                        st.rerun()
                
                with col3:
                    # Export options
                    st.markdown("**Export:**")
                    
                    if st.button("💾 Save Portfolio"):
                        portfolio_df = pd.DataFrame({'Symbol': st.session_state.portfolio})
                        csv = portfolio_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            data=csv,
                            file_name=f"my_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
            else:
                st.info("Your portfolio is empty. Add some stocks to get started!")
        
        with portfolio_tab2:
            st.subheader("🔄 Smart Portfolio Rebalancing")
            st.markdown("Optimize your portfolio by removing underperformers and adding high-scoring opportunities")
            
            if not st.session_state.portfolio:
                st.warning("⚠️ Your portfolio is empty. Add stocks in the 'My Portfolio' tab first.")
            else:
                # Rebalancing configuration
                st.markdown("### ⚙️ Rebalancing Configuration")
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    target_size = st.slider(
                        "🎯 Target Portfolio Size", 
                        min_value=5, 
                        max_value=25, 
                        value=min(15, max(10, len(st.session_state.portfolio))),
                        help="Ideal number of stocks in your portfolio"
                    )
                    
                    min_keep_score = st.slider(
                        "✅ Minimum Score to Keep", 
                        min_value=4.0, 
                        max_value=8.0, 
                        value=6.0,
                        step=0.1,
                        help="Stocks below this score will be suggested for removal"
                    )
                
                with config_col2:
                    min_add_score = st.slider(
                        "➕ Minimum Score to Add", 
                        min_value=6.0, 
                        max_value=9.0, 
                        value=7.5,
                        step=0.1,
                        help="Only add stocks with scores above this threshold"
                    )
                    
                    rebalance_market = st.selectbox(
                        "🌍 Source for New Stocks",
                        ["S&P 500", "NASDAQ 100", "Danish Stocks", "European Stocks"],
                        help="Market to search for new investment opportunities"
                    )
                
                with config_col3:
                    rebalance_frequency = st.selectbox(
                        "📅 Rebalancing Frequency",
                        ["Manual Only", "Monthly", "Quarterly", "Semi-Annually"],
                        index=2,
                        help="How often to suggest rebalancing"
                    )
                    
                    aggressive_mode = st.checkbox(
                        "⚡ Aggressive Mode",
                        value=False,
                        help="Allow larger portfolio changes for better optimization"
                    )
                
                st.markdown("---")
                
                # Current portfolio analysis summary
                st.markdown("### 📊 Current Portfolio Overview")
                
                if len(st.session_state.portfolio) > 0:
                    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
                    
                    with overview_col1:
                        st.metric("Current Size", len(st.session_state.portfolio))
                    
                    with overview_col2:
                        target_change = target_size - len(st.session_state.portfolio)
                        st.metric("Target Change", f"{target_change:+d}")
                    
                    with overview_col3:
                        # Quick estimate of diversification
                        diversification = "Good" if len(st.session_state.portfolio) >= 10 else "Limited"
                        st.metric("Diversification", diversification)
                    
                    with overview_col4:
                        # Last rebalance info
                        last_rebalance = st.session_state.get('last_rebalance_date', 'Never')
                        st.metric("Last Rebalance", last_rebalance)
                
                # Main rebalancing action
                st.markdown("---")
                
                rebalance_col1, rebalance_col2 = st.columns([2, 1])
                
                with rebalance_col1:
                    if st.button("🔄 Analyze Rebalancing Opportunities", type="primary", use_container_width=True):
                        with st.spinner("Analyzing your portfolio for rebalancing opportunities..."):
                            try:
                                actions, analysis_df = rebalance_portfolio_manual(
                                    st.session_state.portfolio,
                                    target_size=target_size,
                                    min_keep_score=min_keep_score,
                                    min_add_score=min_add_score,
                                    source_market=rebalance_market
                                )
                                
                                if actions and analysis_df is not None:
                                    # Store results in session state for potential application
                                    st.session_state['rebalance_suggestions'] = actions
                                    st.session_state['rebalance_analysis'] = analysis_df
                                    
                                    # Display the analysis
                                    display_rebalancing_analysis(actions, analysis_df)
                                    
                                    # Show simulation
                                    create_rebalancing_simulation(st.session_state.portfolio, actions)
                                    
                                    st.success("✅ Rebalancing analysis complete!")
                                else:
                                    st.error("❌ Could not generate rebalancing suggestions. Please try again.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during rebalancing analysis: {str(e)}")
                
                with rebalance_col2:
                    st.markdown("**🎯 Rebalancing Goals:**")
                    st.write("• Improve average score")
                    st.write("• Remove underperformers") 
                    st.write("• Add high-potential stocks")
                    st.write("• Maintain diversification")
                    st.write("• Optimize portfolio size")
                
                # Apply rebalancing if suggestions exist
                if 'rebalance_suggestions' in st.session_state and st.session_state.rebalance_suggestions:
                    st.markdown("---")
                    st.markdown("### ⚡ Apply Rebalancing")
                    
                    apply_col1, apply_col2, apply_col3 = st.columns(3)
                    
                    with apply_col1:
                        if st.button("✅ Apply All Changes", type="secondary"):
                            # Apply the rebalancing
                            st.session_state.portfolio = st.session_state.rebalance_suggestions['new_portfolio'].copy()
                            st.session_state['last_rebalance_date'] = datetime.now().strftime("%Y-%m-%d")
                            
                            # Clear suggestions
                            del st.session_state['rebalance_suggestions']
                            if 'rebalance_analysis' in st.session_state:
                                del st.session_state['rebalance_analysis']
                            
                            st.success("🎉 Portfolio rebalanced successfully!")
                            st.balloons()
                            st.rerun()
                    
                    with apply_col2:
                        if st.button("📋 Apply Removals Only"):
                            # Only remove underperformers
                            for symbol in st.session_state.rebalance_suggestions['remove']:
                                if symbol in st.session_state.portfolio:
                                    st.session_state.portfolio.remove(symbol)
                            
                            st.success("🗑️ Underperforming stocks removed!")
                            st.rerun()
                    
                    with apply_col3:
                        if st.button("➕ Apply Additions Only"):
                            # Only add new stocks
                            for symbol in st.session_state.rebalance_suggestions['add']:
                                if symbol not in st.session_state.portfolio:
                                    st.session_state.portfolio.append(symbol)
                            
                            st.success("✨ New stocks added to portfolio!")
                            st.rerun()
                    
                    # Export rebalancing plan
                    st.markdown("---")
                    
                    if st.button("📥 Export Rebalancing Plan"):
                        actions = st.session_state.rebalance_suggestions
                        
                        plan_text = f"""
Portfolio Rebalancing Plan - {datetime.now().strftime('%Y-%m-%d')}
=================================================================

Current Portfolio Size: {len(st.session_state.portfolio)}
Target Portfolio Size: {actions['target_size']}

REMOVE (Underperformers):
{chr(10).join(f"• {symbol}" for symbol in actions['remove']) if actions['remove'] else "• None"}

KEEP (Good Performers):
{chr(10).join(f"• {symbol}" for symbol in actions['keep']) if actions['keep'] else "• None"}

ADD (New Opportunities):
{chr(10).join(f"• {symbol}" for symbol in actions['add']) if actions['add'] else "• None"}

New Portfolio:
{', '.join(actions['new_portfolio'])}

Generated by Advanced Stock Scoring System
"""
                        
                        st.download_button(
                            "📥 Download Rebalancing Plan",
                            data=plan_text,
                            file_name=f"rebalancing_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                
                # Rebalancing education
                with st.expander("📚 Learn About Portfolio Rebalancing"):
                    st.markdown("""
                    ### 🎯 **Why Rebalance Your Portfolio?**
                    
                    **Performance Optimization:**
                    - Remove stocks with declining fundamentals
                    - Add stocks with improving prospects
                    - Maintain exposure to best opportunities
                    
                    **Risk Management:**
                    - Prevent over-concentration in any single stock
                    - Maintain appropriate diversification
                    - Manage overall portfolio volatility
                    
                    **Discipline & Process:**
                    - Systematic approach to portfolio management
                    - Remove emotion from investment decisions
                    - Stay aligned with investment strategy
                    
                    ### 📅 **When to Rebalance:**
                    - **Quarterly**: Good balance of optimization and transaction costs
                    - **Semi-Annually**: Lower cost, still captures major changes
                    - **Event-Driven**: When stocks hit specific score thresholds
                    - **Market Conditions**: During major market shifts
                    
                    ### ⚠️ **Rebalancing Best Practices:**
                    - Don't rebalance too frequently (increases costs)
                    - Consider tax implications in taxable accounts
                    - Gradual implementation for large changes
                    - Monitor correlation between new additions
                    - Keep some cash for opportunities
                    """)
        
        with portfolio_tab3:
            st.subheader("🔍 Weekly Market Screener")
            st.markdown("Automatically screen the market weekly for new investment opportunities")
            
            # Screener configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Screening Parameters**")
                
                screener_market = st.selectbox(
                    "Market to Screen:",
                    ["S&P 500", "NASDAQ 100", "Danish Stocks", "European Stocks", "All Markets"],
                    key="weekly_screener_market"
                )
                
                min_score_market = st.slider(
                    "Minimum Score Threshold:",
                    min_value=5.0,
                    max_value=9.0,
                    value=7.0,
                    step=0.1,
                    key="weekly_screener_min_score"
                )
                
                max_results = st.number_input(
                    "Maximum Results:",
                    min_value=5,
                    max_value=50,
                    value=20,
                    key="weekly_screener_max_results"
                )
            
            with col2:
                st.markdown("**📅 Schedule Settings**")
                
                screening_day = st.selectbox(
                    "Screening Day:",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=0,
                    key="screening_day"
                )
                
                screening_time = st.time_input(
                    "Screening Time:",
                    value=datetime.strptime("09:00", "%H:%M").time(),
                    key="screening_time"
                )
                
                enable_email = st.checkbox(
                    "Enable Email Notifications",
                    value=False,
                    key="enable_screener_email"
                )
                
                if enable_email:
                    email_address = st.text_input(
                        "Email Address:",
                        placeholder="your.email@example.com",
                        key="screener_email"
                    )
            
            # Test screening
            st.markdown("---")
            st.markdown("**🧪 Test Screening**")
            
            if st.button("🚀 Run Test Screening Now", type="primary"):
                with st.spinner("Running market screening..."):
                    test_results = run_market_screening(
                        screener_market, 
                        min_score_market, 
                        max_results
                    )
                    
                    if not test_results.empty:
                        st.success(f"Found {len(test_results)} promising stocks!")
                        
                        # Display results
                        display_columns = [
                            'Original_Symbol', 'Company', 'Final_Score', 'Recommendation', 
                            'Sector', 'Current_Price', 'P/E_Ratio', 'ROE'
                        ]
                        
                        available_columns = [col for col in display_columns if col in test_results.columns]
                        if available_columns:
                            st.dataframe(
                                test_results[available_columns].head(max_results),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.dataframe(test_results.head(max_results), use_container_width=True, hide_index=True)
                    
                    else:
                        st.warning("No stocks found meeting the criteria. Try lowering the minimum score.")
        
        with portfolio_tab4:
            st.subheader("📈 Portfolio Monitoring & Alerts")
            st.markdown("Monitor your portfolio stocks and get weekly recommendations")
            
            if not st.session_state.portfolio:
                st.warning("⚠️ Your portfolio is empty. Add stocks in the 'My Portfolio' tab first.")
            else:
                # Initialize monitoring settings
                if 'monitoring_settings' not in st.session_state:
                    st.session_state.monitoring_settings = {
                        'email_alerts': False,
                        'email_address': '',
                        'alert_threshold': 1.0,
                        'weekly_reports': False,
                        'last_analysis': None
                    }
                
                # Alert Configuration
                st.subheader("� Alert Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.monitoring_settings['email_alerts'] = st.checkbox(
                        "📧 Enable Email Alerts", 
                        value=st.session_state.monitoring_settings['email_alerts'],
                        help="Get notified when stock scores change significantly"
                    )
                    
                    if st.session_state.monitoring_settings['email_alerts']:
                        st.session_state.monitoring_settings['email_address'] = st.text_input(
                            "Email Address:",
                            value=st.session_state.monitoring_settings['email_address'],
                            placeholder="your.email@example.com",
                            key="monitoring_email"
                        )
                
                with col2:
                    st.session_state.monitoring_settings['alert_threshold'] = st.slider(
                        "🎯 Alert Threshold (Score Change)",
                        min_value=0.5,
                        max_value=3.0,
                        value=st.session_state.monitoring_settings['alert_threshold'],
                        step=0.1,
                        help="Minimum score change to trigger an alert"
                    )
                    
                    st.session_state.monitoring_settings['weekly_reports'] = st.checkbox(
                        "📊 Weekly Performance Reports",
                        value=st.session_state.monitoring_settings['weekly_reports'],
                        help="Receive weekly portfolio performance summaries"
                    )
                
                # Automated Portfolio Analysis
                st.markdown("---")
                st.subheader("🤖 Automated Portfolio Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚀 Run Full Portfolio Analysis", type="primary"):
                        run_automated_portfolio_analysis(st.session_state.portfolio)
                
                with col2:
                    if st.button("📈 Quick Health Check"):
                        run_portfolio_health_check(st.session_state.portfolio)
                
                with col3:
                    if st.button("⚠️ Risk Assessment"):
                        run_portfolio_risk_assessment(st.session_state.portfolio)
                
                # Performance Tracking
                st.markdown("---")
                st.subheader("📊 Performance Tracking")
                
                # Initialize portfolio history
                if 'portfolio_history' not in st.session_state:
                    st.session_state.portfolio_history = []
                
                if st.button("📸 Save Current Portfolio Snapshot"):
                    save_portfolio_snapshot(st.session_state.portfolio)
                
                # Display historical performance if available
                if st.session_state.portfolio_history:
                    display_portfolio_performance_history()
                
                # Weekly Report Generator
                st.markdown("---")
                st.subheader("📋 Weekly Report Generator")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Generate Weekly Report"):
                        generate_weekly_portfolio_report(st.session_state.portfolio)
                
                with col2:
                    report_format = st.selectbox(
                        "Report Format:",
                        ["HTML Summary", "Detailed PDF", "CSV Data", "JSON Export"],
                        key="report_format_selector"
                    )
                
                # Alert History
                if 'alert_history' not in st.session_state:
                    st.session_state.alert_history = []
                
                if st.session_state.alert_history:
                    st.markdown("---")
                    st.subheader("🔔 Recent Alerts")
                    
                    # Display last 5 alerts
                    recent_alerts = st.session_state.alert_history[-5:]
                    for alert in reversed(recent_alerts):
                        alert_type = "🔴" if alert['type'] == 'warning' else "🟡" if alert['type'] == 'info' else "🟢"
                        st.write(f"{alert_type} **{alert['timestamp']}** - {alert['message']}")
                
                # Monitoring Status
                st.markdown("---")
                st.subheader("📱 Monitoring Status")
                
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    status = "🟢 Active" if st.session_state.monitoring_settings['email_alerts'] else "🔴 Inactive"
                    st.metric("Email Alerts", status)
                
                with status_col2:
                    status = "🟢 Enabled" if st.session_state.monitoring_settings['weekly_reports'] else "🔴 Disabled"
                    st.metric("Weekly Reports", status)
                
                with status_col3:
                    portfolio_count = len(st.session_state.portfolio)
                    st.metric("Monitored Stocks", portfolio_count)
        
        with portfolio_tab5:
            st.subheader("⚙️ Advanced Screener Settings")
            st.markdown("Configure advanced settings for screening")
            
            st.info("🚧 Advanced screener settings coming soon!")
            st.markdown("**Planned Features:**")
            st.markdown("- Custom weight adjustments")
            st.markdown("- Sector preferences")
            st.markdown("- Market cap filters")

    # --- Danish Stocks Manager ---
    with tab5:
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
        search_term = st.text_input("🔍 Search stocks", placeholder="Search by symbol or name...", key="danish_search")
        
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
            new_name = st.text_input("Display Name", placeholder="e.g., NOVO-B", key="new_danish_name")
            new_symbol = st.text_input("Yahoo Symbol", placeholder="e.g., NOVO-B.CO", key="new_danish_symbol")
            
            if st.button("➕ Add Stock") and new_name and new_symbol:
                if new_name not in DANISH_STOCKS:
                    st.success(f"Would add: {new_name} -> {new_symbol}")
                    st.info("Note: This is a demo. In production, this would update the stock database.")
                else:
                    st.warning(f"{new_name} already exists")

    # --- Performance Benchmarking ---
    with tab6:
        # Initialize score tracking
        initialize_score_tracking()
        
        # Create the performance benchmarking dashboard
        create_performance_dashboard()
        
        st.markdown("---")
        
        # Score tracking section
        display_score_tracking()

    # --- Help & Documentation ---
    with tab7:
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
            ### 🎯 **Scoring Methodology**
            
            Each stock receives a score from 0-10 based on multiple financial metrics:
            
            #### 📈 **Core Metrics:**
            
            **Valuation Metrics:**
            - **P/E Ratio**: Price-to-Earnings (lower is generally better)
            - **PEG Ratio**: P/E relative to growth (< 1.0 is ideal)
            - **P/B Ratio**: Price-to-Book (< 1.0 indicates undervaluation)
            - **EV/EBITDA**: Enterprise value to EBITDA
            
            **Profitability Metrics:**
            - **ROE**: Return on Equity (higher is better)
            - **Gross Margin**: Profitability efficiency
            - **Free Cash Flow**: Cash generation ability
            
            **Growth Metrics:**
            - **Revenue Growth**: Year-over-year revenue increase
            - **EPS Growth**: Earnings per share growth
            
            **Financial Health:**
            - **Debt/Equity**: Financial leverage (lower is safer)
            - **Dividend Yield**: Income generation
            
            #### 🎯 **Score Interpretation:**
            - **8.0-10.0**: 🚀 Strong Buy - Exceptional opportunity
            - **6.5-7.9**: 📈 Buy - Good investment potential
            - **4.0-6.4**: 🔄 Hold - Adequate performance
            - **2.0-3.9**: 📉 Weak Sell - Below average
            - **0.0-1.9**: 🛑 Strong Sell - Poor fundamentals
            """)
        
        with help_tab3:
            st.subheader("📈 Trading Signals Guide")
            st.markdown("""
            ### 🎯 **Technical Analysis Signals**
            
            #### 📊 **Key Indicators:**
            
            **RSI (Relative Strength Index)**
            - **< 30**: Oversold (potential buy)
            - **> 70**: Overbought (potential sell)
            - **30-70**: Normal trading range
            
            **Moving Averages**
            - **Golden Cross**: 20-day MA crosses above 50-day MA (bullish)
            - **Death Cross**: 20-day MA crosses below 50-day MA (bearish)
            - **Price above both MAs**: Strong uptrend
            
            **MACD (Moving Average Convergence Divergence)**
            - **MACD > Signal**: Bullish momentum
            - **MACD < Signal**: Bearish momentum
            - **Histogram**: Shows momentum strength
            """)
        
        with help_tab4:
            st.subheader("🔧 Advanced Features")
            st.markdown("""
            ### 🎯 **Advanced Capabilities**
            
            #### 🔍 **Multi-Market Screening**
            - Screen S&P 500, NASDAQ 100, European, and Danish stocks
            - Custom symbol lists for personalized screening
            - Sector and market cap filtering
            - Export results for further analysis
            
            #### 💼 **Portfolio Management**
            - **Personal Portfolio**: Track your holdings
            - **Automated Screening**: Weekly market scans
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
    with tab8:
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
            
            symbols_for_comparison = st.text_input(
                "Enter symbols to compare (comma-separated)",
                placeholder="AAPL, MSFT, GOOGL",
                key="manual_comparison"
            )
            
            if st.button("🔍 Compare Stocks", type="primary") and symbols_for_comparison:
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
