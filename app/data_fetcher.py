# app/data_fetcher.py
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from app.config import DANISH_STOCKS

# --- Asynchronous Data Fetching ---
async def fetch_stock_data_async(session, symbol):
    """Asynchronously fetch stock data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        # yfinance is not fully async, so we run blocking calls in a thread pool
        info = await asyncio.to_thread(ticker.info.get)
        return symbol, info
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return symbol, None

async def get_batch_yahoo_info(symbols):
    """Fetch info for a batch of symbols asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_stock_data_async(session, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbol: info for symbol, info in results if info}

# --- Standard Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_yahoo_info(symbol):
    """Fetches comprehensive stock information from Yahoo Finance."""
    # ... (keep your existing fetch_yahoo_info logic here, or adapt it)
    # This function can be simplified if the async version is used primarily
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception:
        return None

@st.cache_data(ttl=86400)
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
            except Exception:
                pass
            
            # Net Income history
            try:
                if 'Net Income' in financials.index:
                    net_income_data = financials.loc['Net Income'].head(3)
                    if not net_income_data.empty:
                        metrics_3y['net_income_trend'] = net_income_data.tolist()
            except Exception:
                pass
            
            # Free Cash Flow
            try:
                if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditures' in cashflow.index:
                    operating_cf = cashflow.loc['Operating Cash Flow'].head(3)
                    capex = cashflow.loc['Capital Expenditures'].head(3)
                    if not operating_cf.empty and not capex.empty:
                        fcf = operating_cf + capex # Capex is usually negative
                        metrics_3y['fcf_trend'] = fcf.tolist()
            except Exception:
                pass
            
            # ROE calculation
            try:
                if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
                    net_income = financials.loc['Net Income'].head(3)
                    equity = balance_sheet.loc['Total Stockholder Equity'].head(3)
                    if not net_income.empty and not equity.empty:
                        roe_series = (net_income / equity) * 100
                        metrics_3y['roe_trend'] = roe_series.tolist()
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
            if symbol in DANISH_STOCKS:
                danish_symbol = DANISH_STOCKS[symbol]
                result = try_fetch_financials(danish_symbol)
                if result:
                    return result
            
            co_symbol = f"{symbol}.CO"
            result = try_fetch_financials(co_symbol)
            if result:
                return result
        
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
            from datetime import datetime, timedelta
            import numpy as np
            
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
            if symbol in DANISH_STOCKS:
                danish_symbol = DANISH_STOCKS[symbol]
                result = try_fetch_price_data(danish_symbol)
                if result:
                    return result
            
            co_symbol = f"{symbol}.CO"
            result = try_fetch_price_data(co_symbol)
            if result:
                return result
        
        return {}
        
    except Exception as e:
        st.error(f"Error fetching price performance for {symbol}: {e}")
        return {}

class PortfolioDataFetcher:
    """Fetches and holds data for a portfolio of multiple stocks."""
    def __init__(self, symbols):
        self.symbols = symbols
        self.all_data = {}

    @st.cache_data(ttl=1800)
    def fetch_all_data(_self):
        """
        Fetches fundamental data for all symbols in the portfolio.
        Using _self to work with st.cache_data as it hashes the arguments.
        """
        data = {}
        for symbol in _self.symbols:
            try:
                info = fetch_yahoo_info(symbol)
                if info and info.get('regularMarketPrice') is not None:
                    data[symbol] = info
                else:
                    # Attempt to fetch with .CO for Danish stocks if initial fails
                    if not symbol.endswith('.CO') and info is None:
                        danish_symbol = f"{symbol}.CO"
                        info = fetch_yahoo_info(danish_symbol)
                        if info and info.get('regularMarketPrice') is not None:
                            data[symbol] = info # Store with original symbol key
            except Exception as e:
                st.error(f"Failed to fetch data for {symbol}: {e}")
        _self.all_data = data
        return _self.all_data

class StockDataFetcher:
    def __init__(self, symbol=None):
        self.symbol = symbol
        self.info = None
        self.financials_3y = None
        self.price_performance = None
        self.technical_data = None

    @st.cache_data(ttl=1800)
    def fetch_all_data(_self):
        """
        Fetches all necessary data for a single stock: info, financials, and price performance.
        Using _self to work with st.cache_data.
        """
        _self.info = fetch_yahoo_info(_self.symbol)
        _self.financials_3y = get_3year_financial_history(_self.symbol)
        _self.price_performance = get_3year_price_performance(_self.symbol)

        # Basic validation to ensure some data was fetched
        if _self.info and _self.info.get('regularMarketPrice') is not None:
            return True
        # Try with .CO if the first attempt failed for a non-.CO symbol
        elif not _self.symbol.endswith('.CO'):
            danish_symbol = f"{_self.symbol}.CO"
            _self.info = fetch_yahoo_info(danish_symbol)
            if _self.info and _self.info.get('regularMarketPrice') is not None:
                _self.financials_3y = get_3year_financial_history(danish_symbol)
                _self.price_performance = get_3year_price_performance(danish_symbol)
                return True

        return False

    async def async_fetch_batch_info(self, symbols):
        """Fetch info for a batch of symbols asynchronously."""
        return await get_batch_yahoo_info(symbols)

    async def async_fetch_info(self):
        """Asynchronously fetch stock info."""
        async with aiohttp.ClientSession() as session:
            self.info = await fetch_stock_data_async(session, self.symbol)

    def calculate_technical_indicators(self):
        """Calculate technical indicators for the stock."""
        if not self.price_performance or not self.price_performance.get('price_data'):
            self.technical_data = None
            return

        try:
            import pandas as pd
            import numpy as np
            
            # Get price data
            df = self.price_performance['price_data'].copy()
            
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            rolling_mean = df['Close'].rolling(window=20).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = rolling_mean + (rolling_std * 2)
            df['BB_Lower'] = rolling_mean - (rolling_std * 2)
            df['BB_Middle'] = rolling_mean
            
            self.technical_data = df
            
        except Exception as e:
            print(f"Error calculating technical indicators for {self.symbol}: {e}")
            self.technical_data = None

    async def fetch_and_process_batch(self, symbols):
        """
        Fetches and processes a batch of symbols asynchronously, returning
        a list of dictionaries with info and technical data.
        """
        st.write(f"Starting async batch fetch for {len(symbols)} symbols...")
        raw_infos = await self.async_fetch_batch_info(symbols)
        
        processed_data = []
        for symbol, info in raw_infos.items():
            if info and info.get('longName'):
                # Create a temporary fetcher to process technical data
                temp_fetcher = StockDataFetcher(symbol)
                temp_fetcher.info = info
                temp_fetcher.calculate_technical_indicators()
                processed_data.append({
                    "symbol": symbol,
                    "info": info,
                    "technical_data": temp_fetcher.technical_data
                })
        st.write(f"Finished processing for {len(processed_data)} symbols.")
        return processed_data
