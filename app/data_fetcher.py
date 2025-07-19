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
    # ... (keep your existing get_3year_price_performance logic)
    pass

class StockDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = None
        self.technical_data = None

    async def async_fetch_info(self):
        """Asynchronously fetch stock info."""
        async with aiohttp.ClientSession() as session:
            self.info = await fetch_stock_data_async(session, self.symbol)

    def calculate_technical_indicators(self):
        """Calculate technical indicators for the stock."""
        if not self.info:
            return

        # Example calculation: Moving Average
        try:
            close_prices = self.info['historicalData'][-30:]  # Last 30 days
            self.technical_data = {
                'moving_average': sum(day['close'] for day in close_prices) / len(close_prices)
            }
        except Exception as e:
            print(f"Error calculating technical indicators for {self.symbol}: {e}")

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
