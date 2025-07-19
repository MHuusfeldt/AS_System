"""
Advanced Caching System - Multi-Level Caching Strategy
=====================================================
Intelligent caching with different TTL for different data types,
date-aware caching, and cache invalidation strategies.
"""

import streamlit as st
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable
import pickle
import os
import json
from functools import wraps

class AdvancedCache:
    def __init__(self, cache_dir: str = ".streamlit_cache"):
        """Initialize advanced caching system"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache TTL configurations (in seconds)
        self.cache_configs = {
            # Fast-changing data
            'current_price': 300,        # 5 minutes
            'intraday_data': 300,        # 5 minutes
            'volume_data': 300,          # 5 minutes
            
            # Medium-changing data
            'technical_indicators': 3600,  # 1 hour
            'analyst_data': 3600,          # 1 hour
            'market_cap': 3600,            # 1 hour
            
            # Slow-changing data
            'fundamental_data': 86400,     # 24 hours
            'company_info': 86400,         # 24 hours
            'sector_industry': 86400,      # 24 hours
            'financial_statements': 86400, # 24 hours
            
            # Very slow-changing data
            'historical_data': 604800,     # 7 days
            'dividend_history': 604800,    # 7 days
            'stock_splits': 2592000,       # 30 days
        }
    
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict, 
                      include_date: bool = False) -> str:
        """Generate unique cache key"""
        # Create base key from function and parameters
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': {k: v for k, v in kwargs.items() if k != 'cache_type'}
        }
        
        # Include date for historical data caching
        if include_date:
            key_data['date'] = datetime.now().date().isoformat()
        
        # Create hash
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.cache")
    
    def _is_cache_valid(self, cache_file: str, ttl: int) -> bool:
        """Check if cache file is still valid"""
        if not os.path.exists(cache_file):
            return False
        
        # Check file age
        file_age = time.time() - os.path.getmtime(cache_file)
        return file_age < ttl
    
    def _save_to_cache(self, cache_file: str, data: Any) -> bool:
        """Save data to cache file"""
        try:
            cache_data = {
                'data': data,
                'timestamp': time.time(),
                'created_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
            return False
    
    def _load_from_cache(self, cache_file: str) -> Optional[Any]:
        """Load data from cache file"""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            return cache_data['data']
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            return None
    
    def cached_function(self, cache_type: str = 'default', include_date: bool = False):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get TTL for this cache type
                ttl = self.cache_configs.get(cache_type, 3600)  # Default 1 hour
                
                # Generate cache key
                cache_key = self._get_cache_key(
                    func.__name__, args, kwargs, include_date
                )
                cache_file = self._get_cache_file_path(cache_key)
                
                # Check if valid cache exists
                if self._is_cache_valid(cache_file, ttl):
                    cached_data = self._load_from_cache(cache_file)
                    if cached_data is not None:
                        return cached_data
                
                # Execute function and cache result
                try:
                    result = func(*args, **kwargs)
                    self._save_to_cache(cache_file, result)
                    return result
                except Exception as e:
                    print(f"⚠️ Error in cached function {func.__name__}: {e}")
                    # Try to return stale cache if available
                    if os.path.exists(cache_file):
                        return self._load_from_cache(cache_file)
                    raise
            
            return wrapper
        return decorator
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache files matching pattern"""
        try:
            if pattern:
                # Remove specific cache files
                for filename in os.listdir(self.cache_dir):
                    if pattern in filename and filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
            else:
                # Remove all cache files
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
            
            print("✅ Cache invalidated successfully")
        except Exception as e:
            print(f"⚠️ Error invalidating cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
            
            total_size = 0
            file_ages = []
            
            for filename in cache_files:
                filepath = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(filepath)
                file_age = time.time() - os.path.getmtime(filepath)
                file_ages.append(file_age)
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'avg_age_hours': sum(file_ages) / len(file_ages) / 3600 if file_ages else 0,
                'oldest_cache_hours': max(file_ages) / 3600 if file_ages else 0
            }
        except Exception as e:
            print(f"⚠️ Error getting cache stats: {e}")
            return {}

# Global cache instance
advanced_cache = AdvancedCache()

# Streamlit-integrated caching decorators
def cache_current_price(ttl: int = 300):
    """Cache current price data (5 minutes default)"""
    return st.cache_data(ttl=ttl, show_spinner=False)

def cache_fundamental_data(ttl: int = 86400):
    """Cache fundamental data (24 hours default)"""
    return st.cache_data(ttl=ttl, show_spinner=False)

def cache_technical_data(ttl: int = 3600):
    """Cache technical indicators (1 hour default)"""
    return st.cache_data(ttl=ttl, show_spinner=False)

def cache_historical_data(ttl: int = 604800):
    """Cache historical data (7 days default)"""
    return st.cache_data(ttl=ttl, show_spinner=False)

# Date-aware caching for historical data
def cache_with_date(ttl: int = 86400):
    """Cache with date included in key for historical data"""
    def decorator(func):
        @st.cache_data(ttl=ttl, show_spinner=False)
        def wrapper(*args, **kwargs):
            # Include current date in cache key for historical data
            current_date = datetime.now().date().isoformat()
            return func(*args, **kwargs, _cache_date=current_date)
        return wrapper
    return decorator

# Intelligent cache clearing
def clear_price_caches():
    """Clear all price-related caches"""
    st.cache_data.clear()
    advanced_cache.invalidate_cache('current_price')
    advanced_cache.invalidate_cache('intraday_data')

def clear_fundamental_caches():
    """Clear fundamental data caches"""
    advanced_cache.invalidate_cache('fundamental_data')
    advanced_cache.invalidate_cache('company_info')

def clear_all_caches():
    """Clear all caches"""
    st.cache_data.clear()
    advanced_cache.invalidate_cache()

# Cache warming functions
def warm_cache_for_symbols(symbols: list):
    """Pre-warm cache for multiple symbols"""
    print(f"🔥 Warming cache for {len(symbols)} symbols...")
    
    # This would be called with async data loading
    # Implementation depends on your data fetching functions
    pass

# Cache monitoring
def get_cache_health() -> Dict:
    """Get overall cache health metrics"""
    streamlit_cache_info = {}
    try:
        # Get Streamlit cache info if available
        streamlit_cache_info = {
            'streamlit_cache_enabled': hasattr(st, 'cache_data'),
        }
    except:
        pass
    
    advanced_cache_info = advanced_cache.get_cache_stats()
    
    return {
        'streamlit': streamlit_cache_info,
        'advanced': advanced_cache_info,
        'total_cache_size_mb': advanced_cache_info.get('total_size_mb', 0),
        'cache_efficiency': 'Good' if advanced_cache_info.get('total_files', 0) > 0 else 'Cold'
    }

# Example usage functions with different cache strategies
@cache_fundamental_data(ttl=86400)  # 24 hours
def get_company_fundamentals(symbol: str):
    """Get company fundamental data with long cache"""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract only fundamental data that changes slowly
        fundamental_data = {
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'marketCap': info.get('marketCap'),
            'enterpriseValue': info.get('enterpriseValue'),
            'pegRatio': info.get('pegRatio'),
            'priceToBook': info.get('priceToBook'),
            'returnOnEquity': info.get('returnOnEquity'),
            'grossMargins': info.get('grossMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'revenueGrowth': info.get('revenueGrowth'),
            'earningsGrowth': info.get('earningsGrowth'),
            'debtToEquity': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'quickRatio': info.get('quickRatio'),
        }
        
        return fundamental_data
    except Exception as e:
        print(f"⚠️ Error fetching fundamentals for {symbol}: {e}")
        return None

@cache_current_price(ttl=300)  # 5 minutes
def get_current_price_data(symbol: str):
    """Get current price data with short cache"""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract only rapidly changing data
        price_data = {
            'currentPrice': info.get('currentPrice'),
            'regularMarketPrice': info.get('regularMarketPrice'),
            'regularMarketDayHigh': info.get('regularMarketDayHigh'),
            'regularMarketDayLow': info.get('regularMarketDayLow'),
            'regularMarketVolume': info.get('regularMarketVolume'),
            'averageVolume': info.get('averageVolume'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
        }
        
        return price_data
    except Exception as e:
        print(f"⚠️ Error fetching price data for {symbol}: {e}")
        return None

@cache_technical_data(ttl=3600)  # 1 hour
def get_technical_indicators(symbol: str, period: str = "1y"):
    """Get technical indicators with medium cache"""
    import yfinance as yf
    import pandas as pd
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return None
        
        # Calculate technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Return latest values
        latest = hist.iloc[-1]
        return {
            'sma_20': latest.get('SMA_20'),
            'sma_50': latest.get('SMA_50'),
            'sma_200': latest.get('SMA_200'),
            'rsi': latest.get('RSI'),
            'current_price': latest.get('Close'),
            'volume': latest.get('Volume')
        }
        
    except Exception as e:
        print(f"⚠️ Error calculating technical indicators for {symbol}: {e}")
        return None

@cache_with_date(ttl=604800)  # 7 days with date key
def get_historical_data_cached(symbol: str, period: str = "1y", _cache_date: str = None):
    """Get historical data with date-aware caching"""
    import yfinance as yf
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist.to_dict('records')
    except Exception as e:
        print(f"⚠️ Error fetching historical data for {symbol}: {e}")
        return None
