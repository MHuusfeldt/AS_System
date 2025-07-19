"""
Asynchronous Data Loading System
===============================
High-performance concurrent data fetching for multiple stocks
with progress tracking and error handling.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
import threading
from queue import Queue

@dataclass
class StockAnalysisResult:
    """Data class for stock analysis results"""
    symbol: str
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: float = 0.0

class AsyncStockDataLoader:
    def __init__(self, max_workers: int = 10, timeout: int = 30):
        """Initialize async data loader"""
        self.max_workers = max_workers
        self.timeout = timeout
        self.results = {}
        self.progress_callback = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback function"""
        self.progress_callback = callback
    
    async def fetch_stock_data_async(self, symbol: str) -> StockAnalysisResult:
        """Fetch single stock data asynchronously"""
        start_time = time.time()
        
        try:
            # Run yfinance in thread pool since it's not async-native
            loop = asyncio.get_event_loop()
            
            # Fetch data in thread pool
            ticker_data = await loop.run_in_executor(
                None, self._fetch_ticker_data, symbol
            )
            
            execution_time = time.time() - start_time
            
            if ticker_data:
                return StockAnalysisResult(
                    symbol=symbol,
                    success=True,
                    data=ticker_data,
                    execution_time=execution_time
                )
            else:
                return StockAnalysisResult(
                    symbol=symbol,
                    success=False,
                    error="No data returned",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return StockAnalysisResult(
                symbol=symbol,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _fetch_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data using yfinance (synchronous)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Only return data if we got meaningful information
            if info and 'symbol' in info:
                return info
            return None
            
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
            return None
    
    async def fetch_multiple_stocks_async(self, symbols: List[str]) -> Dict[str, StockAnalysisResult]:
        """Fetch multiple stocks concurrently"""
        print(f"🚀 Starting async fetch for {len(symbols)} symbols...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_fetch(symbol):
            async with semaphore:
                return await self.fetch_stock_data_async(symbol)
        
        # Create tasks
        tasks = [bounded_fetch(symbol) for symbol in symbols]
        
        # Process tasks with progress tracking
        results = {}
        completed_count = 0
        
        # Wait for tasks to complete
        for coro in asyncio.as_completed(tasks):
            try:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
                results[result.symbol] = result
                
                completed_count += 1
                
                # Update progress
                if self.progress_callback:
                    self.progress_callback(
                        completed_count, 
                        len(symbols), 
                        f"Completed {result.symbol}"
                    )
                    
            except asyncio.TimeoutError:
                print(f"⏰ Timeout for one of the stocks")
            except Exception as e:
                print(f"⚠️ Error processing stock: {e}")
        
        return results
    
    def fetch_multiple_stocks_threaded(self, symbols: List[str]) -> Dict[str, StockAnalysisResult]:
        """Alternative: Use ThreadPoolExecutor for CPU-bound tasks"""
        print(f"🧵 Starting threaded fetch for {len(symbols)} symbols...")
        
        results = {}
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._fetch_stock_data_threaded, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results[symbol] = result
                    
                    completed_count += 1
                    
                    # Update progress
                    if self.progress_callback:
                        self.progress_callback(
                            completed_count,
                            len(symbols),
                            f"Completed {symbol}"
                        )
                        
                except Exception as e:
                    results[symbol] = StockAnalysisResult(
                        symbol=symbol,
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    def _fetch_stock_data_threaded(self, symbol: str) -> StockAnalysisResult:
        """Fetch stock data in thread (synchronous)"""
        start_time = time.time()
        
        try:
            ticker_data = self._fetch_ticker_data(symbol)
            execution_time = time.time() - start_time
            
            if ticker_data:
                return StockAnalysisResult(
                    symbol=symbol,
                    success=True,
                    data=ticker_data,
                    execution_time=execution_time
                )
            else:
                return StockAnalysisResult(
                    symbol=symbol,
                    success=False,
                    error="No data returned",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return StockAnalysisResult(
                symbol=symbol,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

class StreamlitAsyncInterface:
    """Streamlit-compatible async interface"""
    
    def __init__(self):
        self.loader = AsyncStockDataLoader()
        self.progress_bar = None
        self.status_text = None
    
    def analyze_multiple_stocks(self, symbols: List[str], 
                              analysis_function: Callable = None) -> Dict[str, Any]:
        """Analyze multiple stocks with Streamlit progress tracking"""
        
        # Setup progress tracking
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
        def update_progress(completed: int, total: int, message: str):
            progress = completed / total
            self.progress_bar.progress(progress)
            self.status_text.text(f"{message} ({completed}/{total})")
        
        self.loader.set_progress_callback(update_progress)
        
        # Choose execution method
        start_time = time.time()
        
        try:
            # Use threaded approach for Streamlit compatibility
            results = self.loader.fetch_multiple_stocks_threaded(symbols)
            
            # Process results through analysis function if provided
            analyzed_results = {}
            
            if analysis_function:
                self.status_text.text("🔬 Analyzing fetched data...")
                
                for symbol, result in results.items():
                    if result.success and result.data:
                        try:
                            analyzed_data = analysis_function(result.data)
                            analyzed_results[symbol] = {
                                'raw_data': result.data,
                                'analysis': analyzed_data,
                                'success': True,
                                'fetch_time': result.execution_time
                            }
                        except Exception as e:
                            analyzed_results[symbol] = {
                                'error': f"Analysis failed: {e}",
                                'success': False,
                                'fetch_time': result.execution_time
                            }
                    else:
                        analyzed_results[symbol] = {
                            'error': result.error,
                            'success': False,
                            'fetch_time': result.execution_time
                        }
            else:
                # Just return raw results
                for symbol, result in results.items():
                    analyzed_results[symbol] = {
                        'data': result.data,
                        'success': result.success,
                        'error': result.error,
                        'fetch_time': result.execution_time
                    }
            
            total_time = time.time() - start_time
            
            # Show completion summary
            successful = sum(1 for r in analyzed_results.values() if r['success'])
            self.status_text.text(
                f"✅ Completed! {successful}/{len(symbols)} successful in {total_time:.1f}s"
            )
            
            # Clear progress bar after a delay
            time.sleep(1)
            self.progress_bar.empty()
            
            return analyzed_results
            
        except Exception as e:
            self.status_text.text(f"❌ Error: {e}")
            return {}
    
    def async_portfolio_analysis(self, portfolio_symbols: List[str]) -> Dict[str, Any]:
        """Analyze entire portfolio asynchronously"""
        
        def portfolio_analysis_function(ticker_info):
            """Custom analysis function for portfolio stocks"""
            from AS_MH_v6 import calculate_scores_yahoo, get_industry_pe
            
            try:
                # Calculate scores using your existing function
                industry_pe = get_industry_pe(ticker_info)
                scores = calculate_scores_yahoo(ticker_info, industry_pe)
                
                if scores:
                    total_score = sum(scores.values()) / len(scores)
                    
                    return {
                        'scores': scores,
                        'total_score': total_score,
                        'current_price': ticker_info.get('currentPrice', 0),
                        'sector': ticker_info.get('sector', 'Unknown'),
                        'industry': ticker_info.get('industry', 'Unknown'),
                        'market_cap': ticker_info.get('marketCap', 0)
                    }
                
                return None
                
            except Exception as e:
                return {'error': str(e)}
        
        return self.analyze_multiple_stocks(portfolio_symbols, portfolio_analysis_function)
    
    def async_screener_analysis(self, symbols: List[str], min_score: float = 5.0) -> pd.DataFrame:
        """Run screener analysis asynchronously"""
        
        def screener_analysis_function(ticker_info):
            """Analysis function for screening"""
            from AS_MH_v6 import calculate_scores_yahoo, get_industry_pe, get_recommendation
            
            try:
                industry_pe = get_industry_pe(ticker_info)
                scores = calculate_scores_yahoo(ticker_info, industry_pe)
                
                if scores:
                    total_score = sum(scores.values()) / len(scores)
                    recommendation_data = get_recommendation(total_score)
                    
                    return {
                        'total_score': total_score,
                        'recommendation': recommendation_data['recommendation'],
                        'color': recommendation_data['color'],
                        'current_price': ticker_info.get('currentPrice', 0),
                        'market_cap': ticker_info.get('marketCap', 0),
                        'sector': ticker_info.get('sector', 'Unknown'),
                        'pe_ratio': ticker_info.get('trailingPE', 0),
                        'scores': scores
                    }
                
                return None
                
            except Exception as e:
                return {'error': str(e)}
        
        # Analyze all stocks
        results = self.analyze_multiple_stocks(symbols, screener_analysis_function)
        
        # Convert to DataFrame for easy filtering and sorting
        screener_data = []
        
        for symbol, result in results.items():
            if result['success'] and result.get('analysis'):
                analysis = result['analysis']
                if analysis.get('total_score', 0) >= min_score:
                    screener_data.append({
                        'Symbol': symbol,
                        'Score': analysis['total_score'],
                        'Recommendation': analysis['recommendation'],
                        'Price': analysis['current_price'],
                        'Market Cap': analysis['market_cap'],
                        'Sector': analysis['sector'],
                        'P/E': analysis['pe_ratio'],
                        'Fetch Time': result['fetch_time']
                    })
        
        df = pd.DataFrame(screener_data)
        
        if not df.empty:
            # Sort by score descending
            df = df.sort_values('Score', ascending=False)
        
        return df

# Performance monitoring
class AsyncPerformanceMonitor:
    def __init__(self):
        self.execution_logs = []
    
    def log_execution(self, operation: str, symbol_count: int, 
                     execution_time: float, success_rate: float):
        """Log execution metrics"""
        self.execution_logs.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'symbol_count': symbol_count,
            'execution_time': execution_time,
            'success_rate': success_rate,
            'symbols_per_second': symbol_count / execution_time if execution_time > 0 else 0
        })
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.execution_logs:
            return {}
        
        recent_logs = self.execution_logs[-10:]  # Last 10 executions
        
        avg_time = sum(log['execution_time'] for log in recent_logs) / len(recent_logs)
        avg_success_rate = sum(log['success_rate'] for log in recent_logs) / len(recent_logs)
        avg_throughput = sum(log['symbols_per_second'] for log in recent_logs) / len(recent_logs)
        
        return {
            'average_execution_time': avg_time,
            'average_success_rate': avg_success_rate,
            'average_throughput': avg_throughput,
            'total_executions': len(self.execution_logs)
        }

# Global instances
async_loader = StreamlitAsyncInterface()
performance_monitor = AsyncPerformanceMonitor()

# Utility functions for easy integration
def analyze_portfolio_async(symbols: List[str]) -> Dict[str, Any]:
    """Easy function to analyze portfolio asynchronously"""
    return async_loader.async_portfolio_analysis(symbols)

def run_screener_async(symbols: List[str], min_score: float = 5.0) -> pd.DataFrame:
    """Easy function to run screener asynchronously"""
    return async_loader.async_screener_analysis(symbols, min_score)

def benchmark_async_performance(symbols: List[str]) -> Dict[str, float]:
    """Benchmark async vs sync performance"""
    
    # Sync benchmark
    start_time = time.time()
    sync_results = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sync_results[symbol] = info
        except:
            pass
    sync_time = time.time() - start_time
    
    # Async benchmark
    loader = AsyncStockDataLoader()
    start_time = time.time()
    async_results = loader.fetch_multiple_stocks_threaded(symbols)
    async_time = time.time() - start_time
    
    sync_success = len([r for r in sync_results.values() if r])
    async_success = len([r for r in async_results.values() if r.success])
    
    return {
        'sync_time': sync_time,
        'async_time': async_time,
        'speedup': sync_time / async_time if async_time > 0 else 0,
        'sync_success_count': sync_success,
        'async_success_count': async_success,
        'symbols_tested': len(symbols)
    }
