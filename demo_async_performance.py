#!/usr/bin/env python3
"""
Quick Demo: Async Data Loading Performance
==========================================
Demonstrates the performance benefits of async data loading.
"""

import time
import yfinance as yf
from async_data_loader import AsyncStockDataLoader

def sync_data_loading(symbols):
    """Traditional synchronous data loading"""
    print("🐌 Synchronous Loading Test")
    print("-" * 30)
    
    start_time = time.time()
    results = {}
    
    for i, symbol in enumerate(symbols):
        print(f"   Loading {symbol}... ({i+1}/{len(symbols)})")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            results[symbol] = {
                'success': True,
                'current_price': info.get('currentPrice', 0),
                'sector': info.get('sector', 'Unknown')
            }
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
    
    sync_time = time.time() - start_time
    successful = len([r for r in results.values() if r.get('success', False)])
    
    print(f"   ✅ Completed in {sync_time:.2f} seconds")
    print(f"   📊 Success rate: {successful}/{len(symbols)}")
    
    return sync_time, results

def async_data_loading(symbols):
    """Enhanced asynchronous data loading"""
    print("🚀 Asynchronous Loading Test")
    print("-" * 30)
    
    loader = AsyncStockDataLoader()
    
    def progress_callback(completed, total, message):
        print(f"   Progress: {completed}/{total} - {message}")
    
    loader.set_progress_callback(progress_callback)
    
    start_time = time.time()
    results = loader.fetch_multiple_stocks_threaded(symbols)
    async_time = time.time() - start_time
    
    successful = len([r for r in results.values() if r.success])
    
    print(f"   ✅ Completed in {async_time:.2f} seconds")
    print(f"   📊 Success rate: {successful}/{len(symbols)}")
    
    return async_time, results

def demo_performance_comparison():
    """Compare sync vs async performance"""
    
    print("⚡ ASYNC DATA LOADING PERFORMANCE DEMO")
    print("=" * 50)
    
    # Test with different sized portfolios
    test_cases = [
        {
            'name': 'Small Portfolio (5 stocks)',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        },
        {
            'name': 'Medium Portfolio (10 stocks)', 
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'JNJ']
        }
    ]
    
    performance_results = []
    
    for test_case in test_cases:
        print(f"\n🔬 Testing: {test_case['name']}")
        print("=" * 40)
        
        symbols = test_case['symbols']
        print(f"Symbols: {', '.join(symbols)}")
        print()
        
        # Synchronous test
        sync_time, sync_results = sync_data_loading(symbols)
        
        print()
        
        # Asynchronous test  
        async_time, async_results = async_data_loading(symbols)
        
        # Calculate performance improvement
        speedup = sync_time / async_time if async_time > 0 else 0
        time_saved = sync_time - async_time
        
        print(f"\n📈 Performance Results:")
        print(f"   Sync Time: {sync_time:.2f}s")
        print(f"   Async Time: {async_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time Saved: {time_saved:.2f} seconds")
        
        performance_results.append({
            'test': test_case['name'],
            'symbols': len(symbols),
            'sync_time': sync_time,
            'async_time': async_time,
            'speedup': speedup,
            'time_saved': time_saved
        })
    
    # Overall summary
    print(f"\n🎯 PERFORMANCE SUMMARY")
    print("=" * 30)
    
    total_sync_time = sum(r['sync_time'] for r in performance_results)
    total_async_time = sum(r['async_time'] for r in performance_results)
    overall_speedup = total_sync_time / total_async_time if total_async_time > 0 else 0
    total_time_saved = total_sync_time - total_async_time
    
    print(f"📊 Overall Results:")
    print(f"   Total Sync Time: {total_sync_time:.2f}s")
    print(f"   Total Async Time: {total_async_time:.2f}s") 
    print(f"   Overall Speedup: {overall_speedup:.1f}x")
    print(f"   Total Time Saved: {total_time_saved:.2f}s")
    
    print(f"\n✅ Key Benefits:")
    print(f"   🚀 {overall_speedup:.1f}x faster data loading")
    print(f"   ⏰ {total_time_saved:.1f} seconds saved per analysis")
    print(f"   📱 Non-blocking UI with progress tracking")
    print(f"   🔄 Better error handling per stock")
    print(f"   📈 Scales better with portfolio size")
    
    print(f"\n🎉 Async Data Loading Demo Complete!")

if __name__ == "__main__":
    demo_performance_comparison()
