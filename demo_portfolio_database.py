#!/usr/bin/env python3
"""
Quick Demo: Enhanced Portfolio Database
======================================
Demonstrates the SQLite portfolio system in action.
"""

from portfolio_database import portfolio_db
import pandas as pd
from datetime import datetime

def demo_enhanced_portfolio():
    """Demonstrate enhanced portfolio features"""
    
    print("🗄️ ENHANCED PORTFOLIO DATABASE DEMO")
    print("=" * 50)
    
    # 1. Add some stocks to portfolio
    print("📝 Adding stocks to portfolio...")
    
    stocks = [
        ('AAPL', 10, 150.00, 'Technology'),
        ('MSFT', 15, 320.00, 'Technology'), 
        ('JNJ', 8, 160.00, 'Healthcare'),
        ('JPM', 12, 140.00, 'Financials')
    ]
    
    for symbol, shares, price, sector in stocks:
        success = portfolio_db.add_to_portfolio(
            symbol=symbol,
            shares=shares, 
            cost_per_share=price,
            sector=sector,
            notes=f"Added via demo on {datetime.now().strftime('%Y-%m-%d')}"
        )
        print(f"   {symbol}: {'✅ Added' if success else '❌ Failed'}")
    
    # 2. Display portfolio
    print(f"\n📊 Current Portfolio:")
    portfolio = portfolio_db.get_portfolio()
    
    if portfolio:
        df = pd.DataFrame(portfolio)
        print(df[['symbol', 'shares', 'average_cost', 'sector', 'added_date']].to_string(index=False))
        
        # Calculate portfolio value (mock current prices)
        mock_prices = {'AAPL': 175.00, 'MSFT': 350.00, 'JNJ': 170.00, 'JPM': 150.00}
        total_value = 0
        total_cost = 0
        
        print(f"\n💰 Portfolio Valuation:")
        for holding in portfolio:
            symbol = holding['symbol']
            shares = holding['shares']
            cost = holding['average_cost']
            current_price = mock_prices.get(symbol, cost)
            
            position_cost = shares * cost
            position_value = shares * current_price
            gain_loss = position_value - position_cost
            gain_loss_pct = (gain_loss / position_cost) * 100 if position_cost > 0 else 0
            
            total_cost += position_cost
            total_value += position_value
            
            print(f"   {symbol}: ${position_value:,.0f} ({gain_loss_pct:+.1f}%) - {shares} shares @ ${current_price:.2f}")
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
        
        print(f"\n📈 Total Portfolio:")
        print(f"   Cost Basis: ${total_cost:,.0f}")
        print(f"   Current Value: ${total_value:,.0f}")
        print(f"   Gain/Loss: ${total_gain_loss:,.0f} ({total_gain_loss_pct:+.1f}%)")
    
    # 3. Transaction history
    print(f"\n📋 Transaction History:")
    transactions = portfolio_db.get_transaction_history()
    
    if transactions:
        print(f"   Total transactions: {len(transactions)}")
        for tx in transactions[-5:]:  # Last 5 transactions
            print(f"   {tx['transaction_date'][:10]}: {tx['type']} {tx['shares']} {tx['symbol']} @ ${tx['price']:.2f}")
    
    # 4. Add to watchlist
    print(f"\n👀 Watchlist Management:")
    
    watchlist_stocks = [
        ('TSLA', 200.00, 'Waiting for price drop'),
        ('NVDA', 400.00, 'AI growth potential')
    ]
    
    for symbol, target_price, notes in watchlist_stocks:
        portfolio_db.add_to_watchlist(symbol, target_price, notes)
        print(f"   Added {symbol} to watchlist (target: ${target_price:.2f})")
    
    watchlist = portfolio_db.get_watchlist()
    for item in watchlist:
        print(f"   📌 {item['symbol']}: Target ${item['target_price']:.2f} - {item['notes']}")
    
    # 5. Database statistics
    print(f"\n📊 Database Statistics:")
    stats = portfolio_db.get_database_stats()
    
    for key, value in stats.items():
        if key == 'db_size_mb':
            print(f"   {key.replace('_', ' ').title()}: {value:.3f} MB")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # 6. Save portfolio snapshot
    portfolio_snapshot = {
        'total_value': total_value,
        'total_cost': total_cost, 
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_pct': total_gain_loss_pct,
        'average_score': 7.5,  # Mock score
        'symbol_count': len(portfolio),
        'sector_breakdown': {'Technology': 60, 'Healthcare': 20, 'Financials': 20},
        'top_performers': [{'symbol': 'MSFT', 'score': 8.5}]
    }
    
    portfolio_db.save_portfolio_snapshot(portfolio_snapshot)
    print(f"\n💾 Portfolio snapshot saved")
    
    print(f"\n🎉 Enhanced Portfolio Database Demo Complete!")
    print(f"✅ Features demonstrated:")
    print(f"   • SQLite data persistence")
    print(f"   • Transaction history tracking") 
    print(f"   • Portfolio valuation")
    print(f"   • Watchlist management")
    print(f"   • Performance snapshots")
    print(f"   • Database statistics")

if __name__ == "__main__":
    demo_enhanced_portfolio()
