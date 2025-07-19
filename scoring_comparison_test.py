#!/usr/bin/env python3
"""
Stock Scoring System Comparison Test
====================================
This script compares the enhanced scoring system with dynamic benchmarking
against the old static scoring system to demonstrate improvements.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OLD SCORING SYSTEM (Static Thresholds)
# ============================================================================

def score_roe_old(roe):
    """Original static ROE scoring"""
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

def score_gross_margin_old(gm):
    """Original static gross margin scoring"""
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

def score_revenue_growth_old(growth):
    """Original static revenue growth scoring"""
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

def score_debt_equity_old(de):
    """Original static debt/equity scoring"""
    if de < 0:
        return 5
    if de < 30:
        return 10
    elif de < 50:
        return 8
    elif de < 100:
        return 6
    elif de < 200:
        return 4
    else:
        return 2

# ============================================================================
# NEW SCORING SYSTEM (Dynamic Benchmarking)
# ============================================================================

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
    "Unknown": {
        "roe": 15.0, "gross_margin": 45.0, "revenue_growth": 7.0, "debt_equity": 50.0,
        "current_ratio": 1.8, "interest_coverage": 12.0
    }
}

def get_industry_benchmarks(sector):
    """Get industry benchmark values for dynamic scoring"""
    return INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS["Unknown"])

def score_roe_new(roe, sector):
    """Enhanced ROE scoring with dynamic benchmarking"""
    if roe <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_roe = benchmarks["roe"]
    
    # Calculate relative performance
    relative_roe = roe / industry_roe
    
    # Score based on relative performance
    if relative_roe >= 2.0:
        return 10
    elif relative_roe >= 1.5:
        return 8
    elif relative_roe >= 1.2:
        return 7
    elif relative_roe >= 1.0:
        return 6
    elif relative_roe >= 0.8:
        return 5
    elif relative_roe >= 0.6:
        return 3
    elif relative_roe >= 0.4:
        return 2
    else:
        return 1

def score_gross_margin_new(gm, sector):
    """Enhanced gross margin scoring with dynamic benchmarking"""
    if gm <= 0:
        return 0
    
    benchmarks = get_industry_benchmarks(sector)
    industry_gm = benchmarks["gross_margin"]
    
    relative_gm = gm / industry_gm
    
    if relative_gm >= 1.4:
        return 10
    elif relative_gm >= 1.2:
        return 8
    elif relative_gm >= 1.1:
        return 7
    elif relative_gm >= 0.95:
        return 6
    elif relative_gm >= 0.85:
        return 5
    elif relative_gm >= 0.75:
        return 3
    else:
        return 2

def score_revenue_growth_new(growth, sector):
    """Enhanced revenue growth scoring with dynamic benchmarking"""
    benchmarks = get_industry_benchmarks(sector)
    industry_growth = benchmarks["revenue_growth"]
    
    if growth is None or growth == 0:
        return 5
    
    # For negative growth, score more harshly
    if growth < 0:
        return max(0, min(4, 4 + (growth / 5)))
    
    # Calculate relative growth performance
    if industry_growth > 0:
        relative_growth = growth / industry_growth
        if relative_growth >= 2.0:
            return 10
        elif relative_growth >= 1.5:
            return 8
        elif relative_growth >= 1.2:
            return 7
        elif relative_growth >= 1.0:
            return 6
        elif relative_growth >= 0.8:
            return 5
        elif relative_growth >= 0.6:
            return 3
        else:
            return 2
    else:
        # Fallback to absolute scoring if industry growth is 0 or negative
        return score_revenue_growth_old(growth)

def score_debt_equity_new(de, sector):
    """Enhanced debt/equity scoring with dynamic benchmarking"""
    if de < 0:
        return 5
    
    benchmarks = get_industry_benchmarks(sector)
    industry_de = benchmarks["debt_equity"]
    
    # Financial sector has different debt characteristics
    if sector == "Financials":
        if de < industry_de * 0.8:
            return 8
        elif de < industry_de * 1.0:
            return 7
        elif de < industry_de * 1.2:
            return 6
        elif de < industry_de * 1.5:
            return 4
        else:
            return 2
    else:
        # For non-financial sectors, lower debt is better
        relative_de = de / industry_de
        if relative_de <= 0.5:
            return 10
        elif relative_de <= 0.8:
            return 8
        elif relative_de <= 1.0:
            return 6
        elif relative_de <= 1.5:
            return 4
        elif relative_de <= 2.0:
            return 2
        else:
            return 1

# ============================================================================
# TEST STOCKS FROM DIFFERENT SECTORS
# ============================================================================

TEST_STOCKS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
    "Financials": ["JPM", "BAC", "WFC", "GS"],
    "Industrials": ["BA", "CAT", "GE", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "EOG"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD"]
}

def safe_float(value, default=0):
    """Safely convert value to float"""
    try:
        if value is None or value == "None" or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def get_stock_data(symbol):
    """Fetch stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key metrics
        data = {
            'symbol': symbol,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'roe': safe_float(info.get('returnOnEquity', 0)) * 100 if info.get('returnOnEquity') else 0,
            'gross_margin': safe_float(info.get('grossMargins', 0)) * 100 if info.get('grossMargins') else 0,
            'revenue_growth': safe_float(info.get('revenueGrowth', 0)) * 100 if info.get('revenueGrowth') else 0,
            'debt_equity': safe_float(info.get('debtToEquity', 0)),
            'current_price': safe_float(info.get('currentPrice', 0))
        }
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def compare_scoring_systems():
    """Compare old vs new scoring systems across different sectors"""
    
    print("=" * 80)
    print("STOCK SCORING SYSTEM COMPARISON")
    print("=" * 80)
    print("Testing Enhanced Dynamic Benchmarking vs Old Static Scoring")
    print("=" * 80)
    
    all_results = []
    
    for sector, symbols in TEST_STOCKS.items():
        print(f"\n🏭 {sector.upper()} SECTOR")
        print("-" * 50)
        
        sector_results = []
        
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            data = get_stock_data(symbol)
            if not data:
                continue
                
            # Calculate old scores (static)
            old_roe = score_roe_old(data['roe'])
            old_gm = score_gross_margin_old(data['gross_margin'])
            old_rg = score_revenue_growth_old(data['revenue_growth'])
            old_de = score_debt_equity_old(data['debt_equity'])
            old_avg = (old_roe + old_gm + old_rg + old_de) / 4
            
            # Calculate new scores (dynamic)
            new_roe = score_roe_new(data['roe'], data['sector'])
            new_gm = score_gross_margin_new(data['gross_margin'], data['sector'])
            new_rg = score_revenue_growth_new(data['revenue_growth'], data['sector'])
            new_de = score_debt_equity_new(data['debt_equity'], data['sector'])
            new_avg = (new_roe + new_gm + new_rg + new_de) / 4
            
            # Calculate improvements
            improvement = new_avg - old_avg
            improvement_pct = (improvement / old_avg * 100) if old_avg > 0 else 0
            
            result = {
                'Symbol': symbol,
                'Sector': data['sector'],
                'ROE': f"{data['roe']:.1f}%",
                'Old ROE Score': old_roe,
                'New ROE Score': new_roe,
                'GM': f"{data['gross_margin']:.1f}%",
                'Old GM Score': old_gm,
                'New GM Score': new_gm,
                'RG': f"{data['revenue_growth']:.1f}%",
                'Old RG Score': old_rg,
                'New RG Score': new_rg,
                'DE': f"{data['debt_equity']:.1f}",
                'Old DE Score': old_de,
                'New DE Score': new_de,
                'Old Avg': f"{old_avg:.1f}",
                'New Avg': f"{new_avg:.1f}",
                'Improvement': f"{improvement:+.1f}",
                'Improvement %': f"{improvement_pct:+.1f}%"
            }
            
            sector_results.append(result)
            all_results.append(result)
            
            # Print individual result
            print(f"  ROE: {data['roe']:.1f}% | Old: {old_roe} → New: {new_roe} | Δ: {new_roe-old_roe:+}")
            print(f"  GM:  {data['gross_margin']:.1f}% | Old: {old_gm} → New: {new_gm} | Δ: {new_gm-old_gm:+}")
            print(f"  RG:  {data['revenue_growth']:.1f}% | Old: {old_rg} → New: {new_rg} | Δ: {new_rg-old_rg:+}")
            print(f"  DE:  {data['debt_equity']:.1f} | Old: {old_de} → New: {new_de} | Δ: {new_de-old_de:+}")
            print(f"  📈 Average: {old_avg:.1f} → {new_avg:.1f} | Improvement: {improvement:+.1f} ({improvement_pct:+.1f}%)")
        
        # Sector summary
        if sector_results:
            avg_improvement = np.mean([float(r['Improvement'].replace('+', '')) for r in sector_results])
            print(f"\n🎯 {sector} Sector Average Improvement: {avg_improvement:+.1f} points")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("📊 OVERALL COMPARISON SUMMARY")
    print("=" * 80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n📈 Stocks Analyzed: {len(all_results)}")
        
        improvements = [float(r['Improvement'].replace('+', '')) for r in all_results]
        positive_improvements = [x for x in improvements if x > 0]
        negative_improvements = [x for x in improvements if x < 0]
        
        print(f"✅ Stocks with Better Scores: {len(positive_improvements)} ({len(positive_improvements)/len(all_results)*100:.1f}%)")
        print(f"❌ Stocks with Worse Scores: {len(negative_improvements)} ({len(negative_improvements)/len(all_results)*100:.1f}%)")
        print(f"📊 Average Improvement: {np.mean(improvements):+.2f} points")
        print(f"📊 Maximum Improvement: {max(improvements):+.1f} points")
        print(f"📊 Minimum Change: {min(improvements):+.1f} points")
        
        # Show top improvements and declines
        df['Improvement_Value'] = df['Improvement'].str.replace('+', '').astype(float)
        
        print(f"\n🚀 TOP 5 IMPROVEMENTS:")
        top_improvements = df.nlargest(5, 'Improvement_Value')[['Symbol', 'Sector', 'Old Avg', 'New Avg', 'Improvement']]
        for _, row in top_improvements.iterrows():
            print(f"  {row['Symbol']} ({row['Sector'][:15]}): {row['Old Avg']} → {row['New Avg']} ({row['Improvement']})")
        
        if len(negative_improvements) > 0:
            print(f"\n📉 LARGEST DECLINES:")
            top_declines = df.nsmallest(5, 'Improvement_Value')[['Symbol', 'Sector', 'Old Avg', 'New Avg', 'Improvement']]
            for _, row in top_declines.iterrows():
                print(f"  {row['Symbol']} ({row['Sector'][:15]}): {row['Old Avg']} → {row['New Avg']} ({row['Improvement']})")
        
        # Export detailed results
        output_file = f"scoring_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("🎯 KEY BENEFITS OF ENHANCED SYSTEM:")
    print("=" * 80)
    print("✅ Sector-Relative Scoring: Compares companies to industry peers")
    print("✅ Contextual Evaluation: Tech companies scored against tech benchmarks")
    print("✅ Fairer Comparisons: Energy companies not penalized for lower margins")
    print("✅ Dynamic Adaptation: Scoring adapts to industry characteristics")
    print("✅ Momentum Integration: Technical analysis complements fundamentals")
    print("✅ Financial Health: Comprehensive risk assessment")
    print("=" * 80)

if __name__ == "__main__":
    compare_scoring_systems()
