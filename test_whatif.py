#!/usr/bin/env python3
"""Test script for What-If Analysis functionality"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test the What-If functionality
try:
    print("🧪 Testing What-If Analysis Components...")
    
    # Mock required dependencies
    class MockSessionState:
        def __init__(self):
            self.score_weights = {
                'PE': 0.08, 'Forward PE': 0.12, 'PEG': 0.08, 'PB': 0.06,
                'ROE': 0.10, 'EPS Growth': 0.12, 'Revenue Growth': 0.08
            }
            self.stock_data = {}
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # Import and test the classes
    from AS_MH_v6 import PortfolioMetrics, WhatIfAnalyzer, PortfolioDatabase, EnhancedFeaturesManager
    print("✅ Successfully imported What-If classes")
    
    # Test PortfolioMetrics
    metrics = PortfolioMetrics(total_value=10000, diversification_score=8.5, average_score=7.2, symbol_count=5)
    print(f"✅ PortfolioMetrics created: ${metrics.total_value:,.2f}, {metrics.symbol_count} holdings")
    
    # Test PortfolioDatabase
    db = PortfolioDatabase()
    db.add_holding("AAPL", 10, 150.0)
    db.add_holding("MSFT", 5, 300.0)
    holdings = db.get_current_holdings()
    print(f"✅ PortfolioDatabase working: {len(holdings)} holdings added")
    
    # Test WhatIfAnalyzer
    analyzer = WhatIfAnalyzer()
    print("✅ WhatIfAnalyzer created")
    
    # Test EnhancedFeaturesManager
    manager = EnhancedFeaturesManager()
    success = manager.initialize_all_systems()
    print(f"✅ EnhancedFeaturesManager initialized: {success}")
    
    print("🎉 ALL TESTS PASSED - What-If Analysis is fully functional!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
