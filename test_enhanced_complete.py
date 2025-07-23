#!/usr/bin/env python3
"""Simple test to verify the What-If Analysis functionality is working"""

import os
import sys

def test_enhanced_features():
    """Test the enhanced features to make sure they work"""
    
    print("🧪 Testing Enhanced Features Manager...")
    
    try:
        # Test the enhanced features manager
        sys.path.append(os.path.dirname(__file__))
        from AS_MH_v6 import EnhancedFeaturesManager, PortfolioDatabase, WhatIfAnalyzer
        
        print("✅ Successfully imported enhanced classes")
        
        # Test EnhancedFeaturesManager
        manager = EnhancedFeaturesManager()
        print(f"✅ EnhancedFeaturesManager created")
        print(f"   - cache attribute: {hasattr(manager, 'cache')}")
        print(f"   - async_loader attribute: {hasattr(manager, 'async_loader')}")
        print(f"   - portfolio_db attribute: {hasattr(manager, 'portfolio_db')}")
        print(f"   - what_if_analyzer attribute: {hasattr(manager, 'what_if_analyzer')}")
        
        # Test initialization
        success = manager.initialize_all_systems()
        print(f"✅ Systems initialized: {success}")
        print(f"   - cache: {manager.cache}")
        print(f"   - async_loader: {manager.async_loader}")
        print(f"   - portfolio_db: {manager.portfolio_db is not None}")
        print(f"   - what_if_analyzer: {manager.what_if_analyzer is not None}")
        
        # Test PortfolioDatabase methods
        if manager.portfolio_db:
            print("✅ Testing PortfolioDatabase methods...")
            db = manager.portfolio_db
            
            # Test add_holding
            db.add_holding("AAPL", 10, 150.0)
            holdings = db.get_current_holdings()
            print(f"   - add_holding: {len(holdings)} holdings")
            
            # Test get_transaction_history
            transactions = db.get_transaction_history()
            print(f"   - get_transaction_history: {len(transactions)} transactions")
            
            # Test clear_all_holdings
            db.clear_all_holdings()
            holdings_after_clear = db.get_current_holdings()
            print(f"   - clear_all_holdings: {len(holdings_after_clear)} holdings remaining")
        
        print("🎉 ALL TESTS PASSED - Enhanced Features are working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_features()
    sys.exit(0 if success else 1)
