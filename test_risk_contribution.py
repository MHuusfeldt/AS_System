#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import AS_MH_v6
    import pandas as pd
    
    print("✅ Module imported successfully")
    
    # Test the AdvancedRiskAnalyzer with NOVO scenario data
    analyzer = AS_MH_v6.AdvancedRiskAnalyzer()
    print("✅ AdvancedRiskAnalyzer created")
    
    # Create test scenario data (similar to what NOVO scenario would create)
    test_portfolio_data = pd.DataFrame([
        {'Symbol': 'NOVO', 'Quantity': 100, 'Current Price': 120.0, 'Total Value': 12000.0},
        {'Symbol': 'AAPL', 'Quantity': 10, 'Current Price': 150.0, 'Total Value': 1500.0}
    ])
    
    print("✅ Test portfolio data created:")
    print(test_portfolio_data)
    
    # Test risk contribution calculation directly
    print("\n🧪 Testing risk contribution calculation...")
    holdings_data = test_portfolio_data.to_dict('records')  # This is the format passed to risk analysis
    
    risk_contribution = analyzer.calculate_risk_contribution(pd.DataFrame(), holdings_data)
    print(f"✅ Risk contribution calculated: {risk_contribution}")
    
    print("\n🎉 Risk contribution feature ready for NOVO scenario!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
