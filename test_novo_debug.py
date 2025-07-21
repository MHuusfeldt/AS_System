#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import AS_MH_v6
    print("✅ AS_MH_v6 imported successfully")
    
    import what_if_analysis
    print("✅ what_if_analysis imported successfully")
    
    # Test NOVO price fetching
    price = AS_MH_v6.get_simple_current_price('NOVO')
    print(f"✅ NOVO price: {price}")
    
    # Test creating analyzer
    analyzer = what_if_analysis.WhatIfAnalyzer(None)
    print("✅ WhatIfAnalyzer created")
    
    # Test scenario portfolio creation
    test_scenario = [{'symbol': 'NOVO', 'quantity': 100}]
    print("✅ Test scenario created")
    
    print("\n🎉 All tests completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
