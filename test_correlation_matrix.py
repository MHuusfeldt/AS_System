#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import AS_MH_v6
    import pandas as pd
    import numpy as np
    
    print("✅ Module imported successfully")
    
    # Test the AdvancedRiskAnalyzer correlation matrix
    analyzer = AS_MH_v6.AdvancedRiskAnalyzer()
    print("✅ AdvancedRiskAnalyzer created")
    
    # Create test returns data (similar to what would be calculated from price data)
    test_returns = pd.DataFrame({
        'NOVO': [0.01, -0.005, 0.02, -0.01, 0.015],
        'AAPL': [0.012, -0.008, 0.018, -0.012, 0.011]
    })
    
    print("✅ Test returns data created:")
    print(test_returns)
    
    # Test correlation matrix calculation
    print("\n🧪 Testing correlation matrix calculation...")
    correlation_matrix = analyzer.calculate_correlation_matrix(test_returns)
    
    print(f"✅ Correlation matrix calculated:")
    print(f"Shape: {correlation_matrix.shape}")
    print(f"Columns: {list(correlation_matrix.columns) if not correlation_matrix.empty else 'Empty'}")
    
    if not correlation_matrix.empty:
        print("Values:")
        print(correlation_matrix)
    
    print("\n🎉 Correlation matrix feature ready for NOVO scenario!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
