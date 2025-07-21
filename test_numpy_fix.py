#!/usr/bin/env python3
"""
Test script to verify NumPy compatibility fixes in the risk analysis system
"""

import warnings
import os
import sys

# Set up comprehensive warning suppression
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=Warning)

try:
    import pandas as pd
    import numpy as np
    from AS_MH_v6 import AdvancedRiskAnalyzer
    
    print("Testing NumPy compatibility fixes...")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Create test data
    test_portfolio = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Total Value': [1000, 1500, 800]
    })
    
    print(f"\nTest portfolio: {test_portfolio['Symbol'].tolist()}")
    
    # Initialize risk analyzer
    print("\nInitializing AdvancedRiskAnalyzer...")
    risk_analyzer = AdvancedRiskAnalyzer()
    
    # Test correlation matrix calculation
    print("\nTesting correlation matrix calculation...")
    
    # Create sample returns data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'AAPL': np.random.randn(100) * 0.02,
        'MSFT': np.random.randn(100) * 0.015,
        'GOOGL': np.random.randn(100) * 0.025
    }, index=dates)
    
    print(f"Sample returns data shape: {returns_data.shape}")
    
    # Test the correlation matrix calculation
    try:
        correlation_matrix = risk_analyzer.calculate_correlation_matrix(returns_data)
        if correlation_matrix.empty:
            print("❌ Correlation matrix calculation returned empty DataFrame")
        else:
            print("✅ Correlation matrix calculation successful!")
            print(f"Matrix shape: {correlation_matrix.shape}")
            print(f"Matrix columns: {list(correlation_matrix.columns)}")
    except Exception as e:
        print(f"❌ Correlation matrix calculation failed: {e}")
    
    # Test comprehensive risk metrics
    print("\nTesting comprehensive risk metrics calculation...")
    try:
        risk_metrics = risk_analyzer.calculate_comprehensive_risk_metrics(test_portfolio)
        if risk_metrics:
            print("✅ Risk metrics calculation successful!")
            print(f"Available metrics: {list(risk_metrics.keys())}")
        else:
            print("❌ Risk metrics calculation returned empty results")
    except Exception as e:
        print(f"❌ Risk metrics calculation failed: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print("If you see ✅ for both tests, the NumPy compatibility issues are resolved!")
    print("If you see ❌, there may still be compatibility issues to address.")
    print("="*60)
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed.")
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
