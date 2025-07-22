#!/usr/bin/env python3
"""
Test script to verify None handling fixes in scoring functions
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions we need to test
try:
    from AS_MH_v6 import (
        safe_float, safe_comparison,
        score_pe, score_forward_pe, score_peg, score_pb, score_roe,
        score_eps_growth, score_revenue_growth, score_debt_equity,
        score_dividend_yield, score_gross_margin, score_ev_ebitda,
        score_price_sales, score_analyst_upside
    )
    print("✅ Successfully imported all scoring functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_safe_float():
    """Test safe_float function with various None cases"""
    print("\n🧪 Testing safe_float function:")
    
    test_cases = [
        (None, 0, "None value"),
        ("None", 0, "String 'None'"),
        ("", 0, "Empty string"),
        ("nan", 0, "String 'nan'"),
        (float('nan'), 0, "NaN value"),
        ("10.5", 10.5, "Valid string number"),
        (15.7, 15.7, "Valid float"),
        ([], 0, "Invalid type (list)"),
    ]
    
    for value, expected, description in test_cases:
        try:
            result = safe_float(value)
            if result == expected or (expected == 0 and result == 0):
                print(f"  ✅ {description}: {value} → {result}")
            else:
                print(f"  ❌ {description}: {value} → {result} (expected {expected})")
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")

def test_safe_comparison():
    """Test safe_comparison function"""
    print("\n🧪 Testing safe_comparison function:")
    
    test_cases = [
        (None, 10, "<", False, "None < 10"),
        (5, None, "<", True, "5 < None"),
        (None, None, "<", False, "None < None"),
        (5, 10, "<", True, "5 < 10"),
        (15, 10, "<", False, "15 < 10"),
    ]
    
    for val1, val2, op, expected, description in test_cases:
        try:
            result = safe_comparison(val1, val2, op)
            if result == expected:
                print(f"  ✅ {description}: {result}")
            else:
                print(f"  ❌ {description}: {result} (expected {expected})")
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")

def test_scoring_functions():
    """Test all scoring functions with None values"""
    print("\n🧪 Testing scoring functions with None values:")
    
    scoring_tests = [
        (score_pe, [None, 15], "score_pe with None"),
        (score_forward_pe, [None, 15], "score_forward_pe with None"),
        (score_peg, [None], "score_peg with None"),
        (score_pb, [None], "score_pb with None"),
        (score_roe, [None], "score_roe with None"),
        (score_eps_growth, [None], "score_eps_growth with None"),
        (score_revenue_growth, [None], "score_revenue_growth with None"),
        (score_debt_equity, [None], "score_debt_equity with None"),
        (score_dividend_yield, [None], "score_dividend_yield with None"),
        (score_gross_margin, [None], "score_gross_margin with None"),
        (score_ev_ebitda, [None], "score_ev_ebitda with None"),
        (score_price_sales, [None], "score_price_sales with None"),
        (score_analyst_upside, [None], "score_analyst_upside with None"),
    ]
    
    for func, args, description in scoring_tests:
        try:
            result = func(*args)
            if isinstance(result, (int, float)) and result >= 0:
                print(f"  ✅ {description}: {result}")
            else:
                print(f"  ❌ {description}: Invalid result {result}")
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")

def test_comparison_operations():
    """Test that None values don't cause comparison errors"""
    print("\n🧪 Testing None value comparisons:")
    
    test_values = [None, "None", "", float('nan'), 0, 10.5, -5]
    
    for val in test_values:
        try:
            # Test safe conversion
            safe_val = safe_float(val, 0)
            
            # Test comparisons that used to fail
            result1 = safe_val < 10
            result2 = safe_val > 5
            result3 = safe_val <= 0
            
            print(f"  ✅ {repr(val)} → {safe_val}: < 10 = {result1}, > 5 = {result2}, <= 0 = {result3}")
            
        except Exception as e:
            print(f"  ❌ {repr(val)}: Error - {e}")

if __name__ == "__main__":
    print("🔧 Testing None handling fixes for scoring functions")
    print("=" * 60)
    
    test_safe_float()
    test_safe_comparison()
    test_scoring_functions()
    test_comparison_operations()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\nIf you see this message without errors, the None handling fixes are working correctly.")
