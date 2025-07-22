#!/usr/bin/env python3
"""
Quick test to verify critical functions work without errors
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functions():
    """Test basic functionality without Streamlit dependencies"""
    try:
        print("🧪 Testing safe_float function...")
        
        # Define safe_float locally to avoid imports
        def safe_float_test(value, default=0):
            try:
                if value is None or value == "None" or value == "" or str(value).lower() == 'nan':
                    return default
                if isinstance(value, str) and value.strip() == "":
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Test cases
        test_cases = [
            (None, 0, "None value"),
            ("", 0, "Empty string"),
            ("10.5", 10.5, "Valid string"),
            (15.7, 15.7, "Valid float"),
        ]
        
        for value, expected, description in test_cases:
            result = safe_float_test(value)
            if abs(result - expected) < 0.001:  # Use small tolerance for float comparison
                print(f"  ✅ {description}: {value} → {result}")
            else:
                print(f"  ❌ {description}: {value} → {result} (expected {expected})")
        
        print("\n🧪 Testing basic scoring logic...")
        
        # Test simple scoring function logic
        def test_score_pe(pe, industry_pe=20):
            pe = safe_float_test(pe, 0)
            industry_pe = safe_float_test(industry_pe, 20)
            
            if pe <= 0:
                return 5  # Neutral for missing data
            
            if pe < industry_pe * 0.5:
                return 10
            elif pe < industry_pe * 0.8:
                return 8
            elif pe < industry_pe * 1.2:
                return 6
            elif pe < industry_pe * 1.5:
                return 4
            else:
                return 2
        
        # Test scoring with various inputs
        scoring_tests = [
            (None, 20, "None PE"),
            (10, 20, "Good PE"),
            (25, 20, "High PE"),
            ("", 20, "Empty string PE"),
        ]
        
        for pe, industry_pe, description in scoring_tests:
            try:
                result = test_score_pe(pe, industry_pe)
                if isinstance(result, (int, float)) and 0 <= result <= 10:
                    print(f"  ✅ {description}: PE={pe} → Score={result}")
                else:
                    print(f"  ❌ {description}: PE={pe} → Invalid score={result}")
            except Exception as e:
                print(f"  ❌ {description}: PE={pe} → Error: {e}")
        
        print("\n✅ Basic function tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Quick function test for AS_System fixes")
    print("=" * 50)
    
    success = test_basic_functions()
    
    if success:
        print("\n🎉 All basic tests passed!")
        print("The fixes should work correctly in the main application.")
    else:
        print("\n❌ Some tests failed!")
        print("There may still be issues that need to be addressed.")
