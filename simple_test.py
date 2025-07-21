#!/usr/bin/env python3
"""
Simple test to verify the fixes work in isolation
"""

import warnings
import os

# Global warning suppression
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

print("Testing correlation calculation...")

# Create sample data
data = pd.DataFrame({
    'AAPL': [0.01, -0.005, 0.02, -0.01, 0.015],
    'MSFT': [0.008, -0.003, 0.018, -0.008, 0.012],
    'GOOGL': [0.012, -0.007, 0.025, -0.012, 0.018]
})

print(f"Data shape: {data.shape}")
print(f"Data:\n{data}")

try:
    # Test correlation
    corr_matrix = data.corr()
    print("\n✅ Correlation calculation successful!")
    print(f"Correlation matrix:\n{corr_matrix}")
except Exception as e:
    print(f"\n❌ Correlation calculation failed: {e}")

print("\nTest completed!")
