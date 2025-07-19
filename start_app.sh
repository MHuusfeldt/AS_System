#!/bin/bash

echo "🚀 Starting Stock Analysis Hub with Enhanced Portfolio Management..."
echo "📊 Features: Purchase Price Tracking, P&L Analysis, Portfolio Alerts"
echo "🌐 Server will be available at: http://localhost:8501"
echo ""

# Change to the correct directory
cd /Users/magnushuusfeldt/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/AS_System

# Start Streamlit with specified port
streamlit run AS_MH_v6.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo "✅ Stock Analysis Hub started successfully!"
