#!/usr/bin/env python3

import subprocess
import sys
import os

def start_streamlit_app():
    """Start the Stock Analysis Hub application"""
    
    print("🚀 Starting Stock Analysis Hub with Enhanced Portfolio Management...")
    print("📊 Features: Purchase Price Tracking, P&L Analysis, Portfolio Alerts")
    print("🌐 Server will be available at: http://localhost:8501")
    print("")
    
    # Change to the correct directory
    app_dir = "/Users/magnushuusfeldt/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/AS_System"
    os.chdir(app_dir)
    
    # Command to run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "AS_MH_v6.py",
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        print("📱 Starting server...")
        # Start the Streamlit app
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    
    return True

if __name__ == "__main__":
    start_streamlit_app()
