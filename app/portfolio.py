# app/portfolio.py
import streamlit as st
import json
import os
from datetime import datetime
import sqlite3

# --- Database Management ---
DB_PATH = 'portfolio.db'

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL UNIQUE,
            shares REAL,
            purchase_price REAL
        )
    ''')
    conn.commit()
    conn.close()

def load_portfolio_from_db():
    """Load portfolio from the database."""
    if not os.path.exists(DB_PATH):
        init_db()
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM portfolio")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def save_portfolio_to_db(symbols):
    """Save the entire portfolio to the database, overwriting the old one."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio") # Clear existing portfolio
    for symbol in symbols:
        cursor.execute("INSERT INTO portfolio (symbol) VALUES (?)", (symbol,))
    conn.commit()
    conn.close()

# --- Portfolio Sync (JSON for automated monitor) ---
def save_portfolio_to_file(symbols):
    """Save portfolio to JSON file for the automated monitoring script."""
    portfolio_data = {
        'symbols': symbols,
        'last_updated': datetime.now().isoformat(),
        'total_stocks': len(symbols)
    }
    try:
        with open('portfolio_config.json', 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio file: {e}")
        return False

def sync_portfolio():
    """Syncs the DB portfolio to the JSON file."""
    symbols = load_portfolio_from_db()
    if save_portfolio_to_file(symbols):
        st.success("✅ Portfolio synced with automated monitor!")
    else:
        st.error("❌ Portfolio sync failed.")

# --- Portfolio Analysis ---
def analyze_portfolio_optimized(portfolio_symbols):
    """Optimized portfolio analysis using batch processing."""
    # This would now use the async data fetcher
    # from app.data_fetcher import get_batch_yahoo_info
    # ... implementation ...
    pass

def display_what_if_analysis(current_portfolio, symbol_to_add=None, symbol_to_remove=None):
    """Shows a 'what-if' analysis for portfolio changes."""
    st.subheader("🤔 What-If Analysis")
    
    temp_portfolio = set(current_portfolio)
    action = ""

    if symbol_to_add:
        temp_portfolio.add(symbol_to_add)
        action = f"Adding {symbol_to_add}"
    elif symbol_to_remove and symbol_to_remove in temp_portfolio:
        temp_portfolio.remove(symbol_to_remove)
        action = f"Removing {symbol_to_remove}"
    else:
        st.info("Hover over an action to see its impact.")
        return

    st.write(f"**Action:** {action}")
    st.write(f"**New Portfolio Size:** {len(temp_portfolio)}")
    # Here you would calculate and display the hypothetical new
    # average score, diversification, etc.
    st.warning("This is a preview. Changes are not saved yet.")
