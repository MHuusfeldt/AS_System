# app/portfolio.py
import streamlit as st
import json
import os
from datetime import datetime
import sqlite3

# --- Database Management ---
DB_PATH = 'portfolio.db'

def init_db(db_path=DB_PATH):
    """Initialize the SQLite database."""
    conn = sqlite3.connect(db_path)
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

class PortfolioManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def get_portfolio(self):
        """Load portfolio symbols from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM portfolio ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            return symbols
        except sqlite3.Error:
            return []

    def add_stock(self, symbol):
        """Add a stock to the portfolio."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO portfolio (symbol) VALUES (?)", (symbol,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0

    def remove_stock(self, symbol):
        """Remove a stock from the portfolio."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0

    def update_portfolio(self, symbols):
        """Overwrite the portfolio with a new list of symbols."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio")
        cursor.executemany("INSERT INTO portfolio (symbol) VALUES (?)", [(s,) for s in symbols])
        conn.commit()
        conn.close()

    def save_portfolio_to_json(self):
        """Save the current DB portfolio to the JSON config file."""
        symbols = self.get_portfolio()
        return save_portfolio_to_file(symbols)

    def get_portfolio_analysis(self):
        """Fetch and analyze all stocks in the portfolio."""
        # Import locally to prevent circular dependencies
        from app.data_fetcher import StockDataFetcher
        from app.scoring import ScoreCalculator

        symbols = self.get_portfolio()
        if not symbols:
            return []

        calculator = ScoreCalculator()
        
        all_data = []
        for symbol in symbols:
            fetcher = StockDataFetcher(symbol)
            success = fetcher.fetch_all_data()
            if success and fetcher.info:
                scores, final_score = calculator.calculate_total_score(fetcher.info)
                all_data.append({
                    "symbol": symbol,
                    "info": fetcher.info,
                    "scores": scores,
                    "score": final_score
                })
        return all_data

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
