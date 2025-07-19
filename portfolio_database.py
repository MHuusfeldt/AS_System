"""
Portfolio Database Module - SQLite Implementation
================================================
Robust portfolio management with SQLite for data persistence,
transaction history, and portfolio analytics.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional, Tuple
import os

class PortfolioDatabase:
    def __init__(self, db_path: str = "portfolio.db"):
        """Initialize portfolio database"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Portfolio holdings table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    shares REAL DEFAULT 0,
                    average_cost REAL DEFAULT 0,
                    sector TEXT,
                    industry TEXT,
                    added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    UNIQUE(symbol)
                )
            ''')
            
            # Transaction history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL, -- 'BUY', 'SELL', 'DIVIDEND'
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    total_amount REAL NOT NULL,
                    transaction_date TEXT NOT NULL,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            ''')
            
            # Portfolio snapshots for performance tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT NOT NULL,
                    total_value REAL,
                    total_cost REAL,
                    total_gain_loss REAL,
                    total_gain_loss_pct REAL,
                    average_score REAL,
                    symbol_count INTEGER,
                    sector_breakdown TEXT, -- JSON string
                    top_performers TEXT,   -- JSON string
                    portfolio_data TEXT    -- JSON snapshot of full portfolio
                )
            ''')
            
            # Watchlist table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    added_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    target_price REAL,
                    notes TEXT
                )
            ''')
            
            # Analysis history for tracking stock evaluations
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    total_score REAL,
                    recommendation TEXT,
                    sector TEXT,
                    current_price REAL,
                    target_price REAL,
                    score_breakdown TEXT, -- JSON string of all scores
                    fundamental_data TEXT -- JSON string of key metrics
                )
            ''')
            
            conn.commit()
            print("✅ Portfolio database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def add_to_portfolio(self, symbol: str, shares: float = 0, cost_per_share: float = 0, 
                        sector: str = None, industry: str = None, notes: str = None) -> bool:
        """Add or update stock in portfolio"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if stock already exists
            existing = conn.execute(
                "SELECT shares, average_cost FROM portfolio_holdings WHERE symbol = ?", 
                (symbol,)
            ).fetchone()
            
            if existing:
                # Update existing holding
                if shares > 0 and cost_per_share > 0:
                    # Calculate new average cost
                    old_shares, old_cost = existing
                    total_shares = old_shares + shares
                    total_cost = (old_shares * old_cost) + (shares * cost_per_share)
                    new_avg_cost = total_cost / total_shares if total_shares > 0 else 0
                    
                    conn.execute('''
                        UPDATE portfolio_holdings 
                        SET shares = ?, average_cost = ?, last_updated = CURRENT_TIMESTAMP,
                            sector = COALESCE(?, sector), industry = COALESCE(?, industry),
                            notes = COALESCE(?, notes)
                        WHERE symbol = ?
                    ''', (total_shares, new_avg_cost, sector, industry, notes, symbol))
                    
                    # Record transaction
                    if shares > 0:
                        self._record_transaction(symbol, 'BUY', shares, cost_per_share, 
                                               shares * cost_per_share, conn=conn)
                else:
                    # Just update metadata
                    conn.execute('''
                        UPDATE portfolio_holdings 
                        SET last_updated = CURRENT_TIMESTAMP,
                            sector = COALESCE(?, sector), industry = COALESCE(?, industry),
                            notes = COALESCE(?, notes)
                        WHERE symbol = ?
                    ''', (sector, industry, notes, symbol))
            else:
                # Insert new holding
                conn.execute('''
                    INSERT INTO portfolio_holdings 
                    (symbol, shares, average_cost, sector, industry, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, shares, cost_per_share, sector, industry, notes))
                
                # Record transaction if shares were added
                if shares > 0 and cost_per_share > 0:
                    self._record_transaction(symbol, 'BUY', shares, cost_per_share, 
                                           shares * cost_per_share, conn=conn)
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error adding to portfolio: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def remove_from_portfolio(self, symbol: str, shares_to_sell: float = None) -> bool:
        """Remove stock from portfolio (partially or completely)"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get current holding
            holding = conn.execute(
                "SELECT shares, average_cost FROM portfolio_holdings WHERE symbol = ?", 
                (symbol,)
            ).fetchone()
            
            if not holding:
                return False
                
            current_shares, avg_cost = holding
            
            if shares_to_sell is None or shares_to_sell >= current_shares:
                # Remove completely
                conn.execute("DELETE FROM portfolio_holdings WHERE symbol = ?", (symbol,))
                if current_shares > 0:
                    self._record_transaction(symbol, 'SELL', current_shares, avg_cost, 
                                           current_shares * avg_cost, conn=conn)
            else:
                # Partial sale
                remaining_shares = current_shares - shares_to_sell
                conn.execute('''
                    UPDATE portfolio_holdings 
                    SET shares = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (remaining_shares, symbol))
                
                self._record_transaction(symbol, 'SELL', shares_to_sell, avg_cost, 
                                       shares_to_sell * avg_cost, conn=conn)
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error removing from portfolio: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_portfolio(self) -> List[Dict]:
        """Get current portfolio holdings"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT symbol, shares, average_cost, sector, industry, 
                       added_date, last_updated, notes
                FROM portfolio_holdings
                ORDER BY symbol
            ''')
            
            holdings = []
            for row in cursor.fetchall():
                holdings.append({
                    'symbol': row[0],
                    'shares': row[1],
                    'average_cost': row[2],
                    'sector': row[3],
                    'industry': row[4],
                    'added_date': row[5],
                    'last_updated': row[6],
                    'notes': row[7]
                })
            
            return holdings
            
        except Exception as e:
            print(f"❌ Error getting portfolio: {e}")
            return []
        finally:
            conn.close()
    
    def get_portfolio_symbols(self) -> List[str]:
        """Get list of symbols in portfolio"""
        portfolio = self.get_portfolio()
        return [holding['symbol'] for holding in portfolio]
    
    def _record_transaction(self, symbol: str, transaction_type: str, shares: float, 
                          price: float, total_amount: float, transaction_date: str = None,
                          notes: str = None, conn=None):
        """Record a transaction in the database"""
        if transaction_date is None:
            transaction_date = datetime.now().isoformat()
            
        close_conn = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT INTO transactions 
                (symbol, transaction_type, shares, price, total_amount, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, transaction_type, shares, price, total_amount, transaction_date, notes))
            
            if close_conn:
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error recording transaction: {e}")
            if close_conn:
                conn.rollback()
        finally:
            if close_conn:
                conn.close()
    
    def get_transaction_history(self, symbol: str = None, days: int = None) -> List[Dict]:
        """Get transaction history, optionally filtered by symbol and/or date range"""
        conn = sqlite3.connect(self.db_path)
        try:
            query = '''
                SELECT symbol, transaction_type, shares, price, total_amount, 
                       transaction_date, created_date, notes
                FROM transactions
            '''
            params = []
            conditions = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("transaction_date >= ?")
                params.append(cutoff_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY transaction_date DESC"
            
            cursor = conn.execute(query, params)
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append({
                    'symbol': row[0],
                    'type': row[1],
                    'shares': row[2],
                    'price': row[3],
                    'total_amount': row[4],
                    'transaction_date': row[5],
                    'created_date': row[6],
                    'notes': row[7]
                })
            
            return transactions
            
        except Exception as e:
            print(f"❌ Error getting transaction history: {e}")
            return []
        finally:
            conn.close()
    
    def save_portfolio_snapshot(self, portfolio_data: Dict) -> bool:
        """Save portfolio snapshot for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        try:
            snapshot_date = datetime.now().isoformat()
            
            conn.execute('''
                INSERT INTO portfolio_snapshots 
                (snapshot_date, total_value, total_cost, total_gain_loss, total_gain_loss_pct,
                 average_score, symbol_count, sector_breakdown, top_performers, portfolio_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_date,
                portfolio_data.get('total_value', 0),
                portfolio_data.get('total_cost', 0),
                portfolio_data.get('total_gain_loss', 0),
                portfolio_data.get('total_gain_loss_pct', 0),
                portfolio_data.get('average_score', 0),
                portfolio_data.get('symbol_count', 0),
                json.dumps(portfolio_data.get('sector_breakdown', {})),
                json.dumps(portfolio_data.get('top_performers', [])),
                json.dumps(portfolio_data)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error saving portfolio snapshot: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_portfolio_performance_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio performance history"""
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor = conn.execute('''
                SELECT snapshot_date, total_value, total_cost, total_gain_loss, 
                       total_gain_loss_pct, average_score, symbol_count
                FROM portfolio_snapshots
                WHERE snapshot_date >= ?
                ORDER BY snapshot_date ASC
            ''', (cutoff_date,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'date': row[0],
                    'total_value': row[1],
                    'total_cost': row[2],
                    'gain_loss': row[3],
                    'gain_loss_pct': row[4],
                    'average_score': row[5],
                    'symbol_count': row[6]
                })
            
            return history
            
        except Exception as e:
            print(f"❌ Error getting performance history: {e}")
            return []
        finally:
            conn.close()
    
    def add_to_watchlist(self, symbol: str, target_price: float = None, notes: str = None) -> bool:
        """Add stock to watchlist"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO watchlist (symbol, target_price, notes, added_date)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, target_price, notes))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error adding to watchlist: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_watchlist(self) -> List[Dict]:
        """Get watchlist"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT symbol, target_price, added_date, notes
                FROM watchlist
                ORDER BY added_date DESC
            ''')
            
            watchlist = []
            for row in cursor.fetchall():
                watchlist.append({
                    'symbol': row[0],
                    'target_price': row[1],
                    'added_date': row[2],
                    'notes': row[3]
                })
            
            return watchlist
            
        except Exception as e:
            print(f"❌ Error getting watchlist: {e}")
            return []
        finally:
            conn.close()
    
    def remove_from_watchlist(self, symbol: str) -> bool:
        """Remove stock from watchlist"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error removing from watchlist: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def save_analysis(self, symbol: str, analysis_data: Dict) -> bool:
        """Save stock analysis for historical tracking"""
        conn = sqlite3.connect(self.db_path)
        try:
            analysis_date = datetime.now().isoformat()
            
            conn.execute('''
                INSERT INTO analysis_history 
                (symbol, analysis_date, total_score, recommendation, sector, current_price,
                 target_price, score_breakdown, fundamental_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                analysis_date,
                analysis_data.get('total_score', 0),
                analysis_data.get('recommendation', ''),
                analysis_data.get('sector', ''),
                analysis_data.get('current_price', 0),
                analysis_data.get('target_price', 0),
                json.dumps(analysis_data.get('score_breakdown', {})),
                json.dumps(analysis_data.get('fundamental_data', {}))
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_analysis_history(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """Get analysis history for tracking stock evaluation changes"""
        conn = sqlite3.connect(self.db_path)
        try:
            query = '''
                SELECT symbol, analysis_date, total_score, recommendation, sector,
                       current_price, target_price, score_breakdown
                FROM analysis_history
            '''
            params = []
            conditions = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if days:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                conditions.append("analysis_date >= ?")
                params.append(cutoff_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY analysis_date DESC"
            
            cursor = conn.execute(query, params)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'symbol': row[0],
                    'analysis_date': row[1],
                    'total_score': row[2],
                    'recommendation': row[3],
                    'sector': row[4],
                    'current_price': row[5],
                    'target_price': row[6],
                    'score_breakdown': json.loads(row[7]) if row[7] else {}
                })
            
            return history
            
        except Exception as e:
            print(f"❌ Error getting analysis history: {e}")
            return []
        finally:
            conn.close()
    
    def migrate_from_json(self, json_file_path: str) -> bool:
        """Migrate existing portfolio from JSON format"""
        try:
            if not os.path.exists(json_file_path):
                print(f"📂 JSON file not found: {json_file_path}")
                return False
                
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Migrate portfolio symbols
            symbols = data.get('symbols', [])
            for symbol in symbols:
                self.add_to_portfolio(symbol)
            
            print(f"✅ Successfully migrated {len(symbols)} symbols from JSON")
            
            # Create backup of original file
            backup_path = f"{json_file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(json_file_path, backup_path)
            print(f"📁 Original JSON backed up to: {backup_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error migrating from JSON: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        try:
            stats = {}
            
            # Portfolio holdings count
            stats['portfolio_holdings'] = conn.execute(
                "SELECT COUNT(*) FROM portfolio_holdings"
            ).fetchone()[0]
            
            # Transactions count
            stats['total_transactions'] = conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]
            
            # Snapshots count
            stats['portfolio_snapshots'] = conn.execute(
                "SELECT COUNT(*) FROM portfolio_snapshots"
            ).fetchone()[0]
            
            # Watchlist count
            stats['watchlist_items'] = conn.execute(
                "SELECT COUNT(*) FROM watchlist"
            ).fetchone()[0]
            
            # Analysis history count
            stats['analysis_records'] = conn.execute(
                "SELECT COUNT(*) FROM analysis_history"
            ).fetchone()[0]
            
            # Database file size
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
        finally:
            conn.close()
    
    # Additional methods needed for the main application integration
    def get_current_holdings(self) -> pd.DataFrame:
        """Get current portfolio holdings as DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get portfolio holdings
            holdings_df = pd.read_sql_query("""
                SELECT symbol, shares as quantity, average_cost, sector, industry, 
                       added_date as date_added, last_updated
                FROM portfolio_holdings 
                WHERE shares > 0
                ORDER BY symbol
            """, conn)
            
            return holdings_df
            
        except Exception as e:
            print(f"❌ Error getting current holdings: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def add_holding(self, symbol: str, quantity: float, price: float, notes: str = None) -> bool:
        """Add a new holding (wrapper for add_to_portfolio)"""
        return self.add_to_portfolio(symbol, quantity, price, notes=notes)
    
    def remove_holding(self, symbol: str) -> bool:
        """Remove a holding completely (wrapper for remove_from_portfolio)"""
        return self.remove_from_portfolio(symbol, shares_to_sell=None)  # Remove all shares
    
    def record_transaction(self, symbol: str, transaction_type: str, quantity: float, 
                          price: float, notes: str = None) -> bool:
        """Record a transaction (wrapper for _record_transaction)"""
        return self._record_transaction(symbol, transaction_type, quantity, price, notes)
    
    def get_portfolio_snapshots(self, limit: int = 30) -> pd.DataFrame:
        """Get portfolio snapshots as DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            snapshots_df = pd.read_sql_query("""
                SELECT snapshot_date, total_value, total_cost, total_gain_loss as unrealized_pnl, 
                       symbol_count as number_of_holdings, 'Portfolio snapshot' as performance_notes
                FROM portfolio_snapshots 
                ORDER BY snapshot_date DESC
                LIMIT ?
            """, conn, params=(limit,))
            
            return snapshots_df
            
        except Exception as e:
            print(f"❌ Error getting portfolio snapshots: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

# Global database instance
portfolio_db = PortfolioDatabase()
