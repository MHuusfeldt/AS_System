#!/usr/bin/env python3
"""
Enhanced automated portfolio monitoring script with exact scoring system from AS_MH_v6.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os
import json
import sys
import time

# Portfolio loading function
def load_portfolio_from_file():
    """Load portfolio from JSON file created by AS_MH_v6"""
    try:
        if os.path.exists('portfolio_config.json'):
            with open('portfolio_config.json', 'r') as f:
                portfolio_data = json.load(f)
            
            symbols = portfolio_data.get('symbols', [])
            last_updated = portfolio_data.get('last_updated', 'Unknown')
            
            print(f"✅ Loaded portfolio from file: {len(symbols)} stocks")
            print(f"📅 Last updated: {last_updated}")
            print(f"📊 Stocks: {', '.join(symbols)}")
            
            return symbols
        else:
            print("⚠️ No portfolio file found. Using default symbols.")
            return None
    except Exception as e:
        print(f"❌ Error loading portfolio: {e}")
        return None

# Import your exact scoring weights and methodology
DEFAULT_SCORE_WEIGHTS = {
    "PE": 0.10,
    "Forward PE": 0.15,
    "PEG": 0.10,
    "PB": 0.08,
    "EV/EBITDA": 0.12,
    "ROE": 0.12,
    "EPS Growth": 0.15,
    "Revenue Growth": 0.10,
    "FCF Trend": 0.05,
    "Debt/Equity": 0.04,
    "Dividend Yield": 0.03,
    "Gross Margin": 0.03,
    "Price/Sales": 0.05,
    "Analyst Upside": 0.08
}

# Your exact Danish stock mappings
DANISH_STOCKS = {
    "NOVO-B": "NOVO-B.CO",
    "NOVO": "NOVO-B.CO",
    "MAERSK-B": "MAERSK-B.CO",
    "MAERSK": "MAERSK-B.CO",
    "ORSTED": "ORSTED.CO",
    "DSV": "DSV.CO",
    "CARLB": "CARL-B.CO",
    "CARL-B": "CARL-B.CO",
    # ... (include all your Danish stocks)
}

# Your exact industry PE mapping
INDUSTRY_PE_MAP = {
    "Technology": 28,
    "Consumer Discretionary": 22,
    "Consumer Staples": 18,
    "Health Care": 25,
    "Financials": 12,
    "Energy": 15,
    "Materials": 16,
    "Industrials": 18,
    "Utilities": 20,
    "Real Estate": 25,
    "Communication Services": 22,
    "Consumer Cyclical": 22,
    "Healthcare": 25,
    "Financial Services": 12,
    "Basic Materials": 16,
    "Unknown": 20
}

# Your exact sector scoring models
SECTOR_SCORING_MODELS = {
    "Technology": {
        "weight_adjustments": {
            "Forward PE": 1.2,
            "Revenue Growth": 1.4,
            "Gross Margin": 1.3,
            "Price/Sales": 1.2,
            "EPS Growth": 1.3
        }
    },
    "Industrials": {
        "weight_adjustments": {
            "EV/EBITDA": 1.3,
            "ROE": 1.2,
            "Debt/Equity": 1.1,
            "FCF Trend": 1.4,
            "Revenue Growth": 1.1
        }
    },
    # ... (include all your sector models)
}

def safe_float(value, default=0):
    """Your exact safe_float function"""
    try:
        if value is None or value == "None" or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def score_pe(pe, industry_pe):
    """Your exact P/E scoring function"""
    if pe <= 0:
        return 0
    
    relative_pe = pe / industry_pe if industry_pe > 0 else 1
    
    if relative_pe < 0.6:
        return 10
    elif relative_pe < 0.8:
        return 8
    elif relative_pe < 1.0:
        return 6
    elif relative_pe < 1.2:
        return 4
    elif relative_pe < 1.5:
        return 3
    elif relative_pe < 2.0:
        return 2
    elif relative_pe < 2.5:
        return 1
    else:
        return 0

def score_forward_pe(forward_pe, industry_pe):
    """Score based on forward P/E (more predictive than trailing)"""
    if forward_pe <= 0:
        return 0
    
    relative_pe = forward_pe / industry_pe if industry_pe > 0 else 1
    
    if relative_pe < 0.5:
        return 10
    elif relative_pe < 0.7:
        return 8
    elif relative_pe < 0.9:
        return 6
    elif relative_pe < 1.1:
        return 5
    elif relative_pe < 1.3:
        return 3
    elif relative_pe < 1.5:
        return 2
    else:
        return 1

def score_peg(peg):
    """Enhanced PEG scoring"""
    if peg <= 0:
        return 0
    if peg < 0.5:
        return 10
    elif peg < 0.75:
        return 8
    elif peg < 1.0:
        return 6
    elif peg < 1.5:
        return 4
    elif peg < 2.0:
        return 2
    else:
        return 0

def score_pb(pb):
    """Price-to-Book scoring"""
    if pb <= 0:
        return 0
    if pb < 1.0:
        return 10
    elif pb < 1.5:
        return 8
    elif pb < 2.5:
        return 6
    elif pb < 4.0:
        return 4
    else:
        return 2

def score_roe(roe):
    """Return on Equity scoring"""
    if roe <= 0:
        return 0
    if roe > 25:
        return 10
    elif roe > 20:
        return 8
    elif roe > 15:
        return 6
    elif roe > 10:
        return 4
    elif roe > 5:
        return 2
    else:
        return 0

def score_eps_growth(growth):
    """EPS growth scoring"""
    if growth > 25:
        return 10
    elif growth > 15:
        return 8
    elif growth > 10:
        return 6
    elif growth > 5:
        return 4
    elif growth > 0:
        return 2
    else:
        return 0

def score_revenue_growth(growth, has_data=True):
    """Revenue growth scoring"""
    if not has_data or growth is None:
        return 0
    if growth > 20:
        return 10
    elif growth > 15:
        return 8
    elif growth > 10:
        return 6
    elif growth > 5:
        return 4
    elif growth > 0:
        return 2
    else:
        return 0

def score_fcf_trend(fcf_values, has_data=True):
    """Free cash flow trend scoring"""
    if not has_data or not fcf_values or len(fcf_values) < 2 or all(x == 0 for x in fcf_values):
        return 0
    
    positive_count = sum(1 for x in fcf_values if x > 0)
    
    # Check for improvement trend
    if len(fcf_values) >= 3:
        recent_avg = np.mean(fcf_values[:2])
        older_avg = np.mean(fcf_values[1:])
        if recent_avg > older_avg and positive_count >= 2:
            return 10
    
    if positive_count == len(fcf_values):
        return 8
    elif positive_count >= len(fcf_values) * 0.6:
        return 6
    elif positive_count > 0:
        return 4
    else:
        return 0

def score_debt_equity(de):
    """Debt-to-equity scoring"""
    if de < 0:
        return 0
    if de < 30:
        return 10
    elif de < 50:
        return 8
    elif de < 100:
        return 6
    elif de < 200:
        return 4
    else:
        return 2

def score_dividend_yield(dy):
    """Dividend yield scoring"""
    if dy <= 0:
        return 0
    if dy > 5:
        return 10
    elif dy > 3:
        return 8
    elif dy > 1:
        return 6
    else:
        return 4

def score_gross_margin(gm):
    """Gross margin scoring"""
    if gm <= 0:
        return 0
    if gm > 60:
        return 10
    elif gm > 40:
        return 8
    elif gm > 25:
        return 6
    elif gm > 15:
        return 4
    else:
        return 2

def score_ev_ebitda(ev_ebitda):
    """EV/EBITDA scoring"""
    if ev_ebitda <= 0:
        return 0
    if ev_ebitda < 8:
        return 10
    elif ev_ebitda < 12:
        return 8
    elif ev_ebitda < 15:
        return 6
    elif ev_ebitda < 20:
        return 4
    elif ev_ebitda < 25:
        return 2
    else:
        return 0

def score_price_sales(ps_ratio):
    """Price-to-Sales scoring"""
    if ps_ratio <= 0:
        return 0
    if ps_ratio < 1:
        return 10
    elif ps_ratio < 2:
        return 8
    elif ps_ratio < 4:
        return 6
    elif ps_ratio < 6:
        return 4
    elif ps_ratio < 10:
        return 2
    else:
        return 0

def score_analyst_upside(upside_percent):
    """Score based on analyst price target upside"""
    if upside_percent > 25:
        return 10
    elif upside_percent > 15:
        return 8
    elif upside_percent > 5:
        return 6
    elif upside_percent > -5:
        return 5
    elif upside_percent > -15:
        return 3
    else:
        return 1

# Include ALL your exact scoring functions:
# score_forward_pe, score_peg, score_pb, score_roe, score_eps_growth, 
# score_revenue_growth, score_fcf_trend, score_debt_equity, score_dividend_yield,
# score_gross_margin, score_ev_ebitda, score_price_sales, score_analyst_upside

def get_industry_pe(info):
    """Your exact industry PE function"""
    industry = info.get("industry", "")
    sector = info.get("sector", "")
    
    for key in INDUSTRY_PE_MAP:
        if key.lower() in industry.lower() or key.lower() in sector.lower():
            return INDUSTRY_PE_MAP[key]
    
    return INDUSTRY_PE_MAP["Unknown"]

def apply_sector_adjustments(scores, sector):
    """Your exact sector adjustment function"""
    if not sector or sector not in SECTOR_SCORING_MODELS:
        return scores
    
    model = SECTOR_SCORING_MODELS[sector]
    weight_adjustments = model.get("weight_adjustments", {})
    adjusted_scores = {}
    
    for metric, score in scores.items():
        if metric in weight_adjustments:
            multiplier = weight_adjustments[metric]
            adjusted_score = min(10, score * multiplier)
            adjusted_scores[metric] = adjusted_score
        else:
            adjusted_scores[metric] = score
    
    return adjusted_scores

def fetch_yahoo_info_exact(symbol):
    """Your exact Yahoo Finance data fetching function"""
    def try_fetch_symbol(sym):
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            
            if not info:
                return None
            
            price_indicators = [
                info.get("regularMarketPrice"),
                info.get("currentPrice"),
                info.get("previousClose"),
                info.get("open")
            ]
            
            has_price = any(price is not None for price in price_indicators)
            has_company_info = info.get("longName") or info.get("shortName")
            
            if not (has_price or has_company_info):
                return None
            
            enhanced_info = {
                "name": info.get("longName", info.get("shortName", "Unknown")),
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "pe": info.get("trailingPE"),
                "peg": info.get("trailingPegRatio"),
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "eps_growth": info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "de": info.get("debtToEquity"),
                "dy": info.get("dividendYield"),
                "gm": info.get("grossMargins"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "marketCap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
                "symbol": info.get("symbol", sym),
                "forwardPE": info.get("forwardPE"),
                "enterpriseToEbitda": info.get("enterpriseToEbitda"),
                "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                "targetMeanPrice": info.get("targetMeanPrice"),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice"))
            }
            
            return enhanced_info
            
        except Exception as e:
            print(f"Failed to fetch {sym}: {str(e)}")
            return None
    
    # Try original symbol first, then Danish variations
    result = try_fetch_symbol(symbol)
    if result:
        return result
    
    if not symbol.endswith('.CO'):
        if symbol in DANISH_STOCKS:
            danish_symbol = DANISH_STOCKS[symbol]
            result = try_fetch_symbol(danish_symbol)
            if result:
                return result
        
        co_symbol = f"{symbol}.CO"
        result = try_fetch_symbol(co_symbol)
        if result:
            return result
    
    return None

def calculate_scores_exact(info, industry_pe=20, score_weights=None):
    """Your exact scoring calculation function"""
    if not info:
        return None, None
    
    if score_weights is None:
        score_weights = DEFAULT_SCORE_WEIGHTS
    
    try:
        # Extract values exactly as in your system
        pe = safe_float(info.get("pe", 0))
        forward_pe = safe_float(info.get("forwardPE", 0))
        peg = safe_float(info.get("peg", 0))
        pb = safe_float(info.get("pb", 0))
        ev_ebitda = safe_float(info.get("enterpriseToEbitda", 0))
        price_sales = safe_float(info.get("priceToSalesTrailing12Months", 0))
        roe = safe_float(info.get("roe", 0)) * 100 if info.get("roe") else 0
        de = safe_float(info.get("de", 0))
        dy = safe_float(info.get("dy", 0))  # Use as-is, already a percentage
        gm = safe_float(info.get("gm", 0)) * 100 if info.get("gm") else 0
        eps_growth = safe_float(info.get("eps_growth", 0)) * 100 if info.get("eps_growth") else 0
        rev_growth = safe_float(info.get("revenue_growth", 0)) * 100 if info.get("revenue_growth") else 0
        
        # Analyst data
        target_price = safe_float(info.get("targetMeanPrice", 0))
        current_price = safe_float(info.get("currentPrice", 0))
        analyst_upside = 0
        if target_price > 0 and current_price > 0:
            analyst_upside = ((target_price - current_price) / current_price) * 100
        
        # Calculate scores using your exact functions
        scores = {
            "PE": score_pe(pe, industry_pe),
            "PEG": score_peg(peg),
            "PB": score_pb(pb),
            "ROE": score_roe(roe),
            "EPS Growth": score_eps_growth(eps_growth),
            "Revenue Growth": score_revenue_growth(rev_growth, True),
            "FCF Trend": score_fcf_trend([], True),  # Simplified for automation
            "Debt/Equity": score_debt_equity(de),
            "Dividend Yield": score_dividend_yield(dy),
            "Gross Margin": score_gross_margin(gm)
        }
        
        # Add enhanced metrics
        if forward_pe > 0:
            scores["Forward PE"] = score_forward_pe(forward_pe, industry_pe)
        
        if ev_ebitda > 0:
            scores["EV/EBITDA"] = score_ev_ebitda(ev_ebitda)
        
        if price_sales > 0:
            scores["Price/Sales"] = score_price_sales(price_sales)
        
        if target_price > 0 and current_price > 0:
            scores["Analyst Upside"] = score_analyst_upside(analyst_upside)
        
        # Filter out None values
        scores = {k: v for k, v in scores.items() if v is not None}
        
        # Apply sector adjustments exactly as in your system
        sector = info.get("sector", "")
        if sector:
            scores = apply_sector_adjustments(scores, sector)
        
        return scores, info
        
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return None, None

def analyze_stock_complete_exact(symbol, score_weights=None):
    """Your exact complete stock analysis"""
    try:
        info = fetch_yahoo_info_exact(symbol)
        
        if not info or info.get('name') == 'Unknown':
            return None
        
        industry_pe = get_industry_pe(info)
        scores, debug_data = calculate_scores_exact(info, industry_pe, score_weights)
        
        if not scores:
            return None
        
        # Calculate overall score exactly as in your system
        if score_weights is None:
            score_weights = DEFAULT_SCORE_WEIGHTS
            
        available_weights = {k: score_weights.get(k, 0) 
                           for k in scores if k in score_weights}
        
        if available_weights:
            total_weight = sum(available_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
                overall_score = sum(scores[k] * normalized_weights[k] for k in available_weights)
            else:
                overall_score = sum(scores.values()) / len(scores)
        else:
            overall_score = sum(scores.values()) / len(scores)
        
        # Your exact recommendation system
        if overall_score >= 8:
            recommendation = "🚀 Strong Buy"
        elif overall_score >= 6.5:
            recommendation = "📈 Buy"
        elif overall_score >= 4:
            recommendation = "🔄 Hold"
        elif overall_score >= 2:
            recommendation = "📉 Weak Sell"
        else:
            recommendation = "🛑 Strong Sell"
        
        return {
            'symbol': symbol,
            'company': info.get('name', 'Unknown'),
            'price': info.get('currentPrice', 0),
            'score': overall_score,
            'recommendation': recommendation,
            'sector': info.get('sector', 'Unknown'),
            'scores': scores
        }
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def main():
    """Main monitoring function with exact scoring system"""
    print("🚀 Starting automated portfolio monitoring with exact AS_MH_v6 scoring...")
    print(f"📅 Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get custom weights from environment if available
    custom_weights = os.environ.get('CUSTOM_SCORE_WEIGHTS')
    if custom_weights:
        try:
            score_weights = json.loads(custom_weights)
        except:
            score_weights = DEFAULT_SCORE_WEIGHTS
    else:
        score_weights = DEFAULT_SCORE_WEIGHTS
    
    # Try to load portfolio from file first
    portfolio_symbols = load_portfolio_from_file()
    
    # If no portfolio file, fall back to environment variable or default
    if not portfolio_symbols:
        portfolio_symbols_env = os.environ.get('PORTFOLIO_SYMBOLS', 'AAPL,MSFT,GOOGL')
        portfolio_symbols = [s.strip() for s in portfolio_symbols_env.split(',')]
        print(f"📈 Using environment/default symbols: {', '.join(portfolio_symbols)}")
    
    portfolio = portfolio_symbols
    
    print(f"📊 Analyzing {len(portfolio)} stocks with exact AS_MH_v6 scoring: {', '.join(portfolio)}")
    print(f"⚖️ Using {len(score_weights)} weighted metrics")
    
    # Analyze each stock with exact scoring
    results = []
    failed_symbols = []
    
    for symbol in portfolio:
        print(f"   Analyzing {symbol}...")
        result = analyze_stock_complete_exact(symbol, score_weights)
        if result:
            results.append(result)
            print(f"   ✅ {symbol}: Score {result['score']:.1f} - {result['recommendation']}")
        else:
            failed_symbols.append(symbol)
            print(f"   ❌ {symbol}: Failed to analyze")
    
    if results:
        print(f"\n📊 Analysis complete! {len(results)} stocks analyzed successfully.")
        
        # Check for significant changes using exact scoring
        significant_changes = check_significant_changes_exact(results)
        
        if significant_changes:
            print(f"🚨 {len(significant_changes)} significant changes detected:")
            for change in significant_changes:
                print(f"   • {change['symbol']}: {change['previous_score']:.1f} → {change['current_score']:.1f} ({change['change']:+.1f})")
        else:
            print("📊 No significant changes detected")
        
        # Send email with exact scoring data
        send_scheduled = os.environ.get('SEND_SCHEDULED_REPORTS', 'false').lower() == 'true'
        if significant_changes or send_scheduled:
            print("📧 Sending email alert with exact scoring analysis...")
            email_sent = send_email_alert_exact(results, significant_changes)
            if email_sent:
                print("✅ Email alert sent successfully")
            else:
                print("❌ Failed to send email alert")
        
        # Print summary with exact methodology
        print(f"\n📊 Portfolio Summary (AS_MH_v6 Scoring):")
        print(f"   • Average Score: {sum(r['score'] for r in results) / len(results):.1f}/10")
        print(f"   • Strong Buys (≥8.0): {len([r for r in results if r['score'] >= 8.0])}")
        print(f"   • Buys (≥6.5): {len([r for r in results if 6.5 <= r['score'] < 8.0])}")
        print(f"   • Holds (4.0-6.5): {len([r for r in results if 4.0 <= r['score'] < 6.5])}")
        print(f"   • Sells (≤4.0): {len([r for r in results if r['score'] < 4.0])}")
        
        if failed_symbols:
            print(f"⚠️  Failed to analyze: {', '.join(failed_symbols)}")
    
    else:
        print("❌ No stocks were successfully analyzed")
        sys.exit(1)
    
    print("✅ Monitoring complete with exact AS_MH_v6 scoring system!")

def check_significant_changes_exact(results):
    """Check for significant changes in stock scores"""
    significant_changes = []
    
    try:
        # Try to load previous scores
        if os.path.exists('previous_scores.json'):
            with open('previous_scores.json', 'r') as f:
                previous_scores = json.load(f)
        else:
            previous_scores = {}
        
        # Check for significant changes (threshold of 1.0 points)
        for result in results:
            symbol = result['symbol']
            current_score = result['score']
            
            if symbol in previous_scores:
                previous_score = previous_scores[symbol]
                change = current_score - previous_score
                
                if abs(change) >= 1.0:  # Significant change threshold
                    significant_changes.append({
                        'symbol': symbol,
                        'previous_score': previous_score,
                        'current_score': current_score,
                        'change': change
                    })
        
        # Save current scores for next run
        current_scores = {result['symbol']: result['score'] for result in results}
        with open('previous_scores.json', 'w') as f:
            json.dump(current_scores, f, indent=2)
        
        return significant_changes
        
    except Exception as e:
        print(f"Warning: Could not check for significant changes: {e}")
        return []

def send_email_alert_exact(results, significant_changes):
    """Send email alert with exact scoring analysis"""
    try:
        # Only send email if environment variables are set
        gmail_user = os.environ.get('GMAIL_USER')
        gmail_password = os.environ.get('GMAIL_PASSWORD')
        alert_email = os.environ.get('ALERT_EMAIL')
        
        if not all([gmail_user, gmail_password, alert_email]):
            print("📧 Email credentials not configured - skipping email alert")
            return False
        
        # Create email content
        subject = f"Portfolio Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Get portfolio information
        portfolio_info = ""
        if os.path.exists('portfolio_config.json'):
            try:
                with open('portfolio_config.json', 'r') as f:
                    portfolio_data = json.load(f)
                
                symbols = portfolio_data.get('symbols', [])
                last_updated = portfolio_data.get('last_updated', 'Unknown')
                
                portfolio_info = f"""
📊 Portfolio Information:
• Portfolio Size: {len(symbols)} stocks
• Symbols: {', '.join(symbols)}
• Last Sync: {last_updated}
• Source: AS_MH_v6 Application

"""
            except:
                portfolio_info = "📊 Portfolio: Using default symbols\n\n"
        else:
            portfolio_info = "📊 Portfolio: Using environment/default symbols\n\n"
        
        body = f"""
Portfolio Monitoring Alert - AS_MH_v6 Scoring System

Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{portfolio_info}"""
        
        if significant_changes:
            body += "🚨 SIGNIFICANT CHANGES DETECTED:\n\n"
            for change in significant_changes:
                body += f"• {change['symbol']}: {change['previous_score']:.1f} → {change['current_score']:.1f} ({change['change']:+.1f})\n"
            body += "\n"
        
        body += "📊 CURRENT PORTFOLIO STATUS:\n\n"
        
        # Sort by score (highest first)
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        for result in sorted_results:
            body += f"• {result['symbol']}: {result['score']:.1f}/10 - {result['recommendation']}\n"
        
        body += f"\n📈 PORTFOLIO SUMMARY:\n"
        body += f"• Average Score: {sum(r['score'] for r in results) / len(results):.1f}/10\n"
        body += f"• Strong Buys (≥8.0): {len([r for r in results if r['score'] >= 8.0])}\n"
        body += f"• Buys (≥6.5): {len([r for r in results if 6.5 <= r['score'] < 8.0])}\n"
        body += f"• Holds (4.0-6.5): {len([r for r in results if 4.0 <= r['score'] < 6.5])}\n"
        body += f"• Sells (≤4.0): {len([r for r in results if r['score'] < 4.0])}\n"
        
        body += "\n\nGenerated by AS_MH_v6 Automated Portfolio Monitor"
        
        # Send email
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = alert_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

if __name__ == "__main__":
    main()