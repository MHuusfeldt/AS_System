# app/scoring.py
import numpy as np
from app.config import SECTOR_SCORING_MODELS, INDUSTRY_PE_MAP

def safe_float(value, default=0):
    """Safely convert value to float"""
    try:
        if value is None or value == "None" or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

# --- New Scoring Functions ---
def score_current_ratio(ratio):
    """Score based on Current Ratio (Liquidity)."""
    if ratio > 2: return 10
    if ratio > 1.5: return 8
    if ratio > 1: return 6
    if ratio > 0.75: return 4
    if ratio > 0.5: return 2
    return 0

def score_interest_coverage(ratio):
    """Score based on Interest Coverage Ratio."""
    if ratio > 10: return 10
    if ratio > 7: return 8
    if ratio > 5: return 6
    if ratio > 3: return 4
    if ratio > 1: return 2
    return 0

def score_momentum(price, sma50, sma200):
    """Score based on price momentum relative to moving averages."""
    if price > sma50 > sma200: return 10 # Strong uptrend
    if price > sma50 and price > sma200: return 8 # Uptrend
    if price > sma50 or price > sma200: return 6 # Mild uptrend
    if price < sma50 and price < sma200: return 2 # Downtrend
    return 4 # Neutral/sideways

# --- Existing (Refactored) Scoring Functions ---
def score_pe(pe, industry_pe):
    """Enhanced P/E scoring with industry comparison"""
    if pe <= 0: return -2 # Penalize negative earnings
    relative_pe = pe / industry_pe if industry_pe > 0 else 1
    if relative_pe < 0.6: return 10
    if relative_pe < 0.8: return 8
    if relative_pe < 1.0: return 6
    if relative_pe < 1.2: return 4
    if relative_pe < 1.5: return 3
    if relative_pe < 2.0: return 2
    return 0

def score_roe(roe, industry_roe):
    """Dynamic ROE scoring based on industry comparison."""
    if industry_roe == 0: industry_roe = 15 # Fallback
    if roe < 0: return -5 # Penalize negative ROE heavily
    
    relative_roe = roe / industry_roe
    if relative_roe > 1.5: return 10
    if relative_roe > 1.2: return 8
    if relative_roe > 1.0: return 6
    if relative_roe > 0.8: return 4
    if relative_roe > 0.5: return 2
    return 0

# ... (add all other existing score_* functions here, refactored as needed)
# For brevity, I will omit them, but you should move them all.

def get_industry_pe(info):
    """Get industry P/E ratio"""
    industry = info.get("industry", "")
    sector = info.get("sector", "")
    for key in INDUSTRY_PE_MAP:
        if key.lower() in industry.lower() or key.lower() in sector.lower():
            return INDUSTRY_PE_MAP[key]
    return INDUSTRY_PE_MAP["Unknown"]

def apply_sector_adjustments(scores, sector):
    """Apply sector-specific weight adjustments to improve accuracy"""
    if not sector or sector not in SECTOR_SCORING_MODELS:
        return scores
    model = SECTOR_SCORING_MODELS[sector]
    weight_adjustments = model.get("weight_adjustments", {})
    adjusted_scores = {}
    for metric, score in scores.items():
        adjustment = weight_adjustments.get(metric, 1.0)
        adjusted_scores[metric] = min(10, score * adjustment) # Cap score at 10
    return adjusted_scores

def calculate_scores(info, technical_data):
    """Main function to calculate all scores for a stock."""
    if not info:
        return None, None

    industry_pe = get_industry_pe(info)
    # A placeholder for industry ROE, you'd need to fetch this data
    industry_roe = info.get('industryRoe', 15) 

    scores = {
        "PE": score_pe(safe_float(info.get("pe")), industry_pe),
        "ROE": score_roe(safe_float(info.get("roe", 0) * 100), industry_roe),
        # ... call all other scoring functions
    }

    # Add new scores
    scores["Current Ratio"] = score_current_ratio(safe_float(info.get("currentRatio")))
    scores["Interest Coverage"] = score_interest_coverage(safe_float(info.get("interestCoverage")))
    
    if technical_data is not None and not technical_data.empty:
        latest_tech = technical_data.iloc[-1]
        scores["Momentum"] = score_momentum(
            latest_tech['Close'],
            latest_tech.get('SMA_50'),
            latest_tech.get('SMA_200')
        )

    # Apply sector adjustments
    sector = info.get("sector", "")
    if sector:
        scores = apply_sector_adjustments(scores, sector)

    return scores, info
