# app/scoring.py
import numpy as np
from app.config import SECTOR_SCORING_MODELS, INDUSTRY_PE_MAP, DEFAULT_SCORE_WEIGHTS

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

def score_pb(pb):
    """Price-to-Book scoring"""
    if pb <= 0: return 0
    if pb < 1.0: return 10
    if pb < 1.5: return 8
    if pb < 2.5: return 6
    if pb < 4.0: return 4
    return 2

def score_eps_growth(growth):
    """EPS growth scoring"""
    if growth > 25: return 10
    if growth > 15: return 8
    if growth > 10: return 6
    if growth > 5: return 4
    if growth > 0: return 2
    return 0

def score_revenue_growth(growth):
    """Revenue growth scoring"""
    if growth > 20: return 10
    if growth > 15: return 8
    if growth > 10: return 6
    if growth > 5: return 4
    if growth > 0: return 2
    return 0

def score_fcf_trend(fcf_values):
    """Free cash flow trend scoring"""
    if not fcf_values or len(fcf_values) < 2 or all(x == 0 for x in fcf_values):
        return 0
    
    positive_count = sum(1 for x in fcf_values if x > 0)
    
    if len(fcf_values) >= 3:
        recent_avg = np.mean(fcf_values[:2])
        older_avg = np.mean(fcf_values[1:])
        if recent_avg > older_avg and positive_count >= 2:
            return 10
    
    if positive_count == len(fcf_values): return 8
    if positive_count >= len(fcf_values) * 0.6: return 6
    if positive_count > 0: return 4
    return 0

def score_debt_equity(de):
    """Debt-to-equity scoring"""
    if de < 0: return 0
    if de < 30: return 10
    if de < 50: return 8
    if de < 100: return 6
    if de < 200: return 4
    return 2

def score_dividend_yield(dy):
    """Dividend yield scoring"""
    if dy <= 0: return 0
    if dy > 5: return 10
    if dy > 3: return 8
    if dy > 1: return 6
    return 4

def score_gross_margin(gm):
    """Gross margin scoring"""
    if gm <= 0: return 0
    if gm > 60: return 10
    if gm > 40: return 8
    if gm > 25: return 6
    if gm > 15: return 4
    return 2

def score_ev_ebitda(ev_ebitda):
    """EV/EBITDA scoring"""
    if ev_ebitda <= 0: return 0
    if ev_ebitda < 8: return 10
    if ev_ebitda < 12: return 8
    if ev_ebitda < 15: return 6
    if ev_ebitda < 20: return 4
    return 2

def score_price_sales(ps_ratio):
    """Price-to-Sales scoring"""
    if ps_ratio <= 0: return 0
    if ps_ratio < 1: return 10
    if ps_ratio < 2: return 8
    if ps_ratio < 4: return 6
    if ps_ratio < 6: return 4
    return 2

def score_analyst_upside(upside_percent):
    """Score based on analyst price target upside"""
    if upside_percent > 25: return 10
    if upside_percent > 15: return 8
    if upside_percent > 5: return 6
    if upside_percent > -5: return 5
    if upside_percent > -15: return 3
    return 1


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

class ScoreCalculator:
    """Calculates scores for stocks based on various financial metrics."""

    def __init__(self, weights=None):
        """Initializes the ScoreCalculator with a specific set of weights."""
        self.weights = weights if weights is not None else DEFAULT_SCORE_WEIGHTS.copy()

    def calculate_individual_scores(self, info, technical_data=None):
        """Calculates all individual metric scores for a stock."""
        if not info:
            return {}

        industry_pe = get_industry_pe(info)
        industry_roe = info.get('industryRoe', 15) # Fallback

        scores = {
            "PE": score_pe(safe_float(info.get("trailingPE")), industry_pe),
            "Forward PE": score_pe(safe_float(info.get("forwardPE")), industry_pe),
            "PEG": score_peg(safe_float(info.get("pegRatio"))),
            "PB": score_pb(safe_float(info.get("priceToBook"))),
            "EV/EBITDA": score_ev_ebitda(safe_float(info.get("enterpriseToEbitda"))),
            "ROE": score_roe(safe_float(info.get("returnOnEquity", 0) * 100), industry_roe),
            "EPS Growth": score_eps_growth(safe_float(info.get("earningsQuarterlyGrowth", 0) * 100)),
            "Revenue Growth": score_revenue_growth(safe_float(info.get("revenueGrowth", 0) * 100)),
            "Debt/Equity": score_debt_equity(safe_float(info.get("debtToEquity"))),
            "Dividend Yield": score_dividend_yield(safe_float(info.get("dividendYield", 0) * 100)),
            "Gross Margin": score_gross_margin(safe_float(info.get("grossMargins", 0) * 100)),
            "Price/Sales": score_price_sales(safe_float(info.get("priceToSalesTrailing12Months"))),
            "Current Ratio": score_current_ratio(safe_float(info.get("currentRatio"))),
            "Interest Coverage": score_interest_coverage(safe_float(info.get("interestCoverage"))),
        }
        
        # Analyst upside
        target_price = safe_float(info.get('targetMeanPrice'))
        current_price = safe_float(info.get('currentPrice'))
        if target_price and current_price:
            upside = ((target_price - current_price) / current_price) * 100
            scores["Analyst Upside"] = score_analyst_upside(upside)
        else:
            scores["Analyst Upside"] = 0

        # Momentum Score
        if technical_data is not None and not technical_data.empty:
            latest_tech = technical_data.iloc[-1]
            scores["Momentum"] = score_momentum(
                latest_tech['Close'],
                latest_tech.get('SMA_50', 0),
                latest_tech.get('SMA_200', 0)
            )
        else:
            scores["Momentum"] = 0
            
        # FCF Trend (requires historical data not in 'info')
        # This would need to be passed in if available
        scores["FCF Trend"] = 0 # Placeholder

        return scores

    def calculate_total_score(self, info, technical_data=None):
        """Calculates the total weighted score for a stock based on its info."""
        if not info:
            return {}, 0.0

        # First, calculate the individual raw scores
        scores = self.calculate_individual_scores(info, technical_data)

        # Then, apply sector-specific adjustments
        sector = info.get("sector")
        adjusted_scores = apply_sector_adjustments(scores, sector)

        total_score = 0
        total_weight = 0
        
        for metric, score in adjusted_scores.items():
            if metric in self.weights:
                total_score += score * self.weights[metric]
                total_weight += self.weights[metric]
        
        # Normalize score
        if total_weight > 0:
            final_score = (total_score / total_weight) * 10
        else:
            final_score = 0
            
        return adjusted_scores, final_score

def score_peg(peg):
    """Enhanced PEG scoring"""
    if peg <= 0: return 0
    if peg < 0.5: return 10
    if peg < 0.75: return 8
    if peg < 1.0: return 6
    if peg < 1.5: return 4
    if peg < 2.0: return 2
    return 0
