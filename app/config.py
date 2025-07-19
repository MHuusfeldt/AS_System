# app/config.py

# Configuration
API_KEY = "7J1AJVC9MAYLRRA7"
REQUEST_DELAY = 0.5

# Danish stock mappings - Comprehensive list
DANISH_STOCKS = {
    # Large Cap - Major Danish companies
    "NOVO-B": "NOVO-B.CO",           # Novo Nordisk
    "NOVO": "NOVO-B.CO",             # Alternative Novo symbol
    "MAERSK-B": "MAERSK-B.CO",       # A.P. Moller-Maersk
    "MAERSK": "MAERSK-B.CO",         # Alternative Maersk symbol
    "ORSTED": "ORSTED.CO",           # Orsted
    "DSV": "DSV.CO",                 # DSV
    "CARLB": "CARL-B.CO",            # Carlsberg
    "CARL-B": "CARL-B.CO",           # Carlsberg B shares
    "NZYM-B": "NZYM-B.CO",          # Novozymes
    "NOVOZYMES": "NZYM-B.CO",        # Alternative Novozymes symbol
    "TRYG": "TRYG.CO",               # Tryg
    "DEMANT": "DEMANT.CO",           # Demant
    "COLO-B": "COLO-B.CO",          # Coloplast
    "COLOPLAST": "COLO-B.CO",        # Alternative Coloplast symbol
    "GMAB": "GMAB.CO",               # Genmab
    "GENMAB": "GMAB.CO",             # Alternative Genmab symbol
    
    # Mid Cap
    "AMBU-B": "AMBU-B.CO",          # Ambu
    "BAVA": "BAVA.CO",               # Bavarian Nordic
    "CHR": "CHR.CO",                 # Chr. Hansen
    "DANSKE": "DANSKE.CO",           # Danske Bank
    "FLS": "FLS.CO",                 # FLSmidth
    "GN": "GN.CO",                   # GN Store Nord
    "ISS": "ISS.CO",                 # ISS
    "JYSK": "JYSK.CO",               # Jyske Bank
    "NETC": "NETC.CO",               # NetCompany
    "PNDORA": "PNDORA.CO",           # Pandora
    "PANDORA": "PNDORA.CO",          # Alternative Pandora symbol
    "RBREW": "RBREW.CO",             # Royal Unibrew
    "ROCK-B": "ROCK-B.CO",           # Rockwool
    "SIM": "SIM.CO",                 # SimCorp
    "SYDB": "SYDB.CO",               # Sydbank
    "VWS": "VWS.CO",                 # Vestas Wind Systems
    "VESTAS": "VWS.CO",              # Alternative Vestas symbol
    
    # Small Cap
    "ALKA-B": "ALKA-B.CO",          # Alkane
    "BIOPRT": "BIOPRT.CO",          # Bioporto
    "CAPD": "CAPD.CO",               # Capdan
    "DKSH": "DKSH.CO",               # DKSH
    "ERHV": "ERHV.CO",               # Erhvervs
    "FLUG-B": "FLUG-B.CO",          # Flugger
    "GYLD": "GYLD.CO",               # Gyldendal
    "HPRO": "HPRO.CO",               # H+H
    "LUXOR-B": "LUXOR-B.CO",        # Luxor
    "MATAS": "MATAS.CO",             # Matas
    "NNIT": "NNIT.CO",               # NNIT
    "OSKAR": "OSKAR.CO",             # Oskar
    "RILBA": "RILBA.CO",             # Rilba
    "SANT": "SANT.CO",               # Santander Consumer Bank
    "SPNO": "SPNO.CO",               # Spar Nord
    "TLSN": "TLSN.CO",               # Tl
    
    # Additional banking and financial
    "NYKR": "NYKR.CO",               # Nykredit
    "TOPDM": "TOPDM.CO",             # TopDanmark
    "ALMB": "ALMB.CO",               # Alm. Brand
    
    # Additional healthcare and pharma
    "BAVB": "BAVA.CO",               # Bavarian Nordic (alternative)
    "NOVO-A": "NOVO-A.CO",           # Novo Nordisk A shares
    "NZYM-A": "NZYM-A.CO",          # Novozymes A shares
    "CARL-A": "CARL-A.CO",           # Carlsberg A shares
    "COLO-A": "COLO-A.CO",          # Coloplast A shares
    "ROCK-A": "ROCK-A.CO",          # Rockwool A shares
    "MAERSK-A": "MAERSK-A.CO",      # Maersk A shares
}

# S&P 500 major stocks (representative sample)
SP500_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX", "ADBE",
    "CRM", "CMCSA", "XOM", "VZ", "KO", "INTC", "ABT", "NKE", "PFE", "TMO",
    "AVGO", "CVX", "WMT", "COST", "NEE", "DHR", "ABBV", "ACN", "TXN", "LIN",
    "HON", "BMY", "UPS", "QCOM", "LOW", "AMD", "ORCL", "LMT", "T", "IBM"
]

# NASDAQ 100 major stocks (representative sample)
NASDAQ100_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE", "PYPL",
    "INTC", "CMCSA", "CSCO", "PEP", "AVGO", "TXN", "QCOM", "COST", "SBUX", "INTU",
    "AMD", "ISRG", "AMAT", "BKNG", "ADP", "GILD", "MU", "MELI", "LRCX", "FISV",
    "REGN", "CSX", "ATVI", "VRTX", "ILMN", "JD", "EA", "EXC", "KDP", "SIRI",
    "CTSH", "BIIB", "WBA", "MRNA", "ZM", "DOCU", "PTON", "DXCM", "OKTA", "TEAM"
]

# European major stocks (representative sample)
EUROPEAN_STOCKS = [
    # UK
    "SHEL", "AZN", "BP.L", "ULVR.L", "HSBA.L", "VOD.L", "GSK.L", "DGE.L", "BT-A.L", "BARC.L",
    # Germany  
    "SAP", "ASML", "TTE", "OR.PA", "SAN", "INGA.AS", "MC.PA", "RMS.PA", "AIR.PA", "BNP.PA",
    # France
    "LVMH.PA", "NESN.SW", "RHHBY", "NOVN.SW", "UG.PA", "CAP.PA", "SU.PA", "BN.PA", "EL.PA",
    # Netherlands/Other
    "RDS-A", "ING", "PHIA.AS", "UNA.AS", "HEIA.AS", "DSM.AS", "ASML.AS", "RDSA.AS"
]

DEFAULT_SCORE_WEIGHTS = {
    "PE": 0.08,
    "Forward PE": 0.12,
    "PEG": 0.10,
    "PB": 0.07,
    "EV/EBITDA": 0.10,
    "ROE": 0.10,
    "EPS Growth": 0.12,
    "Revenue Growth": 0.08,
    "FCF Trend": 0.05,
    "Debt/Equity": 0.05,
    "Current Ratio": 0.03, # New
    "Interest Coverage": 0.03, # New
    "Dividend Yield": 0.02,
    "Gross Margin": 0.02,
    "Price/Sales": 0.04,
    "Analyst Upside": 0.06,
    "Momentum": 0.03 # New
}

# Industry P/E mapping
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

# Sector-specific scoring models
SECTOR_SCORING_MODELS = {
    "Technology": {
        "weight_adjustments": {
            "Forward PE": 1.2,
            "Revenue Growth": 1.4,
            "Gross Margin": 1.3,
            "Price/Sales": 1.2,
            "EPS Growth": 1.3
        },
        "benchmarks": {
            "revenue_growth_excellent": 25,
            "gross_margin_excellent": 70,
            "forward_pe_max": 35
        }
    },
    "Industrials": {
        "weight_adjustments": {
            "EV/EBITDA": 1.3,
            "ROE": 1.2,
            "Debt/Equity": 1.1,
            "FCF Trend": 1.4,
            "Revenue Growth": 1.1
        },
        "benchmarks": {
            "roe_excellent": 18,
            "ev_ebitda_excellent": 12,
            "debt_equity_max": 0.6
        }
    },
    "Financials": {
        "weight_adjustments": {
            "PB": 1.5,
            "ROE": 1.4,
            "Dividend Yield": 1.2,
            "PE": 1.2
        },
        "benchmarks": {
            "roe_excellent": 15,
            "pb_excellent": 1.2,
            "dividend_yield_good": 3.5
        }
    },
    "Healthcare": {
        "weight_adjustments": {
            "Forward PE": 1.1,
            "Revenue Growth": 1.2,
            "Gross Margin": 1.3,
            "EPS Growth": 1.2
        },
        "benchmarks": {
            "revenue_growth_excellent": 15,
            "gross_margin_excellent": 75
        }
    },
    "Consumer Staples": {
        "weight_adjustments": {
            "Dividend Yield": 1.4,
            "ROE": 1.2,
            "Debt/Equity": 1.1,
            "Gross Margin": 1.2
        },
        "benchmarks": {
            "dividend_yield_excellent": 4,
            "roe_good": 20
        }
    }
}
