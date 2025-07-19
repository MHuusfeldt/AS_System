# 📈 Stock Analysis Hub - Enhanced Portfolio Management

## 🚀 Quick Start

### Option 1: Direct Command (Recommended)
```bash
cd /Users/magnushuusfeldt/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/AS_System
streamlit run AS_MH_v6.py --server.port 8501
```

### Option 2: Using Python Script
```bash
cd /Users/magnushuusfeldt/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/AS_System
python run_app.py
```

### Option 3: Using Shell Script
```bash
cd /Users/magnushuusfeldt/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/AS_System
./start_app.sh
```

## 🌐 Access the Application

Once started, the application will be available at:
**http://localhost:8501**

## 🎉 New Enhanced Features

### 💼 Portfolio Management with Purchase Price Tracking
- **Add Purchase Prices**: Track what you paid for each stock
- **Quantity Management**: Monitor share quantities
- **Purchase Date Tracking**: Record when you bought each position
- **Automatic Migration**: Existing portfolios are seamlessly upgraded

### 💰 Real-time P&L Analysis
- **Profit & Loss Calculation**: See current gains/losses for each holding
- **Portfolio Value Tracking**: Total investment vs. current market value
- **Performance Metrics**: Win rate, best/worst performers
- **Detailed Holdings Table**: Complete breakdown of all positions

### 🚨 Portfolio Alerts & Risk Monitoring
- **Gain/Loss Alerts**: Get notified when positions exceed thresholds
- **Daily Price Movement Alerts**: Monitor significant daily changes
- **Portfolio Risk Assessment**: Volatility analysis and risk scoring
- **Customizable Thresholds**: Set your own alert parameters

### 🔄 Integrated Portfolio Features
- **Portfolio Rebalancing**: Updated to work with holdings structure
- **Weekly Market Screener**: Portfolio-aware screening 
- **Automated Analysis**: Full portfolio health checks
- **Enhanced Monitoring**: Comprehensive tracking tools

## 📋 How to Use the New Features

### 1. Adding Holdings with Purchase Prices
1. Go to "Portfolio Manager" → "My Portfolio" tab
2. Enter Symbol, Quantity, and Purchase Price
3. Set Purchase Date (defaults to today)
4. Click "Add to Portfolio"

### 2. Viewing P&L Analysis
1. Navigate to "Portfolio Alerts" tab
2. Click "💰 P&L Analysis" button
3. View real-time profit/loss calculations
4. See top performers and underperformers

### 3. Setting Up Alerts
1. In "Portfolio Alerts" tab
2. Set gain/loss thresholds (e.g., +10%, -10%)
3. Set daily price change alerts (e.g., 5%)
4. Click "🚨 Portfolio Alerts & Risk Check"

### 4. Portfolio Rebalancing
1. Go to "Portfolio Rebalancing" tab
2. Set target portfolio size and score thresholds
3. Click "🔄 Rebalance Portfolio"
4. Review recommendations and implement gradually

## 🛠️ Requirements

- Python 3.7+
- Streamlit
- yfinance
- pandas
- plotly
- All dependencies listed in `requirements.txt`

## 🔧 Installation

If you need to install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Application Structure

- **Stock Analysis Hub**: Multi-source stock analysis with scoring
- **Trading Signals Hub**: Buy/sell recommendations with technical analysis
- **Market Screeners**: Screen multiple markets for opportunities
- **Portfolio Manager**: Enhanced portfolio management with P&L tracking
- **Performance Benchmarking**: Historical analysis and score tracking
- **Danish Stocks Manager**: Specialized Copenhagen Exchange tools

## 🎯 Key Benefits

### 📊 Complete Investment Tracking
- Know exactly how much you've invested vs. current value
- Track performance of individual holdings over time
- Identify your best and worst performing investments

### ⚡ Proactive Risk Management
- Get alerted to significant gains that might warrant profit-taking
- Be notified of losses before they become major problems
- Monitor portfolio risk levels and volatility

### 🧠 Intelligent Portfolio Optimization
- Automated suggestions for portfolio improvements
- Integration with scoring system for rebalancing decisions
- Weekly screening for new opportunities

### 📈 Professional-Grade Analytics
- Win rate calculations and performance metrics
- Detailed holding period analysis
- Risk-adjusted portfolio assessment

## 🌟 What's Enhanced

- ✅ Purchase price tracking with full cost basis calculation
- ✅ Real-time P&L monitoring with current market prices
- ✅ Portfolio alerts with customizable thresholds
- ✅ Risk assessment based on historical volatility
- ✅ Professional-grade portfolio analytics
- ✅ Seamless integration across all portfolio features

## 📞 Support

The enhanced portfolio management system is now fully integrated and ready to help you track, monitor, and optimize your investments with professional-grade tools!
