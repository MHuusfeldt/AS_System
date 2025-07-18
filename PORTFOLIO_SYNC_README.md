# Portfolio Sync Feature

## Overview
The Portfolio Sync feature allows you to automatically sync your portfolio between the AS_MH_v6 Streamlit application and the Automated_monitor.py script. This ensures that your automated monitoring always uses the latest portfolio configuration.

## How It Works

### 1. Portfolio Configuration File
When you sync your portfolio, a `portfolio_config.json` file is created containing:
- Your portfolio symbols
- Last update timestamp
- Total number of stocks

### 2. AS_MH_v6 Application Features
- **Sync Status Indicator**: Shows if your portfolio is synced, out of sync, or not synced yet
- **Manual Sync Button**: Click "🔄 Sync with Automated Monitor" to sync immediately
- **Auto-Sync Toggle**: Enable to automatically sync whenever you add/remove stocks
- **Real-time Feedback**: Get immediate confirmation when sync completes

### 3. Automated Monitor Integration
The `Automated_monitor.py` script automatically:
- Checks for `portfolio_config.json` first
- Falls back to environment variables if no config file exists
- Includes portfolio information in email alerts
- Shows portfolio sync status in console output

## Usage Instructions

### Setting Up Sync
1. Open AS_MH_v6 in Streamlit
2. Go to "📊 Portfolio Manager & Weekly Screeners" tab
3. Navigate to "💼 My Portfolio" sub-tab
4. Add your desired stocks to the portfolio
5. Click "🔄 Sync with Automated Monitor"
6. Optionally enable "🔄 Auto-sync on changes"

### Monitoring Status
- **🟢 Synced**: Portfolio is synchronized with automated monitor
- **🟡 Out of sync**: Portfolio has changed since last sync
- **🔵 Not synced**: No sync file exists yet
- **🔴 Error**: Sync system encountered an error

### GitHub Actions Integration
The automated monitoring workflow will:
- Use your synced portfolio if `portfolio_config.json` exists
- Fall back to environment variables if no config file
- Include portfolio sync information in alerts

## File Structure
```
AS_System/
├── AS_MH_v6.py                 # Main Streamlit app with sync features
├── Automated_monitor.py         # Monitor script with portfolio loading
├── portfolio_config.json       # Auto-generated portfolio config
└── .github/workflows/
    └── automated_portfolio_monitoring.yml  # Updated workflow
```

## Benefits
✅ **Automatic Sync**: Keep your monitoring in sync with portfolio changes
✅ **Real-time Updates**: Changes in AS_MH_v6 immediately available to monitor
✅ **Fallback Support**: Works with existing environment variable setup
✅ **Visual Feedback**: Clear status indicators and sync confirmations
✅ **Email Integration**: Portfolio info included in monitoring alerts

## Example Portfolio Config
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "NOVO-B.CO", "ORSTED.CO"],
  "last_updated": "2025-07-18T13:32:12.538632",
  "total_stocks": 5
}
```

## Troubleshooting
- **Sync not working**: Check file permissions and ensure both scripts are in same directory
- **Config file missing**: Manually click sync button to create initial config
- **Out of sync warning**: Click sync button to update the config file
- **Auto-sync not working**: Ensure the toggle is enabled and you're making changes through the UI

This feature ensures seamless integration between your interactive portfolio management and automated monitoring systems!
