# app/ui_components.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
import asyncio

# Import new backend functions
from app.config import (
    DANISH_STOCKS, SP500_STOCKS, NASDAQ100_STOCKS, EUROPEAN_STOCKS,
    DEFAULT_SCORE_WEIGHTS, SECTOR_SCORING_MODELS, INDUSTRY_PE_MAP
)
from app.data_fetcher import (
    StockDataFetcher, PortfolioDataFetcher, get_3year_financial_history, get_3year_price_performance
)
from app.scoring import ScoreCalculator
from app.portfolio import PortfolioManager, init_db

def format_currency(value, currency='USD', decimals=2):
    """Format a monetary value with the appropriate currency symbol."""
    if value is None or pd.isna(value) or value == 0:
        return "N/A"
    
    symbols = {'DKK': 'DKK', 'EUR': '€', 'GBP': '£', 'USD': '$'}
    symbol = symbols.get(currency, '$')
    
    if symbol == 'DKK':
        return f"{value:,.{decimals}f} {symbol}"
    else:
        return f"{symbol}{value:,.{decimals}f}"

def create_enhanced_score_chart(scores, symbol):
    """Create an enhanced score visualization using Plotly."""
    if not scores:
        return go.Figure().update_layout(title_text=f"No scores available for {symbol}", showlegend=False)

    score_df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score']).sort_values('Score', ascending=False)
    
    colors = px.colors.sequential.Viridis_r
    color_scale = np.linspace(0, 1, len(score_df))
    marker_colors = [colors[int(c * (len(colors)-1))] for c in color_scale]

    fig = go.Figure(go.Bar(
        x=score_df['Score'],
        y=score_df['Metric'],
        orientation='h',
        marker=dict(
            color=marker_colors,
            line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
        ),
        text=score_df['Score'].apply(lambda x: f'{x:.1f}'),
        textposition='outside'
    ))

    fig.update_layout(
        title=f'<b>Fundamental Score Breakdown for {symbol}</b>',
        xaxis_title='Score (out of 10)',
        yaxis_title='Metric',
        yaxis=dict(autorange="reversed"),
        height=400 + len(score_df) * 20,
        margin=dict(l=120, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        xaxis=dict(range=[0, 11])
    )
    return fig

def create_comprehensive_chart(fetcher):
    """Create a comprehensive technical chart with indicators."""
    if fetcher.technical_data is None or fetcher.technical_data.empty:
        return go.Figure().update_layout(title_text="No technical data available.")

    df = fetcher.technical_data
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])

    # Candlestick Chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df.get('SMA_50'), name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.get('SMA_200'), name='SMA 200', line=dict(color='red', width=1.5)), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BB_Upper'), name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BB_Lower'), name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df.get('RSI'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green", line_width=1)

    # MACD
    fig.add_trace(go.Bar(x=df.index, y=df.get('MACD_Histogram'), name='Histogram', marker_color=np.where(df.get('MACD_Histogram', 0) > 0, 'green', 'red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.get('MACD'), name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.get('MACD_Signal'), name='Signal', line=dict(color='orange', width=1)), row=3, col=1)

    fig.update_layout(
        title=f'<b>Technical Analysis for {fetcher.symbol}</b>',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig

def display_comprehensive_analysis(fetcher, scores, final_score):
    """Display comprehensive analysis combining fundamental and technical analysis."""
    st.subheader(f"Analysis for {fetcher.info.get('longName', fetcher.symbol)}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", format_currency(fetcher.info.get('currentPrice'), fetcher.info.get('currency')), delta=f"{fetcher.info.get('regularMarketChangePercent', 0):.2f}%")
    col2.metric("Market Cap", f"{format_currency(fetcher.info.get('marketCap'), fetcher.info.get('currency'))}")
    col3.metric("Sector", f"{fetcher.info.get('sector', 'N/A')}")

    st.markdown("---")

    # --- Main Tabs ---
    fund_tab, tech_tab, chart_tab = st.tabs(["📊 Fundamental Analysis", "📈 Technical Signals", "📉 Charts & Data"])

    with fund_tab:
        st.subheader("Fundamental Score")
        # Clamp the value between 0.0 and 1.0 for st.progress
        progress_value = max(0.0, min(1.0, final_score / 10.0))
        st.progress(progress_value)
        st.markdown(f"### Final Score: **{final_score:.2f} / 10**")

        st.plotly_chart(create_enhanced_score_chart(scores, fetcher.symbol), use_container_width=True)
        
        with st.expander("View Raw Fundamental Data"):
            st.json(fetcher.info)

    with tech_tab:
        st.subheader("Technical Indicators")
        if fetcher.technical_data is not None and not fetcher.technical_data.empty:
            latest_tech = fetcher.technical_data.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("RSI", f"{latest_tech.get('RSI', 0):.2f}")
            c2.metric("SMA 50", format_currency(latest_tech.get('SMA_50'), fetcher.info.get('currency')))
            c3.metric("SMA 200", format_currency(latest_tech.get('SMA_200'), fetcher.info.get('currency')))
        else:
            st.warning("Technical data not available.")

    with chart_tab:
        st.plotly_chart(create_comprehensive_chart(fetcher), use_container_width=True)

def display_single_stock_analysis():
    """UI for the single stock analysis tab."""
    st.header("Single Stock Analysis")
    
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, NOVO-B.CO)", "AAPL").upper()

    if st.button("Analyze"):
        if not symbol:
            st.warning("Please enter a stock symbol.")
            return

        with st.spinner(f"Analyzing {symbol}..."):
            # 1. Fetch data
            fetcher = StockDataFetcher(symbol)
            success = fetcher.fetch_all_data()

            if not success:
                st.error(f"Could not retrieve data for {symbol}. Please check the symbol and try again.")
                return

            # 2. Calculate scores
            calculator = ScoreCalculator()
            # Pass the fetched info directly to the calculator
            scores, final_score = calculator.calculate_total_score(fetcher.info)

            # 3. Calculate technical indicators
            # This part is missing, let's assume it's in the fetcher or needs to be called
            # fetcher.calculate_technical_indicators() # You would need to implement this

            # 4. Display results
            display_comprehensive_analysis(fetcher, scores, final_score)

def display_portfolio_manager():
    """Displays the portfolio management interface."""
    st.title("💼 Portfolio Manager")

    portfolio_manager = st.session_state.portfolio_manager

    # --- Single, unified form for adding stocks ---
    with st.form("add_stock_form"):
        new_stock = st.text_input("Add Stock Symbol (e.g., AAPL, ORSTED)")
        submitted = st.form_submit_button("Add Stock")
        if submitted and new_stock:
            if portfolio_manager.add_stock(new_stock.upper()):
                st.success(f"Added {new_stock.upper()} to your portfolio.")
                st.rerun()
            else:
                st.error(f"{new_stock.upper()} is already in your portfolio or could not be validated.")

    st.markdown("---")

    # --- Display Current Portfolio ---
    st.subheader("Current Portfolio")
    portfolio = portfolio_manager.get_portfolio()

    if not portfolio:
        st.info("Your portfolio is empty. Add stocks using the form above to get started.", icon="ℹ️")
    else:
        # Display the stocks in a clear list with remove buttons
        for stock_symbol in sorted(portfolio):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<h4>{stock_symbol}</h4>", unsafe_allow_html=True)
            with col2:
                if st.button(f"Remove", key=f"remove_{stock_symbol}"):
                    portfolio_manager.remove_stock(stock_symbol)
                    st.success(f"Removed {stock_symbol} from portfolio.")
                    st.rerun()
        
        st.markdown("---")

        # --- Portfolio-level actions ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sync with Monitor")
            if st.button("Sync Now", help="Saves the current portfolio to portfolio_config.json for the automated monitor."):
                if portfolio_manager.save_portfolio_to_json():
                    st.success("Portfolio synced successfully!")
                else:
                    st.error("Failed to sync portfolio.")

        with col2:
            st.subheader("Analyze Full Portfolio")
            if st.button("Analyze Portfolio"):
                with st.spinner("Fetching data and analyzing portfolio..."):
                    fetcher = PortfolioDataFetcher(portfolio)
                    all_data = fetcher.fetch_all_data()

                    if not all_data:
                        st.warning("Could not fetch complete data for the portfolio analysis.")
                    else:
                        # Display analysis results
                        st.success("Analysis complete!")
                        df = pd.DataFrame.from_dict(all_data, orient='index')
                        st.dataframe(df[['longName', 'sector', 'country', 'regularMarketPrice', 'marketCap']].head(10))


def display_market_screener():
    """UI for the Market Screener tab."""
    st.header("📈 Market Screener")

    from app.config import DANISH_STOCKS, SP500_STOCKS, NASDAQ100_STOCKS, EUROPEAN_STOCKS

    market_options = {
        "🇩🇰 Danish Stocks": list(DANISH_STOCKS.values()),
        "🇺🇸 S&P 500": SP500_STOCKS,
        "💻 NASDAQ 100": NASDAQ100_STOCKS,
        "🇪🇺 European Stocks": EUROPEAN_STOCKS
    }

    selected_market = st.selectbox("Select Market to Screen", list(market_options.keys()))
    
    num_stocks_to_screen = st.slider("Number of stocks to screen", 5, 50, 10)

    if st.button("Run Screener"):
        stock_list = market_options[selected_market][:num_stocks_to_screen]
        
        st.info(f"Screening {len(stock_list)} stocks from {selected_market} asynchronously...")
        
        with st.spinner("Fetching data in parallel... this will be fast!"):
            fetcher = StockDataFetcher()
            # Run the async batch fetch
            batch_data = asyncio.run(fetcher.fetch_and_process_batch(stock_list))

        st.info("Calculating scores...")
        results = []
        calculator = ScoreCalculator()
        progress_bar = st.progress(0)
        
        if not batch_data:
            st.warning("No data was returned from the fetcher. The market may be closed or symbols are invalid.")
            return

        total_stocks = len(batch_data)
        for i, data in enumerate(batch_data):
            try:
                if data and data.get('info'):
                    scores, final_score = calculator.calculate_total_score(data['info'])
                    results.append({
                        'Symbol': data['symbol'],
                        'Company': data['info'].get('longName', 'N/A'),
                        'Score': final_score,
                        'Price': data['info'].get('currentPrice', 0),
                        'Currency': data['info'].get('currency', 'USD'),
                        'Sector': data['info'].get('sector', 'N/A'),
                        'P/E': data['info'].get('trailingPE', 0)
                    })
            except Exception as e:
                st.warning(f"Could not process {data.get('symbol', 'Unknown')}: {e}")
            
            progress_bar.progress((i + 1) / total_stocks)

        if results:
            df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
            
            # Formatting for display
            df_display = df.copy()
            df_display['Price'] = df.apply(lambda row: format_currency(row['Price'], row['Currency']), axis=1)
            df_display['Score'] = df['Score'].apply(lambda x: f"{x:.2f}")
            df_display['P/E'] = df['P/E'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(df_display[['Symbol', 'Company', 'Score', 'Price', 'P/E', 'Sector']], use_container_width=True)
        else:
            st.warning("No results found. The market may be closed or data is unavailable.")


def display_main_dashboard():
    """The main dashboard UI."""
    st.title("Advanced Stock Analysis System v7")
    st.markdown("Refactored for performance, maintainability, and enhanced scoring.")

    # ... (rebuild your UI here by calling functions from other modules)
    
    # Example:
    # from app.portfolio import display_portfolio_manager
    # from app.screener import display_screener
    
    tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Portfolio Manager", "Market Screener"])

    with tab1:
        display_single_stock_analysis()
    
    with tab2:
        display_portfolio_manager()
        
    with tab3:
        st.header("Market Screener")
        display_market_screener()

def display_main_dashboard():
    """Sets up the main dashboard with tabs."""
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio("Go to", ["Single Stock Analysis", "Portfolio Manager", "Market Screener"])

    if selected_tab == "Single Stock Analysis":
        display_single_stock_analysis()
    elif selected_tab == "Portfolio Manager":
        display_portfolio_manager()
    elif selected_tab == "Market Screener":
        display_market_screener()
