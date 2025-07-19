# main_app.py
import streamlit as st
import warnings
from app.ui_components import display_main_dashboard
from app.portfolio import PortfolioManager

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="AS System v7")

def main():
    """
    Main function to run the Streamlit app.
    The database is initialized automatically by the PortfolioManager.
    """
    # Initialize PortfolioManager in session state if it doesn't exist
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()

    # Display the main user interface
    display_main_dashboard()

if __name__ == "__main__":
    main()
