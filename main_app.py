# main_app.py
import streamlit as st
import warnings
from app.ui_components import display_main_dashboard

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="AS System v7")

def main():
    """
    Main function to run the Streamlit app.
    The database is initialized automatically by the PortfolioManager.
    """
    # Display the main user interface
    display_main_dashboard()

if __name__ == "__main__":
    main()
