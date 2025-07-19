# main_app.py
import streamlit as st
import warnings
from app.ui_components import display_main_dashboard
from app.portfolio import init_db

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="AS System v7")

def main():
    """
    Main function to run the Streamlit app.
    Initializes the database and displays the main dashboard.
    """
    # Initialize the database on first run
    init_db()
    
    # Display the main user interface
    display_main_dashboard()

if __name__ == "__main__":
    main()
