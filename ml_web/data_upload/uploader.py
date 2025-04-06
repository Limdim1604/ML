import streamlit as st
import pandas as pd
import os

def upload_file():
    """Handle file upload via Streamlit's file uploader widget"""
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success("Data uploaded successfully!")
            return data
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    return None

def load_sample_data():
    """Load the sample Heart.csv dataset"""
    try:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Heart.csv")
        data = pd.read_csv("Heart.csv")
        st.session_state.data = data
        st.success("Heart.csv loaded successfully!")
        return data
    except FileNotFoundError:
        st.error("Heart.csv not found. Please make sure the file exists in the application directory.")
        return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def render_upload_page():
    """Render the data upload page"""
    upload_option = st.radio("Choose data source:", ["Upload CSV", "Use Heart.csv sample"])
    
    if upload_option == "Upload CSV":
        data = upload_file()
    else:
        data = load_sample_data()
    
    if data is not None:
        st.write("### Data Preview")
        st.write(data.head())
        
        st.write("### Data Information")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")
