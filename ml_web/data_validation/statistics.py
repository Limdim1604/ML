import streamlit as st
import pandas as pd
import numpy as np

def render_statistics(data):
    """Display basic statistics for the dataset"""
    st.subheader("Basic Statistics")
    st.write(data.describe())
    
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    
    # Show percentage of missing values
    missing_percent = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percent
    })
    st.write(missing_df[missing_df['Missing Values'] > 0])
    
    # Data Types
    st.subheader("Data Types")
    data_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    st.write(data_types)
    
    # Summary of categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    if not categorical_columns.empty:
        st.subheader("Categorical Columns Summary")
        for col in categorical_columns:
            st.write(f"**{col}** - Unique values: {data[col].nunique()}")
            st.write(data[col].value_counts())
