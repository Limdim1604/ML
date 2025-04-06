import streamlit as st
import pandas as pd
import numpy as np

def handle_missing_values(data):
    """Handle missing values in the dataset"""
    # Check if there are any missing values
    if data.isnull().sum().sum() == 0:
        st.info("No missing values found in the dataset.")
        return data
    
    # Display missing value counts
    missing_values = data.isnull().sum()
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': (missing_values / len(data)) * 100
    })
    st.write(missing_df[missing_df['Missing Values'] > 0])
    
    # Choose missing value strategy
    missing_strategy = st.radio("Choose strategy for missing values:", 
                              ["Remove rows", "Fill with mean/mode", "Fill with median", "Fill with a constant value", "Skip"])
    
    if missing_strategy == "Remove rows":
        original_shape = data.shape
        data = data.dropna()
        st.success(f"Removed rows with missing values. Rows before: {original_shape[0]}, Rows after: {data.shape[0]}")
    
    elif missing_strategy == "Fill with mean/mode":
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if data[column].dtype in [np.number]:
                    data[column] = data[column].fillna(data[column].mean())
                    st.info(f"Filled missing values in '{column}' with mean: {data[column].mean():.2f}")
                else:
                    mode_value = data[column].mode()[0]
                    data[column] = data[column].fillna(mode_value)
                    st.info(f"Filled missing values in '{column}' with mode: {mode_value}")
    
    elif missing_strategy == "Fill with median":
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if data[column].dtype in [np.number]:
                    data[column] = data[column].fillna(data[column].median())
                    st.info(f"Filled missing values in '{column}' with median: {data[column].median():.2f}")
                else:
                    mode_value = data[column].mode()[0]
                    data[column] = data[column].fillna(mode_value)
                    st.info(f"Filled missing values in '{column}' with mode: {mode_value}")
    
    elif missing_strategy == "Fill with a constant value":
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                st.subheader(f"Fill missing values in '{column}'")
                if data[column].dtype in [np.number]:
                    fill_value = st.number_input(f"Enter value for {column}:", value=0.0)
                    data[column] = data[column].fillna(fill_value)
                else:
                    fill_value = st.text_input(f"Enter value for {column}:", value="unknown")
                    data[column] = data[column].fillna(fill_value)
    
    if missing_strategy != "Skip" and data.isnull().sum().sum() > 0:
        st.warning("There are still missing values in the dataset.")
    
    return data
