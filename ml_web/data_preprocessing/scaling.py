import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_features(data):
    """Apply feature scaling to numeric columns"""
    # Get numeric columns
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_columns:
        st.info("No numeric columns found for scaling.")
        return data
    
    # Select columns to scale
    st.write("Select columns to scale:")
    cols_to_scale = []
    for col in numeric_columns:
        if st.checkbox(f"Scale {col}", value=True):
            cols_to_scale.append(col)
    
    if not cols_to_scale:
        st.info("No columns selected for scaling.")
        return data
    
    # Choose scaling method
    scaling_option = st.radio("Choose scaling method:", 
                             ["StandardScaler", "MinMaxScaler", "RobustScaler", "Skip"])
    
    if scaling_option == "Skip":
        return data
    
    # Create a copy to avoid modifying the original data
    scaled_data = data.copy()
    
    # Apply selected scaler
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
        scaled_data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        st.success(f"StandardScaler applied to {len(cols_to_scale)} column(s)")
        
    elif scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        scaled_data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        st.success(f"MinMaxScaler applied to {len(cols_to_scale)} column(s)")
        
    elif scaling_option == "RobustScaler":
        scaler = RobustScaler()
        scaled_data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        st.success(f"RobustScaler applied to {len(cols_to_scale)} column(s)")
    
    # Show before and after statistics
    st.write("### Before and After Scaling")
    
    for col in cols_to_scale[:3]:  # Show first 3 columns to avoid clutter
        st.write(f"**{col}**")
        before_after = pd.DataFrame({
            'Before': [data[col].min(), data[col].max(), data[col].mean(), data[col].std()],
            'After': [scaled_data[col].min(), scaled_data[col].max(), scaled_data[col].mean(), scaled_data[col].std()]
        }, index=['Min', 'Max', 'Mean', 'Std'])
        st.write(before_after)
    
    if len(cols_to_scale) > 3:
        st.info(f"Statistics for {len(cols_to_scale) - 3} more columns are not shown.")
    
    return scaled_data
