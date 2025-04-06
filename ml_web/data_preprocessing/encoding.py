import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def encode_categorical(data):
    """Encode categorical variables in the dataset"""
    # Get categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        st.info("No categorical columns found for encoding.")
        return data
    
    # Select columns to encode
    st.write("Select columns to encode:")
    cols_to_encode = []
    for col in categorical_columns:
        if st.checkbox(f"Encode {col}", value=True):
            cols_to_encode.append(col)
    
    if not cols_to_encode:
        st.info("No columns selected for encoding.")
        return data
    
    # Choose encoding method
    encoding_option = st.radio("Choose encoding method:", 
                             ["One-hot encoding", "Label encoding", "Ordinal encoding", "Skip"])
    
    if encoding_option == "Skip":
        return data
        
    # Create a copy to avoid modifying the original data
    encoded_data = data.copy()
    
    # Apply selected encoding method
    if encoding_option == "One-hot encoding":
        # Ask if we should drop the first category to avoid dummy variable trap
        drop_first = st.checkbox("Drop first category (recommended for modeling)", value=True)
        
        # Create dummies for each selected column
        for col in cols_to_encode:
            # Get dummies and drop the original column
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=drop_first)
            encoded_data = pd.concat([encoded_data.drop(col, axis=1), dummies], axis=1)
            st.info(f"One-hot encoded '{col}' into {dummies.shape[1]} columns")
    
    elif encoding_option == "Label encoding":
        # Apply label encoder to each selected column
        label_mappings = {}
        for col in cols_to_encode:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(data[col].astype(str))
            
            # Store label mappings
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            label_mappings[col] = mapping
            
            st.info(f"Label encoded '{col}'")
        
        # Display mappings
        st.write("### Label Encoding Mappings")
        for col, mapping in label_mappings.items():
            st.write(f"**{col}**")
            st.json(mapping)
    
    elif encoding_option == "Ordinal encoding":
        # For each column, let user specify the order
        ordinal_mappings = {}
        
        for col in cols_to_encode:
            unique_values = sorted(data[col].unique())
            st.write(f"**Specify order for '{col}'**")
            st.write("Drag to reorder (first item = 0, last item = n-1)")
            
            # Use a text input for simplicity in this example
            # In a real app, you'd want a drag-and-drop interface
            default_order = ", ".join([str(val) for val in unique_values])
            order_input = st.text_input(f"Order for {col} (comma-separated)", value=default_order)
            
            try:
                # Parse the order
                ordered_categories = [category.strip() for category in order_input.split(',')]
                
                # Create mapping
                mapping = {category: idx for idx, category in enumerate(ordered_categories)}
                ordinal_mappings[col] = mapping
                
                # Apply ordinal encoding
                encoded_data[col] = data[col].map(mapping)
                st.info(f"Ordinal encoded '{col}'")
            except Exception as e:
                st.error(f"Error encoding {col}: {e}")
        
        # Display mappings
        st.write("### Ordinal Encoding Mappings")
        for col, mapping in ordinal_mappings.items():
            st.write(f"**{col}**")
            st.json(mapping)
    
    return encoded_data
