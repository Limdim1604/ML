import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def render_visualizations(data):
    """Render various data visualizations"""
    st.subheader("Data Visualization")
    
    # Select visualization type
    viz_option = st.selectbox("Select visualization type:", 
                             ["Distribution Plot", "Correlation Matrix", "Box Plot", "Scatter Plot", "Pair Plot"])
    
    if viz_option == "Distribution Plot":
        render_distribution_plot(data)
            
    elif viz_option == "Correlation Matrix":
        render_correlation_matrix(data)
            
    elif viz_option == "Box Plot":
        render_box_plot(data)
            
    elif viz_option == "Scatter Plot":
        render_scatter_plot(data)
        
    elif viz_option == "Pair Plot":
        render_pair_plot(data)

def render_distribution_plot(data):
    """Render distribution plot for selected column"""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for distribution plot.")
        return
        
    col = st.selectbox("Select column:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[col], kde=True, ax=ax)
    plt.title(f'Distribution of {col}')
    st.pyplot(fig)

def render_correlation_matrix(data):
    """Render correlation matrix for numeric columns"""
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.empty:
        st.warning("No numeric columns available for correlation matrix.")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = numeric_data.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

def render_box_plot(data):
    """Render box plot for selected column"""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for box plot.")
        return
        
    col = st.selectbox("Select column for box plot:", numeric_cols)
    
    # Option to group by a categorical column
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    use_categorical = st.checkbox("Group by categorical variable", value=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if use_categorical and categorical_cols:
        group_by = st.selectbox("Group by:", categorical_cols)
        sns.boxplot(data=data, x=group_by, y=col, ax=ax)
        plt.title(f'Box Plot of {col} grouped by {group_by}')
    else:
        sns.boxplot(data=data, y=col, ax=ax)
        plt.title(f'Box Plot of {col}')
        
    st.pyplot(fig)

def render_scatter_plot(data):
    """Render scatter plot for selected columns"""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot.")
        return
        
    col1 = st.selectbox("Select X-axis column:", numeric_cols)
    col2 = st.selectbox("Select Y-axis column:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    # Option to color by a categorical column
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    use_color = st.checkbox("Color by categorical variable", value=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if use_color and categorical_cols:
        color_by = st.selectbox("Color by:", categorical_cols)
        sns.scatterplot(data=data, x=col1, y=col2, hue=color_by, ax=ax)
        plt.title(f'Scatter Plot: {col1} vs {col2}, colored by {color_by}')
    else:
        sns.scatterplot(data=data, x=col1, y=col2, ax=ax)
        plt.title(f'Scatter Plot: {col1} vs {col2}')
        
    st.pyplot(fig)

def render_pair_plot(data):
    """Render pair plot for selected columns"""
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for pair plot.")
        return
    
    # Let user select columns to include (max 5 for performance)
    st.write("Select columns to include (max 5 recommended for performance):")
    selected_cols = []
    for i, col in enumerate(numeric_cols[:10]):  # Limit to first 10 columns
        if st.checkbox(col, value=(i < 3)):  # Default select first 3
            selected_cols.append(col)
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return
    
    if len(selected_cols) > 5:
        st.warning("You selected more than 5 columns. This might be slow to render.")
    
    # Optional: color by categorical variable
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    hue_col = None
    if categorical_cols:
        use_hue = st.checkbox("Color by categorical variable", value=False)
        if use_hue:
            hue_col = st.selectbox("Color by:", categorical_cols)
    
    with st.spinner("Generating pair plot..."):
        fig = sns.pairplot(data[selected_cols + ([hue_col] if hue_col else [])], 
                          hue=hue_col,
                          diag_kind='kde')
        fig.fig.suptitle('Pair Plot', y=1.02)
        st.pyplot(fig)
