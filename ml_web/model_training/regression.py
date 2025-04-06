import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time

from model_training.evaluation import split_data, evaluate_regression_model, plot_feature_importance

def render_regression_page(X, y):
    """Render the regression model training page"""
    st.subheader("Regression Model Training")
    
    # Let users know what the target variable looks like
    st.write("Target Variable Preview:")
    st.write(pd.DataFrame({
        'Statistic': ['Min', 'Max', 'Mean', 'Std Dev'],
        'Value': [y.min(), y.max(), y.mean(), y.std()]
    }))
    
    # Show target distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y, bins=30, alpha=0.7)
    ax.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
    ax.axvline(y.median(), color='green', linestyle='--', label=f'Median: {y.median():.2f}')
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Target Variable Distribution')
    ax.legend()
    st.pyplot(fig)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Model selection
    st.subheader("Model Selection")
    
    model_option = st.selectbox("Select a regression model:", [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Support Vector Regression (SVR)",
        "XGBoost"
    ])
    
    # Hyperparameter settings
    st.subheader("Hyperparameters")
    
    if model_option == "Linear Regression":
        model = LinearRegression()
    
    elif model_option == "Ridge Regression":
        alpha = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0, 0.01)
        solver = st.selectbox("Solver:", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        
        model = Ridge(
            alpha=alpha,
            solver=solver
        )
    
    elif model_option == "Lasso Regression":
        alpha = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.slider("Maximum iterations:", 100, 3000, 1000, 100)
        
        model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            random_state=42
        )
    
    elif model_option == "Elastic Net":
        alpha = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0, 0.01)
        l1_ratio = st.slider("L1 ratio:", 0.0, 1.0, 0.5, 0.01)
        max_iter = st.slider("Maximum iterations:", 100, 3000, 1000, 100)
        
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=42
        )
    
    elif model_option == "Decision Tree":
        max_depth = st.slider("Maximum depth:", 1, 30, 5, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Minimum samples in leaf:", 1, 20, 1, 1)
        
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    elif model_option == "Random Forest":
        n_estimators = st.slider("Number of trees:", 10, 300, 100, 10)
        max_depth = st.slider("Maximum depth:", 1, 30, 5, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_option == "Gradient Boosting":
        n_estimators = st.slider("Number of boosting stages:", 10, 300, 100, 10)
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.slider("Maximum depth:", 1, 10, 3, 1)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_option == "Support Vector Regression (SVR)":
        kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
        C = st.slider("Regularization parameter (C):", 0.1, 10.0, 1.0, 0.1)
        epsilon = st.slider("Epsilon:", 0.01, 1.0, 0.1, 0.01)
        
        model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon
        )
    
    elif model_option == "XGBoost":
        n_estimators = st.slider("Number of boosting rounds:", 10, 300, 100, 10)
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.slider("Maximum depth:", 1, 10, 3, 1)
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            st.success(f"Model trained in {training_time:.2f} seconds!")
            
            # Evaluate the model
            metrics = evaluate_regression_model(model, X_test, y_test)
            
            # Plot feature importance for tree-based models
            plot_feature_importance(model, X.columns)
