import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, mean_squared_error,
                           mean_absolute_error, r2_score)

def determine_problem_type(y):
    """Determine if it's a classification or regression problem"""
    # Check number of unique values
    unique_values = len(np.unique(y))
    
    # If less than 10 unique values or data type is object/boolean, assume classification
    if unique_values <= 10 or y.dtype == 'object' or y.dtype == 'bool':
        problem_type = st.radio("Problem type:", ["Classification", "Regression"], index=0)
    else:
        problem_type = st.radio("Problem type:", ["Classification", "Regression"], index=1)
    
    return problem_type

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    st.subheader("Train-Test Split")
    test_size = st.slider("Test size ratio:", 0.1, 0.5, test_size, 0.05)
    random_state = st.number_input("Random state:", value=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    st.success(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test

def evaluate_classification_model(model, X_test, y_test, class_names=None):
    """Evaluate classification model performance"""
    y_pred = model.predict(X_test)
    
    st.subheader("Model Evaluation")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Accuracy: {accuracy:.4f}")
    
    # Multi-class or binary classification
    if len(np.unique(y_test)) > 2:
        # Multi-class
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        st.write(f"Precision (weighted): {precision:.4f}")
        st.write(f"Recall (weighted): {recall:.4f}")
        st.write(f"F1 Score (weighted): {f1:.4f}")
    else:
        # Binary classification
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
    
    # Classification report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if class_names:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate regression model performance"""
    y_pred = model.predict(X_test)
    
    st.subheader("Model Evaluation")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")
    
    # Predicted vs Actual plot
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    st.pyplot(fig)
    
    # Residual plot
    st.subheader("Residuals")
    residuals = y_test - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    st.pyplot(fig)
    
    # Residual distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    plt.xlabel('Residual')
    plt.title('Residual Distribution')
    st.pyplot(fig)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), ax=ax)
        plt.title('Top 15 Feature Importance')
        st.pyplot(fig)
        
        # Show the table of all features
        st.write("### Feature Importance Table")
        st.write(feature_importance)
