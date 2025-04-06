import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time

from model_training.evaluation import split_data, evaluate_classification_model, plot_feature_importance

def render_classification_page(X, y):
    """Render the classification model training page"""
    st.subheader("Classification Model Training")
    
    # Let users know what the target variable looks like
    st.write("Target Variable Preview:")
    st.write(pd.DataFrame({
        'Target': y.value_counts().index,
        'Count': y.value_counts().values
    }))
    
    # If binary classification, show class distribution
    if len(np.unique(y)) == 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(['Class 0', 'Class 1'], [
            (y == 0).sum() / len(y) * 100, 
            (y == 1).sum() / len(y) * 100
        ])
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Class Distribution')
        for i, v in enumerate([(y == 0).sum(), (y == 1).sum()]):
            ax.text(i, 5, f"{v} samples", ha='center')
        st.pyplot(fig)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Model selection
    st.subheader("Model Selection")
    
    model_option = st.selectbox("Select a classification model:", [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbors (KNN)",
        "XGBoost"
    ])
    
    # Hyperparameter settings
    st.subheader("Hyperparameters")
    
    if model_option == "Logistic Regression":
        solver = st.selectbox("Solver:", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
        C = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.slider("Maximum iterations:", 100, 2000, 1000, 100)
        
        model = LogisticRegression(
            solver=solver,
            C=C,
            max_iter=max_iter,
            random_state=42
        )
    
    elif model_option == "Decision Tree":
        max_depth = st.slider("Maximum depth:", 1, 30, 5, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Minimum samples in leaf:", 1, 20, 1, 1)
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    elif model_option == "Random Forest":
        n_estimators = st.slider("Number of trees:", 10, 300, 100, 10)
        max_depth = st.slider("Maximum depth:", 1, 30, 5, 1)
        min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2, 1)
        
        model = RandomForestClassifier(
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
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_option == "Support Vector Machine (SVM)":
        kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
        C = st.slider("Regularization parameter (C):", 0.1, 10.0, 1.0, 0.1)
        
        model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=42
        )
    
    elif model_option == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Number of neighbors:", 1, 20, 5, 1)
        weights = st.selectbox("Weight function:", ["uniform", "distance"])
        
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1
        )
    
    elif model_option == "XGBoost":
        n_estimators = st.slider("Number of boosting rounds:", 10, 300, 100, 10)
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.slider("Maximum depth:", 1, 10, 3, 1)
        
        model = XGBClassifier(
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
            metrics = evaluate_classification_model(model, X_test, y_test)
            
            # If binary classification, plot ROC curve
            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                st.subheader("ROC Curve")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)
            
            # Plot feature importance for tree-based models
            plot_feature_importance(model, X.columns)
