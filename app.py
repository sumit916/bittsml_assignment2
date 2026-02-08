import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="ML Classifier App", layout="wide")

st.title(" Machine Learning Classification Dashboard")
st.markdown("Upload a CSV file to generate predictions using trained models.")

@st.cache_resource
def load_resources():
    resources = {}
    model_names = [
        "logistic_regression", "decision_tree", "knn", 
        "naive_bayes", "random_forest", "xgboost"
    ]
    
    # Load Models
    for name in model_names:
        try:
            with open(f'models/{name}.pkl', 'rb') as f:
                resources[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file 'models/{name}.pkl' not found. Please run train_model.py first.")
            return None

    # Load Scaler and Features
    try:
        with open('models/scaler.pkl', 'rb') as f:
            resources['scaler'] = pickle.load(f)
        with open('models/features.pkl', 'rb') as f:
            resources['features'] = pickle.load(f)
    except FileNotFoundError:
        st.error("Scaler or Feature list not found.")
        return None
        
    return resources

resources = load_resources()

if resources:
    # 2. Sidebar - Model Selection
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Model", 
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )
    
    # Map choice to file key
    key_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    selected_model = resources[key_map[model_choice]]

    # 3. Main Area - File Upload
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(data.head())

            # Data Preprocessing
            # Expecting 'target' column for metrics calculation
            if 'target' in data.columns:
                y_true = data['target']
                X = data.drop(columns=['target'])
            else:
                st.warning("No 'target' column found. Metrics cannot be calculated, only predictions will be shown.")
                X = data
                y_true = None

            # Ensure columns match training
            # (In a real app, you would add more robust checks here)
            if list(X.columns) != resources['features']:
                st.error("Feature mismatch! Please ensure your CSV has the same columns as the training data.")
            else:
                # Scale Data
                X_scaled = resources['scaler'].transform(X)

                # Prediction
                y_pred = selected_model.predict(X_scaled)
                
                # Try getting probabilities for AUC
                if hasattr(selected_model, "predict_proba"):
                    y_prob = selected_model.predict_proba(X_scaled)[:, 1]
                else:
                    y_prob = y_pred  # Fallback

                # Display Results
                st.subheader(f"Results using {model_choice}")
                
                # Metrics (Only if ground truth is available)
                if y_true is not None:
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                    col2.metric("AUC", f"{roc_auc_score(y_true, y_prob):.2f}")
                    col3.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
                    col4.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
                    col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")
                    col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")

                    # Confusion Matrix
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                
                # Show Predictions
                st.write("### Predictions")
                results_df = X.copy()
                results_df['Predicted_Class'] = y_pred
                st.dataframe(results_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")