import pandas as pd
import numpy as np
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

if not os.path.exists('models'):
    os.makedirs('models')

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("test_data.csv saved. Use this file to test your Streamlit app.")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("Training Models...")
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results[name] = [acc, auc, prec, rec, f1, mcc]
    
    # Save Model
    filename = f'models/{name.replace(" ", "_").lower()}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} to {filename}")

# 4. Print Comparison Table for README
print("\n=== Model Performance (Copy this to your README) ===")
print(f"{'Model':<20} | {'Acc':<5} | {'AUC':<5} | {'Prec':<5} | {'Rec':<5} | {'F1':<5} | {'MCC':<5}")
print("-" * 75)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics[0]:.3f} | {metrics[1]:.3f} | {metrics[2]:.3f} | {metrics[3]:.3f} | {metrics[4]:.3f} | {metrics[5]:.3f}")