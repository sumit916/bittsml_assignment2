import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# 1. Load Dataset (Replace 'dataset.csv' with your actual file name)
# Ensure your dataset is in the same folder or update path
df = pd.read_csv('dataset.csv')

# 2. Preprocessing
# Drop ID columns if they exist (customize this list based on your data)
# df = df.drop(columns=['CustomerID', 'Name'], errors='ignore')

# Handle Missing Values
df = df.dropna()

# Encode Categorical Variables (Target and Features)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Define X (Features) and y (Target)
# ASSUMPTION: The LAST column is the Target. Change if needed.
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale Features (Crucial for KNN and Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 4. Train, Evaluate, and Save
results = {}

print("Training Models and Saving...")
print("-" * 30)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred # For AUC

    # Calculate Metrics [cite: 40-46]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y)) == 2 else 0, # AUC mostly for binary
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
    results[name] = metrics

    # Save Model
    # We save a pipeline-like structure or just the model.
    # ideally save scaler too, but for simplicity here we save just model.
    with open(f'model/{name.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

print("Training Complete! Models saved in 'model/' folder.")

# 5. Print Table for README [cite: 70]
print("\n=== METRICS TABLE (Copy to README) ===")
results_df = pd.DataFrame(results).T
print(results_df)

# Save scaler for the app to use
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)