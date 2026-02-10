# Machine Learning Assignment 2

## a. Problem Statement
The objective of this assignment is to implement and compare multiple machine learning classification models to predict whether a breast cancer tumor is **Malignant** (harmful) or **Benign** (non-harmful) based on diagnostic measurements. The project involves building an end-to-end workflow including data preprocessing, model training, evaluation, and deployment of an interactive web application using Streamlit.

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
* **Source:** Scikit-Learn (Originally from UCI Machine Learning Repository)
* **Type:** Binary Classification
* **Features:** 30 numeric features (Meets minimum requirement of >12)
* **Instances:** 569 (Meets minimum requirement of >500)
* **Target Variable:** Diagnosis (0 = Malignant, 1 = Benign)

## c. Models Used
The following 6 classification models were implemented on the dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Evaluation Metrics Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.974 | 0.997 | 0.972 | 0.986 | 0.979 | 0.944 |
| **Decision Tree** | 0.939 | 0.932 | 0.944 | 0.958 | 0.951 | 0.869 |
| **KNN** | 0.947 | 0.982 | 0.958 | 0.958 | 0.958 | 0.888 |
| **Naive Bayes** | 0.965 | 0.997 | 0.959 | 0.986 | 0.972 | 0.925 |
| **Random Forest (Ensemble)** | 0.965 | 0.995 | 0.959 | 0.986 | 0.972 | 0.925 |
| **XGBoost (Ensemble)** | 0.956 | 0.991 | 0.958 | 0.972 | 0.965 | 0.906 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed well as the dataset is linearly separable. It achieved high accuracy and balanced precision/recall, making it a strong baseline. |
| **Decision Tree** | Showed slightly lower performance compared to others, likely due to overfitting on the training data. It struggled to generalize as well as the ensemble methods. |
| **KNN** | Performed well but computation time increases with dataset size. It was effective here due to distinct clusters in the data. |
| **Naive Bayes** | Surprisingly effective despite the assumption of feature independence. It provided a very high recall, which is crucial for medical diagnoses (minimizing false negatives). |
| **Random Forest (Ensemble)** | Outperformed single Decision Trees by reducing variance. It provided robust predictions and high AUC scores. |
| **XGBoost (Ensemble)** | Achieved the best overall performance with the highest MCC score. It effectively handled complex patterns in the feature space. |