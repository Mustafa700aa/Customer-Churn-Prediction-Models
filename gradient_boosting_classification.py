import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import joblib
import pickle

# Load data
data = pd.read_excel('Telco_customer_churn.xlsx')

# Clean data
data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
data = data.dropna(subset=['Total Charges'])

# Feature Engineering
data['Charge Ratio'] = data['Monthly Charges'] / (data['Total Charges'] + 1e-6)
data['Tenure Category'] = pd.cut(data['Tenure Months'], bins=[0, 12, 24, 36, 48, 60, 72], labels=[1, 2, 3, 4, 5, 6])
data['Num Services'] = (
    data['Phone Service'].map({'Yes': 1, 'No': 0}) +
    data['Multiple Lines'].map({'Yes': 1, 'No': 0, 'No phone service': 0}) +
    data['Internet Service'].map({'DSL': 1, 'Fiber optic': 1, 'No': 0}) +
    data['Online Security'].map({'Yes': 1, 'No': 0, 'No internet service': 0}) +
    data['Online Backup'].map({'Yes': 1, 'No': 0, 'No internet service': 0}) +
    data['Device Protection'].map({'Yes': 1, 'No': 0, 'No internet service': 0}) +
    data['Tech Support'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
)
data['Contract_Tenure'] = data['Contract'] + '_' + data['Tenure Category'].astype(str)

# Define variables
categorical_cols = [
    'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
    'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support', 'Contract', 'Payment Method', 
    'Tenure Category', 'Contract_Tenure'
]
numerical_cols = ['Monthly Charges', 'Tenure Months', 'Charge Ratio', 'Num Services', 'Churn Score', 'CLTV']

# One-Hot Encoding
data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
X = pd.concat([data_encoded, data[numerical_cols]], axis=1)
y = data['Churn Label'].map({'Yes': 1, 'No': 0})

# Remove low-importance features
low_importance_features = ['Gender_Male', 'Paperless Billing_Yes', 'Streaming TV_Yes', 'Streaming Movies_Yes']
X = X.drop(columns=[col for col in low_importance_features if col in X.columns], errors='ignore')

# Save column names
feature_columns = X.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("Feature Columns saved")

# Split data
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42, stratify=y_temp)

# Standardization
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
X_holdout[numerical_cols] = scaler.transform(X_holdout[numerical_cols])

# SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Compute sample weights
sample_weights = np.where(y_train_smote == 1, 2.0, 1.0)

# Train model with best parameters
gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=2, min_samples_split=5,
    min_samples_leaf=10, subsample=0.6, random_state=42, n_iter_no_change=20, validation_fraction=0.05
)

# Feature Selection with RFE
rfe = RFE(estimator=gb, n_features_to_select=15)
rfe.fit(X_train_smote, y_train_smote, sample_weight=sample_weights)
X_train_smote = X_train_smote[X_train_smote.columns[rfe.support_]]
X_test = X_test[X_test.columns[rfe.support_]]
X_holdout = X_holdout[X_holdout.columns[rfe.support_]]

# Train model
gb.fit(X_train_smote, y_train_smote, sample_weight=sample_weights)

# Evaluate on Test Data (Optimal Threshold = 0.6420)
optimal_threshold = 0.6420
y_test_prob = gb.predict_proba(X_test)[:, 1]
y_test_pred_optimal = (y_test_prob >= optimal_threshold).astype(int)
print("\nTest Classification Report (Optimal Threshold):")
print(classification_report(y_test, y_test_pred_optimal))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_prob))

# Evaluate on Holdout Set (Optimal Threshold = 0.6420)
y_holdout_prob = gb.predict_proba(X_holdout)[:, 1]
y_holdout_pred_optimal = (y_holdout_prob >= optimal_threshold).astype(int)
print("\nHoldout Set Classification Report (Optimal Threshold):")
print(classification_report(y_holdout, y_holdout_pred_optimal))
print("Holdout Set ROC-AUC:", roc_auc_score(y_holdout, y_holdout_prob))

# Save model and unique Scaler
joblib.dump(gb, 'best_gb_model.pkl')
joblib.dump(scaler, 'scaler_gb.pkl')
print("\nModel saved as 'best_gb_model.pkl'")
print("Scaler saved as 'scaler_gb.pkl'")