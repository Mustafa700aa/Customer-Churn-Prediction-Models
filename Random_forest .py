import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import pickle

# تحميل الداتا
data = pd.read_excel('Telco_customer_churn.xlsx')

# تنظيف الداتا
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
    data['Tech Support'].map({'Yes': 1, 'No': 0, 'No internet service': 0}) +
    data['Streaming TV'].map({'Yes': 1, 'No': 0, 'No internet service': 0}) +
    data['Streaming Movies'].map({'Yes': 1, 'No': 0, 'No internet service': 0})
)

# تحديد المتغيرات
categorical_cols = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
    'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
    'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
    'Contract', 'Paperless Billing', 'Payment Method', 'Tenure Category'
]
numerical_cols = ['Monthly Charges', 'Total Charges', 'Tenure Months', 'Charge Ratio', 'Num Services', 'Churn Score', 'CLTV']

# One-Hot Encoding
data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
X = pd.concat([data_encoded, data[numerical_cols]], axis=1)
y = data['Churn Label'].map({'Yes': 1, 'No': 0})

# حفظ أسماء الأعمدة
feature_columns = X.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("Feature Columns saved as 'feature_columns.pkl'")

# تقسيم الداتا
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardization
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# SMOTE
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Grid Search
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}
grid = GridSearchCV(rf, param_grid, cv=5, scoring='recall', n_jobs=4)
grid.fit(X_train_smote, y_train_smote)

# أفضل موديل
best_rf = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# تدريب أفضل موديل
best_rf.fit(X_train_smote, y_train_smote)

# تقييم على الـ Training Data
y_train_pred = best_rf.predict(X_train_smote)
y_train_prob = best_rf.predict_proba(X_train_smote)[:, 1]
print("\nTraining Classification Report:")
print(classification_report(y_train_smote, y_train_pred))
print("Training ROC-AUC:", roc_auc_score(y_train_smote, y_train_prob))

# تقييم على الـ Test Data (Default Threshold)
y_test_pred = best_rf.predict(X_test)
y_test_prob = best_rf.predict_proba(X_test)[:, 1]
print("\nTest Classification Report (Default Threshold):")
print(classification_report(y_test, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_prob))

# إيجاد Optimal Threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)
optimal_idx = np.where(recalls >= 0.85)[0]
thresholds = thresholds[optimal_idx]
precisions = precisions[optimal_idx]
optimal_threshold = thresholds[np.argmax(precisions)]
print(f"\nOptimal Threshold: {optimal_threshold:.4f}, Precision: {precisions[np.argmax(precisions)]:.4f}, Recall: {recalls[np.argmax(precisions)]:.4f}")

# تقييم عند Optimal Threshold
y_test_pred_optimal = (y_test_prob >= optimal_threshold).astype(int)
print("\nTest Classification Report (Optimal Threshold):")
print(classification_report(y_test, y_test_pred_optimal))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance.head(10))

# Cross-Validation
cv_scores = cross_val_score(best_rf, X_train_smote, y_train_smote, cv=5, scoring='recall')
print("\nCross-Validation Recall:", cv_scores.mean(), "+/-", cv_scores.std())
cv_scores_auc = cross_val_score(best_rf, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print("\nCross-Validation ROC-AUC:", cv_scores_auc.mean(), "+/-", cv_scores_auc.std())

# حفظ الموديل والـ Scaler
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler_rf.pkl')
print("\nModel saved as 'best_rf_model.pkl'")
print("Scaler saved as 'scaler_rf.pkl'")