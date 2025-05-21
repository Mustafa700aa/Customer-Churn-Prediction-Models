import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

# Load and clean data
df = pd.read_excel('Telco_customer_churn.xlsx')
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)

# Feature engineering
df['Charge Ratio'] = df['Monthly Charges'] / (df['Total Charges'] + 1e-6)
df['Tenure Category'] = pd.cut(df['Tenure Months'], [0, 12, 24, 36, 48, 60, 72], labels=[1, 2, 3, 4, 5, 6])
df['Num Services'] = sum(df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
                         for col in ['Phone Service', 'Multiple Lines', 'Online Security', 'Online Backup', 
                                     'Device Protection', 'Tech Support'])
df['Num Services'] += df['Internet Service'].map({'DSL': 1, 'Fiber optic': 1, 'No': 0})
df['Contract_Tenure'] = df['Contract'] + '_' + df['Tenure Category'].astype(str)

# Select features
cat_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
            'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
            'Tech Support', 'Contract', 'Payment Method', 'Tenure Category', 'Contract_Tenure']
num_cols = ['Monthly Charges', 'Tenure Months', 'Charge Ratio', 'Num Services', 'Churn Score', 'CLTV']

# Encoding
X = pd.concat([pd.get_dummies(df[cat_cols], drop_first=True), df[num_cols]], axis=1)
y = df['Churn Label'].map({'Yes': 1, 'No': 0})
X.drop(columns=[c for c in ['Gender_Male', 'Paperless Billing_Yes', 'Streaming TV_Yes', 'Streaming Movies_Yes'] if c in X.columns], inplace=True)

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# SMOTE & feature selection
X_train_sm, y_train_sm = SMOTE(sampling_strategy=0.8, random_state=42).fit_resample(X_train, y_train)
sel = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=15)
X_train_sel = X_train_sm.loc[:, sel.fit(X_train_sm, y_train_sm).support_]
X_test_sel = X_test.loc[:, sel.support_]

# Best K
best_k, best_acc = 0, 0
for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train_sel, y_train_sm)
    acc = accuracy_score(y_test, model.predict(X_test_sel))
    if acc > best_acc: best_k, best_acc, best_model = k, acc, model

print(f"Best k: {best_k} | Accuracy: {best_acc:.4f}")

# Eval
y_prob = best_model.predict_proba(X_test_sel)[:, 1]
y_pred = best_model.predict(X_test_sel)
print("\nDefault Threshold (0.5):")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Optimal threshold
prec, rec, thres = precision_recall_curve(y_test, y_prob)
idx = np.where(rec >= 0.85)[0]
opt_idx = idx[np.argmax(prec[idx])]
opt_thresh = thres[opt_idx]
print(f"\nOptimal Threshold: {opt_thresh:.4f}, Precision: {prec[opt_idx]:.4f}, Recall: {rec[opt_idx]:.4f}")
print("\nClassification Report (Optimal Threshold):")
print(classification_report(y_test, (y_prob >= opt_thresh).astype(int)))

# Feature importance
perm = permutation_importance(best_model, X_test_sel, y_test, n_repeats=10, random_state=42)
feat_imp = pd.DataFrame({'Feature': X_test_sel.columns, 'Importance': perm.importances_mean}).sort_values(by='Importance', ascending=False).head(10)
print("\nTop 10 Features:\n", feat_imp)

# Cross-validation
cv_acc = cross_val_score(best_model, X_train_sel, y_train_sm, cv=5, scoring='accuracy')
cv_auc = cross_val_score(best_model, X_train_sel, y_train_sm, cv=5, scoring='roc_auc')
print(f"\nCV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Save model and unique Scaler
joblib.dump(best_model, 'knn_churn_model.pkl')
joblib.dump(scaler, 'scaler_knn.pkl')
print("\nModel saved as 'knn_churn_model.pkl'")
print("Scaler saved as 'scaler_knn.pkl'")
