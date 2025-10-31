# ==========================================================
# Mobile Money Fraud Detection – Supervised Classification
# Dataset: a18ae1df-3812-4dc0-b7cb-9ff2fdaf8655.csv
# Group 6: Valentia Musabeyezu, Narmada Karnati,
#          Jaya Prakash Annam, Delphine Ruzindana, Rajdeep Kaur
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder

# ------------------- STEP 1: Load Dataset -------------------
df = pd.read_csv("a18ae1df-3812-4dc0-b7cb-9ff2fdaf8655.csv")

print("✅ Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ------------------- STEP 2: Handle Missing Data -------------------
df = df.fillna(0)   # replace NaN with 0 for simplicity

# ------------------- STEP 3: Encode Categorical Features -------------------
cat_cols = ['transaction_type', 'location', 'user_id', 'device_id', 'recipient_id']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ------------------- STEP 4: Define Features and Label -------------------
X = df.drop(columns=['is_fraud', 'anomaly_reason', 'timestamp', 'transaction_id'])
y = df['is_fraud']

print("\n✅ Features selected:")
print(X.columns.tolist())

# ------------------- STEP 5: Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ------------------- STEP 6: Train Model -------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("\n🎯 Model training completed")

# ------------------- STEP 7: Predictions -------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------- STEP 8: Evaluation Metrics -------------------
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n🔹 ROC-AUC Score: {roc_auc:.4f}")

# ------------------- STEP 9: Plot ROC Curve -------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Fraud Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# ------------------- STEP 10: Precision–Recall Curve -------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Fraud Detection")
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()

# ------------------- STEP 11: Save Predictions -------------------
output = X_test.copy()
output["actual_is_fraud"] = y_test
output["predicted_is_fraud"] = y_pred
output["fraud_probability"] = y_prob
output.to_csv("fraud_predictions.csv", index=False)

print("\n💾 Predictions saved to fraud_predictions.csv")
print("✅ ROC curve saved as roc_curve.png")
print("✅ Precision–Recall curve saved as precision_recall_curve.png")
 