# ================================
# Terrorism ML Prediction Pipeline
# ================================

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ================================
# 1. Load Dataset
# ================================
zip_path = r"C:\Users\Sankalp Sharma\Downloads\archive.zip"
extract_path = r"C:\Users\Sankalp Sharma\Downloads\terrorism_data"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# find CSV file
files = os.listdir(extract_path)
csv_file = [f for f in files if f.endswith(".csv")][0]
file_path = os.path.join(extract_path, csv_file)

df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
print("✅ Dataset loaded:", df.shape)

# ================================
# 2. Preprocessing
# ================================
# Target = 'success' (0/1)
y = df["success"]
X = df[["nkill", "nwound", "suicide", "attacktype1", "targtype1", "weaptype1"]]

# Handle missing
X = X.fillna(0)

# Scale numeric columns only
scaler = StandardScaler()
X[["nkill", "nwound"]] = scaler.fit_transform(X[["nkill", "nwound"]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# ================================
# 3. Models with Class Weights
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# ================================
# 4. Training + Evaluation
# ================================
results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # metrics
    auc = roc_auc_score(y_test, probs)
    results[name] = auc

    print(f"\n📊 {name} - ROC AUC: {auc:.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{name} - ROC Curve")
    plt.show()

# ================================
# 5. Compare Models
# ================================
print("\n✅ Model ROC-AUC Scores:")
for name, score in results.items():
    print(f"{name}: {score:.3f}")
