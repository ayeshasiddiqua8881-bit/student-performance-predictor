"""
Student Performance Predictor - Model Training Script
Generates synthetic student data and trains a Random Forest Classifier.
Run this script first to generate 'model.pkl'
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N = 1000  # number of synthetic students

# ── Generate synthetic dataset ────────────────────────────────────────────────
def generate_data(n=N):
    study_hours      = np.round(np.random.uniform(0, 10, n), 1)       # hrs/day
    attendance       = np.round(np.random.uniform(40, 100, n), 1)     # %
    prev_score       = np.round(np.random.uniform(30, 100, n), 1)     # out of 100
    assignments_done = np.round(np.random.uniform(0, 100, n), 1)      # % completed
    sleep_hours      = np.round(np.random.uniform(3, 10, n), 1)       # hrs/day
    extra_curricular = np.random.randint(0, 2, n)                     # 0 or 1

    # Label logic — weighted combination
    score = (
        study_hours      * 4.5 +
        attendance       * 0.3 +
        prev_score       * 0.4 +
        assignments_done * 0.2 +
        sleep_hours      * 1.5 +
        extra_curricular * 3.0 +
        np.random.normal(0, 5, n)   # noise
    )

    # Threshold at median to get balanced classes
    threshold = np.median(score)
    result = (score >= threshold).astype(int)  # 1 = Pass, 0 = Fail

    df = pd.DataFrame({
        "study_hours":      study_hours,
        "attendance":       attendance,
        "prev_score":       prev_score,
        "assignments_done": assignments_done,
        "sleep_hours":      sleep_hours,
        "extra_curricular": extra_curricular,
        "result":           result
    })
    return df

# ── Train ─────────────────────────────────────────────────────────────────────
df = generate_data()
df.to_csv("student_data.csv", index=False)
print(f"✅ Dataset saved: student_data.csv ({len(df)} rows)")

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy : {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# Feature importance
feature_names = X.columns.tolist()
importances   = model.feature_importances_
print("📌 Feature Importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name:<22} {imp:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n✅ Model saved: model.pkl")
