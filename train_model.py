import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import (
    logistic, decision_tree, knn,
    naive_bayes, random_forest, xgboost_model
)

os.makedirs("model/saved", exist_ok=True)

TARGET = "default.payment.next.month"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit-card.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Save test set for Streamlit & evaluation
# -----------------------------
import os

os.makedirs("data", exist_ok=True)

test_df = X_test.copy()
test_df[TARGET] = y_test.values

test_df.to_csv("data/test_data.csv", index=False)

print("✅ Saved test data to data/test_data.csv")


# 🔑 SAVE FEATURES
joblib.dump(X_train.columns.tolist(), "model/saved/features.pkl")

# 🔑 SCALER
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "model/saved/scaler.pkl")

# TRAIN + SAVE MODELS
joblib.dump(logistic.train(X_train_scaled, y_train), "model/saved/logistic.pkl")
joblib.dump(knn.train(X_train_scaled, y_train), "model/saved/knn.pkl")

joblib.dump(decision_tree.train(X_train, y_train), "model/saved/decision_tree.pkl")
joblib.dump(naive_bayes.train(X_train, y_train), "model/saved/naive_bayes.pkl")
joblib.dump(random_forest.train(X_train, y_train), "model/saved/random_forest.pkl")
joblib.dump(xgboost_model.train(X_train, y_train), "model/saved/xgboost.pkl")
