# train_model.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

print("Loading enhanced features for training...")
df = pd.read_csv("premier_league_with_elo_best.csv")

feature_cols = [c for c in df.columns if c.startswith(("home_", "away_", "diff_", "norm_prob_", "odds_spread", "elo_"))]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df["Result"].astype(int)

print(f"Training on {len(X):,} matches with {len(feature_cols)} features")

tscv = TimeSeriesSplit(n_splits=5)
model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss")

scores = []
for train_idx, test_idx in tscv.split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    scores.append(accuracy_score(y.iloc[test_idx], pred))

print(f"✅ STEP 3 COMPLETE — Mean CV Accuracy: {np.mean(scores):.4f}")

model.fit(X, y)
joblib.dump(model, "xgboost_premier_league_model.pkl")
print("✅ Model saved as xgboost_premier_league_model.pkl")