# boardgame_baselines.py

from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "boardgame_data_prepared.npz"

data = np.load(NPZ_PATH, allow_pickle=True)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
feature_names = data["feature_names"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Feature count:", len(feature_names))


# ==============================
# Linear Regression (scaled)
# ==============================
lin_model = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LinearRegression())
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# CV: neg_root_mean_squared_error scorer mevcutsa çalışır; yoksa aşağıda fallback var.
try:
    cv_rmse = -cross_val_score(
        lin_model, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=kfold
    ).mean()
except Exception:
    # Daha eski sklearn için fallback: neg_mean_squared_error -> sqrt
    cv_mse = -cross_val_score(
        lin_model, X_train, y_train,
        scoring="neg_mean_squared_error",
        cv=kfold
    ).mean()
    cv_rmse = float(np.sqrt(cv_mse))

lin_model.fit(X_train, y_train)
pred_lr = lin_model.predict(X_test)

mse_lr = mean_squared_error(y_test, pred_lr)
rmse_lr = float(np.sqrt(mse_lr))
r2_lr = r2_score(y_test, pred_lr)

print("\n=== Linear Regression ===")
print(f"CV RMSE: {cv_rmse:.4f}")
print(f"Test RMSE: {rmse_lr:.4f}")
print(f"Test R2:   {r2_lr:.4f}")


# ==============================
# Random Forest
# ==============================
rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

try:
    cv_rmse_rf = -cross_val_score(
        rf, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=kfold
    ).mean()
except Exception:
    cv_mse_rf = -cross_val_score(
        rf, X_train, y_train,
        scoring="neg_mean_squared_error",
        cv=kfold
    ).mean()
    cv_rmse_rf = float(np.sqrt(cv_mse_rf))

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, pred_rf)
rmse_rf = float(np.sqrt(mse_rf))
r2_rf = r2_score(y_test, pred_rf)

print("\n=== Random Forest ===")
print(f"CV RMSE: {cv_rmse_rf:.4f}")
print(f"Test RMSE: {rmse_rf:.4f}")
print(f"Test R2:   {r2_rf:.4f}")
