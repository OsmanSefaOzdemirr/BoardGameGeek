# boardgame_xgboost_shap.py

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap
import pandas as pd


# ==============================
# 0. OUTPUTS DIRECTORY
# ==============================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ==============================
# 1. HAZIR VERİYİ YÜKLE
# ==============================
data = np.load(BASE_DIR / "boardgame_data_prepared.npz", allow_pickle=True)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Outputs folder:", OUT_DIR)


# ==============================
# 2. XGBoost + RandomizedSearchCV
# ==============================
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    nthread=-1,
    tree_method="hist",
    random_state=42
)

param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
}

# sklearn scorer uyumluluğu: neg_root_mean_squared_error yoksa fallback
try:
    _ = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        scoring="neg_root_mean_squared_error",
        n_iter=1,
        cv=2,
        random_state=42,
        n_jobs=-1
    )
    scoring_used = "neg_root_mean_squared_error"
except Exception:
    scoring_used = "neg_mean_squared_error"

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    scoring=scoring_used,
    n_iter=30,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nXGBoost tuning başlıyor...\n")
random_search.fit(X_train, y_train)

print("\nEn iyi hiperparametreler:")
print(random_search.best_params_)

best_xgb = random_search.best_estimator_

y_pred_xgb = best_xgb.predict(X_test)

# RMSE: squared=False yok -> sqrt(MSE)
rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n=== XGBoost (TEST SET) ===")
print(f"Scoring used: {scoring_used}")
print(f"Test RMSE: {rmse_xgb:.4f}")
print(f"Test R^2 : {r2_xgb:.4f}")


# ==============================
# 3. FEATURE IMPORTANCE (PLOT + SAVE)
# ==============================
feature_names_list = list(feature_names)
importances = best_xgb.feature_importances_

fi_df = pd.DataFrame({
    "feature": feature_names_list,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nEn önemli 20 özellik (XGBoost):\n")
print(fi_df.head(20))

top_n = 20
fi_top = fi_df.head(top_n).iloc[::-1]

fig = plt.figure(figsize=(8, 10))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.xlabel("Importance")
plt.title(f"Top {top_n} XGBoost Feature Importances")
fig.tight_layout()
fig.savefig(OUT_DIR / "xgb_feature_importance_top20.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# ==============================
# 4. SHAP Summary + Dependence (SHOW + SAVE)
# ==============================
# Not: shap.initjs() Jupyter içindir, script'te şart değil ama kalsın.
try:
    shap.initjs()
except Exception:
    pass

rng = np.random.default_rng(42)
sample_size = min(20000, X_train.shape[0])
sample_idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
X_train_sample = X_train[sample_idx]

print("SHAP için kullanılan örnek sayısı:", X_train_sample.shape[0])

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train_sample)

# ---- SHAP Summary (beeswarm) save + show ----
shap.summary_plot(
    shap_values,
    X_train_sample,
    feature_names=feature_names_list,
    show=False
)
fig = plt.gcf()
fig.tight_layout()
fig.savefig(OUT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

# ---- SHAP Bar importance save + show ----
shap.summary_plot(
    shap_values,
    X_train_sample,
    feature_names=feature_names_list,
    plot_type="bar",
    show=False
)
fig = plt.gcf()
fig.tight_layout()
fig.savefig(OUT_DIR / "shap_bar_importance.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

# Dependence plots (türetilmiş feature’ları da ekledim)
dependence_features = [
    "GameWeight", "MfgPlaytime", "MinPlayers", "MaxPlayers",
    "log_play_time", "is_solo_supported", "play_time_per_player", "players_range",
    "published_decade", "age_norm"
]

for fname in dependence_features:
    if fname in feature_names_list:
        shap.dependence_plot(
            fname,
            shap_values,
            X_train_sample,
            feature_names=feature_names_list,
            show=False
        )
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"shap_dependence_{fname}.png", dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)

print("\n✅ SHAP plots saved to:", OUT_DIR)


# ==============================
# 5. MODELİ KAYDETME (STREAMLIT İÇİN)
# ==============================
print("\nModel Streamlit için kaydediliyor...")

model_data = {
    "model": best_xgb,
    "feature_names": feature_names
}

with open(BASE_DIR / "xgb_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model 'xgb_model.pkl' olarak kaydedildi.")