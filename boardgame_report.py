from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

NPZ_PATH = BASE_DIR / "boardgame_data_prepared.npz"
ARTEFACTS_PATH = BASE_DIR / "preprocessing_artifacts.json"
XGB_PKL_PATH = BASE_DIR / "xgb_model.pkl"
XGB_JSON_PATH = BASE_DIR / "xgb_model.json"


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        rmse=_rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def load_prepared_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()
    return X_train, X_test, y_train, y_test, feature_names


def load_xgb_model() -> object:
    if XGB_PKL_PATH.exists():
        with open(XGB_PKL_PATH, "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"]

    if XGB_JSON_PATH.exists():
        booster = xgb.Booster()
        booster.load_model(str(XGB_JSON_PATH))
        return booster

    raise FileNotFoundError("No XGBoost model found (xgb_model.pkl / xgb_model.json)")


def predict_with_xgb(model: object, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    # XGBRegressor
    if hasattr(model, "predict") and not isinstance(model, xgb.Booster):
        return np.asarray(model.predict(X), dtype=float)

    # Booster
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    return np.asarray(model.predict(dmat), dtype=float)


def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True AvgRating")
    plt.ylabel("Predicted AvgRating")
    plt.title("Test Set: Predicted vs True (Parity Plot)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(7, 4.5))
    plt.scatter(y_pred, residuals, s=8, alpha=0.35)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted AvgRating")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Test Set: Residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_xgb_with_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_model: object | None,
) -> tuple[object, dict] | tuple[None, None]:
    """Train an XGBoost model with eval tracking to produce a training-curve plot.

    Note: xgboost>=3 removed many sklearn .fit() kwargs (eval_metric, callbacks,
    early_stopping_rounds, evals_result). For compatibility we use low-level
    xgboost.train() here.

    Returns (booster, evals_result) or (None, None) if training can't be performed.
    """

    # Split train -> train/valid (keep test untouched)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    num_boost_round = 600

    # Defaults (reasonable + fast)
    params: dict = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": 42,
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
    }

    if base_model is not None and hasattr(base_model, "get_params"):
        try:
            p = base_model.get_params()
            num_boost_round = max(int(p.get("n_estimators", num_boost_round)), 300)
            if "learning_rate" in p and p["learning_rate"] is not None:
                params["eta"] = float(p["learning_rate"])
            if "max_depth" in p and p["max_depth"] is not None:
                params["max_depth"] = int(p["max_depth"])
            if "subsample" in p and p["subsample"] is not None:
                params["subsample"] = float(p["subsample"])
            if "colsample_bytree" in p and p["colsample_bytree"] is not None:
                params["colsample_bytree"] = float(p["colsample_bytree"])
            if "reg_lambda" in p and p["reg_lambda"] is not None:
                params["lambda"] = float(p["reg_lambda"])
            if "reg_alpha" in p and p["reg_alpha"] is not None:
                params["alpha"] = float(p["reg_alpha"])
        except Exception:
            pass

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    evals_result: dict = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        evals_result=evals_result,
        verbose_eval=False,
        early_stopping_rounds=50,
    )

    return booster, evals_result


def save_training_curve(evals_result: dict, path: Path) -> bool:
    if not evals_result:
        return False

    # Typical keys (xgb.train): train/valid -> rmse
    train_rmse = evals_result.get("train", {}).get("rmse", [])
    valid_rmse = evals_result.get("valid", {}).get("rmse", [])
    if not train_rmse or not valid_rmse:
        return False

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(train_rmse, label="train")
    plt.plot(valid_rmse, label="valid")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training Curve (RMSE)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def update_streamlit_metrics(xgb_metrics: RegressionMetrics) -> None:
    if not ARTEFACTS_PATH.exists():
        return

    artefacts = json.loads(ARTEFACTS_PATH.read_text(encoding="utf-8"))
    artefacts["metrics"] = {
        **asdict(xgb_metrics),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    ARTEFACTS_PATH.write_text(json.dumps(artefacts, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    X_train, X_test, y_train, y_test, feature_names = load_prepared_data()

    # ----- Baselines (test metrics only; CV already exists in boardgame_baselines.py) -----
    print("[1/4] Training LinearRegression baseline...")
    lin = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression()),
    ])
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)
    lin_metrics = compute_metrics(y_test, pred_lin)

    print("[2/4] Training RandomForest baseline...")
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test, pred_rf)

    # ----- XGBoost (load saved model) -----
    print("[3/4] Evaluating saved XGBoost model...")
    xgb_model = load_xgb_model()
    pred_xgb = predict_with_xgb(xgb_model, X_test, feature_names)
    xgb_metrics = compute_metrics(y_test, pred_xgb)

    # Save inference visualizations
    save_parity_plot(y_test, pred_xgb, OUT_DIR / "test_parity_plot.png")
    save_residual_plot(y_test, pred_xgb, OUT_DIR / "test_residuals.png")

    # Training curve plot (refit only for curve)
    print("[4/4] Fitting XGBoost (for training curve only)...")
    base_model_for_params = xgb_model if hasattr(xgb_model, "get_params") else None
    curve_ok = False
    try:
        _, evals_result = train_xgb_with_curves(X_train, y_train, base_model_for_params)
        if evals_result is not None:
            curve_ok = save_training_curve(evals_result, OUT_DIR / "training_curve_rmse.png")
    except Exception:
        curve_ok = False

    # Metrics artifact files
    metrics_payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "split": {"train": int(X_train.shape[0]), "test": int(X_test.shape[0])},
        "models": {
            "linear_regression": asdict(lin_metrics),
            "random_forest": asdict(rf_metrics),
            "xgboost": asdict(xgb_metrics),
        },
        "artifacts": {
            "test_parity_plot": str((OUT_DIR / "test_parity_plot.png").name),
            "test_residuals": str((OUT_DIR / "test_residuals.png").name),
            "training_curve_rmse": str((OUT_DIR / "training_curve_rmse.png").name if curve_ok else "(not generated)"),
        },
    }

    (OUT_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        {"model": "LinearRegression", **asdict(lin_metrics)},
        {"model": "RandomForest", **asdict(rf_metrics)},
        {"model": "XGBoost", **asdict(xgb_metrics)},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    update_streamlit_metrics(xgb_metrics)

    print("✅ Wrote:", OUT_DIR / "metrics_summary.json")
    print("✅ Wrote:", OUT_DIR / "metrics_summary.csv")
    print("✅ Wrote:", OUT_DIR / "test_parity_plot.png")
    print("✅ Wrote:", OUT_DIR / "test_residuals.png")
    if curve_ok:
        print("✅ Wrote:", OUT_DIR / "training_curve_rmse.png")
    else:
        print("⚠️  Training curve not generated (evals_result unavailable)")

    print("\n=== Test metrics ===")
    print("LinearRegression:", lin_metrics)
    print("RandomForest:    ", rf_metrics)
    print("XGBoost:         ", xgb_metrics)


if __name__ == "__main__":
    main()
