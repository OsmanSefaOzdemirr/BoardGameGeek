# streamlit_app.py
# Board Game Rating Prediction - Streamlit Application (fixed)

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
ARTEFACTS_PATH = BASE_DIR / "preprocessing_artifacts.json"
PKL_PATH = BASE_DIR / "xgb_model.pkl"
JSON_MODEL_PATH = BASE_DIR / "xgb_model.json"


def _resolve_csv(name: str) -> Path:
    p1 = BASE_DIR / "data" / name
    p2 = BASE_DIR / name
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"CSV not found: {name}")


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Board Game Rating Predictor", page_icon="🎲", layout="wide")


@st.cache_data
def load_artefacts() -> dict:
    if not ARTEFACTS_PATH.exists():
        st.error("❌ preprocessing_artifacts.json not found. Run boardgame_prepare_data.py first.")
        st.stop()
    return json.loads(ARTEFACTS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_model_and_features():
    """
    Önce pkl (eski format) dene.
    Olmazsa booster json yükle.
    """
    artefacts = load_artefacts()
    feature_cols = artefacts["feature_cols"]

    # 1) PKL
    if PKL_PATH.exists():
        with open(PKL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data["model"]
        feature_names = model_data["feature_names"]
        return model, feature_names

    # 2) JSON booster fallback
    if JSON_MODEL_PATH.exists():
        booster = xgb.Booster()
        booster.load_model(str(JSON_MODEL_PATH))
        # Streamlit'te booster predict için DMatrix gerekir
        return booster, feature_cols

    st.error("❌ Model not found. Run boardgame_xgboost_shap.py first.")
    st.stop()


def compute_age_norm(min_age: int) -> float:
    # age bins: [0-8), [8-12), [12-18), [18+]
    if min_age < 8:
        age_ord = 0
    elif min_age < 12:
        age_ord = 1
    elif min_age < 18:
        age_ord = 2
    else:
        age_ord = 3
    return age_ord / 3.0


def create_feature_row(
    min_players: int,
    max_players: int,
    playtime: int,
    min_age: int,
    game_weight: float,
    year_published: int,
    num_ratings: int,
    selected_mechanics: list[str],
    selected_themes: list[str],
    artefacts: dict,
    feature_names: list[str],
) -> np.ndarray:

    # --- TRAINING-ALIGNED CLIPS ---
    playtime = int(np.clip(playtime, artefacts["clip"]["playtime_min"], artefacts["clip"]["playtime_max"]))
    max_players = int(np.clip(max_players, 1, artefacts["clip"]["max_players_max"]))

    # derived
    players_range = max_players - min_players
    is_solo_supported = 1 if min_players == 1 else 0
    log_play_time = float(np.log1p(playtime))
    play_time_per_player = float(playtime / max_players) if max_players > 0 else float(playtime)
    published_decade = int((year_published // 10) * 10)
    age_norm = float(compute_age_norm(min_age))

    # base numeric features (must match training)
    features = {
        "MinPlayers": min_players,
        "MaxPlayers": max_players,
        "MfgPlaytime": playtime,
        "MfgAgeRec": min_age,
        "GameWeight": game_weight,
        "YearPublished": year_published,
        "NumUserRatings": num_ratings,
        "players_range": players_range,
        "is_solo_supported": is_solo_supported,
        "log_play_time": log_play_time,
        "play_time_per_player": play_time_per_player,
        "published_decade": published_decade,
        "age_norm": age_norm,
    }

    # top mech/theme lists come from artefacts (train-time)
    for m in artefacts["top_mechanics"]:
        features[m] = 1 if m in selected_mechanics else 0
    for t in artefacts["top_themes"]:
        features[t] = 1 if t in selected_themes else 0

    # build row in exact feature order
    row = np.array([features.get(f, 0) for f in feature_names], dtype=float).reshape(1, -1)
    return row


# ==============================
# LOAD
# ==============================
artefacts = load_artefacts()
model, feature_names = load_model_and_features()

top_mechanics = artefacts["top_mechanics"]
top_themes = artefacts["top_themes"]

# ==============================
# UI
# ==============================
st.title("Board Game Rating Predictor")

metrics = artefacts.get("metrics", None)
if metrics:
    c1, c2 = st.columns(2)
    c1.metric("Model R² (test)", f"{metrics['r2']:.3f}")
    c2.metric("Model RMSE (test)", f"{metrics['rmse']:.3f}")

st.header("Enter Game Information")
col1, col2 = st.columns(2)

with col1:
    min_players = st.number_input("Min Players", min_value=1, max_value=20, value=2)
    max_players = st.number_input("Max Players", min_value=1, max_value=50, value=4)
    playtime = st.number_input("Play Time (minutes)", min_value=1, max_value=1000, value=60)
    min_age = st.number_input("Minimum Age", min_value=0, max_value=99, value=10)

with col2:
    game_weight = st.slider("Game Complexity (Weight)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
    year_published = st.number_input("Year Published", min_value=1900, max_value=2025, value=2020)
    num_ratings = st.number_input("Number of User Ratings", min_value=0, value=100)

st.subheader("Mechanics (top features used in training)")
selected_mechanics = st.multiselect("Select mechanics:", options=top_mechanics, default=[])

st.subheader("Themes (top features used in training)")
selected_themes = st.multiselect("Select themes:", options=top_themes, default=[])


# ==============================
# PREDICT
# ==============================
if st.button("Predict Rating"):
    row = create_feature_row(
        min_players=min_players,
        max_players=max_players,
        playtime=playtime,
        min_age=min_age,
        game_weight=game_weight,
        year_published=year_published,
        num_ratings=num_ratings,
        selected_mechanics=selected_mechanics,
        selected_themes=selected_themes,
        artefacts=artefacts,
        feature_names=feature_names,
    )

    # pkl model (XGBRegressor) vs booster json
    if hasattr(model, "predict"):
        pred = float(model.predict(row)[0])
    else:
        dmat = xgb.DMatrix(row, feature_names=feature_names)
        pred = float(model.predict(dmat)[0])

    st.success("Prediction complete.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Rating", f"{pred:.2f}")

    stars = "⭐" * max(0, min(10, int(pred)))
    c2.markdown(f"**Visual:** {stars}")

    if pred >= 8.0:
        c3.success("Excellent game potential")
    elif pred >= 7.0:
        c3.info("Good game potential")
    else:
        c3.warning("Average game potential")
