# boardgame_prepare_data.py
# Data preparation pipeline for BoardGameGeek regression project

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================
# 0. PATHS (ROBUST)
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Eğer data/ yoksa (csv'ler aynı klasördeyse) fallback
def _resolve_csv(name: str) -> Path:
    p1 = DATA_DIR / name
    p2 = BASE_DIR / name
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"CSV not found: {name} (looked in {p1} and {p2})")


# ==============================
# 1. COLUMN NAMES
# ==============================
COL_ID          = "BGGId"
COL_MIN_PLAYERS = "MinPlayers"
COL_MAX_PLAYERS = "MaxPlayers"
COL_PLAY_TIME   = "MfgPlaytime"
COL_MIN_AGE     = "MfgAgeRec"
COL_WEIGHT      = "GameWeight"
COL_YEAR_PUB    = "YearPublished"
COL_USERS_RATED = "NumUserRatings"
COL_TARGET      = "AvgRating"


# ==============================
# 2. LOAD
# ==============================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(_resolve_csv("games.csv"))
    mechanics = pd.read_csv(_resolve_csv("mechanics.csv"))
    themes = pd.read_csv(_resolve_csv("themes.csv"))
    return games, mechanics, themes


# ==============================
# 3. MERGE + CLEAN CORE
# ==============================
def build_base_df(games: pd.DataFrame, mechanics: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
    df = games.merge(mechanics, on=COL_ID, how="left").merge(themes, on=COL_ID, how="left")

    # Target ve temel metrikler boşsa drop (model için gerekli)
    df = df.dropna(subset=[COL_TARGET, COL_USERS_RATED])

    # Negatif/absürt değerleri temizle (defensive)
    for c in [COL_MIN_PLAYERS, COL_MAX_PLAYERS, COL_PLAY_TIME, COL_MIN_AGE, COL_WEIGHT, COL_YEAR_PUB, COL_USERS_RATED]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[COL_MIN_PLAYERS, COL_MAX_PLAYERS, COL_PLAY_TIME, COL_MIN_AGE, COL_WEIGHT, COL_YEAR_PUB])

    # Mantıksız min/max
    df = df[df[COL_MIN_PLAYERS] >= 1]
    df = df[df[COL_MAX_PLAYERS] >= df[COL_MIN_PLAYERS]]
    df = df[df[COL_USERS_RATED] >= 0]

    return df


# ==============================
# 4. DERIVED FEATURES (UPDATED: age_norm)
# ==============================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Outlier trimming / winsorization
    df[COL_PLAY_TIME] = df[COL_PLAY_TIME].clip(lower=5, upper=600)
    df[COL_MAX_PLAYERS] = df[COL_MAX_PLAYERS].clip(upper=50)

    # Derived
    df["players_range"] = df[COL_MAX_PLAYERS] - df[COL_MIN_PLAYERS]
    df["is_solo_supported"] = (df[COL_MIN_PLAYERS] == 1).astype(int)
    df["log_play_time"] = np.log1p(df[COL_PLAY_TIME])

    df["play_time_per_player"] = df[COL_PLAY_TIME] / df[COL_MAX_PLAYERS]
    df["play_time_per_player"] = df["play_time_per_player"].fillna(df[COL_PLAY_TIME])

    df["published_decade"] = (df[COL_YEAR_PUB] // 10) * 10

    # Age bin (ordinal + normalized)  -> hocanın dediği “normalized gibi”
    df["age_bin"] = pd.cut(
        df[COL_MIN_AGE],
        bins=[0, 8, 12, 18, 99],
        labels=["kid", "preteen", "teen", "adult"],
        include_lowest=True
    )

    age_map = {"kid": 0, "preteen": 1, "teen": 2, "adult": 3}
    df["age_ord"] = df["age_bin"].map(age_map).astype(float)
    df["age_norm"] = (df["age_ord"] / 3.0).fillna(0.0)

    return df


# ==============================
# 5. TOP MECH/THEME SELECTION
# ==============================
def select_top_binary_features(df: pd.DataFrame, top_mech: int = 80, top_theme: int = 80) -> tuple[pd.DataFrame, list[str], list[str]]:
    mech_cols = [c for c in df.columns if c.startswith("mechanic_")]
    theme_cols = [c for c in df.columns if c.startswith("theme_")]

    # Eğer datasetinizde prefix yoksa (sizin csv’lerde prefix yok)
    # BGGId dışındaki tüm sütunları al:
    if not mech_cols:
        # mechanics.csv merge sonrası gelen kolonlar: BGGId hariç
        mech_cols = [c for c in df.columns if c not in {
            COL_ID, COL_MIN_PLAYERS, COL_MAX_PLAYERS, COL_PLAY_TIME, COL_MIN_AGE, COL_WEIGHT, COL_YEAR_PUB, COL_USERS_RATED, COL_TARGET,
            "players_range", "is_solo_supported", "log_play_time", "play_time_per_player", "published_decade",
            "age_bin", "age_ord", "age_norm"
        } and c not in theme_cols]

    if not theme_cols:
        # themes.csv merge sonrası gelen kolonlar: BGGId hariç (mekaniklerden ayrıştırma zor)
        # Daha sağlam yöntem: orijinal themes.csv başlıklarını oku
        themes_header = pd.read_csv(_resolve_csv("themes.csv"), nrows=1).columns.tolist()
        theme_cols = [c for c in themes_header if c != COL_ID]
        mech_header = pd.read_csv(_resolve_csv("mechanics.csv"), nrows=1).columns.tolist()
        mech_cols = [c for c in mech_header if c != COL_ID]

    # Fill NaN -> 0
    df[mech_cols] = df[mech_cols].fillna(0)
    df[theme_cols] = df[theme_cols].fillna(0)

    mech_sums = df[mech_cols].sum().sort_values(ascending=False)
    theme_sums = df[theme_cols].sum().sort_values(ascending=False)

    top_mech_cols = mech_sums.head(top_mech).index.tolist()
    top_theme_cols = theme_sums.head(top_theme).index.tolist()

    # Drop diğerleri
    drop_mech = [c for c in mech_cols if c not in top_mech_cols]
    drop_theme = [c for c in theme_cols if c not in top_theme_cols]

    df_model = df.drop(columns=drop_mech + drop_theme)
    return df_model, top_mech_cols, top_theme_cols


# ==============================
# 6. BUILD X, y + SAVE ARTEFACTS
# ==============================
def main() -> None:
    games, mechanics, themes = load_data()
    df = build_base_df(games, mechanics, themes)
    df = add_derived_features(df)

    # Top features
    df_model, top_mech_cols, top_theme_cols = select_top_binary_features(df, top_mech=80, top_theme=80)

    # Numeric feature set (age_bin one-hot YOK; age_norm VAR)
    numeric_features = [
        COL_MIN_PLAYERS, COL_MAX_PLAYERS, COL_PLAY_TIME,
        COL_MIN_AGE, COL_WEIGHT, COL_YEAR_PUB,
        COL_USERS_RATED,
        "players_range", "is_solo_supported", "log_play_time",
        "play_time_per_player", "published_decade",
        "age_norm"
    ]

    feature_cols = numeric_features + top_mech_cols + top_theme_cols

    df_model = df_model.copy()
    X = df_model[feature_cols].to_numpy()
    y = df_model[COL_TARGET].to_numpy()
    feature_names = np.array(feature_cols, dtype=object)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save npz
    npz_path = BASE_DIR / "boardgame_data_prepared.npz"
    np.savez(
        npz_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names
    )

    # Save preprocessing artefacts for Streamlit (top lists + clip rules)
    artefacts = {
        "top_mechanics": top_mech_cols,
        "top_themes": top_theme_cols,
        "clip": {"playtime_min": 5, "playtime_max": 600, "max_players_max": 50},
        "age_bins": {"kid": [0, 8], "preteen": [8, 12], "teen": [12, 18], "adult": [18, 99]},
        "feature_cols": feature_cols
    }
    artefacts_path = BASE_DIR / "preprocessing_artifacts.json"
    artefacts_path.write_text(json.dumps(artefacts, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Saved:", npz_path.name)
    print("✅ Saved:", artefacts_path.name)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
