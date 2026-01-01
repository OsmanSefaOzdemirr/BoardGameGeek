# 🎲 Board Game Rating Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Overview
**Predicting the average user rating of board games using supervised machine learning.**

Board games are evaluated by thousands of users on platforms like **BoardGameGeek**, resulting in ratings that reflect a game’s overall reception. These ratings are influenced by complex factors including mechanics, themes, playtime, and complexity.

This project formulates the problem as a **regression task**. Beyond standard modeling, a key objective is to **analyze the impact of domain-informed feature engineering** on model performance using ablation studies and explainability methods (SHAP).

---

## 📂 Dataset

The dataset is derived from **BoardGameGeek (BGG)** and consists of merged CSV files joined by a unique `BGGId`.

| File Name | Description |
| :--- | :--- |
| `data/games.csv` | Core info: player counts, playtime, age, weight, publication year, etc. |
| `data/mechanics.csv` | Binary indicators for specific game mechanics. |
| `data/themes.csv` | Binary indicators for thematic categories. |

**Target Variable:**
* `AvgRating`: The average user rating of the board game.

Column/field notes: `data/bgg_data_documentation.txt`.

---

## ⚙️ Data Preprocessing Pipeline

### 1. Data Cleaning 🧹
To ensure data quality, the following steps were applied:
* **Removal:** Dropped games with missing `AvgRating` or essential numeric attributes.
* **Filtering:** Removed inconsistent values (e.g., cases where `MaxPlayers < MinPlayers`).
* **Outlier Handling (Clipping):**
    * Playtime clipped to range **[5, 600] minutes**.
    * Max players clipped to **50**.

### 2. Feature Engineering 🛠️
Domain knowledge was used to create derived features to capture better signal:

| Derived Feature | Description |
| :--- | :--- |
| `players_range` | Difference between maximum and minimum number of players. |
| `is_solo_supported` | Binary indicator (1 if MinPlayers = 1, else 0). |
| `log_play_time` | Log-transformed playtime to reduce skewness. |
| `play_time_per_player` | Playtime normalized by the maximum number of players. |
| `published_decade` | The decade in which the game was published. |
| `age_norm` | Normalized ordinal representation of age recommendation. |

### 3. Dimensionality Reduction 📉
Mechanics and Themes are high-dimensional categorical features.
* Retained only the **top 80** most frequent mechanics and themes.
* Less frequent categories were discarded.
* Missing values filled with zero (indicating absence).

---

## 🤖 Modeling Strategy

The dataset was split into **80% Training** and **20% Testing**. Features and labels were stored in a `boardgame_data_prepared.npz` archive for consistency.

### Models Evaluated
1.  **Baseline Models:**
    * Linear Regression
    * Random Forest
2.  **Advanced Model:**
    * Optimized **XGBoost Regressor**

### Evaluation Metrics
* **RMSE** (Root Mean Squared Error)
* **R²** (Coefficient of Determination)

### Explainability
* **Ablation Studies:** To quantify the gain from engineered features.
* **SHAP Analysis:** To interpret model decisions and feature importance.

---

## 🧰 Environment

You can set up dependencies using either **pip** or **conda**.

### Option A: pip

```bash
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate boardgamefinal
```

Note: `environment.yml` pins `python=3.13`.

---

## 🗂️ Source Code (What’s Where)

| File | Purpose |
| --- | --- |
| `boardgame_prepare_data.py` | Loads/merges raw CSVs, cleans data, builds engineered features, exports artifacts |
| `boardgame_baselines.py` | Trains/evaluates baseline models (Linear Regression, Random Forest) |
| `boardgame_xgboost_shap.py` | Trains XGBoost (with tuning) and generates SHAP/importance plots |
| `boardgame_report.py` | Produces a compact report: metrics + parity/residual plots + training curve |
| `streamlit_app.py` | Streamlit demo app for interactive prediction |

Key generated artifacts:

- `boardgame_data_prepared.npz` (train/test arrays)
- `preprocessing_artifacts.json` (feature metadata + saved metrics)
- `outputs/` (plots + `metrics_summary.*`)

---

## 🚀 How to Run (End-to-End)

Run these from the project root:

```bash
# 1) Prepare data + artifacts
python boardgame_prepare_data.py

# 2) Baseline models
python boardgame_baselines.py

# 3) XGBoost training + SHAP/importance plots
python boardgame_xgboost_shap.py

# 4) Generate metrics + test plots + training curve
python boardgame_report.py
```

Optional (demo UI):

```bash
streamlit run streamlit_app.py
```

---

## 📈 Model Results (Test Set)

Metrics are saved to `outputs/metrics_summary.json` and `outputs/metrics_summary.csv`.

Dataset split (from the report): **Train = 17,394**, **Test = 4,349**.

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.7193 | 0.5491 | 0.4084 |
| Random Forest | 0.6117 | 0.4527 | 0.5722 |
| XGBoost | **0.5911** | **0.4385** | **0.6005** |

---

## 🧪 Inference Visualizations (Test)

- Parity plot: `outputs/test_parity_plot.png`
- Residual plot: `outputs/test_residuals.png`

---

## 📉 Training Process Plot

- Training curve (RMSE): `outputs/training_curve_rmse.png`

---

## 🔍 Explainability (SHAP / Importance)

Generated by `boardgame_xgboost_shap.py`:

- SHAP summary: `outputs/shap_summary.png`
- SHAP bar importance: `outputs/shap_bar_importance.png`
- Feature importance (Top-20): `outputs/xgb_feature_importance_top20.png`
- SHAP dependence examples (selected):
    - `outputs/shap_dependence_GameWeight.png`
    - `outputs/shap_dependence_log_play_time.png`
    - `outputs/shap_dependence_players_range.png`

