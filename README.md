# Board Game Rating Prediction with Machine Learning

## Description of the Problem

Board games are evaluated by thousands of users on platforms such as BoardGameGeek, resulting in average user ratings that reflect a game’s overall reception.  
However, these ratings are influenced by many factors simultaneously, including game complexity, player count, playtime, age recommendations, and design characteristics such as mechanics and themes.

The goal of this project is to **predict the average user rating (AvgRating)** of a board game using supervised machine learning techniques.  
This problem is formulated as a **regression task**, where the model learns the relationship between game-related attributes and the resulting average rating.

In addition to building predictive models, a key objective of the project is to **analyze the impact of engineered (derived) features** on model performance.  
Rather than relying only on raw dataset attributes, domain-informed feature engineering is applied and its contribution is evaluated quantitatively using ablation studies and explainability methods.

---

## Dataset and Preprocessing

### Dataset Description

The dataset is derived from **BoardGameGeek** and consists of multiple CSV files that are merged using a unique game identifier (`BGGId`):

- `games.csv`: Core game information such as player counts, playtime, age recommendation, game weight, year of publication, number of ratings, and average rating.
- `mechanics.csv`: Binary indicators representing the presence of specific game mechanics.
- `themes.csv`: Binary indicators representing the thematic categories of each game.

The target variable for prediction is:
- **AvgRating**: Average user rating of a board game.

---

### Data Cleaning

The following preprocessing steps are applied:

- Games with missing target values (`AvgRating`) or missing essential numeric attributes are removed.
- Invalid or inconsistent values are filtered (e.g., `MinPlayers < 1`, `MaxPlayers < MinPlayers`).
- Extremely large outliers are handled using clipping:
  - Playtime is clipped to the range **[5, 600] minutes**.
  - Maximum number of players is clipped to **50**.

These steps reduce noise and prevent extreme values from dominating the learning process.

---

### Feature Engineering

In addition to raw features, several **derived features** are created using domain knowledge:

- `players_range`: Difference between maximum and minimum number of players.
- `is_solo_supported`: Binary indicator showing whether a game supports solo play.
- `log_play_time`: Log-transformed playtime to reduce skewness.
- `play_time_per_player`: Playtime normalized by the maximum number of players.
- `published_decade`: Decade in which the game was published.
- `age_norm`: Normalized ordinal representation of age recommendation.

Age recommendations are treated as an **ordinal variable** rather than one-hot encoded, preserving the natural ordering between age groups.

---

### Mechanics and Themes Selection

Mechanics and themes are high-dimensional categorical features.  
To control dimensionality:

- Only the **top 80 most frequent mechanics** and **top 80 most frequent themes** are retained.
- Less frequent mechanics and themes are discarded.

Missing values in mechanics and themes are filled with zero, indicating absence.

---

### Final Dataset Construction

After preprocessing:

- The dataset is split into **80% training** and **20% test** sets.
- Features and labels are stored in a NumPy archive file (`boardgame_data_prepared.npz`) for reuse across experiments.
- Additional preprocessing metadata (selected mechanics, themes, clipping rules) is saved to ensure consistency between training and inference.

---

### Modeling Overview

The prepared dataset is used to train and evaluate:

- Baseline models (Linear Regression, Random Forest)
- An optimized **XGBoost regressor**

Model performance is evaluated using **RMSE** and **R²**, and the contribution of derived features is analyzed through **ablation studies** and **SHAP-based explainability**.

---

This project demonstrates how feature engineering and explainable machine learning can be applied to a real-world regression problem in the board game domain.
