# preprocessing_pipeline.py
"""
Pipeline completo: Preprocesado + Hyper-tuned MLP + Predicciones
===============================================================

* Limpieza e imputación
* Ingeniería de variables
* Winsorización (outliers)
* Codificación categórica (ordinal + one-hot)
* Estandarización robusta
* **Búsqueda de hiper-parámetros (RandomizedSearchCV) para un MLPRegressor**
* Split train/validación + re-entrenado final + `submissions.csv`

Probado con **scikit-learn ≥ 1.4**.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

###############################################################################
# 1. Feature engineering
###############################################################################

class FeatureEngineer(BaseEstimator, TransformerMixin):
    _orientation_map = {
        "Norte": 0,
        "Nordeste": 45,
        "Este": 90,
        "Sudeste": 135,
        "Sur": 180,
        "Sudoeste": 225,
        "Oeste": 270,
        "Noroeste": 315,
    }

    def __init__(self, current_year: int = 2025):
        self.current_year = current_year

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["antiguedad"] = self.current_year - X["ano_construccion"]
        X["superficie_total_m2"] = (
            X["superficie_interior_m2"].fillna(0) + X["superficie_exterior_m2"].fillna(0)
        )
        X["ratio_exterior_total"] = X["superficie_exterior_m2"] / X["superficie_total_m2"].replace(0, np.nan)
        X["densidad_arboles"] = X["numero_arboles_xardin"] / (X["superficie_exterior_m2"] + 1)
        X["orient_deg"] = X["orientacion"].map(self._orientation_map)
        X["orient_sin"] = np.sin(np.deg2rad(X["orient_deg"]))
        X["orient_cos"] = np.cos(np.deg2rad(X["orient_deg"]))
        return X.drop(columns=["orientacion", "orient_deg"])

###############################################################################
# 2. Winsorizer
###############################################################################

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        for col in df.columns:
            self.bounds_[col] = (df[col].quantile(self.lower), df[col].quantile(self.upper))
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            df[col] = df[col].clip(lo, hi)
        return df.values

###############################################################################
# 3. ColumnTransformer
###############################################################################

def build_preprocessor(df_sample: pd.DataFrame):
    target = "prezo_euros"
    numeric_cols = df_sample.select_dtypes("number").columns.tolist()
    numeric_cols.remove(target)
    engineered = [
        "antiguedad",
        "superficie_total_m2",
        "ratio_exterior_total",
        "densidad_arboles",
        "orient_sin",
        "orient_cos",
    ]
    numeric_cols += engineered

    categorical_cols = [c for c in df_sample.select_dtypes("object").columns if c != "orientacion"]

    ordinal_maps = {
        "calidade_materiais": ["Baixa", "Media", "Alta"],
        "acceso_transporte_publico": ["Malo", "Regular", "Bo", "Moi bo"],
        "eficiencia_enerxetica": ["G", "F", "E", "D", "C", "B", "A"],
    }
    ord_feats = list(ordinal_maps)
    ord_cats = [ordinal_maps[c] for c in ord_feats]
    onehot_feats = [c for c in categorical_cols if c not in ord_feats]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer()),
        ("scale", RobustScaler()),
    ])

    ord_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(categories=ord_cats)),
    ])

    ohe_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("ord", ord_pipe, ord_feats),
        ("ohe", ohe_pipe, onehot_feats),
    ], n_jobs=-1)

    return pre, target

###############################################################################
# 4. MLP + RandomizedSearchCV
###############################################################################

def build_model_with_search(df_sample: pd.DataFrame, random_state: int = 42):
    pre, target = build_preprocessor(df_sample)
    base_mlp = MLPRegressor(max_iter=500, early_stopping=True, solver="adam", random_state=random_state)
    pipe = Pipeline([
        ("eng", FeatureEngineer()),
        ("pre", pre),
        ("var", VarianceThreshold(0.0)),
        ("mlp", base_mlp),
    ])

    param_dist = {
        "mlp__hidden_layer_sizes": [(256,128,64), (256,128), (128,128,64), (256,256,128,64)],
        "mlp__alpha": loguniform(1e-5, 1e-2),
        "mlp__learning_rate_init": loguniform(1e-4, 5e-3),
        "mlp__activation": ["relu", "tanh"],
    }

    rmse = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=25, cv=3, scoring=rmse, random_state=random_state, n_jobs=-1, verbose=1)
    return search, target

###############################################################################
# 5. Train + submit
###############################################################################

def train_and_submit(train_csv: str, test_csv: str, submission_csv: str, val_size: float = 0.2):
    df = pd.read_csv(train_csv)
    search, target = build_model_with_search(df)
    X = df.drop(columns=[target])
    y = df[target].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=42, shuffle=True)

    search.fit(X_tr, y_tr)
    best_model = search.best_estimator_
    print("Mejores hiper-parámetros:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    val_rmse = mean_squared_error(y_val, best_model.predict(X_val), squared=False)
    print(f"Validation RMSE: {val_rmse:.2f}")

    best_model.fit(X, y)
    joblib.dump(best_model, "nn_model.joblib")

    df_test = pd.read_csv(test_csv)
    preds = best_model.predict(df_test)
    pd.DataFrame({"id": df_test["id"], "prezo_euros": preds}).to_csv(submission_csv, index=False)
    print(f"Submission guardada en {submission_csv}")

###############################################################################
# 6. CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--test_csv", default="test.csv")
    parser.add_argument("--submission", default="submissions.csv")
    args = parser.parse_args()
    train_and_submit(args.train_csv, args.test_csv, args.submission)