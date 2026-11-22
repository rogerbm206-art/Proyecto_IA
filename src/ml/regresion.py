"""Regresion: predecir el tiempo programado de un tramo (`tiempo_s`).

Modelos: regresion lineal (linea base), arbol de decision y red neuronal (MLP).
Todos reciben las mismas variables y se evaluan con la misma validacion cruzada
AGRUPADA POR TRAMO: un tramo nunca esta a la vez en entrenamiento y en prueba, asi
la metrica mide la capacidad de predecir tramos que el modelo no ha visto.

Variables de entrada (no se usan `n_obs` ni la velocidad: no se conocen al predecir
y la velocidad contiene el objetivo):
  hora (y su forma ciclica sin/cos), dist_m, n_rutas, viajes_tramo, lat, lon,
  grado_u, grado_v y tipo_dia (one-hot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

OBJETIVO = "tiempo_s"
NUMERICAS = ["hora", "hora_sin", "hora_cos", "dist_m", "n_rutas", "viajes_tramo",
             "lat", "lon", "grado_u", "grado_v"]
CATEGORICAS = ["tipo_dia"]
SEMILLA = 0


def atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de entrada de los modelos (misma para regresion y clasificacion)."""
    X = df.copy()
    ang = 2 * np.pi * X["hora"] / 24
    X["hora_sin"], X["hora_cos"] = np.sin(ang), np.cos(ang)
    return X[NUMERICAS + CATEGORICAS]


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, grupos). El grupo de cada fila es su tramo."""
    return atributos(df), df[OBJETIVO], df["u"] + "|" + df["v"]


def _preproceso() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])


def modelos() -> dict[str, tuple[Pipeline, dict]]:
    """nombre -> (pipeline, rejilla de hiperparametros para GridSearchCV)."""
    pre = _preproceso
    return {
        "Línea base (media)": (Pipeline([("m", DummyRegressor())]), {}),
        "Regresión lineal": (Pipeline([("pre", pre()), ("m", LinearRegression())]), {}),
        "Árbol de decisión": (
            Pipeline([("pre", pre()), ("m", DecisionTreeRegressor(random_state=SEMILLA))]),
            {"m__max_depth": [4, 6, 8, 12, 16], "m__min_samples_leaf": [5, 20, 50]}),
        "Red neuronal": (
            Pipeline([("pre", pre()), ("m", MLPRegressor(
                max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=SEMILLA))]),
            {"m__hidden_layer_sizes": [(32,), (64, 32)], "m__alpha": [1e-4, 1e-2]}),
    }


def metricas(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"mae_s": mean_absolute_error(y, pred),
            "rmse_s": mean_squared_error(y, pred) ** 0.5,
            "r2": r2_score(y, pred)}


def evaluar(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, k: int = 5, k_interno: int = 3):
    """Validacion cruzada agrupada por tramo, con seleccion de hiperparametros anidada.

    Devuelve (filas por pliegue, predicciones fuera de muestra por modelo).
    """
    externo = GroupKFold(n_splits=k)
    filas = []
    oof = {n: np.zeros(len(y)) for n in modelos()}
    for fold, (tr, te) in enumerate(externo.split(X, y, grupos), start=1):
        for nombre, (pipe, rejilla) in modelos().items():
            if rejilla:
                modelo = GridSearchCV(pipe, rejilla, cv=GroupKFold(n_splits=k_interno),
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
                modelo.fit(X.iloc[tr], y.iloc[tr], groups=grupos.iloc[tr])
                params = {p.removeprefix("m__"): v for p, v in modelo.best_params_.items()}
            else:
                modelo, params = pipe.fit(X.iloc[tr], y.iloc[tr]), {}
            pred = modelo.predict(X.iloc[te])
            oof[nombre][te] = pred
            filas.append({"modelo": nombre, "pliegue": fold, "hiperparametros": str(params),
                          **metricas(y.iloc[te].to_numpy(), pred)})
    return pd.DataFrame(filas), oof


def ajustar_final(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, nombre: str):
    """Reajusta el modelo `nombre` con todos los datos (seleccionando hiperparametros)."""
    pipe, rejilla = modelos()[nombre]
    if not rejilla:
        return pipe.fit(X, y)
    gs = GridSearchCV(pipe, rejilla, cv=GroupKFold(n_splits=5),
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X, y, groups=grupos)
    return gs.best_estimator_
