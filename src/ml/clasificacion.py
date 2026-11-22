"""Clasificacion: nivel de congestion de un tramo (baja, media, alta).

Etiqueta. Para cada (tramo, ruta) se toma como velocidad libre el percentil 90 de sus
velocidades programadas en el dia (v_libre) y el multiplicador de congestion de cada fila es

    m = v_libre / vel_kmh

    baja  : m <= 1,25   (velocidad >= 80 % de la libre)
    media : 1,25 < m <= 1,67
    alta  : m > 1,67    (velocidad < 60 % de la libre)

Los grupos con menos de 8 filas no tienen una velocidad libre fiable y se descartan.

Entrada. Las mismas variables que la regresion (`regresion.NUMERICAS` y `CATEGORICAS`).
Ni `vel_kmh` ni `tiempo_s` ni `n_obs` entran como variable: de la velocidad sale la
etiqueta, y el tiempo dividido entre la distancia es la velocidad.

Modelos: arbol de decision y red neuronal (MLP), mas una linea base que siempre
predice la clase mas frecuente. Validacion cruzada agrupada por tramo, igual que en
la regresion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import regresion

CLASES = ["baja", "media", "alta"]
UMBRAL_MEDIA = 1.25
UMBRAL_ALTA = 1.67
PERCENTIL_LIBRE = 0.9
MIN_FILAS_GRUPO = 8
SEMILLA = 0


def etiquetar(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con `v_libre_kmh`, `m_cong` y `nivel` (descarta grupos pequenos)."""
    d = df.copy()
    claves = ["u", "v", "ruta"]
    d = d[d.groupby(claves)["vel_kmh"].transform("size") >= MIN_FILAS_GRUPO].copy()
    d["v_libre_kmh"] = d.groupby(claves)["vel_kmh"].transform(lambda s: s.quantile(PERCENTIL_LIBRE))
    d["m_cong"] = np.maximum(1.0, d["v_libre_kmh"] / d["vel_kmh"])
    d["nivel"] = pd.cut(d["m_cong"], [0, UMBRAL_MEDIA, UMBRAL_ALTA, np.inf],
                        labels=CLASES, right=True).astype(str)
    return d.reset_index(drop=True)


def preparar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """(X, y, grupos, df etiquetado)."""
    d = etiquetar(df)
    X, _, grupos = regresion.preparar(d)
    return X, d["nivel"], grupos, d


def modelos() -> dict[str, tuple[Pipeline, dict]]:
    pre = regresion._preproceso
    return {
        "Línea base (clase más frecuente)": (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), {}),
        "Árbol de decisión": (
            Pipeline([("pre", pre()), ("m", DecisionTreeClassifier(random_state=SEMILLA))]),
            {"m__max_depth": [3, 5, 8, 12], "m__min_samples_leaf": [5, 20, 50]}),
        "Red neuronal": (
            Pipeline([("pre", pre()), ("m", MLPClassifier(
                alpha=1e-3, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=SEMILLA))]),
            {"m__hidden_layer_sizes": [(32,), (64, 32)]}),
    }


def metricas(y, pred) -> dict[str, float]:
    return {"exactitud": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", labels=CLASES, zero_division=0)}


def evaluar(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, k: int = 5, k_interno: int = 3):
    """Validacion cruzada agrupada por tramo con seleccion anidada de hiperparametros."""
    filas = []
    oof = {n: np.empty(len(y), dtype=object) for n in modelos()}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y, grupos), start=1):
        for nombre, (pipe, rejilla) in modelos().items():
            if rejilla:
                modelo = GridSearchCV(pipe, rejilla, cv=GroupKFold(n_splits=k_interno),
                                      scoring="f1_macro", n_jobs=-1)
                modelo.fit(X.iloc[tr], y.iloc[tr], groups=grupos.iloc[tr])
                params = {p.removeprefix("m__"): v for p, v in modelo.best_params_.items()}
            else:
                modelo, params = pipe.fit(X.iloc[tr], y.iloc[tr]), {}
            pred = modelo.predict(X.iloc[te])
            oof[nombre][te] = pred
            filas.append({"modelo": nombre, "pliegue": fold, "hiperparametros": str(params),
                          **metricas(y.iloc[te], pred)})
    return pd.DataFrame(filas), oof


def ajustar_final(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, nombre: str):
    pipe, rejilla = modelos()[nombre]
    if not rejilla:
        return pipe.fit(X, y)
    gs = GridSearchCV(pipe, rejilla, cv=GroupKFold(n_splits=5), scoring="f1_macro", n_jobs=-1)
    gs.fit(X, y, groups=grupos)
    return gs.best_estimator_
