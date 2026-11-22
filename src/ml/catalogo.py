"""Catalogo de modelos para la aplicacion: todos los modelos entrenados, uno por clave.

La web deja cambiar entre estos modelos, asi que se guardan todos (no solo el ganador de
cada tarea):

  clave    regresion (tiempo del tramo)   clasificacion (nivel de congestion)
  base     media de entrenamiento         clase mas frecuente
  lineal   regresion lineal               (no clasifica)
  arbol    arbol de decision              arbol de decision
  rna      red neuronal (MLP)             red neuronal (MLP)

Se ajustan con todos los datos y con los hiperparametros que mas veces eligio la
validacion cruzada anidada (ver `data/resultados_regresion.json` y
`data/resultados_clasificacion.json`), sin repetir la busqueda.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import clasificacion, regresion

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOGO = ROOT / "data" / "modelos_ml.joblib"

NOMBRES = {"base": "Línea base", "lineal": "Regresión lineal", "arbol": "Árbol de decisión",
           "rna": "Red neuronal"}
CLASIFICAN = ("base", "arbol", "rna")

# hiperparametros mas elegidos por la validacion cruzada anidada
HIPER_REG = {
    "arbol": {"m__max_depth": 6, "m__min_samples_leaf": 5},
    "rna": {"m__hidden_layer_sizes": (32,), "m__alpha": 1e-4},
}
HIPER_CLF = {
    "arbol": {"m__max_depth": 8, "m__min_samples_leaf": 50},
    "rna": {"m__hidden_layer_sizes": (32,)},
}
_REG = {"base": "Línea base (media)", "lineal": "Regresión lineal", "arbol": "Árbol de decisión",
        "rna": "Red neuronal"}
_CLF = {"base": "Línea base (clase más frecuente)", "arbol": "Árbol de decisión", "rna": "Red neuronal"}


def entrenar(df: pd.DataFrame) -> dict:
    """Ajusta todos los modelos con todos los datos. Devuelve {"regresion": {...}, "clasificacion": {...}}."""
    X, y, _ = regresion.preparar(df)
    reg = {}
    for clave, nombre in _REG.items():
        pipe = regresion.modelos()[nombre][0]
        reg[clave] = pipe.set_params(**HIPER_REG.get(clave, {})).fit(X, y)

    Xc, yc, _, _ = clasificacion.preparar(df)
    clf = {}
    for clave, nombre in _CLF.items():
        pipe = clasificacion.modelos()[nombre][0]
        clf[clave] = pipe.set_params(**HIPER_CLF.get(clave, {})).fit(Xc, yc)
    return {"regresion": reg, "clasificacion": clf, "columnas": list(X.columns)}


def guardar(catalogo: dict) -> None:
    joblib.dump(catalogo, CATALOGO)


def cargar() -> dict:
    return joblib.load(CATALOGO)


def disponible() -> bool:
    return CATALOGO.exists()
