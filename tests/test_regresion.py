"""Pruebas del modulo de regresion (sobre una muestra pequena del dataset)."""
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from src.ml import dataset, regresion

pytestmark = pytest.mark.skipif(not dataset.CSV_PATH.exists(), reason="falta data/dataset_ml.csv")


@pytest.fixture(scope="module")
def datos():
    df = dataset.cargar()
    tramos = df[["u", "v"]].drop_duplicates().sample(60, random_state=0)
    df = df.merge(tramos, on=["u", "v"]).sample(3000, random_state=0).reset_index(drop=True)
    return regresion.preparar(df)


def test_sin_fuga_de_variables(datos):
    X, _, _ = datos
    assert not {"vel_kmh", "tiempo_s", "n_obs"} & set(X.columns)


def test_hora_ciclica(datos):
    X, _, _ = datos
    assert np.allclose(X["hora_sin"] ** 2 + X["hora_cos"] ** 2, 1.0)


def test_los_pliegues_no_comparten_tramos(datos):
    X, y, g = datos
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        assert not set(g.iloc[tr]) & set(g.iloc[te])


@pytest.mark.parametrize("nombre", ["Regresión lineal", "Árbol de decisión", "Red neuronal"])
def test_modelo_predice_mejor_que_la_media(datos, nombre):
    X, y, g = datos
    tr, te = next(GroupKFold(n_splits=4).split(X, y, g))
    base = regresion.modelos()["Línea base (media)"][0].fit(X.iloc[tr], y.iloc[tr])
    pipe = regresion.modelos()[nombre][0].fit(X.iloc[tr], y.iloc[tr])
    mae_base = regresion.metricas(y.iloc[te], base.predict(X.iloc[te]))["mae_s"]
    mae = regresion.metricas(y.iloc[te], pipe.predict(X.iloc[te]))["mae_s"]
    assert mae < mae_base


def test_metricas_de_prediccion_perfecta():
    y = np.array([10.0, 20.0, 30.0])
    m = regresion.metricas(y, y)
    assert m["mae_s"] == 0 and m["rmse_s"] == 0 and m["r2"] == 1
