"""Pruebas del modulo de clasificacion (sobre una muestra pequena del dataset)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.ml import clasificacion as C
from src.ml import dataset

pytestmark = pytest.mark.skipif(not dataset.CSV_PATH.exists(), reason="falta data/dataset_ml.csv")


@pytest.fixture(scope="module")
def datos():
    df = dataset.cargar()
    tramos = df[["u", "v"]].drop_duplicates().sample(80, random_state=0)
    return C.preparar(df.merge(tramos, on=["u", "v"]).reset_index(drop=True))


def test_etiquetas_validas_y_umbrales(datos):
    _, y, _, d = datos
    assert set(y) <= set(C.CLASES)
    assert (d.loc[d.nivel == "baja", "m_cong"] <= C.UMBRAL_MEDIA).all()
    assert (d.loc[d.nivel == "alta", "m_cong"] > C.UMBRAL_ALTA).all()
    medias = d.loc[d.nivel == "media", "m_cong"]
    assert ((medias > C.UMBRAL_MEDIA) & (medias <= C.UMBRAL_ALTA)).all()


def test_m_cong_nunca_menor_que_uno(datos):
    assert (datos[3]["m_cong"] >= 1.0).all()


def test_hay_las_tres_clases(datos):
    assert set(datos[1]) == set(C.CLASES)


def test_sin_fuga_de_variables(datos):
    X = datos[0]
    assert not {"vel_kmh", "tiempo_s", "n_obs", "m_cong", "v_libre_kmh", "nivel"} & set(X.columns)


def test_los_pliegues_no_comparten_tramos(datos):
    X, y, g, _ = datos
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        assert not set(g.iloc[tr]) & set(g.iloc[te])


def test_grupos_pequenos_se_descartan():
    df = pd.DataFrame({"u": ["a"] * 10 + ["b"] * 3, "v": ["c"] * 13, "ruta": ["r"] * 13,
                       "vel_kmh": np.linspace(10, 30, 13)})
    assert set(C.etiquetar(df)["u"]) == {"a"}


@pytest.mark.parametrize("nombre", ["Árbol de decisión", "Red neuronal"])
def test_modelo_supera_a_la_linea_base(datos, nombre):
    X, y, g, _ = datos
    tr, te = next(GroupKFold(n_splits=4).split(X, y, g))
    base = C.modelos()["Línea base (clase más frecuente)"][0].fit(X.iloc[tr], y.iloc[tr])
    pipe = C.modelos()[nombre][0].fit(X.iloc[tr], y.iloc[tr])
    f_base = C.metricas(y.iloc[te], base.predict(X.iloc[te]))["f1_macro"]
    f = C.metricas(y.iloc[te], pipe.predict(X.iloc[te]))["f1_macro"]
    assert f > f_base
