"""Pruebas del agrupamiento con K-Medias."""
import numpy as np
import pandas as pd
import pytest

from src.ml import agrupamiento as A
from src.ml import dataset


def _perfiles_sinteticos(n_por_grupo=30, niveles=(8, 15, 30), ruido=0.5, seed=0):
    rng = np.random.default_rng(seed)
    filas = [nivel + rng.normal(0, ruido, len(A.HORAS)) for nivel in niveles for _ in range(n_por_grupo)]
    return pd.DataFrame(filas, columns=A.HORAS)


def test_recupera_grupos_sinteticos():
    P = _perfiles_sinteticos()
    etiquetas, _ = A.ajustar(P, 3)
    for g in range(3):
        assert len(set(etiquetas[g * 30:(g + 1) * 30])) == 1   # cada nivel cae en un solo grupo
    assert len(set(etiquetas)) == 3


def test_grupos_ordenados_por_velocidad_media():
    _, centros = A.ajustar(_perfiles_sinteticos(), 3)
    medias = centros.mean(axis=1)
    assert (np.diff(medias) > 0).all()


def test_etiquetas_independientes_de_la_semilla_del_orden():
    P = _perfiles_sinteticos()
    e1, _ = A.ajustar(P, 3)
    e2, _ = A.ajustar(P.sample(frac=1, random_state=1).sort_index(), 3)
    assert (e1 == e2).all()


def test_silueta_alta_con_grupos_separados():
    sel = A.evaluar_k(_perfiles_sinteticos(), ks=range(2, 5))
    assert sel.loc[sel.k == 3, "silueta"].iloc[0] > 0.8
    assert sel.loc[sel.silueta.idxmax(), "k"] == 3


def test_estabilidad_alta_con_grupos_separados():
    assert A.estabilidad(_perfiles_sinteticos(), 3, repeticiones=5)["ari_medio"] > 0.95


def test_asignar_devuelve_el_centroide_mas_cercano():
    P = _perfiles_sinteticos()
    etiquetas, centros = A.ajustar(P, 3)
    assert (A.asignar(P, centros) == etiquetas).all()


@pytest.mark.skipif(not dataset.CSV_PATH.exists(), reason="falta data/dataset_ml.csv")
def test_perfiles_del_dataset_real():
    P = A.perfiles(dataset.cargar())
    assert list(P.columns) == A.HORAS
    assert not P.isna().any().any()
    assert (P.values > 0).all() and len(P) > 400
