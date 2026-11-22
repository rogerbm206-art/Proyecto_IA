"""Invariantes del dataset de ML (data/dataset_ml.csv)."""
import pytest

from src.ml import dataset

pytestmark = pytest.mark.skipif(not dataset.CSV_PATH.exists(), reason="falta data/dataset_ml.csv")


@pytest.fixture(scope="module")
def df():
    return dataset.cargar()


def test_columnas(df):
    esperadas = {"u", "v", "ruta", "hora", "tipo_dia", "dist_m", "n_rutas", "viajes_tramo",
                 "lat", "lon", "grado_u", "grado_v", "n_obs", "tiempo_s", "vel_kmh"}
    assert esperadas <= set(df.columns)


def test_sin_nulos_ni_duplicados(df):
    assert not df.isna().any().any()
    assert not df.duplicated(["u", "v", "ruta", "hora", "tipo_dia"]).any()


def test_rangos_depurados(df):
    assert df["hora"].between(0, 23).all()
    assert (df["tiempo_s"] > 0).all() and (df["tiempo_s"] < dataset.DT_MAX_S).all()
    assert (df["dist_m"] >= dataset.DIST_MIN_M).all()
    assert (df["vel_kmh"] <= dataset.VEL_MAX_KMH + 1).all()
    assert (df["n_obs"] >= dataset.MIN_OBS).all()
    assert set(df["tipo_dia"]) <= {"habil", "sabado", "domingo"}


def test_velocidad_consistente(df):
    esperada = df["dist_m"] / df["tiempo_s"] * 3.6
    assert (esperada - df["vel_kmh"]).abs().max() < 0.05


def test_tramos_existen_en_el_grafo(df):
    from src.gtfs_graph import load_zonal_graph
    g = load_zonal_graph()
    pares = df[["u", "v"]].drop_duplicates().itertuples(index=False)
    assert all(v in g.adj[u] for u, v in pares)
