import pytest

from src.gtfs_graph import load_zonal_graph
from src.traffic import ConfigTrafico, ModeloTrafico


@pytest.fixture(scope="session")
def g():
    """Grafo de la red zonal (300 paradas), compartido por todas las pruebas."""
    return load_zonal_graph()


@pytest.fixture(scope="session")
def modelo(g):
    """Modelo de trafico con los parametros por defecto (peso 0.5, semilla 0)."""
    return ModeloTrafico(g, cfg=ConfigTrafico(peso=0.5), semilla=0)
