"""Pruebas individuales de BFS (busqueda en anchura)."""
import pytest

from src.search import search
from tests.helpers import OTROS, PARES, camino_valido


@pytest.mark.parametrize("o,d", PARES)
def test_bfs_encuentra_ruta_valida(g, o, d):
    r = search(g, o, d, "bfs")
    assert r.found
    assert r.path[0] == g.resolve(o) and r.path[-1] == g.resolve(d)
    assert camino_valido(g, r.path)
    assert r.path_edges == len(r.path) - 1


@pytest.mark.parametrize("o,d", PARES)
def test_bfs_minimiza_numero_de_tramos(g, o, d):
    """Ningun otro algoritmo encuentra una ruta con menos tramos que BFS."""
    base = search(g, o, d, "bfs").path_edges
    for strat in OTROS:
        assert base <= search(g, o, d, strat).path_edges


def test_bfs_metricas_coherentes(g):
    r = search(g, *PARES[0], "bfs")
    assert r.nodes_expanded > 0
    assert r.nodes_generated >= r.path_edges
    assert r.max_frontier > 0
    assert r.elapsed_ms >= 0 and r.peak_kib > 0
