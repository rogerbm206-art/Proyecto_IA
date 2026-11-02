"""Pruebas individuales de A* (f = g + h)."""
import pytest

from src.search import search
from tests.helpers import PARES, camino_valido


@pytest.mark.parametrize("o,d", PARES)
def test_astar_encuentra_ruta_valida(g, o, d):
    r = search(g, o, d, "astar")
    assert r.found
    assert camino_valido(g, r.path)


@pytest.mark.parametrize("o,d", PARES)
def test_astar_es_optimo(g, o, d):
    assert search(g, o, d, "astar").path_cost_m == pytest.approx(
        search(g, o, d, "ucs").path_cost_m)


@pytest.mark.parametrize("o,d", PARES)
def test_astar_expande_menos_que_ucs(g, o, d):
    """Con heuristica consistente, A* no expande mas nodos que UCS."""
    assert search(g, o, d, "astar").nodes_expanded <= search(g, o, d, "ucs").nodes_expanded


@pytest.mark.parametrize("o,d", PARES)
def test_astar_no_peor_que_bfs_en_coste(g, o, d):
    assert search(g, o, d, "astar").path_cost_m <= search(g, o, d, "bfs").path_cost_m + 1e-6


def test_astar_metricas_coherentes(g):
    r = search(g, *PARES[0], "astar")
    assert r.nodes_expanded > 0 and r.max_frontier > 0
    assert r.elapsed_ms >= 0 and r.peak_kib > 0
