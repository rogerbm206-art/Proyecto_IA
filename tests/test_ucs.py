"""Pruebas individuales de UCS (coste uniforme / Dijkstra)."""
import pytest

from src.search import STRATEGIES, search
from tests.helpers import PARES, camino_valido


@pytest.mark.parametrize("o,d", PARES)
def test_ucs_encuentra_ruta_valida(g, o, d):
    r = search(g, o, d, "ucs")
    assert r.found
    assert camino_valido(g, r.path)


@pytest.mark.parametrize("o,d", PARES)
def test_ucs_es_optimo_en_coste(g, o, d):
    """UCS obtiene el coste minimo: ningun otro algoritmo lo mejora."""
    opt = search(g, o, d, "ucs").path_cost_m
    for strat in STRATEGIES:
        assert search(g, o, d, strat).path_cost_m >= opt - 1e-6


@pytest.mark.parametrize("o,d", PARES)
def test_ucs_minimiza_transbordos_en_estos_pares(g, o, d):
    assert search(g, o, d, "ucs").transbordos <= 1


def test_ucs_metricas_coherentes(g):
    r = search(g, *PARES[0], "ucs")
    assert r.nodes_expanded > 0 and r.max_frontier > 0
    assert r.elapsed_ms >= 0 and r.peak_kib > 0
