"""Pruebas individuales de la busqueda voraz primero-el-mejor (greedy)."""
import pytest

from src.search import search
from tests.helpers import PARES, camino_valido


@pytest.mark.parametrize("o,d", PARES)
def test_greedy_encuentra_ruta_valida(g, o, d):
    r = search(g, o, d, "greedy")
    assert r.found
    assert r.path[0] == g.resolve(o) and r.path[-1] == g.resolve(d)
    assert camino_valido(g, r.path)


@pytest.mark.parametrize("o,d", PARES)
def test_greedy_expande_pocos_nodos(g, o, d):
    """La voraz se guia solo por h: expande <= que UCS (a costa de optimalidad)."""
    assert search(g, o, d, "greedy").nodes_expanded <= search(g, o, d, "ucs").nodes_expanded


def test_greedy_no_siempre_es_optimo(g):
    difs = [search(g, o, d, "greedy").path_cost_m - search(g, o, d, "ucs").path_cost_m
            for o, d in PARES]
    assert max(difs) > 0


def test_greedy_metricas_coherentes(g):
    r = search(g, *PARES[0], "greedy")
    assert r.nodes_expanded > 0 and r.max_frontier > 0
    assert r.elapsed_ms >= 0 and r.peak_kib > 0
