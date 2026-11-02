"""Pruebas individuales de DFS (busqueda en profundidad)."""
import pytest

from src.search import search
from tests.helpers import PARES, camino_valido


@pytest.mark.parametrize("o,d", PARES)
def test_dfs_encuentra_ruta_valida(g, o, d):
    r = search(g, o, d, "dfs")
    assert r.found
    assert r.path[0] == g.resolve(o) and r.path[-1] == g.resolve(d)
    assert camino_valido(g, r.path)


def test_dfs_no_es_optimo(g):
    """En algun par, DFS devuelve una ruta estrictamente mas cara que UCS."""
    difs = [search(g, o, d, "dfs").path_cost_m - search(g, o, d, "ucs").path_cost_m
            for o, d in PARES]
    assert max(difs) > 0


def test_dfs_metricas_coherentes(g):
    r = search(g, *PARES[0], "dfs")
    assert r.nodes_expanded > 0
    assert r.max_frontier > 0
    assert r.elapsed_ms >= 0 and r.peak_kib > 0
