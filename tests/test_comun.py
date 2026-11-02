"""Pruebas transversales a todos los algoritmos y del modelo de rutas fijas."""
import pytest

from src.search import RouteProblem, STRATEGIES, search, ucs
from tests.helpers import PARES, camino_valido, mini_red_con_rutas
from src.gtfs_graph import TransitGraph


@pytest.mark.parametrize("strat", list(STRATEGIES))
@pytest.mark.parametrize("o,d", PARES)
def test_cada_algoritmo_devuelve_camino_valido(g, strat, o, d):
    r = search(g, o, d, strat)
    assert r.found
    assert r.path[0] == g.resolve(o) and r.path[-1] == g.resolve(d)
    assert camino_valido(g, r.path)
    assert r.path_edges == len(r.path) - 1
    assert r.transbordos >= 0
    assert r.path_cost_m >= r.path_dist_m - 1e-6   # coste = distancia + penalizaciones


def test_heuristica_admisible(g):
    """h(n) nunca sobreestima el coste real minimo de n a la meta.

    `path_cost_m` viene redondeado a 0.1 m, de ahi la tolerancia.
    """
    meta = g.resolve("Universidad EAN")
    for n in g.nodes():
        p = RouteProblem(g, n, meta)
        real = ucs(p).path_cost_m if n != meta else 0.0
        assert p.h((n, None)) <= real + 0.1


def test_origen_igual_destino(g):
    for strat in STRATEGIES:
        r = search(g, "Cl 67", "Cl 67", strat)
        assert r.found and r.path_edges == 0 and r.path_cost_m == 0 and r.transbordos == 0


def test_sin_ruta_en_grafo_desconectado():
    e = {"dist_m": 100.0, "rutas": ["R1"], "n_rutas": 1, "trips": 9}
    tg = TransitGraph({"X": {"Y": e}, "Y": {"X": e}, "Z": {}},
                      {"X": (4.65, -74.06), "Y": (4.651, -74.06), "Z": (4.66, -74.05)},
                      {"X": "X", "Y": "Y", "Z": "Z"})
    for strat in STRATEGIES:
        r = search(tg, "X", "Z", strat)
        assert r.found is False and r.path == []


def test_penalizacion_de_transbordo_cambia_la_ruta_elegida():
    tg = mini_red_con_rutas()
    sin = search(tg, "A", "D", "ucs", transfer_penalty_m=0)
    con = search(tg, "A", "D", "ucs", transfer_penalty_m=800)
    assert sin.path == ["A", "C", "D"] and sin.transbordos == 1     # gana la mas corta
    assert con.path == ["A", "B", "D"] and con.transbordos == 0     # gana la de 1 ruta
