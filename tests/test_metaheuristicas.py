"""Pruebas de las metaheuristicas (hc, sa, ga, aco, pso) sobre el mismo problema de rutas."""
import pytest

from src.metaheuristics import METAHEURISTICAS
from src.search import search
from tests.helpers import PARES, camino_valido, mini_red_con_rutas

PAR = PARES[0]   # Br. Colombia -> Universidad De La Salle


@pytest.mark.parametrize("alg", list(METAHEURISTICAS))
@pytest.mark.parametrize("o,d", PARES[:3])
def test_camino_valido_y_no_mejor_que_el_optimo(g, alg, o, d):
    r = search(g, o, d, alg)
    opt = search(g, o, d, "ucs").path_cost_m
    assert r.found
    assert r.path[0] == g.resolve(o) and r.path[-1] == g.resolve(d)
    assert camino_valido(g, r.path)
    assert len(set(r.path)) == len(r.path)          # camino simple
    assert r.path_cost_m >= opt - 0.1               # ninguna supera al optimo
    assert r.path_cost_m >= r.path_dist_m - 1e-6
    assert r.nodes_expanded > 0 and r.max_frontier >= 1


@pytest.mark.parametrize("alg", list(METAHEURISTICAS))
def test_misma_semilla_mismo_resultado(g, alg):
    a = search(g, *PAR, alg, semilla=3)
    b = search(g, *PAR, alg, semilla=3)
    assert a.path == b.path and a.path_cost_m == b.path_cost_m


@pytest.mark.parametrize("alg", list(METAHEURISTICAS))
def test_origen_igual_destino(g, alg):
    r = search(g, PAR[0], PAR[0], alg)
    assert r.found and r.path_edges == 0 and r.path_cost_m == 0


@pytest.mark.parametrize("alg", list(METAHEURISTICAS))
def test_coste_incluye_transbordos(alg):
    """En la red de juguete la ruta por B (sin transbordo, 200 m) y la de C
    (180 m + 800 m de transbordo = 980) deben dar el coste de su camino."""
    g = mini_red_con_rutas()
    r = search(g, "A", "D", alg)
    assert r.found
    esperado = {("A", "B", "D"): 200.0, ("A", "C", "D"): 980.0}
    assert abs(r.path_cost_m - esperado[tuple(r.path)]) < 0.1
    assert r.path_cost_m == 200.0                   # el optimo, facil de encontrar aqui


def test_con_trafico_sigue_dando_camino_valido(g, modelo):
    mult = modelo.mult_esperado(17 * 3600.0)
    for alg in METAHEURISTICAS:
        r = search(g, *PAR, alg, mult=mult, peso_trafico=0.5)
        opt = search(g, *PAR, "ucs", mult=mult, peso_trafico=0.5).path_cost_m
        assert r.found and camino_valido(g, r.path)
        assert r.path_cost_m >= opt - 0.1
        assert r.path_time_s > 0


def test_estrategia_desconocida(g):
    with pytest.raises(ValueError):
        search(g, *PAR, "no_existe")
