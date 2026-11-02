"""Pruebas del modelo de trafico y del coste combinado distancia + trafico."""
import pytest

from src.search import RouteProblem, search, ucs
from src.traffic import (ConfigTrafico, ModeloTrafico, PerfilHorario, coste_combinado,
                         fmt_hora, parse_hora)
from tests.helpers import PARES, mini_red_con_rutas

HORA_PICO = parse_hora("17:00")


# --- horas ---------------------------------------------------------------
def test_parse_y_fmt_hora():
    assert parse_hora("16:30") == 16.5 * 3600
    assert parse_hora("8") == 8 * 3600
    assert parse_hora(0) == 0
    assert parse_hora(25) == 3600            # da la vuelta al dia
    assert fmt_hora(parse_hora("07:05")) == "07:05"


# --- perfil horario ------------------------------------------------------
def test_perfil_cargado_valido():
    p = PerfilHorario.cargar()
    assert len(p.mult) == 24 and all(m >= 1.0 for m in p.mult)
    assert p.mult[3] == pytest.approx(1.0)               # madrugada: flujo libre
    assert max(p.mult) > 1.7                             # tarde: casi el doble de lento
    assert p.mult.index(max(p.mult)) in (15, 16, 17)
    assert 15 < p.v_libre_kmh < 30


def test_perfil_interpola_y_es_continuo():
    p = PerfilHorario.cargar()
    for h in range(24):                                  # exacto en el centro de cada hora
        assert p.m(h * 3600 + 1800) == pytest.approx(p.mult[h])
    t = 0.0
    while t < 86400:                                     # sin saltos (propiedad FIFO)
        assert abs(p.m(t + 60) - p.m(t)) < 0.01
        t += 60


# --- coste combinado -----------------------------------------------------
def test_coste_combinado_interpola_entre_distancia_y_tiempo():
    assert coste_combinado(100, 2.0, 0.0) == 100          # solo distancia
    assert coste_combinado(100, 2.0, 1.0) == 200          # distancia * congestion
    assert coste_combinado(100, 2.0, 0.5) == 150          # mezcla
    assert coste_combinado(100, 1.0, 0.7) == 100          # sin congestion no cambia


# --- modelo (lo que el agente conoce) ----------------------------------
def test_modelo_multiplicadores_validos(g, modelo):
    mult = modelo.mult_esperado(HORA_PICO)
    assert len(mult) == 2 * g.number_of_edges()
    assert all(m >= 1.0 for m in mult.values())
    for (a, b), m in mult.items():
        assert mult[(b, a)] == m                          # simetrico


def test_modelo_reproducible_por_semilla(g):
    a = ModeloTrafico(g, semilla=3).mult_esperado(HORA_PICO)
    b = ModeloTrafico(g, semilla=3).mult_esperado(HORA_PICO)
    c = ModeloTrafico(g, semilla=4).mult_esperado(HORA_PICO)
    assert a == b and a != c


def test_trafico_mayor_en_hora_pico_que_de_madrugada(modelo):
    pico = modelo.mult_esperado(HORA_PICO)
    noche = modelo.mult_esperado(parse_hora("03:00"))
    assert sum(pico.values()) > 1.5 * sum(noche.values())


# --- busqueda con trafico ------------------------------------------------
def test_sin_trafico_el_resultado_es_el_original(g, modelo):
    o, d = PARES[0]
    base = search(g, o, d, "astar")
    mult = modelo.mult_esperado(HORA_PICO)
    con_peso_cero = search(g, o, d, "astar", mult=mult, peso_trafico=0.0)
    sin_mult = search(g, o, d, "astar", mult=None, peso_trafico=0.5)
    for r in (con_peso_cero, sin_mult):
        assert r.path == base.path and r.path_cost_m == base.path_cost_m


@pytest.mark.parametrize("o,d", PARES)
def test_ucs_y_astar_siguen_siendo_optimos_con_trafico(g, modelo, o, d):
    mult = modelo.mult_esperado(HORA_PICO)
    kw = dict(mult=mult, peso_trafico=0.5)
    opt = search(g, o, d, "ucs", **kw).path_cost_m
    assert search(g, o, d, "astar", **kw).path_cost_m == pytest.approx(opt, abs=0.2)
    for alg in ("bfs", "dfs", "greedy"):
        assert search(g, o, d, alg, **kw).path_cost_m >= opt - 0.2


def test_mas_peso_de_trafico_no_abarata_el_coste(g, modelo):
    mult = modelo.mult_esperado(HORA_PICO)
    o, d = PARES[1]
    costes = [search(g, o, d, "ucs", mult=mult, peso_trafico=w).path_cost_m for w in (0.0, 0.5, 1.0)]
    assert costes[0] <= costes[1] <= costes[2]


def test_heuristica_sigue_admisible_con_trafico(g, modelo):
    mult = modelo.mult_esperado(HORA_PICO)
    meta = g.resolve("Universidad EAN")
    for n in g.nodes()[::12]:
        p = RouteProblem(g, n, meta, mult=mult, peso_trafico=0.5)
        real = ucs(p).path_cost_m if n != meta else 0.0
        assert p.h((n, None)) <= real + 0.1


def test_el_trafico_puede_cambiar_la_ruta_elegida(g, modelo):
    """En hora pico y con peso 1 se evitan tramos congestionados aunque sean mas largos."""
    o, d = "Br. Chicó Norte III Sector", "Br. Sucre"
    mult = modelo.mult_esperado(HORA_PICO)
    libre = search(g, o, d, "astar")
    consciente = search(g, o, d, "astar", mult=mult, peso_trafico=1.0)
    assert consciente.path != libre.path
    # la ruta consciente es mejor SEGUN el trafico: menos tiempo estimado que la de solo-distancia
    libre_con_trafico = RouteProblem(g, g.resolve(o), g.resolve(d), mult=mult, peso_trafico=1.0)
    t_libre = (sum(g.cost(a, b) * libre_con_trafico.m(a, b) for a, b in zip(libre.path, libre.path[1:]))
               + libre.transbordos * 800.0) / libre_con_trafico.v_libre
    assert consciente.path_time_s <= t_libre + 1.0


def test_tiempo_estimado_sin_trafico_es_distancia_a_velocidad_libre(g):
    o, d = PARES[0]
    r = search(g, o, d, "astar")
    p = RouteProblem(g, g.resolve(o), g.resolve(d))
    esperado = (r.path_dist_m + r.transbordos * 800.0) / p.v_libre
    assert r.path_time_s == pytest.approx(esperado, abs=0.2)


def test_ruta_inicial_evita_penalizar_seguir_en_el_mismo_bus():
    tg = mini_red_con_rutas()          # A-B-D por R1 (200 m) vs A-C-D por R2+R3
    seguir = search(tg, "A", "D", "ucs", ruta_inicial="R1")
    assert seguir.path == ["A", "B", "D"] and seguir.transbordos == 0 and seguir.path_cost_m == 200
    otro = search(tg, "A", "D", "ucs", ruta_inicial="R2")   # ya voy en R2: cambiar cuesta
    assert otro.path == ["A", "C", "D"] and otro.transbordos == 1 and otro.path_cost_m == 980
