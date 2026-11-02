"""Pruebas del entorno de trafico dinamico y del viaje simulado con replanificacion."""
import pytest

from src.agent import RoutePlannerAgent
from src.search import search
from src.simulacion import POLITICAS, simular_viaje
from src.traffic import EntornoTrafico, parse_hora

PAR = ("Br. Chicó Norte III Sector", "Br. Sucre")
HORA = "17:00"


def _aristas(g, n=40):
    return [(a, b) for a in g.nodes() for b in g.neighbors(a) if a < b][:n]


# --- entorno -------------------------------------------------------------
def test_entorno_reproducible_por_semilla(g, modelo):
    def traza(seed):
        e = EntornoTrafico(modelo, parse_hora(HORA), True, seed)
        out = []
        for _ in range(5):
            e.avanzar(300)
            out.append([round(e.m_real(a, b), 9) for a, b in _aristas(g)])
        return out

    assert traza(1) == traza(1)
    assert traza(1) != traza(2)


def test_entorno_estatico_coincide_con_el_modelo(g, modelo):
    e = EntornoTrafico(modelo, parse_hora(HORA), dinamico=False)
    for _ in range(4):
        for a, b in _aristas(g):
            assert e.m_real(a, b) == pytest.approx(modelo.m_esperado(a, b, e.t))
        e.avanzar(600)


def test_entorno_dinamico_se_aparta_del_patron_tipico(g, modelo):
    e = EntornoTrafico(modelo, parse_hora(HORA), True, 7)
    e.avanzar(1800)
    difs = [abs(e.m_real(a, b) - modelo.m_esperado(a, b, e.t)) for a, b in _aristas(g, 200)]
    assert max(difs) > 0.05


def test_congestion_real_nunca_baja_de_flujo_libre(g, modelo):
    e = EntornoTrafico(modelo, parse_hora("03:00"), True, 3)
    for _ in range(10):
        e.avanzar(240)
        assert all(m >= 1.0 for m in e.todos_m_real().values())


def test_aparecen_picos_repentinos(modelo):
    e = EntornoTrafico(modelo, parse_hora(HORA), True, 5)
    e.avanzar(8 * 3600)                    # 8 h simuladas: con ~4 picos/h tiene que haber
    assert e.n_picos >= 5


# --- viaje simulado ------------------------------------------------------
@pytest.mark.parametrize("politica", POLITICAS)
def test_el_agente_llega_y_el_viaje_es_consistente(g, modelo, politica):
    v = simular_viaje(g, modelo, *PAR, "astar", politica, "local", HORA, True, semilla=2)
    assert v.existe_ruta
    assert v.ruta_recorrida[0] == g.name(g.resolve(PAR[0]))
    assert v.ruta_recorrida[-1] == g.name(g.resolve(PAR[1]))
    assert len(v.pasos) == len(v.ruta_recorrida) - 1
    ids = [g.resolve(n) for n in v.ruta_recorrida]
    assert all(b in g.neighbors(a) for a, b in zip(ids, ids[1:]))      # solo se mueve a vecinas
    assert v.dist_real_m == pytest.approx(sum(g.cost(a, b) for a, b in zip(ids, ids[1:])), abs=0.2)
    assert v.tiempo_real_s > 0 and v.coste_real >= v.dist_real_m - 0.2
    assert v.transbordos == sum(p.transbordo for p in v.pasos)


def test_politica_nunca_sigue_el_plan_inicial(g, modelo):
    v = simular_viaje(g, modelo, *PAR, "astar", "nunca", "local", HORA, True, semilla=2)
    assert v.replanificaciones == 0
    assert v.ruta_recorrida == v.ruta_planeada


def test_politica_cada_parada_replanifica_antes_de_cada_tramo(g, modelo):
    v = simular_viaje(g, modelo, *PAR, "astar", "cada_parada", "local", HORA, True, semilla=2)
    assert v.replanificaciones == len(v.pasos) - 1
    assert sum(p.replanifico for p in v.pasos) == v.replanificaciones


def test_entorno_estatico_no_obliga_a_replanificar_con_umbral(g, modelo):
    v = simular_viaje(g, modelo, *PAR, "astar", "umbral", "local", HORA, dinamico=False, semilla=2)
    assert v.replanificaciones == 0 and v.picos == 0


def test_simulacion_reproducible(g, modelo):
    args = (g, modelo, *PAR, "astar", "umbral", "local", HORA, True)
    a, b = simular_viaje(*args, semilla=9), simular_viaje(*args, semilla=9)
    assert (a.ruta_recorrida, a.tiempo_real_s, a.replanificaciones) == \
           (b.ruta_recorrida, b.tiempo_real_s, b.replanificaciones)


def test_plan_inicial_coincide_con_la_busqueda_directa(g, modelo):
    v = simular_viaje(g, modelo, *PAR, "astar", "nunca", "local", HORA, True, semilla=1)
    mult = modelo.mult_esperado(parse_hora(HORA))
    r = search(g, *PAR, "astar", mult=mult, peso_trafico=modelo.cfg.peso,
               v_libre_ms=modelo.v_libre_ms)
    assert v.ruta_planeada == r.path_names
    assert v.tiempo_planeado_s == r.path_time_s


def test_observabilidad_global_tambien_llega(g, modelo):
    v = simular_viaje(g, modelo, *PAR, "astar", "umbral", "global", HORA, True, semilla=4)
    assert v.existe_ruta


def test_politica_desconocida_lanza_error(g, modelo):
    with pytest.raises(ValueError):
        simular_viaje(g, modelo, *PAR, politica="inexistente")


# --- agente: reutilizar un plan ya calculado --------------------------------
def test_agente_reutiliza_plan_calculado(g):
    o, d = PAR
    plan = search(g, o, d, "astar")
    log = RoutePlannerAgent(g).run(o, d, "astar", plan=plan)
    assert log.plan is plan
    assert log.ruta == plan.path_names and log.transbordos == plan.transbordos
