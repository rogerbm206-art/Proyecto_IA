"""Pruebas de la integracion de los modelos de ML con el agente y la web."""
import sys
from pathlib import Path

import pytest

from src.agent import RoutePlannerAgent
from src.ml import integracion
from src.search import search

pytestmark = pytest.mark.skipif(not integracion.modelos_disponibles(),
                                reason="faltan los modelos entrenados en data/")

PARES = [("Br. Chicó Norte III Sector", "Br. Sucre"), ("Estación Calle 100", "Br. Marly"),
         ("Br. Colombia", "Universidad De La Salle")]


@pytest.fixture(scope="module")
def pred(g):
    return integracion.PredictorTramos(g)


def test_tiempos_positivos_para_todos_los_tramos(g, pred):
    t = pred.tiempos(8, "habil")
    assert len(t) == g.number_of_edges()
    assert all(v >= 1.0 for v in t.values())
    assert all(a < b for a, b in t)


def test_mult_simetrico_y_al_menos_uno(g, pred):
    m = pred.mult(17, "habil")
    assert all(v >= 1.0 for v in m.values())
    assert all(m[(a, b)] == m[(b, a)] for a, b in m)


def test_clases_y_grupos_validos(g, pred):
    assert set(pred.clases(12, "habil").values()) <= {"baja", "media", "alta"}
    assert set(pred.grupos().values()) <= {1, 2, 3, 4}
    a, b = next(iter(pred.tiempos(8)))
    assert pred.clase(a, b, 8) == pred.clase(b, a, 8)


def test_la_hora_es_ciclica(pred):
    assert pred.tiempos(25, "habil") == pred.tiempos(1, "habil")


def test_dia_desconocido_da_error(pred):
    with pytest.raises(ValueError):
        pred.tiempos(8, "lunes")


def test_hora_pico_es_mas_lenta_que_la_madrugada(pred):
    pico = sum(pred.tiempos(17, "habil").values())
    noche = sum(pred.tiempos(23, "habil").values())
    assert pico > noche


def test_mult_desde_tiempos_recorta_a_uno(g):
    (u, v) = next((a, b) for a in g.adj for b in g.adj[a] if a < b)
    rapido = integracion.mult_desde_tiempos(g, {(u, v): 0.001}, 21.44)
    lento = integracion.mult_desde_tiempos(g, {(u, v): 10_000.0}, 21.44)
    assert rapido[(u, v)] == 1.0 and lento[(u, v)] > 1.0


@pytest.mark.parametrize("o,d", PARES)
def test_astar_sigue_siendo_optimo_con_el_coste_de_ml(g, pred, o, d):
    kw = dict(mult=pred.mult(17, "habil"), peso_trafico=1.0, v_libre_ms=pred.perfil.v_libre_ms)
    a = search(g, g.resolve(o), g.resolve(d), "astar", **kw)
    u = search(g, g.resolve(o), g.resolve(d), "ucs", **kw)
    assert a.found and abs(a.path_cost_m - u.path_cost_m) < 1e-6
    assert a.path_cost_m >= a.path_dist_m          # m >= 1: el coste nunca baja de la distancia


@pytest.mark.parametrize("o,d", PARES)
def test_el_agente_planifica_con_el_trafico_de_ml(g, pred, o, d):
    agente = RoutePlannerAgent(g)
    r = agente.plan(o, d, "astar", mult=pred.mult(8, "habil"), peso_trafico=1.0,
                    v_libre_ms=pred.perfil.v_libre_ms)
    assert r.found and r.path_time_s > 0
    log = agente.run(o, d, "astar", plan=r)
    assert log.existe_ruta and log.ruta == r.path_names


@pytest.fixture(scope="module")
def web():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
    import app
    return app.app.test_client()


BASE = "/?origen=Br.+Colombia&destino=Universidad+De+La+Salle&alg=astar&hora=17:00&peso=1"


def test_web_con_fuente_ml(web):
    r = web.get(BASE + "&fuente=ml&dia=sabado")
    t = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "predicho por ML" in t and "Congestión prevista por el clasificador" in t
    assert 'id="colorpor"' in t and 'class="cg c-' in t


def test_web_perfil_no_menciona_ml_en_la_ruta(web):
    r = web.get(BASE)
    assert r.status_code == 200 and "predicho por ML" not in r.get_data(as_text=True)


def test_web_fuente_o_dia_invalidos_vuelven_al_defecto(web):
    t = web.get(BASE + "&fuente=otra&dia=lunes").get_data(as_text=True)
    assert "predicho por ML" not in t


def test_web_imagen_con_fuente_ml(web):
    r = web.get(BASE.replace("/?", "/ruta.png?") + "&fuente=ml")
    assert r.status_code == 200 and r.mimetype == "image/png"
