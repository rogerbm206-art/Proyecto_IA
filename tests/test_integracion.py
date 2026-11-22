"""Pruebas de la integracion de los modelos de ML con el agente y la web."""
import re
import sys
from pathlib import Path

import pytest

from src.agent import RoutePlannerAgent
from src.ml import catalogo, integracion
from src.search import search

pytestmark = pytest.mark.skipif(not integracion.modelos_disponibles(),
                                reason="falta data/modelos_ml.joblib (scripts/guardar_modelos.py)")

PARES = [("Br. Chicó Norte III Sector", "Br. Sucre"), ("Estación Calle 100", "Br. Marly"),
         ("Br. Colombia", "Universidad De La Salle")]
MODELOS = list(catalogo.NOMBRES)               # base, lineal, arbol, rna
CLASIFICAN = [m for m in MODELOS if m in catalogo.CLASIFICAN]


@pytest.fixture(scope="module")
def pred(g):
    return integracion.PredictorTramos(g)


@pytest.mark.parametrize("modelo", MODELOS)
def test_tiempos_positivos_para_todos_los_tramos(g, pred, modelo):
    t = pred.tiempos(8, "habil", modelo)
    assert len(t) == g.number_of_edges()
    assert all(v >= 1.0 for v in t.values())
    assert all(a < b for a, b in t)


def test_la_linea_base_predice_siempre_lo_mismo(pred):
    assert len(set(pred.tiempos(8, "habil", "base").values())) == 1
    assert pred.tiempos(8, "habil", "base") == pred.tiempos(17, "domingo", "base")


@pytest.mark.parametrize("modelo", ["lineal", "arbol", "rna"])
def test_los_modelos_distinguen_tramos_y_horas(pred, modelo):
    assert len(set(pred.tiempos(8, "habil", modelo).values())) > 10   # un arbol de profundidad 6 tiene <= 64 hojas
    assert pred.tiempos(8, "habil", modelo) != pred.tiempos(17, "habil", modelo)


def test_los_modelos_dan_predicciones_distintas(pred):
    t = {m: pred.tiempos(17, "habil", m) for m in MODELOS}
    assert len({tuple(round(v, 3) for v in x.values()) for x in t.values()}) == len(MODELOS)


@pytest.mark.parametrize("modelo", MODELOS)
def test_mult_simetrico_y_al_menos_uno(pred, modelo):
    m = pred.mult(17, "habil", modelo)
    assert all(v >= 1.0 for v in m.values())
    assert all(m[(a, b)] == m[(b, a)] for a, b in m)


@pytest.mark.parametrize("modelo", CLASIFICAN)
def test_clases_validas(pred, modelo):
    assert pred.clasifica(modelo)
    assert set(pred.clases(12, "habil", modelo).values()) <= {"baja", "media", "alta"}
    a, b = next(iter(pred.tiempos(8)))
    assert pred.clase(a, b, 8, "habil", modelo) == pred.clase(b, a, 8, "habil", modelo)


def test_la_regresion_lineal_no_clasifica(pred):
    assert not pred.clasifica("lineal")
    with pytest.raises(ValueError):
        pred.clases(8, "habil", "lineal")


def test_modelo_desconocido_da_error(pred):
    with pytest.raises(ValueError):
        pred.tiempos(8, "habil", "otro")


def test_grupos_validos(pred):
    assert set(pred.grupos().values()) <= {1, 2, 3, 4}


def test_la_hora_es_ciclica(pred):
    assert pred.tiempos(25, "habil") == pred.tiempos(1, "habil")


def test_dia_desconocido_da_error(pred):
    with pytest.raises(ValueError):
        pred.tiempos(8, "lunes")


def test_hora_pico_es_mas_lenta_que_la_madrugada(pred):
    assert sum(pred.tiempos(17, "habil").values()) > sum(pred.tiempos(23, "habil").values())


def test_mult_desde_tiempos_recorta_a_uno(g):
    (u, v) = next((a, b) for a in g.adj for b in g.adj[a] if a < b)
    rapido = integracion.mult_desde_tiempos(g, {(u, v): 0.001}, 21.44)
    lento = integracion.mult_desde_tiempos(g, {(u, v): 10_000.0}, 21.44)
    assert rapido[(u, v)] == 1.0 and lento[(u, v)] > 1.0


@pytest.mark.parametrize("modelo", MODELOS)
@pytest.mark.parametrize("o,d", PARES)
def test_astar_sigue_siendo_optimo_con_el_coste_de_cada_modelo(g, pred, modelo, o, d):
    kw = dict(mult=pred.mult(17, "habil", modelo), peso_trafico=1.0, v_libre_ms=pred.perfil.v_libre_ms)
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


@pytest.mark.parametrize("modelo", MODELOS)
def test_web_con_cada_modelo(web, modelo):
    r = web.get(BASE + f"&modelo={modelo}&dia=sabado")
    t = r.get_data(as_text=True)
    assert r.status_code == 200
    assert f"Modelo: {catalogo.NOMBRES[modelo]}" in t
    assert re.search(r'value="%s" checked' % modelo, t)           # el panel marca el modelo elegido
    assert 'id="colorpor"' in t and 'class="cg c-t"' in t
    if modelo in CLASIFICAN:
        assert re.search(r'class="cg c-(baja|media|alta)"', t)
    else:
        assert "no clasifica la congestión" in t


def test_web_panel_de_modelos_en_la_pagina_inicial(web):
    t = web.get("/").get_data(as_text=True)
    assert 'class="mlpanel"' in t
    assert len(re.findall(r'name="modelo" value=', t)) == 5      # sin ML + los 4 modelos
    assert re.search(r'value="perfil" checked', t)


def test_web_sin_ml_no_muestra_modelo(web):
    r = web.get(BASE)
    assert r.status_code == 200 and "Modelo: " not in r.get_data(as_text=True)


def test_web_modelo_o_dia_invalidos_vuelven_al_defecto(web):
    t = web.get(BASE + "&modelo=otro&dia=lunes&fuente=ml").get_data(as_text=True)
    assert "Modelo: " not in t and re.search(r'value="perfil" checked', t)


def test_web_imagen_con_modelo(web):
    r = web.get(BASE.replace("/?", "/ruta.png?") + "&modelo=arbol")
    assert r.status_code == 200 and r.mimetype == "image/png"


def test_web_la_ruta_cambia_con_el_modelo(web):
    def ruta(m):
        t = web.get("/?origen=Br.+Chic%C3%B3+Norte+III+Sector&destino=Br.+Sucre&alg=astar&hora=17:00&peso=1"
                    f"&modelo={m}").get_data(as_text=True)
        return re.search(r'Tiempo previsto por el modelo[^<]*', t).group(0)
    assert len({ruta(m) for m in MODELOS if m != "perfil"}) > 1
