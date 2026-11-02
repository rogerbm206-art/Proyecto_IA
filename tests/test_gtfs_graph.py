"""Pruebas del modelo del grafo de la red zonal."""
import pytest

from src.gtfs_graph import TARGET_NODES, TransitGraph


def test_tamano(g):
    assert len(g.nodes()) == TARGET_NODES == 300
    assert g.number_of_edges() == 557


def test_nombres_unicos(g):
    """Cada parada tiene un nombre unico (se desambiguan las paradas partidas)."""
    assert len(set(g.names.values())) == len(g.nodes())


def test_conexo(g):
    start = g.nodes()[0]
    visto = {start}
    pila = [start]
    while pila:
        n = pila.pop()
        for m in g.neighbors(n):
            if m not in visto:
                visto.add(m)
                pila.append(m)
    assert visto == set(g.nodes())


def test_aristas_simetricas_y_positivas(g):
    for a in g.nodes():
        for b in g.neighbors(a):
            assert a in g.neighbors(b)
            assert g.cost(a, b) == g.cost(b, a) > 0
            e = g.edge(a, b)
            assert e["n_rutas"] >= 1
            assert isinstance(e["rutas"], list) and len(e["rutas"]) == e["n_rutas"]


def test_coordenadas_en_bogota(g):
    for lat, lon in g.coords.values():
        assert 4.63 <= lat <= 4.70 and -74.085 <= lon <= -74.04


def test_resolve_por_nombre_y_por_id(g):
    sid = g.resolve("Universidad EAN")
    assert g.name(sid) == "Universidad EAN"
    assert g.resolve(sid) == sid          # idempotente con stop_id
    # nombre repetido (paradas de sentido opuesto): devuelve una valida
    rep = g.resolve("Br. Porciúncula")
    assert g.name(rep) == "Br. Porciúncula"


def test_resolve_desconocido_lanza_error(g):
    with pytest.raises(KeyError):
        g.resolve("Parada Inexistente 999")


def test_export_networkx_coincide(g):
    h = g.to_networkx()
    assert h.number_of_nodes() == len(g.nodes())
    assert h.number_of_edges() == g.number_of_edges()


def test_roundtrip_json(tmp_path, g):
    p = tmp_path / "cache.json"
    g.to_json(p)
    g2 = TransitGraph.from_json(p)
    assert set(g2.nodes()) == set(g.nodes())
    assert g2.number_of_edges() == g.number_of_edges()
