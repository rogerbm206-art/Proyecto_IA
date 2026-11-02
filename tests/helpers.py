"""Datos y utilidades compartidas por las pruebas."""
from __future__ import annotations

from src.gtfs_graph import TransitGraph

# Pares origen-destino reales del subgrafo de 60 paradas (Chapinero).
PARES = [
    ("Br. Colombia", "Universidad De La Salle"),
    ("Br. La Cabrera", "Cuerpo Of de Bomberos de Bogotá"),
    ("Clínica del Niño", "C.C. Andino"),
    ("Estación Calle 72", "Cámara de Comercio CL 67"),
    ("U. Sergio Arboleda", "Br. Bellavista"),
]

OTROS = ("bfs", "dfs", "ucs", "greedy", "astar")


def camino_valido(g, path: list[str]) -> bool:
    """True si cada par consecutivo del camino es una arista del grafo."""
    return all(b in g.neighbors(a) for a, b in zip(path, path[1:]))


def mini_red_con_rutas() -> TransitGraph:
    """Red de juguete: A-B-D (ruta R1, 200 m) vs A-C-D (R2+R3, 180 m, 1 transbordo)."""
    def e(dist, rutas):
        return {"dist_m": float(dist), "rutas": rutas, "n_rutas": len(rutas), "trips": 9}

    adj = {
        "A": {"B": e(100, ["R1"]), "C": e(90, ["R2"])},
        "B": {"A": e(100, ["R1"]), "D": e(100, ["R1"])},
        "C": {"A": e(90, ["R2"]), "D": e(90, ["R3"])},
        "D": {"B": e(100, ["R1"]), "C": e(90, ["R3"])},
    }
    coords = {"A": (4.650, -74.060), "B": (4.651, -74.060),
              "C": (4.6505, -74.0605), "D": (4.652, -74.060)}
    return TransitGraph(adj, coords, {k: k for k in adj})
