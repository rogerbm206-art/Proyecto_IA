"""Integracion de los modelos de ML con el agente de busqueda.

`PredictorTramos` aplica a cada tramo del grafo el modelo elegido del catalogo
(`catalogo.py`: base, lineal, arbol, rna):

  * regresion      -> tiempo programado del tramo a una hora y tipo de dia (s),
  * clasificacion  -> nivel de congestion (baja, media, alta); la regresion lineal no
                      clasifica, asi que con ese modelo no hay clase,
  * agrupamiento   -> grupo de K-Medias (1 = lento ... 4 = muy rapido), que no depende
                      del modelo.

El tiempo predicho se convierte en el multiplicador de congestion que ya usa la
busqueda (`RouteProblem.mult`), de modo que el agente lo usa sin cambiar los algoritmos:

    m = max(1, v_libre * t_predicho / dist)        (m >= 1)

Con peso = 1 el coste de un tramo es dist * m = v_libre * t_predicho: la distancia
equivalente en tiempo. Como m >= 1, el coste nunca baja de la distancia y la heuristica
euclidiana sigue siendo admisible y consistente, asi que A* y costo uniforme siguen
siendo optimos para este coste, sea cual sea el modelo.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..gtfs_graph import TransitGraph
from ..traffic import PerfilHorario
from . import catalogo, regresion
from .catalogo import CLASIFICAN, NOMBRES

ROOT = Path(__file__).resolve().parent.parent.parent
GRUPOS_CSV = ROOT / "data" / "agrupamiento_tramos.csv"
DIAS = ("habil", "sabado", "domingo")
NOMBRES_GRUPO = {1: "lento", 2: "medio", 3: "rápido", 4: "muy rápido"}
MEJOR_REGRESION = "lineal"      # menor MAE en la validacion cruzada
MEJOR_CLASIFICACION = "rna"     # mayor F1 macro en la validacion cruzada


def modelos_disponibles() -> bool:
    return catalogo.disponible()


def mult_desde_tiempos(g: TransitGraph, tiempos: dict, v_libre_kmh: float) -> dict:
    """Convierte tiempos por tramo {(u, v): segundos} en {(a, b): m} (ambos sentidos)."""
    v_libre_ms = v_libre_kmh / 3.6
    out = {}
    for (u, v), t in tiempos.items():
        m = max(1.0, v_libre_ms * t / g.edge(u, v)["dist_m"])
        out[(u, v)] = out[(v, u)] = m
    return out


class PredictorTramos:
    """Predicciones de los modelos de ML para todos los tramos del grafo."""

    def __init__(self, g: TransitGraph, perfil: PerfilHorario | None = None):
        self.g = g
        self.perfil = perfil or PerfilHorario.cargar()
        cat = catalogo.cargar()
        self._reg, self._clf = cat["regresion"], cat["clasificacion"]
        self._tramos = sorted({(a, b) for a in g.adj for b in g.adj[a] if a < b})
        self._base = pd.DataFrame([{
            "u": u, "v": v, "dist_m": g.edge(u, v)["dist_m"], "n_rutas": g.edge(u, v)["n_rutas"],
            "viajes_tramo": g.edge(u, v)["trips"],
            "lat": (g.coords[u][0] + g.coords[v][0]) / 2, "lon": (g.coords[u][1] + g.coords[v][1]) / 2,
            "grado_u": len(g.adj[u]), "grado_v": len(g.adj[v])} for u, v in self._tramos])
        self._grupos = self._cargar_grupos()
        self._tiempos = lru_cache(maxsize=256)(self._calc_tiempos)
        self._clases = lru_cache(maxsize=256)(self._calc_clases)

    @staticmethod
    def _cargar_grupos() -> dict:
        if not GRUPOS_CSV.exists():
            return {}
        d = pd.read_csv(GRUPOS_CSV)
        return {(u, v): int(g) + 1 for u, v, g in zip(d["u"], d["v"], d["grupo"])}

    def _X(self, hora: int, dia: str) -> pd.DataFrame:
        if dia not in DIAS:
            raise ValueError(f"dia debe ser uno de {DIAS}")
        d = self._base.copy()
        d["hora"], d["tipo_dia"] = int(hora) % 24, dia
        return regresion.atributos(d)

    def _calc_tiempos(self, hora: int, dia: str, modelo: str) -> dict:
        if modelo not in self._reg:
            raise ValueError(f"modelo debe ser uno de {tuple(self._reg)}")
        pred = self._reg[modelo].predict(self._X(hora, dia))
        return {t: max(1.0, float(p)) for t, p in zip(self._tramos, pred)}

    def _calc_clases(self, hora: int, dia: str, modelo: str) -> dict:
        if modelo not in self._clf:
            raise ValueError(f"el modelo {modelo!r} no clasifica; usa uno de {CLASIFICAN}")
        return dict(zip(self._tramos, self._clf[modelo].predict(self._X(hora, dia))))

    # -- consultas -----------------------------------------------------------
    @staticmethod
    def _clave(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    @staticmethod
    def clasifica(modelo: str) -> bool:
        """True si el modelo predice el nivel de congestion."""
        return modelo in CLASIFICAN

    def tiempos(self, hora: int, dia: str = "habil", modelo: str = MEJOR_REGRESION) -> dict:
        """{(u, v): segundos previstos}, con u < v."""
        return self._tiempos(int(hora), dia, modelo)

    def tiempo_s(self, a: str, b: str, hora: int, dia: str = "habil", modelo: str = MEJOR_REGRESION) -> float:
        return self.tiempos(hora, dia, modelo)[self._clave(a, b)]

    def mult(self, hora: int, dia: str = "habil", modelo: str = MEJOR_REGRESION) -> dict:
        """Snapshot {(a, b): m} listo para `RouteProblem(mult=...)`."""
        return mult_desde_tiempos(self.g, self.tiempos(hora, dia, modelo), self.perfil.v_libre_kmh)

    def clase(self, a: str, b: str, hora: int, dia: str = "habil", modelo: str = MEJOR_CLASIFICACION) -> str:
        return self._clases(int(hora), dia, modelo)[self._clave(a, b)]

    def clases(self, hora: int, dia: str = "habil", modelo: str = MEJOR_CLASIFICACION) -> dict:
        return self._clases(int(hora), dia, modelo)

    def grupo(self, a: str, b: str) -> int | None:
        """Grupo de K-Medias del tramo (1-4), o None si no tiene perfil horario."""
        return self._grupos.get(self._clave(a, b))

    def grupos(self) -> dict:
        return dict(self._grupos)
