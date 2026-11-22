"""Integracion de los modelos de ML con el agente de busqueda.

`PredictorTramos` aplica a cada tramo del grafo los modelos entrenados:

  * regresion      -> tiempo programado del tramo a una hora y tipo de dia (s),
  * clasificacion  -> nivel de congestion (baja, media, alta),
  * agrupamiento   -> grupo de K-Medias (1 = lento ... 4 = muy rapido).

El tiempo predicho se convierte en el multiplicador de congestion que ya usa la
busqueda (`RouteProblem.mult`), de modo que el agente lo usa sin cambiar los algoritmos:

    m = max(1, v_libre * t_predicho / dist)        (m >= 1)

Con peso = 1 el coste de un tramo es dist * m = v_libre * t_predicho: la distancia
equivalente en tiempo. Como m >= 1, el coste nunca baja de la distancia y la heuristica
euclidiana sigue siendo admisible y consistente, asi que A* y costo uniforme siguen
siendo optimos para este coste.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ..gtfs_graph import TransitGraph
from ..traffic import PerfilHorario
from . import regresion

ROOT = Path(__file__).resolve().parent.parent.parent
MODELO_REG = ROOT / "data" / "modelo_regresion.joblib"
MODELO_CLF = ROOT / "data" / "modelo_clasificacion.joblib"
GRUPOS_CSV = ROOT / "data" / "agrupamiento_tramos.csv"
DIAS = ("habil", "sabado", "domingo")
NOMBRES_GRUPO = {1: "lento", 2: "medio", 3: "rápido", 4: "muy rápido"}


def modelos_disponibles() -> bool:
    return MODELO_REG.exists() and MODELO_CLF.exists()


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
        self._reg = joblib.load(MODELO_REG)["modelo"]
        self._clf = joblib.load(MODELO_CLF)["modelo"]
        self._tramos = sorted({(a, b) for a in g.adj for b in g.adj[a] if a < b})
        self._base = pd.DataFrame([{
            "u": u, "v": v, "dist_m": g.edge(u, v)["dist_m"], "n_rutas": g.edge(u, v)["n_rutas"],
            "viajes_tramo": g.edge(u, v)["trips"],
            "lat": (g.coords[u][0] + g.coords[v][0]) / 2, "lon": (g.coords[u][1] + g.coords[v][1]) / 2,
            "grado_u": len(g.adj[u]), "grado_v": len(g.adj[v])} for u, v in self._tramos])
        self._grupos = self._cargar_grupos()
        self._tiempos = lru_cache(maxsize=96)(self._calc_tiempos)
        self._clases = lru_cache(maxsize=96)(self._calc_clases)

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

    def _calc_tiempos(self, hora: int, dia: str) -> dict:
        pred = self._reg.predict(self._X(hora, dia))
        return {t: max(1.0, float(p)) for t, p in zip(self._tramos, pred)}

    def _calc_clases(self, hora: int, dia: str) -> dict:
        return dict(zip(self._tramos, self._clf.predict(self._X(hora, dia))))

    # -- consultas -----------------------------------------------------------
    @staticmethod
    def _clave(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def tiempos(self, hora: int, dia: str = "habil") -> dict:
        """{(u, v): segundos previstos}, con u < v."""
        return self._tiempos(int(hora), dia)

    def tiempo_s(self, a: str, b: str, hora: int, dia: str = "habil") -> float:
        return self.tiempos(hora, dia)[self._clave(a, b)]

    def mult(self, hora: int, dia: str = "habil") -> dict:
        """Snapshot {(a, b): m} listo para `RouteProblem(mult=...)`."""
        return mult_desde_tiempos(self.g, self.tiempos(hora, dia), self.perfil.v_libre_kmh)

    def clase(self, a: str, b: str, hora: int, dia: str = "habil") -> str:
        return self._clases(int(hora), dia)[self._clave(a, b)]

    def clases(self, hora: int, dia: str = "habil") -> dict:
        return self._clases(int(hora), dia)

    def grupo(self, a: str, b: str) -> int | None:
        """Grupo de K-Medias del tramo (1-4), o None si no tiene perfil horario."""
        return self._grupos.get(self._clave(a, b))

    def grupos(self) -> dict:
        return dict(self._grupos)
