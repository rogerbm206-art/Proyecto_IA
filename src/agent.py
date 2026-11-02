"""Agente planificador de rutas para la red zonal (buses azules del SITP).

Modelo PEAS
-----------
Performance : llegar a la parada de destino minimizando el coste de la ruta
              (distancia por la red) y el numero de tramos / transbordos.
Environment : el grafo `TransitGraph` de paradas zonales (300 paradas en
              Chapinero y barrios vecinos), parcialmente observable desde la parada
              actual. Estatico, determinista, discreto, secuencial, agente unico.
Actuators   : avanzar al siguiente tramo (moverse a una parada adyacente) y
              detener el viaje al llegar.
Sensors     : parada actual y atributos de los tramos incidentes
              (distancia, nº de rutas, nº de viajes).

El agente delibera con uno de los algoritmos de `src/search.py`
(bfs / dfs / ucs / greedy / astar) para construir la ruta, y luego la ejecuta
tramo a tramo con un ciclo percepcion -> accion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gtfs_graph import TransitGraph, load_zonal_graph
from .metaheuristics import METAHEURISTICAS
from .search import STRATEGIES, SearchResult, search


@dataclass
class StepLog:
    paso: int
    percepcion: str
    accion: str
    ruta_bus: str
    transbordo: bool
    coste_tramo_m: float


@dataclass
class TripLog:
    existe_ruta: bool
    estrategia: str
    origen: str
    destino: str
    ruta: list[str] = field(default_factory=list)
    coste_total_m: float = 0.0
    transbordos: int = 0
    bitacora: list[StepLog] = field(default_factory=list)
    plan: SearchResult | None = None


class RoutePlannerAgent:
    """Agente que planifica y recorre una ruta en la red zonal."""

    ESTRATEGIAS = tuple(STRATEGIES) + tuple(METAHEURISTICAS)

    def __init__(self, graph: TransitGraph | None = None):
        self.env = graph or load_zonal_graph()
        self.pos: str | None = None
        self.goal: str | None = None
        self._plan: list[str] = []

    # -- deliberacion ---------------------------------------------------------
    def plan(self, origen: str, destino: str, estrategia: str = "astar") -> SearchResult:
        """Calcula la ruta (sin recorrerla) y devuelve el `SearchResult`."""
        return search(self.env, origen, destino, estrategia)

    # -- SENSORES -----------------------------------------------------------
    def perceive(self) -> dict:
        assert self.pos is not None
        return {
            "parada_actual": self.env.name(self.pos),
            "tramos_visibles": {
                self.env.name(v): self.env.edge(self.pos, v)
                for v in self.env.neighbors(self.pos)
            },
            "es_destino": self.pos == self.goal,
        }

    # -- ACTUADOR + ciclo percepcion->accion -------------------------------
    def run(self, origen: str, destino: str, estrategia: str = "astar",
            plan: SearchResult | None = None) -> TripLog:
        """Planifica (o reutiliza `plan`, un `SearchResult` ya calculado) y recorre la ruta."""
        s = self.env.resolve(origen)
        t = self.env.resolve(destino)
        self.pos, self.goal = s, t
        res = plan if plan is not None else self.plan(origen, destino, estrategia)

        log = TripLog(
            existe_ruta=res.found,
            estrategia=estrategia,
            origen=self.env.name(s),
            destino=self.env.name(t),
            ruta=list(res.path_names),
            plan=res,
        )
        if not res.found:
            return log

        log.transbordos = res.transbordos
        self._plan = list(res.path[1:])
        rutas_plan = list(res.path_routes)
        paso = 0
        ruta_previa = None
        while self._plan:
            percept = self.perceive()                     # <-- SENSORES
            siguiente = self._plan.pop(0)                  # <-- accion elegida del plan
            ruta_bus = rutas_plan.pop(0)
            nombre_sig = self.env.name(siguiente)
            if nombre_sig not in percept["tramos_visibles"]:
                raise RuntimeError(f"Accion invalida: {nombre_sig} no es adyacente")
            es_transbordo = ruta_previa is not None and ruta_bus != ruta_previa
            coste = self.env.cost(self.pos, siguiente)
            paso += 1
            verbo = "tomar" if ruta_previa != ruta_bus else "seguir en"
            log.bitacora.append(StepLog(
                paso=paso,
                percepcion=f"en '{percept['parada_actual']}', "
                           f"{len(percept['tramos_visibles'])} tramo(s) visibles",
                accion=f"{verbo} ruta {ruta_bus} -> '{nombre_sig}'",
                ruta_bus=ruta_bus,
                transbordo=es_transbordo,
                coste_tramo_m=coste,
            ))
            log.coste_total_m += coste
            self.pos = siguiente                            # <-- efecto del ACTUADOR
            ruta_previa = ruta_bus
        log.coste_total_m = round(log.coste_total_m, 1)
        return log
