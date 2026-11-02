"""Algoritmos de busqueda sobre el grafo de la red zonal (con rutas fijas).

Implementa, con una interfaz comun, las cinco estrategias vistas en clase:

    bfs     - busqueda en anchura (optima en nº de tramos)
    dfs     - busqueda en profundidad (ni optima ni completa en general)
    ucs     - coste uniforme / Dijkstra (optima en coste)
    greedy  - voraz primero-el-mejor (usa solo la heuristica h)
    astar   - A* (usa f = g + h; optima con heuristica admisible)

Modelo de RUTAS FIJAS (opcion 1)
--------------------------------
Un bus zonal recorre una ruta predeterminada, asi que no se puede "saltar" de una
ruta a otra sin bajarse y esperar otro bus. Para modelarlo, un estado de la
busqueda no es solo la parada sino el par  (parada, ruta_en_la_que_vienes).

- Para avanzar de la parada u a la parada v hay que ir en alguna ruta `r` que
  cubra ese tramo (`r in grafo.edge(u, v)["rutas"]`).
- Coste del paso = dist_m                         si sigues en la misma ruta
                 = dist_m + TRANSFER_PENALTY_M    si cambias de ruta (transbordo)
- Abordar el primer bus en el origen no cuenta como transbordo.

Coste de un tramo  : distancia euclidiana entre paradas (metros) + penalizacion.
Heuristica h(n)    : distancia euclidiana en linea recta de n a la meta (ignora
                     penalizaciones -> sigue siendo admisible y consistente, asi
                     que UCS y A* siguen siendo optimos).

Cada llamada devuelve un `SearchResult` con metricas para comparar: tiempo,
memoria pico, nodos expandidos, nodos generados, frontera maxima, coste,
distancia y nº de transbordos de la ruta.
"""
from __future__ import annotations

import heapq
import itertools
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass

from .geo import euclidean_m
from .gtfs_graph import TransitGraph

TRANSFER_PENALTY_M = 800.0   # coste equivalente (m) de un transbordo entre rutas
V_LIBRE_MS = 21.44 / 3.6     # velocidad en flujo libre (m/s), calibrada con el GTFS

# Un estado es (stop_id, ruta_actual). En el origen ruta_actual = None.
State = tuple


# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    strategy: str
    found: bool
    path: list[str]                # lista de stop_id
    path_names: list[str]          # nombres de parada
    path_routes: list[str]         # ruta usada en cada tramo (len = path_edges)
    path_edges: int                # nº de tramos de la ruta
    transbordos: int               # nº de cambios de ruta a lo largo del camino
    path_dist_m: float             # distancia geografica de la ruta
    path_cost_m: float             # coste total = distancia + penalizaciones
    nodes_expanded: int            # estados sacados de la frontera y desarrollados
    nodes_generated: int           # estados insertados en la frontera
    max_frontier: int              # tamaño maximo de la frontera
    elapsed_ms: float
    peak_kib: float
    path_time_s: float = 0.0       # tiempo estimado de viaje (con el trafico dado, si lo hay)


class RouteProblem:
    """Ir de `start` a `goal` en un `TransitGraph`, respetando rutas fijas.

    Trafico (opcional): `mult` es un snapshot {(a, b): m} de multiplicadores de
    congestion (m >= 1) y `peso_trafico` mezcla distancia y trafico en el coste de
    cada tramo:  coste = dist * (1 + peso * (m - 1)).  Sin `mult` (o con peso 0) el
    coste es solo la distancia, como en el modelo original.

    `ruta_inicial`: ruta del bus en el que ya vas (al replanificar a mitad de viaje);
    seguir en ella no cuesta transbordo, cambiar de ruta si.
    """

    def __init__(self, graph: TransitGraph, start: str, goal: str,
                 transfer_penalty_m: float = TRANSFER_PENALTY_M,
                 mult: dict | None = None, peso_trafico: float = 0.0,
                 ruta_inicial: str | None = None, v_libre_ms: float = V_LIBRE_MS):
        self.g = graph
        self.goal = goal
        self.penalty = transfer_penalty_m
        self.mult = mult or {}
        self.peso = peso_trafico
        self.v_libre = v_libre_ms
        self.initial: State = (start, ruta_inicial)

    def is_goal(self, s: State) -> bool:
        return s[0] == self.goal

    def m(self, u: str, v: str) -> float:
        """Multiplicador de congestion del tramo u-v (1 si no hay datos de trafico)."""
        return self.mult.get((u, v), 1.0)

    def edge_cost(self, u: str, v: str) -> float:
        """Coste del tramo u-v: mezcla de distancia y trafico."""
        d = self.g.edge(u, v)["dist_m"]
        if self.peso and self.mult:
            return d * (1.0 + self.peso * (self.m(u, v) - 1.0))
        return d

    def expand(self, s: State):
        """Genera los pares (estado_sucesor, coste_del_paso)."""
        u, r = s
        for v in self.g.neighbors(u):
            edge = self.g.edge(u, v)
            base = self.edge_cost(u, v)
            for rt in edge["rutas"]:
                pen = 0.0 if (r is None or rt == r) else self.penalty
                yield (v, rt), base + pen

    def transition_cost(self, a: State, b: State) -> float:
        (_, ra), (v, rb) = a, b
        pen = 0.0 if (ra is None or rb == ra) else self.penalty
        return self.edge_cost(a[0], v) + pen

    def h(self, s: State) -> float:
        la1, lo1 = self.g.coords[s[0]]
        la2, lo2 = self.g.coords[self.goal]
        return euclidean_m(la1, lo1, la2, lo2)


# ---------------------------------------------------------------------------
def _finish(strategy, problem, parent, goal_state, expanded, generated, maxf) -> SearchResult:
    states: list[State] = [goal_state]
    while parent[states[-1]] is not None:
        states.append(parent[states[-1]])
    states.reverse()

    stops = [s[0] for s in states]
    routes_seg = [s[1] for s in states[1:]]
    # un transbordo = cambio de ruta respecto a la anterior (incluida `ruta_inicial`)
    transbordos = sum(1 for a, b in zip(states, states[1:])
                      if a[1] is not None and b[1] != a[1])
    dist = sum(problem.g.edge(a, b)["dist_m"] for a, b in zip(stops, stops[1:]))
    cost = sum(problem.transition_cost(a, b) for a, b in zip(states, states[1:]))
    # tiempo estimado: distancia * congestion a velocidad libre + espera de cada transbordo
    tiempo = (sum(problem.g.edge(a, b)["dist_m"] * problem.m(a, b)
                  for a, b in zip(stops, stops[1:]))
              + transbordos * problem.penalty) / problem.v_libre

    return SearchResult(
        strategy=strategy,
        found=True,
        path=stops,
        path_names=[problem.g.name(n) for n in stops],
        path_routes=routes_seg,
        path_edges=len(stops) - 1,
        transbordos=transbordos,
        path_dist_m=round(dist, 1),
        path_cost_m=round(cost, 1),
        nodes_expanded=expanded,
        nodes_generated=generated,
        max_frontier=maxf,
        elapsed_ms=0.0,
        peak_kib=0.0,
        path_time_s=round(tiempo, 1),
    )


def _not_found(strategy, expanded, generated, maxf) -> SearchResult:
    return SearchResult(strategy, False, [], [], [], 0, 0, 0.0, 0.0,
                        expanded, generated, maxf, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Busqueda no informada
# ---------------------------------------------------------------------------
def bfs(problem: RouteProblem) -> SearchResult:
    """Anchura: frontera FIFO, prueba de meta al generar. Minimiza nº de tramos."""
    s0 = problem.initial
    if problem.is_goal(s0):
        return _finish("bfs", problem, {s0: None}, s0, 0, 1, 1)
    frontier = deque([s0])
    parent = {s0: None}
    expanded = generated = 0
    maxf = 1
    while frontier:
        s = frontier.popleft()
        expanded += 1
        for s2, _c in problem.expand(s):
            if s2 not in parent:
                parent[s2] = s
                generated += 1
                if problem.is_goal(s2):
                    return _finish("bfs", problem, parent, s2, expanded, generated, maxf)
                frontier.append(s2)
                maxf = max(maxf, len(frontier))
    return _not_found("bfs", expanded, generated, maxf)


def dfs(problem: RouteProblem) -> SearchResult:
    """Profundidad: frontera LIFO (grafo, con conjunto de alcanzados).

    No garantiza la ruta mas corta ni la de menor coste.
    """
    s0 = problem.initial
    stack = [s0]
    parent = {s0: None}
    expanded = generated = 0
    maxf = 1
    while stack:
        s = stack.pop()
        expanded += 1
        if problem.is_goal(s):
            return _finish("dfs", problem, parent, s, expanded, generated, maxf)
        for s2, _c in problem.expand(s):
            if s2 not in parent:
                parent[s2] = s
                generated += 1
                stack.append(s2)
                maxf = max(maxf, len(stack))
    return _not_found("dfs", expanded, generated, maxf)


# ---------------------------------------------------------------------------
# Busqueda primero-el-mejor (UCS, voraz, A*)
# ---------------------------------------------------------------------------
def _best_first(strategy: str, problem: RouteProblem, priority) -> SearchResult:
    """`priority(state, g_cost) -> float`. Prueba de meta al expandir."""
    s0 = problem.initial
    counter = itertools.count()
    g_cost = {s0: 0.0}
    parent = {s0: None}
    frontier = [(priority(s0, 0.0), next(counter), s0)]
    closed: set = set()
    expanded = 0
    generated = 1
    maxf = 1
    while frontier:
        _, _, s = heapq.heappop(frontier)
        if s in closed:
            continue
        closed.add(s)
        expanded += 1
        if problem.is_goal(s):
            return _finish(strategy, problem, parent, s, expanded, generated, maxf)
        for s2, c in problem.expand(s):
            ng = g_cost[s] + c
            if s2 not in g_cost or ng < g_cost[s2]:
                g_cost[s2] = ng
                parent[s2] = s
                generated += 1
                heapq.heappush(frontier, (priority(s2, ng), next(counter), s2))
                maxf = max(maxf, len(frontier))
    return _not_found(strategy, expanded, generated, maxf)


def ucs(problem: RouteProblem) -> SearchResult:
    """Coste uniforme (Dijkstra): f = g. Optima en coste."""
    return _best_first("ucs", problem, lambda s, g: g)


def greedy(problem: RouteProblem) -> SearchResult:
    """Voraz primero-el-mejor: f = h. Rapida pero no optima."""
    return _best_first("greedy", problem, lambda s, g: problem.h(s))


def astar(problem: RouteProblem) -> SearchResult:
    """A*: f = g + h. Optima con heuristica admisible."""
    return _best_first("astar", problem, lambda s, g: g + problem.h(s))


# ---------------------------------------------------------------------------
STRATEGIES = {"bfs": bfs, "dfs": dfs, "ucs": ucs, "greedy": greedy, "astar": astar}


def search(graph: TransitGraph, origin: str, goal: str, strategy: str = "astar",
           transfer_penalty_m: float = TRANSFER_PENALTY_M,
           mult: dict | None = None, peso_trafico: float = 0.0,
           ruta_inicial: str | None = None, v_libre_ms: float = V_LIBRE_MS,
           semilla: int = 0) -> SearchResult:
    """Resuelve un problema midiendo tiempo y memoria pico del proceso.

    `strategy` puede ser una busqueda (STRATEGIES) o una metaheuristica (hc, sa, ga,
    aco, pso; ver `metaheuristics.py`). `semilla` solo la usan las metaheuristicas.

    `mult` (snapshot de trafico) y `peso_trafico` activan el coste combinado
    distancia + trafico; sin ellos se comporta exactamente como el modelo original.
    """
    from .metaheuristics import METAHEURISTICAS   # import tardio: evita ciclo de modulos
    if strategy in STRATEGIES:
        fn = STRATEGIES[strategy]
    elif strategy in METAHEURISTICAS:
        fn = lambda pr: METAHEURISTICAS[strategy](pr, semilla=semilla)  # noqa: E731
    else:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. "
                         f"Use {list(STRATEGIES) + list(METAHEURISTICAS)}")
    problem = RouteProblem(graph, graph.resolve(origin), graph.resolve(goal),
                           transfer_penalty_m, mult, peso_trafico, ruta_inicial, v_libre_ms)
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(problem)
    dt = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res.elapsed_ms = round(dt, 4)
    res.peak_kib = round(peak / 1024, 2)
    return res
