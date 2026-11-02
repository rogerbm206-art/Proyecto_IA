"""Metaheuristicas para el mismo problema de rutas: ACO, PSO, temple simulado,
algoritmo genetico y ascenso de colinas.

Mismo problema, mismo coste
---------------------------
Una solucion es un camino SIMPLE de paradas origen -> destino. Su coste es el mismo
que usan BFS/UCS/A*:  suma de `problem.edge_cost` de los tramos + 800 m por cada
transbordo. Como el bus de cada tramo es una decision aparte, `evaluar` reparte las
rutas de bus del camino con una programacion dinamica que minimiza los transbordos
(el camino decide POR DONDE ir; el reparto de buses se resuelve exacto).

Ninguna usa Dijkstra ni el coste optimo para construir soluciones: parten de
caminos aleatorios sesgados hacia el destino y los mejoran con su propia logica.

    hc   ascenso de colinas : mejora con la mejor de varias mutaciones; reinicios
    sa   temple simulado    : acepta empeoramientos con prob. exp(-delta/T), T baja
    ga   algoritmo genetico : poblacion, cruce por parada comun, mutacion, elitismo
    aco  colonia de hormigas: feromona en tramos + heuristica de cercania al destino
    pso  enjambre de particulas: vector de prioridades por parada, decodificado a camino

Son ESTOCASTICAS: con la misma `semilla` dan siempre el mismo resultado.
No garantizan el optimo; devuelven el mejor camino que encontraron.

Metricas en `SearchResult` (para poder compararlas con las de busqueda):
    nodes_expanded   evaluaciones de la funcion de coste
    nodes_generated  soluciones candidatas construidas (incluye intentos fallidos)
    max_frontier     tamaño de la poblacion / colonia / enjambre (1 en hc y sa)
"""
from __future__ import annotations

import math
import random

from .geo import euclidean_m
from .search import RouteProblem, SearchResult, _finish, _not_found


# ---------------------------------------------------------------------------
# Funcion objetivo y operadores comunes
# ---------------------------------------------------------------------------
class _Contexto:
    """Evaluacion de caminos y operadores de vecindad compartidos."""

    def __init__(self, problem: RouteProblem, rng: random.Random):
        self.p = problem
        self.g = problem.g
        self.rng = rng
        self.origen = problem.initial[0]
        self.meta = problem.goal
        self.evals = 0
        self.generados = 0
        self._cache: dict[tuple, tuple[float, list[str]]] = {}
        # distancia en linea recta a la meta (guia de las construcciones)
        self.dmeta = {n: problem.h((n, None)) for n in self.g.nodes()}

    # -- funcion objetivo ----------------------------------------------------
    def evaluar(self, camino: list[str]) -> tuple[float, list[str]]:
        """(coste, rutas por tramo) del camino; reparte buses minimizando transbordos."""
        clave = tuple(camino)
        if clave in self._cache:
            return self._cache[clave]
        self.evals += 1
        tramos = [self.g.edge(a, b)["rutas"] for a, b in zip(camino, camino[1:])]
        # mejor[r] = (transbordos, ruta_previa) para llegar al final del tramo usando r
        mejor = [{r: (0, None) for r in tramos[0]}]
        for rs in tramos[1:]:
            prev = mejor[-1]
            mejor_prev = min(prev, key=lambda r: (prev[r][0], r))
            act = {}
            for r in rs:
                if r in prev:
                    act[r] = (prev[r][0], r)
                else:
                    act[r] = (prev[mejor_prev][0] + 1, mejor_prev)
            mejor.append(act)
        r = min(mejor[-1], key=lambda x: (mejor[-1][x][0], x))
        transb = mejor[-1][r][0]
        rutas = [r]
        for act in reversed(mejor[1:]):
            r = act[r][1]
            rutas.append(r)
        rutas.reverse()
        coste = sum(self.p.edge_cost(a, b) for a, b in zip(camino, camino[1:]))
        coste += self.p.penalty * transb
        self._cache[clave] = (coste, rutas)
        return coste, rutas

    def coste(self, camino: list[str]) -> float:
        return self.evaluar(camino)[0]

    # -- construccion de caminos --------------------------------------------
    def _paso_sesgado(self, u: str, candidatos: list[str], destino: str, beta: float) -> str:
        """Elige el siguiente nodo con prob. ∝ (1/distancia al destino)^beta."""
        la, lo = self.g.coords[destino]
        pesos = []
        for v in candidatos:
            d = euclidean_m(*self.g.coords[v], la, lo) if destino != self.meta else self.dmeta[v]
            pesos.append(1.0 / (d + 50.0) ** beta)
        return self.rng.choices(candidatos, weights=pesos)[0]

    def tramo_aleatorio(self, a: str, b: str, prohibidos: set[str], max_pasos: int,
                        beta: float = 3.0) -> list[str] | None:
        """Camino aleatorio sesgado hacia b (camino simple, sin Dijkstra). None si falla."""
        camino, vistos = [a], {a}
        for _ in range(max_pasos):
            u = camino[-1]
            cand = [v for v in self.g.neighbors(u)
                    if v not in vistos and (v == b or v not in prohibidos)]
            if not cand:
                return None
            v = b if b in cand else self._paso_sesgado(u, cand, b, beta)
            camino.append(v)
            vistos.add(v)
            if v == b:
                return camino
        return None

    def camino_dfs(self, prioridad=None) -> list[str]:
        """Camino garantizado (DFS ordenado por `prioridad(v)`); respaldo y decodificador."""
        prioridad = prioridad or (lambda v: 1.0)
        camino, vistos = [self.origen], {self.origen}
        pila = [iter(self._ordenar(self.origen, prioridad, vistos))]
        while pila:
            for v in pila[-1]:
                camino.append(v)
                vistos.add(v)
                if v == self.meta:
                    return camino
                pila.append(iter(self._ordenar(v, prioridad, vistos)))
                break
            else:
                pila.pop()
                camino.pop()
        return []

    def _ordenar(self, u, prioridad, vistos):
        vec = [v for v in self.g.neighbors(u) if v not in vistos]
        vec.sort(key=lambda v: (-prioridad(v) / (self.dmeta[v] + 50.0), v))
        return vec

    def camino_inicial(self, intentos: int = 40) -> list[str]:
        """Camino aleatorio origen -> meta (sesgado hacia la meta); DFS si todo falla."""
        for _ in range(intentos):
            self.generados += 1
            c = self.tramo_aleatorio(self.origen, self.meta, set(), 150)
            if c:
                return c
        self.generados += 1
        return self.camino_dfs()

    # -- vecindad ----------------------------------------------------------------
    def vecino(self, camino: list[str]) -> list[str]:
        """Una mutacion: atajo directo o re-ruteo aleatorio de un segmento."""
        self.generados += 1
        n = len(camino)
        if n < 3:
            return camino
        rng = self.rng
        if rng.random() < 0.35:                      # atajo: saltar a una parada vecina posterior
            i = rng.randrange(0, n - 2)
            pos = {v: k for k, v in enumerate(camino)}
            saltos = [pos[v] for v in self.g.neighbors(camino[i])
                      if v in pos and pos[v] > i + 1]
            if saltos:
                j = rng.choice(saltos)
                return camino[:i + 1] + camino[j:]
        i = rng.randrange(0, n - 2)
        j = min(n - 1, i + 2 + rng.randrange(0, 10))
        prohibidos = set(camino) - {camino[i], camino[j]}
        seg = self.tramo_aleatorio(camino[i], camino[j], prohibidos, 3 * (j - i) + 6)
        if not seg:
            return camino
        return camino[:i] + seg + camino[j + 1:]

    @staticmethod
    def sin_ciclos(camino: list[str]) -> list[str]:
        """Quita bucles: si una parada se repite, descarta lo recorrido entre repeticiones."""
        out: list[str] = []
        pos: dict[str, int] = {}
        for v in camino:
            if v in pos:
                k = pos[v]
                for w in out[k + 1:]:
                    del pos[w]
                del out[k + 1:]
            else:
                pos[v] = len(out)
                out.append(v)
        return out


def _resultado(nombre: str, ctx: _Contexto, mejor: list[str], poblacion: int) -> SearchResult:
    """Convierte el mejor camino en un SearchResult (mismo formato que la busqueda)."""
    if not mejor:
        return _not_found(nombre, ctx.evals, ctx.generados, poblacion)
    _, rutas = ctx.evaluar(mejor)
    estados = [(mejor[0], None)] + [(v, r) for v, r in zip(mejor[1:], rutas)]
    parent = {estados[0]: None}
    for a, b in zip(estados, estados[1:]):
        parent[b] = a
    return _finish(nombre, ctx.p, parent, estados[-1], ctx.evals, ctx.generados, poblacion)


def _trivial(nombre: str, problem: RouteProblem) -> SearchResult | None:
    s0 = problem.initial
    if problem.is_goal(s0):
        return _finish(nombre, problem, {s0: None}, s0, 0, 1, 1)
    return None


# ---------------------------------------------------------------------------
# Ascenso de colinas
# ---------------------------------------------------------------------------
def hill_climbing(problem: RouteProblem, semilla: int = 0, reinicios: int = 6,
                  vecinos: int = 25, paciencia: int = 8) -> SearchResult:
    """Ascenso de colinas con ascenso por la mejor mutacion (steepest) y reinicios.

    En cada iteracion genera `vecinos` mutaciones del camino actual y salta a la
    mejor si mejora; si no mejora `paciencia` veces seguidas, se queda en un optimo
    local y se reinicia desde otro camino aleatorio. Conserva el mejor global.
    """
    if (r := _trivial("hc", problem)):
        return r
    ctx = _Contexto(problem, random.Random(semilla))
    mejor, mejor_c = None, math.inf
    for _ in range(reinicios):
        actual = ctx.camino_inicial()
        if not actual:
            continue
        c_act = ctx.coste(actual)
        fallos = 0
        while fallos < paciencia:
            cand = min((ctx.vecino(actual) for _ in range(vecinos)), key=ctx.coste)
            c = ctx.coste(cand)
            if c < c_act - 1e-9:
                actual, c_act, fallos = cand, c, 0
            else:
                fallos += 1
        if c_act < mejor_c:
            mejor, mejor_c = actual, c_act
    return _resultado("hc", ctx, mejor or [], 1)


# ---------------------------------------------------------------------------
# Temple simulado
# ---------------------------------------------------------------------------
def simulated_annealing(problem: RouteProblem, semilla: int = 0, alfa: float = 0.95,
                        pasos_por_temp: int = 40, t_min_rel: float = 0.001) -> SearchResult:
    """Temple simulado: acepta peores caminos con prob. exp(-delta/T); T = T·alfa.

    T0 = 20 % del coste del camino inicial (escala del problema), T_final = T0·t_min_rel.
    """
    if (r := _trivial("sa", problem)):
        return r
    ctx = _Contexto(problem, random.Random(semilla))
    actual = ctx.camino_inicial()
    if not actual:
        return _resultado("sa", ctx, [], 1)
    c_act = ctx.coste(actual)
    mejor, mejor_c = actual, c_act
    T = 0.2 * c_act
    T_fin = T * t_min_rel
    while T > T_fin:
        for _ in range(pasos_por_temp):
            cand = ctx.vecino(actual)
            c = ctx.coste(cand)
            delta = c - c_act
            if delta < 0 or ctx.rng.random() < math.exp(-delta / T):
                actual, c_act = cand, c
                if c_act < mejor_c:
                    mejor, mejor_c = actual, c_act
        T *= alfa
    return _resultado("sa", ctx, mejor, 1)


# ---------------------------------------------------------------------------
# Algoritmo genetico
# ---------------------------------------------------------------------------
def genetic(problem: RouteProblem, semilla: int = 0, poblacion: int = 40, generaciones: int = 40,
            p_cruce: float = 0.9, p_mut: float = 0.3, elite: int = 2, torneo: int = 3) -> SearchResult:
    """Algoritmo genetico sobre caminos.

    Individuo = camino. Seleccion por torneo; cruce = unir el principio de un padre con
    el final del otro por una parada comun (se quitan bucles); mutacion = re-ruteo de
    un segmento; elitismo de `elite` individuos.
    """
    if (r := _trivial("ga", problem)):
        return r
    ctx = _Contexto(problem, random.Random(semilla))
    rng = ctx.rng
    pop = [ctx.camino_inicial() for _ in range(poblacion)]
    pop = [c for c in pop if c]
    if not pop:
        return _resultado("ga", ctx, [], poblacion)

    def elegir():
        return min(rng.sample(pop, min(torneo, len(pop))), key=ctx.coste)

    def cruzar(a, b):
        comunes = set(a[1:-1]) & set(b[1:-1])
        if not comunes:
            return list(a)
        v = rng.choice(sorted(comunes))
        hijo = a[:a.index(v) + 1] + b[b.index(v) + 1:]
        return ctx.sin_ciclos(hijo)

    for _ in range(generaciones):
        pop.sort(key=ctx.coste)
        nueva = [list(c) for c in pop[:elite]]
        while len(nueva) < poblacion:
            a, b = elegir(), elegir()
            hijo = cruzar(a, b) if rng.random() < p_cruce else list(a)
            ctx.generados += 1
            if rng.random() < p_mut:
                hijo = ctx.vecino(hijo)
            nueva.append(hijo)
        pop = nueva
    mejor = min(pop, key=ctx.coste)
    return _resultado("ga", ctx, mejor, poblacion)


# ---------------------------------------------------------------------------
# Optimizacion por colonia de hormigas
# ---------------------------------------------------------------------------
def aco(problem: RouteProblem, semilla: int = 0, hormigas: int = 20, iteraciones: int = 40,
        alfa: float = 1.0, beta: float = 3.0, rho: float = 0.1,
        tau_min: float = 0.05, tau_max: float = 5.0) -> SearchResult:
    """Colonia de hormigas (estilo MAX-MIN Ant System).

    Cada hormiga construye un camino simple eligiendo el siguiente nodo con prob.
    ∝ tau(arista)^alfa · eta(nodo)^beta, con eta = 1/distancia al destino. Si se
    atasca retrocede marcando el callejon sin salida. Tras cada iteracion la feromona
    se evapora (rho) y depositan las mejores hormigas y la mejor global.
    """
    if (r := _trivial("aco", problem)):
        return r
    ctx = _Contexto(problem, random.Random(semilla))
    rng, g = ctx.rng, ctx.g
    tau: dict[tuple[str, str], float] = {}

    def clave(a, b):
        return (a, b) if a < b else (b, a)

    def construir() -> list[str] | None:
        ctx.generados += 1
        camino, vistos, muertos = [ctx.origen], {ctx.origen}, set()
        for _ in range(600):
            u = camino[-1]
            if u == ctx.meta:
                return camino
            cand = [v for v in g.neighbors(u) if v not in vistos and v not in muertos]
            if not cand:
                if len(camino) == 1:
                    return None
                muertos.add(camino.pop())
                continue
            pesos = [tau.get(clave(u, v), 1.0) ** alfa * (1.0 / (ctx.dmeta[v] + 50.0)) ** beta
                     for v in cand]
            v = rng.choices(cand, weights=pesos)[0]
            camino.append(v)
            vistos.add(v)
        return None

    mejor, mejor_c = None, math.inf
    for _ in range(iteraciones):
        sols = []
        for _h in range(hormigas):
            c = construir()
            if c:
                sols.append((ctx.coste(c), c))
        if not sols:
            continue
        sols.sort(key=lambda x: x[0])
        if sols[0][0] < mejor_c:
            mejor_c, mejor = sols[0]
        for k in tau:                                   # evaporacion
            tau[k] = max(tau_min, tau[k] * (1.0 - rho))
        depositos = sols[:max(1, hormigas // 4)] + [(mejor_c, mejor)]
        for c, cam in depositos:
            aporte = mejor_c / c                         # (0, 1]: mejor camino = 1
            for a, b in zip(cam, cam[1:]):
                k = clave(a, b)
                tau[k] = min(tau_max, tau.get(k, 1.0) + aporte)
    return _resultado("aco", ctx, mejor or ctx.camino_dfs(), hormigas)


# ---------------------------------------------------------------------------
# Enjambre de particulas
# ---------------------------------------------------------------------------
def pso(problem: RouteProblem, semilla: int = 0, particulas: int = 20, iteraciones: int = 40,
        w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> SearchResult:
    """Enjambre de particulas con codificacion por prioridades.

    Particula = vector x con una prioridad en [0.05, 1] por parada. Se decodifica a un
    camino con una busqueda en profundidad que, en cada parada, prueba primero el
    vecino de mayor prioridad / cercania al destino. Velocidad y posicion se actualizan
    con la regla clasica  v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x).
    """
    if (r := _trivial("pso", problem)):
        return r
    ctx = _Contexto(problem, random.Random(semilla))
    rng = ctx.rng
    nodos = sorted(ctx.g.nodes())
    idx = {n: i for i, n in enumerate(nodos)}
    dim = len(nodos)

    def decodificar(x):
        ctx.generados += 1
        return ctx.camino_dfs(lambda v: x[idx[v]])

    xs = [[rng.uniform(0.05, 1.0) for _ in range(dim)] for _ in range(particulas)]
    vs = [[rng.uniform(-0.2, 0.2) for _ in range(dim)] for _ in range(particulas)]
    pbest, pbest_c, pbest_cam = [list(x) for x in xs], [math.inf] * particulas, [None] * particulas
    gbest, gbest_c, gbest_cam = list(xs[0]), math.inf, None

    for _ in range(iteraciones):
        for i in range(particulas):
            cam = decodificar(xs[i])
            c = ctx.coste(cam) if cam else math.inf
            if c < pbest_c[i]:
                pbest_c[i], pbest[i], pbest_cam[i] = c, list(xs[i]), cam
                if c < gbest_c:
                    gbest_c, gbest, gbest_cam = c, list(xs[i]), cam
        for i in range(particulas):
            x, v, pb = xs[i], vs[i], pbest[i]
            for d in range(dim):
                r1, r2 = rng.random(), rng.random()
                v[d] = w * v[d] + c1 * r1 * (pb[d] - x[d]) + c2 * r2 * (gbest[d] - x[d])
                v[d] = max(-0.5, min(0.5, v[d]))
                x[d] = max(0.05, min(1.0, x[d] + v[d]))
    return _resultado("pso", ctx, gbest_cam or [], particulas)


# ---------------------------------------------------------------------------
METAHEURISTICAS = {
    "hc": hill_climbing,
    "sa": simulated_annealing,
    "ga": genetic,
    "aco": aco,
    "pso": pso,
}
