"""Simulacion de un viaje con trafico dinamico y replanificacion.

El agente SABE el patron tipico del trafico (modelo: hora del dia + factor de cada
tramo) pero la REALIDAD (`EntornoTrafico`) cambia mientras avanza: fluctuaciones
aleatorias y picos repentinos de congestion. Percibe el trafico real solo de los
tramos que salen de su parada actual (observabilidad `local`) o de toda la red
(`global`, como un mapa de trafico en vivo).

Politicas de replanificacion:
  nunca        sigue el plan inicial pase lo que pase (linea base)
  umbral       replanifica si el tramo que va a tomar esta mas congestionado de lo
               esperado en mas de `umbral` (relativo; 0.25 = un 25 % peor)
  cada_parada  replanifica en cada parada

Reloj simulado: cada tramo dura  distancia * m / v_libre  (+ espera si cambia de bus),
y el entorno avanza ese tiempo. El coste real usa la misma formula combinada que la
busqueda: distancia * (1 + peso * (m - 1)) (+ penalizacion por transbordo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .gtfs_graph import TransitGraph
from .search import TRANSFER_PENALTY_M, SearchResult, search
from .traffic import EntornoTrafico, ModeloTrafico, parse_hora

POLITICAS = ("nunca", "umbral", "cada_parada")
OBSERVABILIDAD = ("local", "global")


@dataclass
class PasoSim:
    paso: int
    desde: str
    hasta: str
    ruta_bus: str
    transbordo: bool
    m_real: float           # congestion real del tramo al tomarlo
    m_esperado: float       # congestion que el agente esperaba (segun su plan)
    dt_s: float             # duracion real del tramo
    replanifico: bool       # si replanifico justo antes de este paso


@dataclass
class ViajeSimulado:
    existe_ruta: bool
    estrategia: str
    politica: str
    observabilidad: str
    origen: str
    destino: str
    hora_salida_s: float
    ruta_planeada: list[str] = field(default_factory=list)
    tiempo_planeado_s: float = 0.0
    coste_planeado: float = 0.0
    ruta_recorrida: list[str] = field(default_factory=list)
    tiempo_real_s: float = 0.0
    coste_real: float = 0.0
    dist_real_m: float = 0.0
    transbordos: int = 0
    replanificaciones: int = 0
    picos: int = 0
    computo_ms: float = 0.0
    pasos: list[PasoSim] = field(default_factory=list)

    @property
    def retraso_s(self) -> float:
        return self.tiempo_real_s - self.tiempo_planeado_s


def simular_viaje(graph: TransitGraph, modelo: ModeloTrafico, origen: str, destino: str,
                  estrategia: str = "astar", politica: str = "umbral",
                  observabilidad: str = "local", hora_salida="08:00",
                  dinamico: bool = True, semilla: int = 0, umbral: float = 0.25,
                  max_pasos: int = 400,
                  transfer_penalty_m: float = TRANSFER_PENALTY_M) -> ViajeSimulado:
    if politica not in POLITICAS:
        raise ValueError(f"Politica desconocida: {politica!r}. Use {POLITICAS}")
    if observabilidad not in OBSERVABILIDAD:
        raise ValueError(f"Observabilidad desconocida: {observabilidad!r}. Use {OBSERVABILIDAD}")

    s, t = graph.resolve(origen), graph.resolve(destino)
    t0 = parse_hora(hora_salida)
    peso = modelo.cfg.peso
    v = modelo.v_libre_ms
    env = EntornoTrafico(modelo, t0, dinamico=dinamico, semilla=semilla)

    def creencia() -> dict:
        """Lo que el agente cree del trafico AHORA: patron tipico + lo que percibe."""
        if observabilidad == "global":
            return env.todos_m_real()
        base = modelo.mult_esperado(env.t)
        for w in graph.neighbors(pos):
            m = env.m_real(pos, w)
            base[(pos, w)] = m
            base[(w, pos)] = m
        return base

    computo = 0.0

    def planear(desde: str, ruta_actual: str | None, mult: dict) -> SearchResult:
        nonlocal computo
        c0 = time.perf_counter()
        r = search(graph, desde, t, estrategia, transfer_penalty_m, mult, peso, ruta_actual, v)
        computo += (time.perf_counter() - c0) * 1000
        return r

    viaje = ViajeSimulado(True, estrategia, politica, observabilidad,
                          graph.name(s), graph.name(t), t0)
    pos, ruta_act = s, None
    creido = modelo.mult_esperado(env.t)
    plan = planear(pos, ruta_act, creido)
    if not plan.found:
        viaje.existe_ruta = False
        return viaje
    viaje.ruta_planeada = list(plan.path_names)
    viaje.tiempo_planeado_s = plan.path_time_s
    viaje.coste_planeado = plan.path_cost_m
    viaje.ruta_recorrida = [graph.name(s)]
    idx = 0                      # posicion actual dentro de plan.path
    plan_creido = creido         # trafico que supuso el plan vigente

    while pos != t and len(viaje.pasos) < max_pasos:
        siguiente = plan.path[idx + 1]
        ruta_sig = plan.path_routes[idx]
        replan = False
        if politica != "nunca" and viaje.pasos:   # en el origen el plan acaba de calcularse
            ahora = creencia()
            if politica == "cada_parada":
                replan = True
            else:  # umbral: el tramo a tomar esta peor de lo que suponia el plan
                esperado = plan_creido.get((pos, siguiente), 1.0)
                replan = (ahora.get((pos, siguiente), 1.0) - esperado) / esperado > umbral
            if replan:
                plan = planear(pos, ruta_act, ahora)
                if not plan.found:
                    break
                viaje.replanificaciones += 1
                idx, plan_creido = 0, ahora
                siguiente, ruta_sig = plan.path[1], plan.path_routes[0]

        m = env.m_real(pos, siguiente)
        esperado = plan_creido.get((pos, siguiente), 1.0)
        transbordo = ruta_act is not None and ruta_sig != ruta_act
        d = graph.cost(pos, siguiente)
        dt = (d * m + (transfer_penalty_m if transbordo else 0.0)) / v
        viaje.coste_real += d * (1.0 + peso * (m - 1.0)) + (transfer_penalty_m if transbordo else 0.0)
        viaje.dist_real_m += d
        viaje.tiempo_real_s += dt
        viaje.transbordos += int(transbordo)
        viaje.pasos.append(PasoSim(len(viaje.pasos) + 1, graph.name(pos), graph.name(siguiente),
                                   ruta_sig, transbordo, round(m, 3), round(esperado, 3),
                                   round(dt, 1), replan))
        env.avanzar(dt)
        pos, ruta_act = siguiente, ruta_sig
        idx += 1
        viaje.ruta_recorrida.append(graph.name(pos))

    viaje.existe_ruta = pos == t
    viaje.computo_ms = round(computo, 2)
    viaje.picos = env.n_picos
    viaje.coste_real = round(viaje.coste_real, 1)
    viaje.dist_real_m = round(viaje.dist_real_m, 1)
    viaje.tiempo_real_s = round(viaje.tiempo_real_s, 1)
    return viaje
