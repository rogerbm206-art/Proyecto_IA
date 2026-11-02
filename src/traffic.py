"""Modelo de trafico para la red zonal.

El trafico se expresa como un MULTIPLICADOR de congestion m >= 1 por tramo
(1 = flujo libre; 2 = el doble de lento). Se combina con la distancia en el coste:

    coste_tramo = distancia * (1 + peso * (m - 1))

  peso = 0  ->  solo distancia (el modelo original)
  peso = 1  ->  distancia equivalente en tiempo (distancia * m)
  0 < peso < 1  ->  mezcla de ambos (por defecto 0.5)

Como m >= 1, el coste de un tramo nunca baja de su distancia y la heuristica de
linea recta sigue siendo admisible.

Componentes del multiplicador de un tramo e en el instante t:

  1. PERFIL HORARIO (global, calibrado con el GTFS): m_h(t), interpolado entre horas.
  2. FACTOR ESPACIAL (estatico, por tramo): k_e lognormal de media 1, correlacionado
     entre tramos vecinos (se sortea por parada y el tramo promedia sus dos paradas).
     No hay datos de congestion por tramo, asi que es un SUPUESTO con semilla fija.
  3. FLUCTUACION (solo entorno dinamico): proceso de Ornstein-Uhlenbeck en log por
     parada, que cambia mientras el agente avanza.
  4. PICOS REPENTINOS (solo entorno dinamico): eventos aleatorios de congestion en una
     zona (epicentro + vecinos a 1-2 saltos) que luego decaen. NO son cierres de via.

`ModeloTrafico` es lo que el agente CONOCE (patron tipico: 1 x 2). `EntornoTrafico`
es la REALIDAD (1 x 2 x 3 x 4), que el agente solo ve parcialmente al avanzar.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .gtfs_graph import TransitGraph

PERFIL_JSON = Path(__file__).resolve().parent.parent / "data" / "perfil_trafico.json"


# ---------------------------------------------------------------------------
def parse_hora(h) -> float:
    """Convierte una hora a segundos desde la medianoche.

    Acepta "HH:MM", "HH", o un numero en HORAS (16.5 = 16:30).
    """
    if isinstance(h, str):
        partes = h.strip().split(":")
        horas = int(partes[0]) + (int(partes[1]) / 60 if len(partes) > 1 else 0.0)
    else:
        horas = float(h)
    return (horas % 24.0) * 3600.0


def fmt_hora(t_seg: float) -> str:
    t = int(round(t_seg)) % 86400
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}"


def coste_combinado(dist_m: float, m: float, peso: float) -> float:
    """Coste de un tramo: mezcla de distancia y distancia equivalente en tiempo."""
    return dist_m * (1.0 + peso * (m - 1.0))


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PerfilHorario:
    """Multiplicador de congestion por hora del dia (24 valores, 1 = flujo libre)."""

    mult: tuple[float, ...]
    v_libre_kmh: float

    @classmethod
    def cargar(cls, path: Path = PERFIL_JSON) -> "PerfilHorario":
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}. Generalo con: python scripts/calibrar_trafico.py")
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(d["multiplicador"]), float(d["v_libre_kmh"]))

    @property
    def v_libre_ms(self) -> float:
        return self.v_libre_kmh / 3.6

    def m(self, t_seg: float) -> float:
        """m(t) con interpolacion lineal entre los centros de hora (continuo y circular)."""
        x = (t_seg / 3600.0) % 24.0 - 0.5
        i0 = math.floor(x)
        fr = x - i0
        a, b = self.mult[i0 % 24], self.mult[(i0 + 1) % 24]
        return a + (b - a) * fr


@dataclass(frozen=True)
class ConfigTrafico:
    peso: float = 0.5               # peso del trafico en el coste (0 = solo distancia)
    sigma_espacial: float = 0.25    # variacion estatica entre tramos (lognormal)
    sigma_dinamico: float = 0.20    # fluctuacion aleatoria durante el viaje
    tau_reversion_s: float = 900.0  # constante de tiempo de la fluctuacion (15 min)
    picos_por_hora: float = 4.0     # picos repentinos en toda la red (entorno dinamico)
    pico_amp: tuple[float, float] = (0.6, 1.6)  # aumento del multiplicador en el epicentro
    pico_tau_s: float = 600.0       # decaimiento de un pico (10 min)
    pico_radio_saltos: int = 2      # alcance de un pico (saltos desde el epicentro)


# ---------------------------------------------------------------------------
class ModeloTrafico:
    """Lo que el agente CONOCE: patron tipico por hora + factor espacial de cada tramo."""

    def __init__(self, graph: TransitGraph, perfil: PerfilHorario | None = None,
                 cfg: ConfigTrafico | None = None, semilla: int = 0):
        self.g = graph
        self.perfil = perfil or PerfilHorario.cargar()
        self.cfg = cfg or ConfigTrafico()
        self.semilla = semilla
        rng = random.Random(semilla)
        self._z = {n: rng.gauss(0.0, 1.0) for n in sorted(graph.nodes())}
        s = self.cfg.sigma_espacial
        self._k: dict[tuple[str, str], float] = {}
        for a in graph.nodes():
            for b in graph.neighbors(a):
                if a < b:
                    # media 1: exp(s*z - s^2/2); z = promedio normalizado de las 2 paradas
                    self._k[(a, b)] = math.exp(
                        s * (self._z[a] + self._z[b]) / math.sqrt(2.0) - s * s / 2.0)

    @property
    def v_libre_ms(self) -> float:
        return self.perfil.v_libre_ms

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def k(self, a: str, b: str) -> float:
        return self._k[self._key(a, b)]

    def m_esperado(self, a: str, b: str, t_seg: float) -> float:
        return max(1.0, self.perfil.m(t_seg) * self.k(a, b))

    def mult_esperado(self, t_seg: float) -> dict[tuple[str, str], float]:
        """Snapshot {(a, b): m} (en ambos sentidos) del patron tipico en el instante t."""
        mh = self.perfil.m(t_seg)
        out: dict[tuple[str, str], float] = {}
        for (a, b), k in self._k.items():
            m = max(1.0, mh * k)
            out[(a, b)] = m
            out[(b, a)] = m
        return out


# ---------------------------------------------------------------------------
class EntornoTrafico:
    """La REALIDAD: evoluciona con el reloj mientras el agente avanza.

    En modo estatico (`dinamico=False`) coincide con el patron tipico del modelo
    (solo cambia con la hora). En modo dinamico anade fluctuaciones y picos aleatorios.
    """

    def __init__(self, modelo: ModeloTrafico, hora_salida_s: float,
                 dinamico: bool = True, semilla: int = 0):
        self.modelo = modelo
        self.t = float(hora_salida_s)
        self.dinamico = dinamico
        self._rng = random.Random(semilla)
        self._nodos = sorted(modelo.g.nodes())
        sd = modelo.cfg.sigma_dinamico
        self._x = ({n: self._rng.gauss(0.0, sd) for n in self._nodos}
                   if dinamico else {n: 0.0 for n in self._nodos})
        self._picos: list[tuple[float, float, dict[str, float]]] = []
        self.n_picos = 0

    # -- evolucion -------------------------------------------------------
    def avanzar(self, dt_s: float) -> None:
        if not self.dinamico:
            self.t += dt_s
            return
        restante = dt_s
        while restante > 1e-9:
            paso = min(60.0, restante)
            self._paso(paso)
            restante -= paso

    def _vecindad(self, centro: str, radio: int) -> dict[str, float]:
        """{parada: peso} con peso = 1/(1+saltos) para las paradas a <= radio saltos."""
        dist = {centro: 0}
        cola = [centro]
        for n in cola:
            if dist[n] == radio:
                continue
            for v in self.modelo.g.neighbors(n):
                if v not in dist:
                    dist[v] = dist[n] + 1
                    cola.append(v)
        return {n: 1.0 / (1.0 + d) for n, d in dist.items()}

    def _paso(self, dt: float) -> None:
        cfg = self.modelo.cfg
        a = math.exp(-dt / cfg.tau_reversion_s)
        b = cfg.sigma_dinamico * math.sqrt(1.0 - a * a)
        for n in self._nodos:
            self._x[n] = self._x[n] * a + b * self._rng.gauss(0.0, 1.0)
        if self._rng.random() < cfg.picos_por_hora * dt / 3600.0:
            centro = self._rng.choice(self._nodos)
            amp = self._rng.uniform(*cfg.pico_amp)
            self._picos.append((self.t, amp, self._vecindad(centro, cfg.pico_radio_saltos)))
            self.n_picos += 1
        self.t += dt
        self._picos = [p for p in self._picos if self.t - p[0] < 5.0 * cfg.pico_tau_s]

    # -- consulta --------------------------------------------------------
    def m_real(self, a: str, b: str) -> float:
        m = self.modelo.perfil.m(self.t) * self.modelo.k(a, b)
        if self.dinamico:
            cfg = self.modelo.cfg
            m *= math.exp((self._x[a] + self._x[b]) / math.sqrt(2.0))
            pico = 0.0
            for t0, amp, afectadas in self._picos:
                w = (afectadas.get(a, 0.0) + afectadas.get(b, 0.0)) / 2.0
                if w:
                    pico += amp * w * math.exp(-(self.t - t0) / cfg.pico_tau_s)
            m *= 1.0 + pico
        return max(1.0, m)

    def todos_m_real(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for a in self._nodos:
            for b in self.modelo.g.neighbors(a):
                if a < b:
                    m = self.m_real(a, b)
                    out[(a, b)] = m
                    out[(b, a)] = m
        return out
