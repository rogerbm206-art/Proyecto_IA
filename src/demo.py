"""Ejecuta algoritmos de busqueda sobre la red y produce su grafica + resumen.

Lo usan los scripts `scripts/run_<algoritmo>.py` (uno por algoritmo) y `run_todos.py`.
Opciones comunes (ver `cli`):

    python scripts/run_astar.py                          # sin trafico (solo distancia)
    python scripts/run_astar.py --hora 17:00             # con trafico a las 17:00
    python scripts/run_astar.py --hora 17:00 --peso 1    # coste = distancia en tiempo
    python scripts/run_astar.py "Br. Colombia" "Universidad EAN" --hora 08:00
"""
from __future__ import annotations

import argparse

from .agent import RoutePlannerAgent
from .gtfs_graph import load_zonal_graph
from .plot_search import plot_result, resumen_texto
from .metaheuristics import METAHEURISTICAS
from .search import STRATEGIES, V_LIBRE_MS, search
from .traffic import ConfigTrafico, ModeloTrafico, fmt_hora, parse_hora

# Par origen-destino de referencia.
ORIGEN_DEF = "Br. Chicó Norte III Sector"
DESTINO_DEF = "Br. Sucre"


def contexto_trafico(g, hora, peso: float, semilla: int = 0):
    """Devuelve (mult, peso, v_libre_ms, texto) para planificar con trafico.

    Sin `hora` o con `peso` 0 no hay trafico: (None, 0.0, V_LIBRE_MS, None).
    """
    if hora is None or not peso:
        return None, 0.0, V_LIBRE_MS, None
    modelo = ModeloTrafico(g, cfg=ConfigTrafico(peso=peso), semilla=semilla)
    t = parse_hora(hora)
    return modelo.mult_esperado(t), peso, modelo.v_libre_ms, f"{fmt_hora(t)}, peso {peso:g}"


def run_demo(strategy: str, origen: str = ORIGEN_DEF, destino: str = DESTINO_DEF,
             hora=None, peso: float = 0.0, semilla: int = 0):
    if strategy not in STRATEGIES and strategy not in METAHEURISTICAS:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. "
                         f"Use {list(STRATEGIES) + list(METAHEURISTICAS)}")
    g = load_zonal_graph()
    mult, peso, v_libre, txt = contexto_trafico(g, hora, peso, semilla)
    res = search(g, origen, destino, strategy, mult=mult, peso_trafico=peso, v_libre_ms=v_libre)

    print("=" * 52)
    print(resumen_texto(res, origen, destino, txt))
    if res.found:
        # ciclo percepcion -> accion del agente con esta estrategia (reutiliza el plan)
        log = RoutePlannerAgent(g).run(origen, destino, strategy, plan=res)
        print("-" * 52)
        print("ciclo percepcion -> accion:")
        for s in log.bitacora:
            marca = "  <transbordo>" if s.transbordo else ""
            print(f"  {s.paso:2d}. {s.accion}  ({s.coste_tramo_m:.0f} m){marca}")
    png = plot_result(g, res, origen=origen, destino=destino, mult=mult, trafico_txt=txt)
    print("-" * 52)
    print(f"grafica: {png}")
    return res


def cli(strategy: str | None = None) -> None:
    """Interfaz de linea de comandos de los `run_*.py`. `strategy=None` -> los 5 + comparacion."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origen", nargs="?", default=ORIGEN_DEF)
    ap.add_argument("destino", nargs="?", default=DESTINO_DEF)
    ap.add_argument("--hora", help="hora de salida HH:MM; activa el trafico (por defecto: sin trafico)")
    ap.add_argument("--peso", type=float, default=0.5,
                    help="peso del trafico en el coste, 0..1 (0 = solo distancia). Defecto 0.5")
    ap.add_argument("--semilla", type=int, default=0, help="semilla del factor espacial del trafico")
    a = ap.parse_args()

    for s in ([strategy] if strategy else list(STRATEGIES)):
        run_demo(s, a.origen, a.destino, a.hora, a.peso, a.semilla)
        print()
    if strategy is None:
        from . import benchmark
        print("=" * 52)
        print("COMPARACION GLOBAL (sin trafico)")
        filas = benchmark.run()
        benchmark.print_table(filas)
        benchmark.save_csv(filas)
        benchmark.plot(filas)
        print(f"\nCSV: {benchmark.CSV_OUT}\nPNG: {benchmark.PNG_OUT}")
