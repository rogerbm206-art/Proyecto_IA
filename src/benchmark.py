"""Comparacion de los algoritmos de busqueda sobre la red zonal.

Para cada par origen-destino ejecuta bfs / dfs / ucs / greedy / astar y compara:

    tramos        - nº de aristas de la ruta encontrada
    coste (m)     - distancia total de la ruta por la red
    optima        - si el coste coincide con el minimo (el de UCS)
    expandidos    - nodos sacados de la frontera y desarrollados
    generados     - nodos insertados en la frontera
    frontera_max  - tamaño maximo de la frontera (memoria del algoritmo)
    tiempo (ms)   - mediana de N repeticiones
    memoria (KiB) - pico de asignacion de memoria del proceso (tracemalloc)

Genera:
  - data/benchmark_resultados.csv
  - output/comparacion_algoritmos.png   (graficas de barras comparativas)
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from .gtfs_graph import TransitGraph, load_zonal_graph
from .search import STRATEGIES, RouteProblem, search

ROOT = Path(__file__).resolve().parent.parent
CSV_OUT = ROOT / "data" / "benchmark_resultados.csv"
PNG_OUT = ROOT / "output" / "comparacion_algoritmos.png"

# Pares origen-destino de prueba (nombres reales del grafo de 300 paradas; los
# dos primeros cruzan varios barrios de norte a sur).
PARES = [
    ("Br. Chicó Norte III Sector", "Br. Sucre"),
    ("Estación Calle 100", "Br. Marly"),
    ("Br. Colombia", "Universidad De La Salle"),
    ("Br. La Cabrera", "Cuerpo Of de Bomberos de Bogotá"),
    ("U. Sergio Arboleda", "Br. Galerías"),
]
REPS = 21  # repeticiones para medir el tiempo (mediana); con 300 paradas cada busqueda cuesta ms


def _time_ms(graph, o, d, strategy) -> float:
    p = RouteProblem(graph, graph.resolve(o), graph.resolve(d))
    fn = STRATEGIES[strategy]
    muestras = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(p)
        muestras.append((time.perf_counter() - t0) * 1000)
    return statistics.median(muestras)


def run(graph: TransitGraph | None = None) -> list[dict]:
    g = graph or load_zonal_graph()
    filas: list[dict] = []
    for o, d in PARES:
        costo_opt = search(g, o, d, "ucs").path_cost_m
        for strat in STRATEGIES:
            r = search(g, o, d, strat)
            filas.append({
                "origen": o,
                "destino": d,
                "algoritmo": strat,
                "tramos": r.path_edges,
                "transbordos": r.transbordos,
                "dist_m": r.path_dist_m,
                "coste_m": r.path_cost_m,
                "optima": "si" if abs(r.path_cost_m - costo_opt) < 1e-6 else "no",
                "expandidos": r.nodes_expanded,
                "generados": r.nodes_generated,
                "frontera_max": r.max_frontier,
                "tiempo_ms": round(_time_ms(g, o, d, strat), 5),
                "memoria_kib": r.peak_kib,
            })
    return filas


def print_table(filas: list[dict]) -> None:
    cols = ["origen", "destino", "algoritmo", "tramos", "transbordos", "dist_m",
            "coste_m", "optima", "expandidos", "generados", "frontera_max", "tiempo_ms"]
    w = {c: max(len(c), *(len(str(f[c])) for f in filas)) for c in cols}
    par_actual = None
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("-" * (sum(w.values()) + 2 * (len(cols) - 1)))
    for f in filas:
        if (f["origen"], f["destino"]) != par_actual:
            par_actual = (f["origen"], f["destino"])
            print()
        print("  ".join(str(f[c]).ljust(w[c]) for c in cols))


def save_csv(filas: list[dict], path: Path = CSV_OUT) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(filas[0]))
        wr.writeheader()
        wr.writerows(filas)


def plot(filas: list[dict], path: Path = PNG_OUT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = list(STRATEGIES)
    metricas = [
        ("expandidos", "Nodos expandidos (media)"),
        ("frontera_max", "Frontera maxima (media)"),
        ("transbordos", "Transbordos de la ruta (media)"),
        ("dist_m", "Distancia de la ruta, m (media)"),
        ("coste_m", "Coste = distancia + transbordos, m (media)"),
        ("tiempo_ms", "Tiempo, ms (media)"),
    ]
    prom = {
        m: [statistics.mean(f[m] for f in filas if f["algoritmo"] == a) for a in algos]
        for m, _ in metricas
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for ax, (m, titulo) in zip(axes.flat, metricas):
        ax.bar(algos, prom[m], color=colores)
        ax.set_title(titulo, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for i, v in enumerate(prom[m]):
            ax.text(i, v, f"{v:.0f}" if v >= 10 else f"{v:.2g}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Comparacion de algoritmos de busqueda - red zonal (300 paradas) "
                 f"({len(PARES)} pares O-D)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    filas = run()
    print_table(filas)
    save_csv(filas)
    plot(filas)
    print(f"\nCSV : {CSV_OUT}")
    print(f"PNG : {PNG_OUT}")
