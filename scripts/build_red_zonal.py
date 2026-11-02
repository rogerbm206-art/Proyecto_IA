"""Dibuja la red de paradas del componente zonal (buses azules SITP).

El grafo se construye en `src/gtfs_graph.py` (a partir del GTFS oficial
`data/GTFS-SIMUR-2026-04-29.zip`, componente zonal, recuadro de Chapinero y barrios vecinos).
Este script solo lo carga y lo dibuja con layout geografico real.

Salida: output/red_zonal_chapinero.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.gtfs_graph import load_zonal_graph

OUT_PNG = Path(__file__).resolve().parent.parent / "output" / "red_zonal_chapinero.png"
EN_ETIQUETA = 45   # nº de paradas con etiqueta en el dibujo


def plot() -> Path:
    tg = load_zonal_graph()
    h = tg.to_networkx()
    pos = {n: (d["lon"], d["lat"]) for n, d in h.nodes(data=True)}
    deg = dict(h.degree())
    # con ~300 nodos se dibuja mas pequeño y solo se etiquetan los mas conectados
    node_sizes = [25 + 22 * deg[n] for n in h.nodes()]
    edge_widths = [0.3 + 0.18 * h[u][v]["n_rutas"] for u, v in h.edges()]
    etiquetados = set(sorted(h.nodes(), key=lambda n: (-deg[n], n))[:EN_ETIQUETA])

    fig, ax = plt.subplots(figsize=(16, 15))
    nx.draw_networkx_edges(h, pos, ax=ax, edge_color="#6C8EBF", width=edge_widths, alpha=0.75)
    nodes = nx.draw_networkx_nodes(
        h, pos, ax=ax, node_size=node_sizes,
        node_color=[deg[n] for n in h.nodes()], cmap="YlOrRd",
        edgecolors="#333", linewidths=0.7,
    )
    for n, (x, y) in pos.items():
        if n in etiquetados:
            ax.annotate(h.nodes[n]["name"], (x, y), fontsize=6.3,
                        xytext=(4, 3), textcoords="offset points")

    cbar = fig.colorbar(nodes, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("grado del nodo (nº de paradas conectadas)")
    ax.set_title(
        "Red de paradas del componente zonal (buses azules SITP)\n"
        f"{tg.meta.get('zona', 'Bogota')} - {h.number_of_nodes()} paradas, "
        f"{h.number_of_edges()} tramos (GTFS SIMUR 2026-04-29)\n"
        f"(se etiquetan las {EN_ETIQUETA} paradas mas conectadas)",
        fontsize=11,
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    return OUT_PNG


if __name__ == "__main__":
    tg = load_zonal_graph()
    print(f"Nodos: {len(tg.nodes())}  Aristas: {tg.number_of_edges()}")
    print(f"Grafica guardada en: {plot()}")
