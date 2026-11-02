"""Dibuja el resultado de una busqueda: la red con el camino resaltado + metricas.

`plot_result(graph, result, out_path)` genera un PNG con:
  - el grafo completo de la red (gris),
  - el camino elegido por el algoritmo resaltado (amarillo), con el origen en
    verde y el destino en rojo,
  - un panel de texto con las metricas (tiempo, memoria, nodos expandidos,
    nodos generados, frontera maxima, tramos, transbordos, distancia, coste).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .gtfs_graph import TransitGraph
from .search import SearchResult

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Con redes grandes y rutas largas (p. ej. DFS) hay que limitar el texto del dibujo.
MAX_ETIQUETAS = 28   # si la ruta tiene mas paradas, solo se rotulan origen y destino
MAX_LISTA = 36       # lineas maximas de la lista "ruta:" del panel (se resume el medio)
M_MIN, M_MAX = 1.0, 2.6   # rango de la escala de color de la congestion


def resumen_texto(res: SearchResult, origen: str, destino: str,
                  trafico: str | None = None) -> str:
    """Texto de una linea-por-metrica para consola o para el panel del grafico."""
    if not res.found:
        return (f"algoritmo : {res.strategy}\n"
                f"origen    : {origen}\n"
                f"destino   : {destino}\n"
                f"RESULTADO : sin ruta\n"
                f"expandidos: {res.nodes_expanded}\n"
                f"tiempo    : {res.elapsed_ms:.3f} ms")
    return (
        f"algoritmo      : {res.strategy}\n"
        f"origen         : {origen}\n"
        f"destino        : {destino}\n"
        f"tramos         : {res.path_edges}\n"
        f"transbordos    : {res.transbordos}\n"
        f"distancia      : {res.path_dist_m:.0f} m\n"
        f"coste (d+transb): {res.path_cost_m:.0f} m\n"
        f"tiempo estimado: {res.path_time_s / 60:.1f} min\n"
        + (f"trafico        : {trafico}\n" if trafico else "")
        + f"optima (coste) : {'si' if res.strategy in ('ucs', 'astar') else '-'}\n"
        f"nodos expandidos: {res.nodes_expanded}\n"
        f"nodos generados : {res.nodes_generated}\n"
        f"frontera maxima : {res.max_frontier}\n"
        f"tiempo         : {res.elapsed_ms:.3f} ms\n"
        f"memoria pico   : {res.peak_kib:.1f} KiB"
    )


def plot_result(graph: TransitGraph, res: SearchResult, out_path: Path | None = None,
                origen: str = "", destino: str = "", mult: dict | None = None,
                trafico_txt: str | None = None) -> Path:
    h = graph.to_networkx()
    pos = {n: (d["lon"], d["lat"]) for n, d in h.nodes(data=True)}
    origen = origen or (graph.name(res.path[0]) if res.path else "")
    destino = destino or (graph.name(res.path[-1]) if res.path else "")

    fig, (axm, axt) = plt.subplots(
        1, 2, figsize=(16, 12.5), gridspec_kw={"width_ratios": [3.1, 1]}
    )

    # --- mapa -----------------------------------------------------------
    if mult:   # tramos coloreados por congestion (verde = libre, rojo = congestionado)
        aristas = list(h.edges())
        nx.draw_networkx_edges(h, pos, ax=axm, edgelist=aristas, width=1.8,
                               edge_color=[mult.get((u, v), 1.0) for u, v in aristas],
                               edge_cmap=plt.cm.RdYlGn_r, edge_vmin=M_MIN, edge_vmax=M_MAX,
                               alpha=0.55 if res.found else 1.0)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(M_MIN, M_MAX), cmap=plt.cm.RdYlGn_r),
                     ax=axm, shrink=0.45, pad=0.01, label="congestion del tramo (x flujo libre)")
    else:
        nx.draw_networkx_edges(h, pos, ax=axm, edge_color="#D5D5D5", width=0.8)
    nx.draw_networkx_nodes(h, pos, ax=axm, node_size=45,
                           node_color="#E8E8E8", edgecolors="#9A9A9A", linewidths=0.5)

    if res.found and len(res.path) > 1:
        camino = list(zip(res.path, res.path[1:]))
        # el camino se atenua el resto de la red y se dibuja con borde oscuro (halo) para destacarlo
        nx.draw_networkx_edges(h, pos, ax=axm, edgelist=camino,
                               edge_color="#1B2631", width=11)
        nx.draw_networkx_edges(h, pos, ax=axm, edgelist=camino,
                               edge_color="#FFD60A", width=7)
        nx.draw_networkx_nodes(h, pos, ax=axm, nodelist=res.path,
                               node_size=170, node_color="#FFD60A",
                               edgecolors="#1B2631", linewidths=1.6)
        nx.draw_networkx_nodes(h, pos, ax=axm, nodelist=[res.path[0]],
                               node_size=260, node_color="#2ECC71", edgecolors="#145A32")
        nx.draw_networkx_nodes(h, pos, ax=axm, nodelist=[res.path[-1]],
                               node_size=260, node_color="#E74C3C", edgecolors="#7B241C")
        rotulados = (res.path if len(res.path) <= MAX_ETIQUETAS
                     else [res.path[0], res.path[-1]])
        for n in rotulados:
            x, y = pos[n]
            axm.annotate(graph.name(n), (x, y), fontsize=7.5,
                         xytext=(4, 3), textcoords="offset points")

    axm.set_title(f"Ruta {res.strategy.upper()}   {origen}  ->  {destino}", fontsize=12)
    axm.set_xlabel("Longitud")
    axm.set_ylabel("Latitud")
    axm.set_aspect("equal", adjustable="datalim")
    axm.grid(True, linestyle=":", alpha=0.35)

    # --- panel de metricas -------------------------------------------------
    axt.axis("off")
    axt.text(0.0, 1.0, resumen_texto(res, origen, destino, trafico_txt), transform=axt.transAxes,
             va="top", ha="left", family="monospace", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#999"))
    if res.found:
        lineas = [f"  {i}. {graph.name(s)}" for i, s in enumerate(res.path)]
        if len(lineas) > MAX_LISTA:
            mitad = MAX_LISTA // 2
            lineas = (lineas[:mitad]
                      + [f"  ... ({len(lineas) - 2 * mitad} paradas intermedias) ..."]
                      + lineas[-mitad:])
        axt.text(0.0, 0.68, "ruta:\n" + "\n".join(lineas),
                 transform=axt.transAxes, va="top", ha="left",
                 family="monospace", fontsize=7.5)

    out_path = out_path or (OUT_DIR / f"ruta_{res.strategy}.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trafico(graph: TransitGraph, mult: dict, titulo: str,
                 out_path: Path | None = None, etiquetas: int = 25) -> Path:
    """Mapa de la red con cada tramo coloreado por su congestion (`mult` = {(a, b): m})."""
    h = graph.to_networkx()
    pos = {n: (d["lon"], d["lat"]) for n, d in h.nodes(data=True)}
    deg = dict(h.degree())
    aristas = list(h.edges())

    fig, ax = plt.subplots(figsize=(13, 12.5))
    nx.draw_networkx_edges(h, pos, ax=ax, edgelist=aristas, width=2.4,
                           edge_color=[mult.get((u, v), 1.0) for u, v in aristas],
                           edge_cmap=plt.cm.RdYlGn_r, edge_vmin=M_MIN, edge_vmax=M_MAX)
    nx.draw_networkx_nodes(h, pos, ax=ax, node_size=[18 + 6 * deg[n] for n in h.nodes()],
                           node_color="#F5F5F5", edgecolors="#555", linewidths=0.5)
    for n in sorted(h.nodes(), key=lambda n: (-deg[n], n))[:etiquetas]:
        x, y = pos[n]
        ax.annotate(graph.name(n), (x, y), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(M_MIN, M_MAX), cmap=plt.cm.RdYlGn_r),
                 ax=ax, shrink=0.5, pad=0.01, label="congestion del tramo (x flujo libre)")
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.35)
    out_path = out_path or (OUT_DIR / "mapa_trafico.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
