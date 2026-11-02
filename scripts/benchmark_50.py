"""Corre cada algoritmo de busqueda N veces y vuelca todas las metricas a un Excel,
con un promedio por parametro. En cada corrida los 5 algoritmos resuelven el mismo
par origen-destino.

Uso:
    python scripts/benchmark_50.py                              # 50 corridas, pares ALEATORIOS (semilla 42)
    python scripts/benchmark_50.py -n 100 -s 7                  # 100 corridas, semilla 7
    python scripts/benchmark_50.py -o "Br. Colombia" -d "Universidad EAN"   # par FIJO (elegido a mano)
    python scripts/benchmark_50.py --listar                     # imprime las 60 paradas y sale

Los nombres aceptan coincidencia parcial (p. ej. -o "colombia").

Salida: data/benchmark_50.xlsx
  - hoja "Resumen"      -> una fila por algoritmo con el promedio de las N corridas
  - hoja por algoritmo  -> las N corridas (con su par O-D) + fila PROMEDIO al final
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from src.gtfs_graph import load_zonal_graph
from src.metaheuristics import METAHEURISTICAS
from src.search import STRATEGIES, search

XLSX_OUT = Path(__file__).resolve().parent.parent / "data" / "benchmark_50.xlsx"

# columna interna -> (titulo, formato de nº)
METRICAS = [
    ("tramos", "Tramos", "0"),
    ("transbordos", "Transbordos", "0"),
    ("dist_m", "Distancia (m)", "0.0"),
    ("coste_m", "Coste = dist + transbordos (m)", "0.0"),
    ("desempeno", "Desempeno (% del optimo)", "0.0"),
    ("expandidos", "Nodos expandidos", "0"),
    ("generados", "Nodos generados", "0"),
    ("frontera_max", "Frontera maxima", "0"),
    ("tiempo_ms", "Tiempo (ms)", "0.000"),
    ("memoria_kib", "Memoria pico (KiB)", "0.00"),
]
_LEAD = ["Corrida", "Origen", "Destino"]

_HDR_FILL = PatternFill("solid", fgColor="D9E1F2")
_AVG_FILL = PatternFill("solid", fgColor="FCE4D6")
_BOLD = Font(bold=True)
_TOP = Border(top=Side(style="medium"))


def _pares_aleatorios(g, n: int, semilla: int) -> list[tuple[str, str]]:
    rng = random.Random(semilla)
    nodos = g.nodes()
    pares = []
    while len(pares) < n:
        o, d = rng.sample(nodos, 2)
        pares.append((o, d))
    return pares


def _fila(g, o, d, strat, i, coste_optimo: float) -> dict:
    r = search(g, o, d, strat, semilla=i)   # semilla distinta por corrida (solo metaheuristicas)
    # Medida de desempeno (PEAS): la tarea es llegar al destino minimizando el
    # coste = distancia + penalizacion por transbordo. Se expresa como % del
    # optimo alcanzable (el de UCS): 100 % = ruta optima, menos = ruta peor.
    desempeno = 100.0 * coste_optimo / r.path_cost_m if r.found and r.path_cost_m else 0.0
    return {
        "corrida": i,
        "origen": g.name(o),
        "destino": g.name(d),
        "tramos": r.path_edges,
        "transbordos": r.transbordos,
        "dist_m": r.path_dist_m,
        "coste_m": r.path_cost_m,
        "desempeno": round(desempeno, 1),
        "expandidos": r.nodes_expanded,
        "generados": r.nodes_generated,
        "frontera_max": r.max_frontier,
        "tiempo_ms": r.elapsed_ms,
        "memoria_kib": r.peak_kib,
    }


def _hoja_algoritmo(wb: Workbook, titulo: str, filas: list[dict]) -> dict:
    ws = wb.create_sheet(titulo)
    ws.append(_LEAD + [t for _, t, _ in METRICAS])
    for c in ws[1]:
        c.font = _BOLD
        c.fill = _HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")

    cols = ["corrida", "origen", "destino"] + [k for k, *_ in METRICAS]
    for f in filas:
        ws.append([f[c] for c in cols])

    prom = {k: statistics.mean(f[k] for f in filas) for k, *_ in METRICAS}
    ws.append(["PROMEDIO", f"({len(filas)} pares aleatorios)", "—"]
              + [prom[k] for k, *_ in METRICAS])
    last = ws.max_row
    for c in ws[last]:
        c.font = _BOLD
        c.fill = _AVG_FILL
        c.border = _TOP

    for j, (_k, _t, fmt) in enumerate(METRICAS, start=len(_LEAD) + 1):
        letra = ws.cell(row=1, column=j).column_letter
        for row in range(2, last + 1):
            ws.cell(row=row, column=j).number_format = fmt
        ws.column_dimensions[letra].width = 16
    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 26
    ws.freeze_panes = "D2"
    return prom


def _hoja_resumen(wb: Workbook, promedios: dict[str, dict], descripcion: str):
    ws = wb.create_sheet("Resumen", 0)
    ws.append([descripcion])
    ws["A1"].font = Font(bold=True, size=12)
    ws.append([])
    ws.append(["Algoritmo"] + [t for _, t, _ in METRICAS])
    for c in ws[3]:
        c.font = _BOLD
        c.fill = _HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")

    for strat, prom in promedios.items():
        ws.append([strat.upper()] + [prom[k] for k, *_ in METRICAS])

    for j, (_k, _t, fmt) in enumerate(METRICAS, start=2):
        for row in range(4, 4 + len(promedios)):
            ws.cell(row=row, column=j).number_format = fmt
        ws.column_dimensions[ws.cell(row=3, column=j).column_letter].width = 16
    ws.column_dimensions["A"].width = 12
    ws.freeze_panes = "B4"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--corridas", type=int, default=50, help="nº de corridas (50)")
    ap.add_argument("-s", "--semilla", type=int, default=42, help="semilla de aleatoriedad (42)")
    ap.add_argument("-o", "--origen", help="parada de origen (par FIJO; requiere --destino)")
    ap.add_argument("-d", "--destino", help="parada de destino (par FIJO; requiere --origen)")
    ap.add_argument("--listar", action="store_true", help="imprime las paradas y sale")
    args = ap.parse_args(argv)

    g = load_zonal_graph()

    if args.listar:
        for nombre in sorted(set(g.names.values())):
            print(" ", nombre)
        return

    n = args.corridas
    if bool(args.origen) ^ bool(args.destino):
        ap.error("--origen y --destino se usan juntos")

    if args.origen and args.destino:
        o, d = g.resolve(args.origen), g.resolve(args.destino)
        pares = [(o, d)] * n
        descripcion = (f"{n} corridas | par FIJO: {g.name(o)}  ->  {g.name(d)} "
                       f"(solo varian tiempo y memoria)")
        print(descripcion)
    else:
        pares = _pares_aleatorios(g, n, args.semilla)
        descripcion = (f"Promedio de {n} corridas con origen y destino ALEATORIOS "
                       f"(semilla = {args.semilla}); mismo par O-D para los 10 algoritmos (5 de busqueda + 5 metaheuristicas)")
        print(f"{n} corridas | origen-destino aleatorios | semilla {args.semilla}")

    # coste optimo de cada par (el de UCS) -> referencia de la medida de desempeno
    optimo = [search(g, o, d, "ucs").path_cost_m for o, d in pares]

    wb = Workbook()
    wb.remove(wb.active)
    promedios: dict[str, dict] = {}
    for strat in [*STRATEGIES, *METAHEURISTICAS]:
        filas = [_fila(g, o, d, strat, i, optimo[i - 1])
                 for i, (o, d) in enumerate(pares, start=1)]
        promedios[strat] = _hoja_algoritmo(wb, strat.upper(), filas)
        p = promedios[strat]
        print(f"  {strat:7s} desempeno {p['desempeno']:5.1f}% | tiempo {p['tiempo_ms']:7.3f} ms "
              f"| mem {p['memoria_kib']:6.1f} KiB | expandidos {p['expandidos']:6.1f} "
              f"| transbordos {p['transbordos']:.2f}")

    _hoja_resumen(wb, promedios, descripcion)
    XLSX_OUT.parent.mkdir(exist_ok=True)
    wb.save(XLSX_OUT)
    print(f"\nExcel: {XLSX_OUT}")


if __name__ == "__main__":
    main()
