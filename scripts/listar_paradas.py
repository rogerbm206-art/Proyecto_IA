"""Genera un Excel con las 60 paradas del grafo de la red zonal.

Salida: data/paradas.xlsx  (una fila por parada)
  Nº | Paradero | Grado | Latitud | Longitud | Nº de rutas | Rutas
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from src.gtfs_graph import load_zonal_graph

XLSX_OUT = Path(__file__).resolve().parent.parent / "data" / "paradas.xlsx"
_HDR_FILL = PatternFill("solid", fgColor="D9E1F2")
_BOLD = Font(bold=True)


def main() -> None:
    g = load_zonal_graph()
    nodos = sorted(g.nodes(), key=lambda q: g.name(q).lower())

    wb = Workbook()
    ws = wb.active
    ws.title = "Paradas"
    ws.append(["Nº", "Paradero", "Grado", "Latitud", "Longitud", "Nº de rutas", "Rutas"])
    for c in ws[1]:
        c.font = _BOLD
        c.fill = _HDR_FILL
        c.alignment = Alignment(vertical="center")

    for i, q in enumerate(nodos, start=1):
        lat, lon = g.coords[q]
        rutas = sorted({r for v in g.neighbors(q) for r in g.edge(q, v)["rutas"]})
        ws.append([i, g.name(q), len(g.adj[q]), round(lat, 6), round(lon, 6),
                   len(rutas), ", ".join(rutas)])

    for col, w in zip("ABCDEFG", [5, 34, 8, 12, 12, 12, 70]):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        row[3].number_format = "0.000000"
        row[4].number_format = "0.000000"
    ws.freeze_panes = "A2"

    XLSX_OUT.parent.mkdir(exist_ok=True)
    wb.save(XLSX_OUT)
    print(f"{len(nodos)} paradas -> {XLSX_OUT}")


if __name__ == "__main__":
    main()
