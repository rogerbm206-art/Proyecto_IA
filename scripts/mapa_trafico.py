"""Dibuja el mapa de la red con cada tramo coloreado por su congestion a una hora.

Uso:  python scripts/mapa_trafico.py [HH:MM ...] [--semilla N]
      python scripts/mapa_trafico.py 03:00 08:00 16:00
Salida: output/mapa_trafico_HHMM.png  (una imagen por hora)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gtfs_graph import load_zonal_graph
from src.plot_search import OUT_DIR, plot_trafico
from src.traffic import ConfigTrafico, ModeloTrafico, fmt_hora, parse_hora

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("horas", nargs="*", default=["03:00", "08:00", "16:00"])
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    g = load_zonal_graph()
    modelo = ModeloTrafico(g, cfg=ConfigTrafico(), semilla=a.semilla)
    for h in a.horas:
        t = parse_hora(h)
        mult = modelo.mult_esperado(t)
        medio = sum(mult.values()) / len(mult)
        png = plot_trafico(
            g, mult, f"Trafico tipico a las {fmt_hora(t)}  (congestion media x{medio:.2f}; "
                     f"semilla del factor espacial = {a.semilla})",
            OUT_DIR / f"mapa_trafico_{fmt_hora(t).replace(':', '')}.png")
        print(f"{fmt_hora(t)}  congestion media x{medio:.2f}  ->  {png}")
