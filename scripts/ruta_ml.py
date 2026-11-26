"""Planifica una ruta con el agente usando los modelos de ML y la compara con el coste solo distancia.

Uso:  python scripts/ruta_ml.py "Br. Chicó Norte III Sector" "Br. Sucre" --hora 17:00 --dia habil

Muestra, para cada criterio, la ruta, los transbordos, el tiempo previsto y, tramo a tramo,
la congestion prevista por el clasificador y el grupo de K-Medias del tramo.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import RoutePlannerAgent
from src.gtfs_graph import load_zonal_graph
from src.ml import integracion
from src.traffic import parse_hora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("origen")
    ap.add_argument("destino")
    ap.add_argument("--hora", default="17:00")
    ap.add_argument("--dia", default="habil", choices=integracion.DIAS)
    ap.add_argument("--algoritmo", default="astar")
    ap.add_argument("--peso", type=float, default=1.0, help="peso del trafico en el coste (0-1)")
    a = ap.parse_args()

    if not integracion.modelos_disponibles():
        sys.exit("faltan los modelos: ejecuta scripts/entrenar_regresion.py y scripts/entrenar_clasificacion.py")
    g = load_zonal_graph()
    pred = integracion.PredictorTramos(g)
    agente = RoutePlannerAgent(g)
    o, d = g.resolve(a.origen), g.resolve(a.destino)
    h = int(parse_hora(a.hora) // 3600)
    v = pred.perfil.v_libre_ms

    criterios = {
        "solo distancia (sin tráfico)": {},
        f"tiempo predicho por ML ({a.dia}, {h:02d} h, peso {a.peso:g})":
            {"mult": pred.mult(h, a.dia), "peso_trafico": a.peso, "v_libre_ms": v},
    }
    for nombre, kw in criterios.items():
        r = agente.plan(o, d, a.algoritmo, **kw)
        t_ml = sum(pred.tiempo_s(x, y, h, a.dia) for x, y in zip(r.path, r.path[1:]))
        print(f"\n== {nombre} ==")
        print(f"{r.path_edges} tramos, {r.transbordos} transbordos, {r.path_dist_m:,.0f} m, "
              f"tiempo previsto por el modelo {t_ml / 60:.1f} min (sin esperas)")
        for (x, y), bus in zip(zip(r.path, r.path[1:]), r.path_routes):
            gr = pred.grupo(x, y)
            print(f"  ruta {bus:>6}  {g.name(x)} -> {g.name(y)}  "
                  f"{pred.tiempo_s(x, y, h, a.dia):5.0f} s  congestión {pred.clase(x, y, h, a.dia):<5}"
                  f"  grupo {gr if gr else '-'}")


if __name__ == "__main__":
    main()
