"""Simula un viaje con trafico DINAMICO: el trafico cambia mientras el agente avanza
y el agente puede replanificar. Compara las politicas de replanificacion con la
misma realidad (misma semilla).

Uso:
  python scripts/simular_viaje.py                       # par por defecto, 17:00, A*
  python scripts/simular_viaje.py "Origen" "Destino" --hora 08:00 --algoritmo ucs --semilla 5
  python scripts/simular_viaje.py --observabilidad global --umbral 0.15
  python scripts/simular_viaje.py --estatico            # sin cambios repentinos (solo la hora)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo import DESTINO_DEF, ORIGEN_DEF
from src.gtfs_graph import load_zonal_graph
from src.metaheuristics import METAHEURISTICAS
from src.search import STRATEGIES
from src.simulacion import OBSERVABILIDAD, POLITICAS, simular_viaje
from src.traffic import ConfigTrafico, ModeloTrafico

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origen", nargs="?", default=ORIGEN_DEF)
    ap.add_argument("destino", nargs="?", default=DESTINO_DEF)
    ap.add_argument("--hora", default="17:00")
    ap.add_argument("--algoritmo", default="astar", choices=list(STRATEGIES) + list(METAHEURISTICAS))
    ap.add_argument("--peso", type=float, default=0.5, help="peso del trafico en el coste (0..1)")
    ap.add_argument("--observabilidad", default="local", choices=OBSERVABILIDAD)
    ap.add_argument("--umbral", type=float, default=0.25)
    ap.add_argument("--semilla", type=int, default=1, help="semilla de la realidad (trafico dinamico)")
    ap.add_argument("--estatico", action="store_true", help="entorno sin cambios repentinos")
    a = ap.parse_args()

    g = load_zonal_graph()
    modelo = ModeloTrafico(g, cfg=ConfigTrafico(peso=a.peso), semilla=0)
    print(f"{a.origen} -> {a.destino} | {a.algoritmo} | salida {a.hora} | peso trafico {a.peso} | "
          f"observabilidad {a.observabilidad} | {'estatico' if a.estatico else 'dinamico'} | semilla {a.semilla}\n")
    print(f"{'politica':12s} {'plan(min)':>9s} {'real(min)':>9s} {'retraso':>8s} {'replan':>6s} "
          f"{'tramos':>6s} {'transb':>6s} {'picos':>5s} {'computo(ms)':>11s}")
    viajes = {}
    for pol in POLITICAS:
        v = simular_viaje(g, modelo, a.origen, a.destino, a.algoritmo, pol, a.observabilidad,
                          a.hora, not a.estatico, a.semilla, a.umbral)
        viajes[pol] = v
        print(f"{pol:12s} {v.tiempo_planeado_s/60:9.1f} {v.tiempo_real_s/60:9.1f} "
              f"{v.retraso_s/60:+8.1f} {v.replanificaciones:6d} {len(v.pasos):6d} "
              f"{v.transbordos:6d} {v.picos:5d} {v.computo_ms:11.0f}")
    v = viajes["umbral"]
    print("\nViaje con politica 'umbral' (R = replanifico antes de este tramo):")
    for p in v.pasos:
        print(f"  {p.paso:2d}. {'R' if p.replanifico else ' '} ruta {p.ruta_bus:6s} {p.desde[:26]:26s} -> "
              f"{p.hasta[:26]:26s} congestion real x{p.m_real:.2f} (esperada x{p.m_esperado:.2f}) "
              f"{p.dt_s:5.0f} s{'  <transbordo>' if p.transbordo else ''}")
