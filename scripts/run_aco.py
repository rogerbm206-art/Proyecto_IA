"""Ejecuta COLONIA DE HORMIGAS (ACO) sobre la red zonal: grafica con el camino resaltado + metricas.

Uso:  python scripts/run_aco.py  ["Origen" "Destino"]  [--hora HH:MM] [--peso 0.5]
Salida: output/ruta_aco.png  +  resumen por consola
Con --hora se activa el trafico (coste = mezcla de distancia y congestion).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo import cli

if __name__ == "__main__":
    cli("aco")
