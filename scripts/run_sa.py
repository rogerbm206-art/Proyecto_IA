"""Ejecuta TEMPLE SIMULADO sobre la red zonal: grafica con el camino resaltado + metricas.

Uso:  python scripts/run_sa.py  ["Origen" "Destino"]  [--hora HH:MM] [--peso 0.5]
Salida: output/ruta_sa.png  +  resumen por consola
Con --hora se activa el trafico (coste = mezcla de distancia y congestion).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo import cli

if __name__ == "__main__":
    cli("sa")
