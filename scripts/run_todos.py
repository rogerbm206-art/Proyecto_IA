"""Ejecuta los 5 algoritmos (bfs, dfs, ucs, greedy, astar) sobre el mismo par
origen-destino, genera una grafica por algoritmo y luego la comparacion global.

Uso:  python scripts/run_todos.py  ["Origen" "Destino"]  [--hora HH:MM] [--peso 0.5]
Salida:
  output/ruta_bfs.png ... output/ruta_astar.png
  output/comparacion_algoritmos.png  +  data/benchmark_resultados.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo import cli

if __name__ == "__main__":
    cli(None)
