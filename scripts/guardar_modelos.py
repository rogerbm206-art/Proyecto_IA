"""Entrena y guarda todos los modelos que usa la web (~2-3 min).

Uso:  python scripts/guardar_modelos.py      ->  data/modelos_ml.joblib
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml import catalogo, dataset


def main() -> None:
    t0 = time.time()
    cat = catalogo.entrenar(dataset.cargar())
    catalogo.guardar(cat)
    print("regresión:", ", ".join(cat["regresion"]))
    print("clasificación:", ", ".join(cat["clasificacion"]))
    print(f"-> {catalogo.CATALOGO} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
