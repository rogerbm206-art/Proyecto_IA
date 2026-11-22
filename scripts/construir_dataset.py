"""Construye el dataset de aprendizaje automatico desde el GTFS (~1-2 min).

Uso:  python scripts/construir_dataset.py   ->  data/dataset_ml.csv + data/dataset_ml_resumen.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml import dataset


def main() -> None:
    df, resumen = dataset.construir()
    dataset.guardar(df, resumen)
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print(df.describe().T[["count", "mean", "min", "max"]].round(2))
    print(f"-> {dataset.CSV_PATH}")


if __name__ == "__main__":
    main()
