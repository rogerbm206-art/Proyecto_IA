"""Analisis exploratorio del dataset de ML -> output/eda_*.png y data/eda_resumen.json.

Uso:  python scripts/eda.py     (requiere data/dataset_ml.csv)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml import dataset

OUT = ROOT / "output"


def _guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / nombre, dpi=130)
    plt.close(fig)


def main() -> None:
    df = dataset.cargar()
    resumen = {"filas": len(df), "tramos": int(df[["u", "v"]].drop_duplicates().shape[0]),
               "rutas": int(df["ruta"].nunique())}

    # 1. distribucion de la velocidad y del tiempo
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df["vel_kmh"], bins=60, color="#2563eb")
    ax[0].set(title="Distribución de la velocidad", xlabel="velocidad (km/h)", ylabel="filas")
    ax[1].hist(df["tiempo_s"], bins=60, color="#0ea5e9")
    ax[1].set(title="Distribución del tiempo por tramo", xlabel="tiempo (s)", ylabel="filas")
    _guardar(fig, "eda_distribuciones.png")
    resumen["vel_kmh"] = df["vel_kmh"].describe().round(2).to_dict()
    resumen["tiempo_s"] = df["tiempo_s"].describe().round(2).to_dict()

    # 2. velocidad por hora y tipo de dia
    fig, ax = plt.subplots(figsize=(8, 4))
    for tipo, color in (("habil", "#2563eb"), ("sabado", "#f59e0b"), ("domingo", "#dc2626")):
        g = df[df.tipo_dia == tipo].groupby("hora")["vel_kmh"]
        med = g.median()
        ax.plot(med.index, med.values, marker="o", ms=3, label=tipo, color=color)
        if tipo == "habil":
            ax.fill_between(med.index, g.quantile(.25), g.quantile(.75), alpha=.15, color=color)
    ax.set(title="Velocidad mediana por hora (banda = rango intercuartil, día hábil)",
           xlabel="hora del día", ylabel="velocidad (km/h)")
    ax.legend()
    _guardar(fig, "eda_velocidad_hora.png")
    resumen["vel_mediana_por_hora_habil"] = (
        df[df.tipo_dia == "habil"].groupby("hora")["vel_kmh"].median().round(2).to_dict())
    resumen["vel_mediana_por_tipo_dia"] = df.groupby("tipo_dia")["vel_kmh"].median().round(2).to_dict()

    # 3. tiempo vs distancia
    muestra = df.sample(min(len(df), 15000), random_state=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(muestra["dist_m"], muestra["tiempo_s"], s=3, alpha=.25, color="#2563eb")
    ax.set(title="Tiempo programado vs distancia del tramo", xlabel="distancia (m)", ylabel="tiempo (s)")
    _guardar(fig, "eda_tiempo_distancia.png")

    # 4. correlaciones
    cols = ["hora", "dist_m", "n_rutas", "viajes_tramo", "lat", "lon", "grado_u", "grado_v",
            "n_obs", "tiempo_s", "vel_kmh"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("Correlación de Pearson")
    _guardar(fig, "eda_correlaciones.png")
    resumen["correlacion_con_tiempo_s"] = corr["tiempo_s"].round(3).to_dict()
    resumen["correlacion_con_vel_kmh"] = corr["vel_kmh"].round(3).to_dict()

    # 5. mapa de velocidad mediana por tramo (hora pico de la tarde, dia habil)
    pico = df[(df.tipo_dia == "habil") & (df.hora.between(16, 17))]
    por_tramo = pico.groupby(["u", "v"]).agg(vel=("vel_kmh", "median"), lat=("lat", "first"),
                                            lon=("lon", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(por_tramo["lon"], por_tramo["lat"], c=por_tramo["vel"], s=16, cmap="RdYlGn")
    fig.colorbar(sc, ax=ax, shrink=.8, label="velocidad mediana (km/h)")
    ax.set(title="Velocidad por tramo, día hábil 16-17 h", xlabel="longitud", ylabel="latitud")
    ax.set_aspect(1 / np.cos(np.radians(4.66)))
    _guardar(fig, "eda_mapa_velocidad.png")

    # 6. velocidad por numero de rutas
    df["rutas_grupo"] = df["n_rutas"].clip(upper=20)
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column="vel_kmh", by="rutas_grupo", ax=ax, showfliers=False)
    ax.set(title="Velocidad según número de rutas del tramo", xlabel="rutas en el tramo (20 = 20 o más)",
           ylabel="velocidad (km/h)")
    fig.suptitle("")
    _guardar(fig, "eda_velocidad_rutas.png")

    (ROOT / "data" / "eda_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("filas", "tramos", "rutas", "vel_mediana_por_tipo_dia")},
                     ensure_ascii=False, indent=1))
    print("vel_mediana_por_hora_habil:", resumen["vel_mediana_por_hora_habil"])
    print("corr con tiempo_s:", resumen["correlacion_con_tiempo_s"])


if __name__ == "__main__":
    main()
