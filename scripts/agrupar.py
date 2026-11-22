"""Agrupa los tramos con K-Medias segun su perfil horario de velocidad.

Uso:  python scripts/agrupar.py [--k 4]
Salidas: data/agrupamiento_tramos.csv, data/resultados_agrupamiento.json, output/agr_*.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml import agrupamiento as A
from src.ml import dataset

OUT = ROOT / "output"
DATA = ROOT / "data"
COLORES = ["#dc2626", "#f59e0b", "#22c55e", "#2563eb", "#7c3aed", "#0ea5e9", "#475569", "#db2777"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="numero de grupos (defecto 4, ver informe)")
    k = ap.parse_args().k

    df = dataset.cargar()
    P = A.perfiles(df)
    print(f"{len(P)} tramos con perfil horario (de {df[['u', 'v']].drop_duplicates().shape[0]})")

    sel = A.evaluar_k(P)
    print(sel.round(3).to_string(index=False))
    etiquetas, centros = A.ajustar(P, k)
    est = A.estabilidad(P, k)
    print(f"k={k} estabilidad (ARI): {est}")

    # atributos de cada tramo para describir los grupos
    attrs = (df.groupby(["u", "v"]).agg(dist_m=("dist_m", "first"), n_rutas=("n_rutas", "first"),
                                        viajes_tramo=("viajes_tramo", "first"), lat=("lat", "first"),
                                        lon=("lon", "first")).reindex(P.index))
    res = attrs.copy()
    res["grupo"] = etiquetas
    res["vel_media_kmh"] = P.mean(axis=1).round(2)
    res["vel_pico_kmh"] = P[[16, 17]].mean(axis=1).round(2)
    res["vel_valle_kmh"] = P[[21, 22, 23]].mean(axis=1).round(2)
    res.reset_index().to_csv(DATA / "agrupamiento_tramos.csv", index=False, encoding="utf-8")

    tabla = res.groupby("grupo").agg(
        tramos=("grupo", "size"), vel_media_kmh=("vel_media_kmh", "mean"), vel_pico_kmh=("vel_pico_kmh", "mean"),
        vel_noche_kmh=("vel_valle_kmh", "mean"), dist_m=("dist_m", "mean"), n_rutas=("n_rutas", "mean"),
        viajes_tramo=("viajes_tramo", "mean")).round(1)
    tabla["caida_pico_pct"] = (100 * (1 - tabla["vel_pico_kmh"] / tabla["vel_noche_kmh"])).round(0)
    print(tabla.to_string())

    (DATA / "resultados_agrupamiento.json").write_text(json.dumps({
        "tramos_con_perfil": len(P), "horas": A.HORAS, "k": k,
        "seleccion_de_k": sel.round(4).to_dict(orient="records"), "estabilidad": est,
        "grupos": tabla.reset_index().to_dict(orient="records"),
        "centroides_kmh": {int(g): c.round(2).tolist() for g, c in enumerate(centros)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- graficas ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(sel["k"], sel["inercia"] / 1000, marker="o", color="#2563eb")
    ax[0].axvline(k, ls="--", color="#94a3b8")
    ax[0].set(xlabel="k (número de grupos)", ylabel="inercia (miles)", title="Método del codo")
    ax[1].plot(sel["k"], sel["silueta"], marker="o", color="#0ea5e9")
    ax[1].axvline(k, ls="--", color="#94a3b8")
    ax[1].set(xlabel="k (número de grupos)", ylabel="coeficiente de silueta", title="Silueta")
    fig.tight_layout(); fig.savefig(OUT / "agr_seleccion_k.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for g in range(k):
        sub = P.values[etiquetas == g]
        ax.fill_between(A.HORAS, np.percentile(sub, 25, axis=0), np.percentile(sub, 75, axis=0),
                        color=COLORES[g], alpha=.15)
        ax.plot(A.HORAS, centros[g], marker="o", ms=3, color=COLORES[g],
                label=f"grupo {g + 1} ({len(sub)} tramos)")
    ax.set(xlabel="hora del día", ylabel="velocidad (km/h)", xticks=range(5, 24, 2),
           title="Perfil horario de cada grupo (línea = centroide, banda = rango intercuartil)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "agr_perfiles.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    for g in range(k):
        s = res[res.grupo == g]
        ax.scatter(s["lon"], s["lat"], s=18, color=COLORES[g], label=f"grupo {g + 1}")
    ax.set(xlabel="longitud", ylabel="latitud", title="Grupos de tramos en el mapa")
    ax.set_aspect(1 / np.cos(np.radians(4.66)))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "agr_mapa.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, col, tit in zip(ax, ["dist_m", "n_rutas", "viajes_tramo"],
                           ["Distancia del tramo (m)", "Rutas por tramo", "Viajes por tramo"]):
        a.bar([f"{g + 1}" for g in range(k)], res.groupby("grupo")[col].mean(), color=COLORES[:k])
        a.set(xlabel="grupo", title=tit)
    fig.tight_layout(); fig.savefig(OUT / "agr_atributos.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
