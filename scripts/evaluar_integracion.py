"""Evalua si planificar con el tiempo predicho por ML produce viajes mas rapidos.

Para pares origen-destino al azar y varias horas de un dia habil, el agente (A*) planifica con
cuatro criterios de coste y se mide cada ruta con el TIEMPO PROGRAMADO del GTFS (el dato de que se
dispone, no un tiempo medido en la calle):

  distancia : solo distancia + 800 m por transbordo (modelo original, sin trafico),
  perfil    : perfil horario del GTFS por factor espacial sintetico (peso 1),
  ml        : tiempo predicho por la regresion lineal (peso 1),
  oraculo   : el propio tiempo programado como coste (cota superior de lo que se puede lograr).

Para que la comparacion sea justa, el criterio `ml` usa predicciones FUERA DE MUESTRA: validacion
cruzada agrupada por tramo, de modo que el modelo que puntua un tramo nunca lo vio al entrenarse.
Solo se comparan pares y horas en que las cuatro rutas tienen tiempo programado en todos sus tramos.

Uso:  python scripts/evaluar_integracion.py [-n 150] [--semilla 42]     (varios minutos)
Salidas: data/resultados_integracion.json, output/int_criterios.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gtfs_graph import load_zonal_graph
from src.ml import dataset, integracion, regresion
from src.search import TRANSFER_PENALTY_M, search
from src.traffic import ModeloTrafico, PerfilHorario

HORAS = (8, 12, 17, 21)
CRITERIOS = ("distancia", "perfil", "ml", "oraculo")


def ic95(x: np.ndarray, repeticiones: int = 5000) -> list[float]:
    """Intervalo de confianza del 95 % de la media (bootstrap, semilla fija)."""
    rng = np.random.default_rng(0)
    medias = [rng.choice(x, len(x)).mean() for _ in range(repeticiones)]
    return [round(float(v), 2) for v in np.percentile(medias, [2.5, 97.5])]


def tiempos_fuera_de_muestra(df: pd.DataFrame) -> dict:
    """{hora: {(u, v): segundos}} de la regresion lineal con validacion cruzada por tramo (dia habil)."""
    X, y, grupos = regresion.preparar(df)
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grupos):
        m = regresion.modelos()["Regresión lineal"][0].fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    d = df[["u", "v", "hora", "tipo_dia"]].assign(p=np.maximum(1.0, oof))
    d = d[d.tipo_dia == "habil"].groupby(["hora", "u", "v"])["p"].mean()
    out: dict = {}
    for (h, u, v), p in d.items():
        out.setdefault(int(h), {})[(u, v)] = float(p)
    return out


def tiempos_programados(df: pd.DataFrame):
    """(por tramo y ruta, por tramo): tiempo programado mediano de un dia habil."""
    d = df[df.tipo_dia == "habil"]
    por_ruta = d.set_index(["u", "v", "ruta", "hora"])["tiempo_s"].to_dict()
    por_tramo = d.groupby(["u", "v", "hora"])["tiempo_s"].mean().to_dict()
    return por_ruta, por_tramo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=150, help="pares origen-destino")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    g = load_zonal_graph()
    df = dataset.cargar()
    perfil = PerfilHorario.cargar()
    v_libre_kmh, v_libre_ms = perfil.v_libre_kmh, perfil.v_libre_ms
    ml_t = tiempos_fuera_de_muestra(df)
    por_ruta, por_tramo = tiempos_programados(df)
    modelo_perfil = ModeloTrafico(g, semilla=0)
    pen_s = TRANSFER_PENALTY_M / v_libre_ms     # espera de un transbordo, en s

    def tramo_hora(k, h):
        """Tiempo programado del tramo a la hora h; si falta, el de la hora vecina mas cercana (+-2 h)."""
        for dh in (0, -1, 1, -2, 2):
            t = por_tramo.get((k[0], k[1], h + dh))
            if t is not None:
                return t
        return None

    def real(u, v, ruta, h):
        k = (u, v) if u < v else (v, u)
        t = por_ruta.get((k[0], k[1], ruta, h))
        return t if t is not None else tramo_hora(k, h)

    rng = random.Random(a.semilla)
    nodos = sorted(g.nodes())
    pares = []
    while len(pares) < a.n:
        o, d = rng.sample(nodos, 2)
        pares.append((o, d))

    filas = []
    for h in HORAS:
        oraculo = {k: (tramo_hora(k, h) or ml_t[h][k]) for k in ml_t[h]}
        mults = {
            "distancia": None,
            "perfil": modelo_perfil.mult_esperado(h * 3600.0),
            "ml": integracion.mult_desde_tiempos(g, ml_t[h], v_libre_kmh),
            "oraculo": integracion.mult_desde_tiempos(g, oraculo, v_libre_kmh),
        }
        for i, (o, d) in enumerate(pares):
            res, tiempos, ok = {}, {}, True
            for c in CRITERIOS:
                kw = {} if c == "distancia" else {"mult": mults[c], "peso_trafico": 1.0, "v_libre_ms": v_libre_ms}
                r = search(g, o, d, "astar", **kw)
                ts = [real(x, y, rt, h) for (x, y), rt in zip(zip(r.path, r.path[1:]), r.path_routes)]
                if not r.found or any(t is None for t in ts):
                    ok = False
                    break
                res[c], tiempos[c] = r, sum(ts)
            if not ok:
                continue
            for c in CRITERIOS:
                filas.append({"hora": h, "par": i, "criterio": c, "tiempo_s": tiempos[c],
                              "transbordos": res[c].transbordos, "tramos": res[c].path_edges,
                              "dist_m": res[c].path_dist_m,
                              "tiempo_total_s": tiempos[c] + res[c].transbordos * pen_s})
        print(f"hora {h:02d}: {sum(1 for f in filas if f['hora'] == h and f['criterio'] == 'ml')} pares comparables")

    t = pd.DataFrame(filas)
    t.to_csv(ROOT / "data" / "resultados_integracion.csv", index=False, encoding="utf-8")
    total = t.groupby("criterio").agg(pares=("par", "size"), tiempo_min=("tiempo_s", lambda s: s.mean() / 60),
                                      transbordos=("transbordos", "mean"), tramos=("tramos", "mean"),
                                      dist_m=("dist_m", "mean"),
                                      tiempo_total_min=("tiempo_total_s", lambda s: s.mean() / 60)).reindex(CRITERIOS)
    print(total.round(2).to_string())

    ancho = t.pivot_table(index=["hora", "par"], columns="criterio", values="tiempo_s")
    ancho_tot = t.pivot_table(index=["hora", "par"], columns="criterio", values="tiempo_total_s")
    comp = {}
    for c in ("perfil", "ml", "oraculo"):
        gan = 1 - ancho[c] / ancho["distancia"]
        gan_tot = 1 - ancho_tot[c] / ancho_tot["distancia"]
        comp[c] = {"ahorro_medio_pct": round(float(gan.mean() * 100), 2),
                   "ahorro_medio_ic95": ic95(gan.to_numpy() * 100),
                   "ahorro_total_medio_pct": round(float(gan_tot.mean() * 100), 2),
                   "ahorro_total_ic95": ic95(gan_tot.to_numpy() * 100),
                   "pares_mas_rapidos_pct": round(float((ancho[c] < ancho["distancia"] - 1e-6).mean() * 100), 1),
                   "pares_mas_lentos_pct": round(float((ancho[c] > ancho["distancia"] + 1e-6).mean() * 100), 1),
                   "pares_igual_pct": round(float((abs(ancho[c] - ancho["distancia"]) <= 1e-6).mean() * 100), 1)}
    horas_ok = [h for h in HORAS if h in ancho.index.get_level_values("hora")]
    por_hora = {int(h): {c: round(float(ancho.loc[h][c].mean() / 60), 2) for c in CRITERIOS} for h in horas_ok}
    print(json.dumps(comp, indent=1))

    (ROOT / "data" / "resultados_integracion.json").write_text(json.dumps({
        "pares_aleatorios": a.n, "semilla": a.semilla, "horas": list(HORAS), "algoritmo": "astar",
        "comparaciones": int(len(ancho)), "espera_transbordo_s": round(pen_s, 1),
        "por_criterio": total.round(3).reset_index().to_dict(orient="records"),
        "frente_a_distancia": comp, "tiempo_medio_min_por_hora": por_hora,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    por_hora_tot = {h: {c: float(ancho_tot.loc[h][c].mean() / 60) for c in CRITERIOS} for h in horas_ok}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.2
    colores = {"distancia": "#94a3b8", "perfil": "#f59e0b", "ml": "#2563eb", "oraculo": "#22c55e"}
    for j, c in enumerate(CRITERIOS):
        ax.bar(np.arange(len(horas_ok)) + (j - 1.5) * w, [por_hora_tot[h][c] for h in horas_ok], w,
               label=c, color=colores[c])
    ax.set_xticks(range(len(horas_ok)), [f"{h:02d} h" for h in horas_ok])
    ax.set(ylabel="tiempo de la ruta con esperas (min)", title="Tiempo de la ruta según el criterio de coste")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(ROOT / "output" / "int_criterios.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
