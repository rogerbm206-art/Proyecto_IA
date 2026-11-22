"""Dataset de aprendizaje automatico: tiempos de desplazamiento por tramo.

Cada fila agrega las observaciones del GTFS (tiempos PROGRAMADOS entre postes
consecutivos de un viaje) de un mismo tramo del grafo, ruta, hora de salida y tipo
de dia. Los postes se asignan a la parada del grafo con la misma clave que usa
`gtfs_graph` (`_parada_key`), eligiendo el nodo mas cercano si la clave se partio.

Columnas:
  u, v           paradas del tramo (ids del grafo, u < v)
  ruta           route_short_name del bus
  hora           hora de llegada al primer poste (0-23)
  tipo_dia       habil | sabado | domingo
  dist_m         distancia euclidiana del tramo en el grafo
  n_rutas        numero de rutas que recorren el tramo
  viajes_tramo   viajes totales del tramo (de `red_zonal.json`)
  lat, lon       punto medio del tramo
  grado_u/v      numero de vecinos de cada parada
  n_obs          observaciones agregadas en la fila
  tiempo_s       mediana del tiempo programado del tramo (variable objetivo)
  vel_kmh        dist_m / tiempo_s * 3.6

Depuracion (cada paso queda contado en el resumen):
  * se descartan pares de postes que no corresponden a un tramo del grafo,
  * 0 < tiempo < 900 s y dist_m >= 40 m (mismos limites que la calibracion de trafico),
  * velocidades > 70 km/h (imposibles para un bus urbano) y filas con < 3 observaciones.
"""
from __future__ import annotations

import csv
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .. import gtfs_graph as G
from ..geo import euclidean_m

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data" / "dataset_ml.csv"
RESUMEN_PATH = ROOT / "data" / "dataset_ml_resumen.json"

DT_MAX_S = 900
DIST_MIN_M = 40.0
VEL_MAX_KMH = 70.0
MIN_OBS = 3
DIAS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _seg(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _tipos_dia(cal: dict) -> set[str]:
    activos = [cal[d] == "1" for d in DIAS]
    tipos = set()
    if any(activos[:5]):
        tipos.add("habil")
    if activos[5]:
        tipos.add("sabado")
    if activos[6]:
        tipos.add("domingo")
    return tipos


def _nodo_de_poste(stop: dict, nodos_por_clave: dict, g: G.TransitGraph) -> str | None:
    cands = nodos_por_clave.get(G._parada_key(stop))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    lat, lon = float(stop["stop_lat"]), float(stop["stop_lon"])
    return min(cands, key=lambda n: euclidean_m(lat, lon, *g.coords[n]))


def construir(g: G.TransitGraph | None = None) -> tuple[pd.DataFrame, dict]:
    """Devuelve (dataset, resumen de la depuracion). Lee `stop_times.txt` (~1-2 min)."""
    g = g or G.load_zonal_graph()
    nodos_por_clave: dict[str, list[str]] = defaultdict(list)
    for n in g.nodes():
        nodos_por_clave[n.split("#")[0]].append(n)

    with zipfile.ZipFile(G.GTFS_ZIP) as z:
        def leer(nombre):
            with z.open(nombre) as fh:
                return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")))

        rutas = {r["route_id"]: r["route_short_name"] for r in leer("routes.txt")
                 if r["agency_id"] == G.ZONAL_AGENCY_ID}
        tipos_servicio = {c["service_id"]: _tipos_dia(c) for c in leer("calendar.txt")}
        viajes = {t["trip_id"]: (rutas[t["route_id"]], tipos_servicio.get(t["service_id"], set()))
                  for t in leer("trips.txt") if t["route_id"] in rutas}
        stops = {s["stop_id"]: s for s in leer("stops.txt")}
        secuencias: dict[str, list] = defaultdict(list)
        with z.open("stop_times.txt") as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")):
                if row["trip_id"] in viajes:
                    secuencias[row["trip_id"]].append(
                        (int(row["stop_sequence"]), row["stop_id"], _seg(row["arrival_time"])))

    poste_nodo: dict[str, str | None] = {}

    def nodo(sid: str) -> str | None:
        if sid not in poste_nodo:
            poste_nodo[sid] = _nodo_de_poste(stops[sid], nodos_por_clave, g)
        return poste_nodo[sid]

    resumen = {"pares_consecutivos": 0, "sin_tramo_en_grafo": 0, "tiempo_o_distancia_invalidos": 0,
               "velocidad_imposible": 0}
    muestras: dict[tuple, list[float]] = defaultdict(list)
    for tid, filas in secuencias.items():
        ruta, tipos = viajes[tid]
        filas.sort()
        for (_, a, ta), (_, b, tb) in zip(filas, filas[1:]):
            if a == b:
                continue
            resumen["pares_consecutivos"] += 1
            na, nb = nodo(a), nodo(b)
            if na is None or nb is None or na == nb or nb not in g.adj.get(na, {}):
                resumen["sin_tramo_en_grafo"] += 1
                continue
            dt, dist = tb - ta, g.cost(na, nb)
            if not (0 < dt < DT_MAX_S) or dist < DIST_MIN_M:
                resumen["tiempo_o_distancia_invalidos"] += 1
                continue
            if dist / dt * 3.6 > VEL_MAX_KMH:
                resumen["velocidad_imposible"] += 1
                continue
            u, v = (na, nb) if na < nb else (nb, na)
            for tipo in tipos:
                muestras[(u, v, ruta, (ta // 3600) % 24, tipo)].append(dt)

    filas_out = []
    for (u, v, ruta, hora, tipo), dts in muestras.items():
        if len(dts) < MIN_OBS:
            continue
        e = g.edge(u, v)
        t_s = statistics.median(dts)
        (la1, lo1), (la2, lo2) = g.coords[u], g.coords[v]
        filas_out.append({
            "u": u, "v": v, "ruta": ruta, "hora": hora, "tipo_dia": tipo,
            "dist_m": round(e["dist_m"], 1), "n_rutas": e["n_rutas"], "viajes_tramo": e["trips"],
            "lat": round((la1 + la2) / 2, 6), "lon": round((lo1 + lo2) / 2, 6),
            "grado_u": len(g.adj[u]), "grado_v": len(g.adj[v]),
            "n_obs": len(dts), "tiempo_s": round(t_s, 1),
            "vel_kmh": round(e["dist_m"] / t_s * 3.6, 2),
        })
    df = pd.DataFrame(filas_out).sort_values(["u", "v", "ruta", "tipo_dia", "hora"]).reset_index(drop=True)
    resumen.update(
        grupos_antes_de_min_obs=len(muestras), filas_finales=len(df),
        tramos=int(df[["u", "v"]].drop_duplicates().shape[0]), rutas=int(df["ruta"].nunique()),
        min_obs=MIN_OBS, dt_max_s=DT_MAX_S, dist_min_m=DIST_MIN_M, vel_max_kmh=VEL_MAX_KMH)
    return df, resumen


def guardar(df: pd.DataFrame, resumen: dict) -> None:
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    RESUMEN_PATH.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")


def cargar() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)
