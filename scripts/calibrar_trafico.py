"""Calibra el perfil horario de trafico a partir del GTFS (tiempos PROGRAMADOS).

Para cada hora del dia mide la velocidad mediana de los buses zonales en los tramos
del recuadro del grafo (distancia euclidiana entre postes consecutivos / diferencia de
tiempos de llegada en `stop_times.txt`). La velocidad "libre" es la mas alta entre las
horas con datos suficientes; el multiplicador de congestion de cada hora es

    m(h) = max(1, v_libre / v(h))        (1 = flujo libre; 1.9 = casi el doble de lento)

Son tiempos programados, no medidos: sirven como perfil de la ciudad, no como atasco
real por tramo.

Uso:  python scripts/calibrar_trafico.py      (~1-2 min)  ->  data/perfil_trafico.json
"""
from __future__ import annotations

import csv
import io
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import gtfs_graph as G
from src.geo import euclidean_m

OUT = Path(__file__).resolve().parent.parent / "data" / "perfil_trafico.json"
MIN_MUESTRAS_HORA = 1000   # horas con menos tramos observados se consideran flujo libre


def _seg(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def main() -> None:
    with zipfile.ZipFile(G.GTFS_ZIP) as z:
        def leer(n):
            with z.open(n) as fh:
                return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")))

        routes = {r["route_id"]: r for r in leer("routes.txt")}
        stops = {s["stop_id"]: s for s in leer("stops.txt")}
        zonales = {rid for rid, r in routes.items() if r["agency_id"] == G.ZONAL_AGENCY_ID}
        viajes = {t["trip_id"] for t in leer("trips.txt") if t["route_id"] in zonales}

        bb = G.BBOX

        def en_recuadro(sid: str) -> bool:
            s = stops[sid]
            try:
                la, lo = float(s["stop_lat"]), float(s["stop_lon"])
            except ValueError:
                return False
            return bb[0] <= la <= bb[1] and bb[2] <= lo <= bb[3]

        dentro = {sid for sid in stops if en_recuadro(sid)}
        seqs = defaultdict(list)
        with z.open("stop_times.txt") as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")):
                if row["trip_id"] in viajes:
                    seqs[row["trip_id"]].append(
                        (int(row["stop_sequence"]), row["stop_id"], _seg(row["arrival_time"])))

    vel = defaultdict(list)
    viajes_hora = defaultdict(set)
    for tid, filas in seqs.items():
        filas.sort()
        for (_, a, ta), (_, b, tb) in zip(filas, filas[1:]):
            if a != b and a in dentro and b in dentro:
                dt = tb - ta
                sa, sb = stops[a], stops[b]
                d = euclidean_m(float(sa["stop_lat"]), float(sa["stop_lon"]),
                                float(sb["stop_lat"]), float(sb["stop_lon"]))
                if 0 < dt < 900 and d > 40:
                    h = (ta // 3600) % 24
                    vel[h].append(d / dt * 3.6)
                    viajes_hora[h].add(tid)

    horas = {}
    for h in range(24):
        v = vel.get(h, [])
        horas[str(h)] = {
            "vel_mediana_kmh": round(statistics.median(v), 2) if v else None,
            "n_tramos": len(v),
            "viajes": len(viajes_hora.get(h, ())),
        }
    validas = [d["vel_mediana_kmh"] for d in horas.values()
               if d["vel_mediana_kmh"] and d["n_tramos"] >= MIN_MUESTRAS_HORA]
    v_libre = max(validas)
    mult = []
    for h in range(24):
        d = horas[str(h)]
        ok = d["vel_mediana_kmh"] and d["n_tramos"] >= MIN_MUESTRAS_HORA
        mult.append(round(max(1.0, v_libre / d["vel_mediana_kmh"]), 3) if ok else 1.0)

    OUT.write_text(json.dumps({
        "fuente": "GTFS SIMUR 2026-04-29, tiempos programados de buses zonales en el recuadro del grafo",
        "recuadro": list(bb),
        "min_muestras_hora": MIN_MUESTRAS_HORA,
        "v_libre_kmh": v_libre,
        "horas": horas,
        "multiplicador": mult,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"v_libre = {v_libre} km/h      ->  {OUT}")
    print("hora  v(km/h)  n_tramos  m(h)")
    for h in range(24):
        d = horas[str(h)]
        print(f"{h:02d}    {d['vel_mediana_kmh'] or 0:6.1f}  {d['n_tramos']:8d}  {mult[h]:5.2f}")


if __name__ == "__main__":
    main()
