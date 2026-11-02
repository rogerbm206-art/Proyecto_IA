"""Parseo del GTFS del SITP / TransMilenio (feed oficial de SIMUR - Transmilenio S.A.).

Fuente del dataset
-----------------
Catalogo  : Mobility Database, feed `mdb-3358` ("SIMUR Bogota", Official Feed).
            https://mobilitydatabase.org/feeds/mdb-3358
Productor : Transmilenio S.A.  (feed_info.feed_publisher_name)
Descarga  : https://storage.googleapis.com/gtfs-estaticos/GTFS-2026-04-29.zip
Licencia  : http://www.simur.gov.co/gtfs
Vigencia  : 2026-01-01 .. 2026-12-31
Local     : data/GTFS-SIMUR-2026-04-29.zip

A diferencia del feed GTFS-2025-07-09 (solo capa zonal), este feed SI incluye el
componente troncal: 109 rutas de agencia "Transmilenio-Troncal", 26 duales y 152
estaciones/portales con `location_type=1`, nombre propio y coordenadas reales.

Uso:  python scripts/parse_gtfs.py
Genera en data/:
  - stations_troncal.csv   (152 estaciones troncales: id, nombre, lat, lon)
  - routes_troncal.csv     (109 rutas troncales)
  - resumen impreso por consola
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GTFS_ZIP = ROOT / "data" / "GTFS-SIMUR-2026-04-29.zip"
OUT_DIR = ROOT / "data"


def read_table(z: zipfile.ZipFile, name: str) -> list[dict]:
    with z.open(name) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")))


def main() -> None:
    if not GTFS_ZIP.exists():
        raise SystemExit(
            f"No se encontro {GTFS_ZIP}.\n"
            "Descargue el feed con:\n"
            "  curl -L -o data/GTFS-SIMUR-2026-04-29.zip "
            "https://storage.googleapis.com/gtfs-estaticos/GTFS-2026-04-29.zip"
        )

    with zipfile.ZipFile(GTFS_ZIP) as z:
        agency = read_table(z, "agency.txt")
        routes = read_table(z, "routes.txt")
        stops = read_table(z, "stops.txt")
        trips = read_table(z, "trips.txt")
        feed_info = read_table(z, "feed_info.txt")
        with z.open("stop_times.txt") as fh:
            n_stop_times = sum(1 for _ in fh) - 1

    ag_name = {a["agency_id"]: a["agency_name"] for a in agency}
    routes_by_agency = Counter(r["agency_id"] for r in routes)
    route_types = Counter(r["route_type"] for r in routes)

    stations = [s for s in stops if s.get("location_type") == "1"]
    platforms = [s for s in stops if s.get("location_type") == "0"]

    # --- resumen -----------------------------------------------------------
    print("=" * 60)
    print("GTFS SITP / TransMilenio  -  feed SIMUR (Transmilenio S.A.)")
    print("=" * 60)
    for k, v in (feed_info[0] if feed_info else {}).items():
        print(f"  {k:22s}: {v}")
    print("\n  Rutas por agencia:")
    for aid, name in ag_name.items():
        print(f"    [{aid}] {name:28s} {routes_by_agency.get(aid, 0):5d}")
    print(f"    {'TOTAL':32s} {len(routes):5d}")
    print(f"\n  route_type: {dict(route_types)}  (3=bus, 6=cable)")
    print(f"  Paradas totales           : {len(stops)}")
    print(f"    estaciones (loc_type=1)  : {len(stations)}")
    print(f"    plataformas (loc_type=0) : {len(platforms)}")
    print(f"  Viajes (trips)            : {len(trips)}")
    print(f"  Registros stop_times      : {n_stop_times}")

    # --- CSV de estaciones troncales -------------------------------------
    st_csv = OUT_DIR / "stations_troncal.csv"
    with open(st_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stop_id", "stop_name", "stop_lat", "stop_lon"])
        for s in sorted(stations, key=lambda s: s["stop_name"].strip().lower()):
            w.writerow([s["stop_id"], s["stop_name"].strip(), s["stop_lat"], s["stop_lon"]])
    print(f"\n  -> {st_csv.relative_to(ROOT)} ({len(stations)} filas)")

    # --- CSV de rutas troncales ----------------------------------------
    troncal = [r for r in routes if r["agency_id"] == "1"]
    rt_csv = OUT_DIR / "routes_troncal.csv"
    with open(rt_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["route_id", "route_short_name", "route_long_name"])
        for r in sorted(troncal, key=lambda r: r["route_short_name"]):
            w.writerow([r["route_id"], r["route_short_name"], r["route_long_name"]])
    print(f"  -> {rt_csv.relative_to(ROOT)} ({len(troncal)} filas)")


if __name__ == "__main__":
    main()
