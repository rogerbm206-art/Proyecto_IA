"""Modelo del grafo de la red zonal (buses azules del SITP) en codigo.

Construccion a partir del GTFS oficial (`data/GTFS-SIMUR-2026-04-29.zip`):

- Nodos  : PARADAS del componente zonal (agencia 3, "Zonal-Urbano") dentro de un
           recuadro geografico (por defecto, Chapinero). Un nodo = una parada
           REAL, no un poste: el GTFS trae un `stop_id` por cada poste (uno por
           sentido de circulacion, mas los de cada calle en las intersecciones),
           asi que se fusionan los postes que comparten `parent_station` o el
           mismo nombre normalizado. La parada guarda su nombre y las coordenadas
           (centroide de sus postes).
- Aristas: par de paradas consecutivas en algun viaje de una ruta zonal,
           observado en >= MIN_TRIPS_EDGE viajes. Atributos:
             * dist_m  : distancia euclidiana entre las dos paradas (coste del tramo)
             * rutas   : lista de rutas zonales (route_short_name) que recorren el
                         tramo -> permite modelar rutas fijas y penalizar transbordos
             * n_rutas : len(rutas)
             * trips   : nº de viajes observados en el tramo

El grafo es no dirigido y se entrega como un objeto `TransitGraph` con una API
minima (`neighbors`, `cost`, `resolve`, ...) pensada para los algoritmos de
busqueda de `src/search.py`. Tambien puede exportarse a `networkx` para dibujarlo.

El resultado se cachea en `data/red_zonal.json` para no releer el zip (~120 MB)
en cada ejecucion; borre ese archivo para forzar la reconstruccion.
"""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .geo import euclidean_m

ROOT = Path(__file__).resolve().parent.parent
GTFS_ZIP = ROOT / "data" / "GTFS-SIMUR-2026-04-29.zip"
CACHE_JSON = ROOT / "data" / "red_zonal.json"

# --- parametros del subgrafo (mismos que la grafica de la red) ---------------
BBOX = (4.6300, 4.7000, -74.0850, -74.0400)   # lat_min, lat_max, lon_min, lon_max
ZONA = "Chapinero y barrios vecinos (Teusaquillo, Chico, Usaquen sur), Bogota"
MIN_TRIPS_EDGE = 8
TARGET_NODES = 300
ZONAL_AGENCY_ID = "3"


# --- fusion de postes en paradas -------------------------------------------
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_WS = re.compile(r"\s+")


def _norm_name(name: str) -> str:
    """Normaliza un nombre de parada para agrupar postes: sin tildes, minusculas,
    sin puntuacion y con las abreviaturas mas comunes expandidas."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = _RE_WS.sub(" ", _RE_PUNCT.sub(" ", s)).strip()
    s = re.sub(r"\bc\s+c\b", "centro comercial", s)
    s = re.sub(r"\bcc\b", "centro comercial", s)
    s = re.sub(r"\bu\b", "universidad", s)
    s = re.sub(r"\bpq\b", "parque", s)
    s = re.sub(r"\bav\b", "avenida", s)
    return s


def _parada_key(stop: dict) -> str:
    """Identidad de la PARADA a la que pertenece un poste (`stop_id`).

    Si el poste declara `parent_station` (plataformas de estacion), se usa esa
    estacion; si no, el nombre normalizado.
    """
    ps = (stop.get("parent_station") or "").strip()
    return f"ST:{ps}" if ps else f"NM:{_norm_name(stop['stop_name'])}"


# Umbral para partir una parada cuyos postes esten absurdamente dispersos
# (nombre de corredor reusado, p. ej. "Av. Caracas"). Es amplio a proposito:
# los pares de sentido opuesto de una misma parada rara vez superan ~500 m.
SPLIT_THR_M = 600


def _split_by_proximity(poles, stops, thr: float) -> list[list[str]]:
    """Parte un conjunto de postes en componentes de cercania (single-linkage).

    Devuelve listas de `stop_id`, ordenadas de forma determinista.
    """
    poles = sorted(poles)
    parent = {p: p for p in poles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ll = {p: (float(stops[p]["stop_lat"]), float(stops[p]["stop_lon"])) for p in poles}
    for i, a in enumerate(poles):
        for b in poles[i + 1:]:
            if euclidean_m(*ll[a], *ll[b]) <= thr:
                parent[find(a)] = find(b)

    comps: dict[str, list[str]] = defaultdict(list)
    for p in poles:
        comps[find(p)].append(p)
    return sorted((sorted(v) for v in comps.values()), key=lambda v: v[0])


# ---------------------------------------------------------------------------
# Estructura de datos del grafo
# ---------------------------------------------------------------------------
@dataclass
class TransitGraph:
    """Grafo no dirigido de paradas. Cada nodo es una parada (id interno
    "ST:<estacion>" o "NM:<nombre normalizado>")."""

    adj: dict[str, dict[str, dict]]          # parada -> vecino -> {dist_m, rutas, n_rutas, trips}
    coords: dict[str, tuple[float, float]]   # parada -> (lat, lon) centroide
    names: dict[str, str]                    # parada -> nombre para mostrar
    meta: dict = field(default_factory=dict)

    # -- consultas basicas ------------------------------------------------
    def nodes(self) -> list[str]:
        return list(self.adj)

    def neighbors(self, node: str) -> list[str]:
        return list(self.adj[node])

    def cost(self, a: str, b: str) -> float:
        """Coste del tramo a->b (metros)."""
        return self.adj[a][b]["dist_m"]

    def edge(self, a: str, b: str) -> dict:
        return self.adj[a][b]

    def name(self, node: str) -> str:
        return self.names.get(node, node)

    def resolve(self, key: str) -> str:
        """Acepta el id interno de una parada o su nombre y devuelve el id.

        Si varias paradas comparten nombre, elige la de mayor grado.
        """
        if key in self.adj:
            return key
        cands = [sid for sid, nm in self.names.items()
                 if sid in self.adj and nm.casefold() == key.casefold()]
        if not cands:
            cands = [sid for sid, nm in self.names.items()
                     if sid in self.adj and key.casefold() in nm.casefold()]
        if not cands:
            raise KeyError(f"No hay ninguna parada que coincida con {key!r}")
        return max(cands, key=lambda s: len(self.adj[s]))

    def number_of_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    # -- interoperabilidad ---------------------------------------------------
    def to_networkx(self):
        import networkx as nx

        g = nx.Graph()
        for sid in self.adj:
            lat, lon = self.coords[sid]
            g.add_node(sid, name=self.names[sid], lat=lat, lon=lon)
        for a, nbrs in self.adj.items():
            for b, d in nbrs.items():
                if a < b:
                    g.add_edge(a, b, **d)
        return g

    def canonical(self) -> "TransitGraph":
        """Reordena nodos y vecinos por id para que el grafo (y su cache,
        y el orden de exploracion de DFS) sean deterministicos entre ejecuciones.
        """
        nodes = sorted(self.adj)
        adj = {a: {b: self.adj[a][b] for b in sorted(self.adj[a])} for a in nodes}
        coords = {a: self.coords[a] for a in nodes}
        names = {a: self.names[a] for a in nodes}
        return TransitGraph(adj, coords, names, dict(self.meta))

    # -- persistencia ------------------------------------------------------
    def to_json(self, path: Path = CACHE_JSON) -> None:
        c = self.canonical()
        edges = [
            [a, b, c.adj[a][b]]
            for a in c.adj for b in c.adj[a] if a < b
        ]
        path.write_text(json.dumps(
            {"coords": c.coords, "names": c.names, "edges": edges, "meta": c.meta},
            ensure_ascii=False, indent=0,
        ), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path = CACHE_JSON) -> "TransitGraph":
        d = json.loads(path.read_text(encoding="utf-8"))
        adj: dict[str, dict[str, dict]] = defaultdict(dict)
        for a, b, attr in d["edges"]:
            adj[a][b] = attr
            adj[b][a] = attr
        coords = {k: tuple(v) for k, v in d["coords"].items()}
        return cls(dict(adj), coords, d["names"], d.get("meta", {})).canonical()


# ---------------------------------------------------------------------------
# Construccion desde el GTFS
# ---------------------------------------------------------------------------
def _read(z: zipfile.ZipFile, name: str) -> list[dict]:
    with z.open(name) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")))


def _desambiguar_nombres(names: dict[str, str], adj: dict) -> dict[str, str]:
    """Da un nombre UNICO a cada parada.

    Cuando una parada se parte en varias (mismo nombre de barrio en puntos
    separados > SPLIT_THR_M) se añade una referencia: "Br. Pasadena (junto a X)",
    con X = vecino de distinto nombre y mas viajes. Si aun asi hay colisiones se
    numera. Determinista (todo se recorre en orden de id).
    """
    cnt = Counter(names.values())
    out = dict(names)
    for q in sorted(names):
        if cnt[names[q]] > 1:
            vecinos = sorted(adj[q], key=lambda v: (-adj[q][v]["trips"], v))
            ref = next((names[v] for v in vecinos if names[v] != names[q]), None)
            if ref:
                out[q] = f"{names[q]} (junto a {ref})"
    cnt2, vistos = Counter(out.values()), Counter()
    for q in sorted(out):
        if cnt2[out[q]] > 1:
            vistos[out[q]] += 1
            out[q] = f"{out[q]} #{vistos[out[q]]}"
    return out


@lru_cache(maxsize=1)
def _pole_data():
    """Parte pesada de la construccion (lee ~9.5 M filas de `stop_times.txt`).

    NO depende del recuadro geografico, asi que se calcula una sola vez por
    proceso y se reutiliza si se construye el grafo con otros parametros.

    Devuelve (stops, trip_count, route_set) a nivel de POSTE (`stop_id`).
    """
    with zipfile.ZipFile(GTFS_ZIP) as z:
        routes = {r["route_id"]: r for r in _read(z, "routes.txt")}
        stops = {s["stop_id"]: s for s in _read(z, "stops.txt")}
        trips = _read(z, "trips.txt")
        zonal_routes = {rid for rid, r in routes.items() if r["agency_id"] == ZONAL_AGENCY_ID}
        zonal_trips = {t["trip_id"]: t["route_id"] for t in trips if t["route_id"] in zonal_routes}

        seqs: dict[str, list[tuple[int, str]]] = defaultdict(list)
        with z.open("stop_times.txt") as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, "utf-8-sig")):
                if row["trip_id"] in zonal_trips:
                    seqs[row["trip_id"]].append((int(row["stop_sequence"]), row["stop_id"]))

    # aristas a nivel de POSTE: pares consecutivos en cada viaje
    trip_count: Counter = Counter()
    route_set: dict[tuple[str, str], set] = defaultdict(set)
    for tid, rows in seqs.items():
        sn = routes[zonal_trips[tid]]["route_short_name"]
        seq = [sid for _, sid in sorted(rows)]
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            trip_count[key] += 1
            route_set[key].add(sn)
    return stops, trip_count, route_set


def _build_from_gtfs(
    bbox=BBOX, min_trips=MIN_TRIPS_EDGE, target_nodes=TARGET_NODES,
) -> TransitGraph:
    stops, trip_count, route_set = _pole_data()

    def in_box(sid: str) -> bool:
        s = stops[sid]
        try:
            la, lo = float(s["stop_lat"]), float(s["stop_lon"])
        except ValueError:
            return False
        return bbox[0] <= la <= bbox[1] and bbox[2] <= lo <= bbox[3]

    # --- fusion de postes en PARADAS -----------------------------------
    valid_edges = [(a, b, c) for (a, b), c in trip_count.items()
                   if c >= min_trips and in_box(a) and in_box(b)]
    pole_trips: Counter = Counter()
    for a, b, c in valid_edges:
        pole_trips[a] += c
        pole_trips[b] += c

    key_poles: dict[str, set] = defaultdict(set)
    for a, b, _c in valid_edges:
        key_poles[_parada_key(stops[a])].add(a)
        key_poles[_parada_key(stops[b])].add(b)

    # Un nombre de corredor ("Av. Caracas") puede etiquetar postes muy lejanos.
    # Si los postes de una parada se dispersan mas de SPLIT_THR_M, se parten en
    # sub-paradas por cercania (single-linkage).
    p2q: dict[str, str] = {}
    for key, poles in key_poles.items():
        # las estaciones (ST:) se respetan tal cual; solo se parten las paradas
        # agrupadas por nombre (NM:)
        comps = ([sorted(poles)] if key.startswith("ST:")
                 else _split_by_proximity(poles, stops, SPLIT_THR_M))
        for k, comp in enumerate(comps):
            qid = key if len(comp) == len(poles) else f"{key}#{k}"
            for s in comp:
                p2q[s] = qid

    q_poles: dict[str, set] = defaultdict(set)
    q_edge_trips: Counter = Counter()
    q_edge_rutas: dict[tuple[str, str], set] = defaultdict(set)
    for a, b, c in valid_edges:
        qa, qb = p2q[a], p2q[b]
        q_poles[qa].add(a)
        q_poles[qb].add(b)
        if qa == qb:
            continue
        k = (qa, qb) if qa < qb else (qb, qa)
        q_edge_trips[k] += c
        q_edge_rutas[k] |= route_set[(a, b)]

    # Todo se recorre en orden de stop_id (no en el orden de hash de los `set`):
    # la suma de floats y el desempate de nombres deben ser identicos entre
    # ejecuciones para que el grafo sea reproducible bit a bit.
    q_coords, q_name = {}, {}
    for q, ps in q_poles.items():
        ps = sorted(ps)
        q_coords[q] = (sum(float(stops[s]["stop_lat"]) for s in ps) / len(ps),
                       sum(float(stops[s]["stop_lon"]) for s in ps) / len(ps))
        cnt: Counter = Counter()
        for s in ps:
            cnt[stops[s]["stop_name"].strip()] += pole_trips[s]
        q_name[q] = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    import networkx as nx

    g = nx.Graph()
    for (qa, qb), c in q_edge_trips.items():
        (la1, lo1), (la2, lo2) = q_coords[qa], q_coords[qb]
        rutas = sorted(q_edge_rutas[(qa, qb)])
        g.add_edge(qa, qb,
                   # sin redondear: si se redondeara hacia abajo, el coste de
                   # una arista podria quedar por debajo de la distancia en linea
                   # recta y romper la admisibilidad de la heuristica.
                   dist_m=euclidean_m(la1, lo1, la2, lo2),
                   rutas=rutas,               # rutas zonales que recorren el tramo
                   n_rutas=len(rutas),
                   trips=c)

    # componente conexa mayor (desempate determinista por menor stop_id)
    big = max(nx.connected_components(g), key=lambda c: (len(c), min(c)))
    g = g.subgraph(big).copy()

    # crecimiento tipo BFS ponderado por nº de viajes hasta `target_nodes`.
    # Todos los desempates son deterministas (no dependen del orden de hash):
    #   arranque   -> mayor grado, luego menor stop_id
    #   expansion  -> arista de mas viajes, luego menor stop_id del vecino
    deg = dict(g.degree())
    maxdeg = max(deg.values())
    start = min(n for n in g.nodes() if deg[n] == maxdeg)
    chosen, seen = [start], {start}
    while len(chosen) < min(target_nodes, g.number_of_nodes()):
        cands = [(-g[n][m]["trips"], m)
                 for n in chosen for m in g.neighbors(n) if m not in seen]
        if not cands:
            break
        _, best = min(cands)
        chosen.append(best)
        seen.add(best)
    h = g.subgraph(chosen).copy()
    big_h = max(nx.connected_components(h), key=lambda c: (len(c), min(c)))
    h = h.subgraph(big_h).copy()

    adj: dict[str, dict[str, dict]] = defaultdict(dict)
    coords, names = {}, {}
    for q in h.nodes():
        coords[q] = q_coords[q]
        names[q] = q_name[q]
    for a, b, d in h.edges(data=True):
        attr = {"dist_m": d["dist_m"], "rutas": list(d["rutas"]),
                "n_rutas": d["n_rutas"], "trips": d["trips"]}
        adj[a][b] = attr
        adj[b][a] = attr

    names = _desambiguar_nombres(names, adj)

    meta = {
        "fuente": "GTFS SIMUR / Transmilenio S.A. 2026-04-29",
        "componente": "zonal (buses azules, agencia 3)",
        "zona": ZONA,
        "nodos": "paradas reales (postes fusionados por parent_station / nombre)",
        "postes_fusionados": sum(len(q_poles[q]) for q in h.nodes()),
        "bbox": list(bbox),
        "min_trips_edge": min_trips,
    }
    return TransitGraph(dict(adj), coords, names, meta)


def load_zonal_graph(rebuild: bool = False) -> TransitGraph:
    """Devuelve el grafo de la red zonal, usando la cache JSON si existe."""
    if CACHE_JSON.exists() and not rebuild:
        return TransitGraph.from_json()
    g = _build_from_gtfs().canonical()
    g.to_json()
    return g


if __name__ == "__main__":
    g = load_zonal_graph(rebuild=True)
    print(f"Nodos   : {len(g.nodes())}")
    print(f"Aristas : {g.number_of_edges()}")
    grados = Counter(len(v) for v in g.adj.values())
    print(f"Grados  : {dict(sorted(grados.items()))}")
    print(f"Cache   : {CACHE_JSON}")
