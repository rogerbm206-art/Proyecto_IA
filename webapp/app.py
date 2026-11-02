"""Interfaz web para probar los algoritmos de busqueda sobre la red zonal.

Eliges origen, destino, algoritmo y (opcional) la hora de salida con el peso del
trafico en el coste. La pagina muestra:
  - las medidas del algoritmo elegido (las mismas columnas del benchmark); se
    ejecuta SOLO ese algoritmo,
  - con «Ejecutar todos», el BENCHMARK de todos los algoritmos sobre ese mismo par
    (con el mejor de cada medida resaltado y el desempeño respecto al optimo),
  - la ruta dibujada sobre el grafo (tramos coloreados por congestion) y el ciclo
    percepcion -> accion del agente,
  - opcionalmente, una SIMULACION con trafico que cambia mientras el agente avanza
    y compara las politicas de replanificacion.

    pip install flask
    python webapp/app.py       # abre http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
from flask import Flask, Response, abort, request, send_file

from src.agent import RoutePlannerAgent
from src.gtfs_graph import load_zonal_graph
from src.plot_search import plot_result
from src.search import STRATEGIES, SearchResult, search
from src.simulacion import POLITICAS, ViajeSimulado, simular_viaje
from src.traffic import ConfigTrafico, ModeloTrafico, fmt_hora, parse_hora

G = load_zonal_graph()
NOMBRES = sorted(G.names.values(), key=str.casefold)
ALGOS = {  # clave -> (nombre largo, nombre corto)
    "bfs": ("BFS · anchura", "BFS"),
    "dfs": ("DFS · profundidad", "DFS"),
    "ucs": ("UCS · costo uniforme", "UCS"),
    "greedy": ("Voraz · greedy", "Voraz"),
    "astar": ("A*", "A*"),
    "hc": ("Ascenso de colinas", "Colinas"),
    "sa": ("Temple simulado", "Temple"),
    "ga": ("Genético", "Genético"),
    "aco": ("Hormigas · ACO", "ACO"),
    "pso": ("Partículas · PSO", "PSO"),
}
POLITICAS_TXT = {
    "nunca": "Sin replanificar",
    "umbral": "Replanificar si el tramo empeora > 25 %",
    "cada_parada": "Replanificar en cada parada",
}
EJEMPLOS = [  # pares que cruzan barrios, para probar con un clic
    ("Br. Chicó Norte III Sector", "Br. Sucre"),
    ("Estación Calle 100", "Br. Marly"),
    ("U. Sergio Arboleda", "Br. Galerías"),
]
HORAS_RAPIDAS = [("Madrugada", "03:00"), ("Pico mañana", "08:00"),
                 ("Mediodía", "12:00"), ("Pico tarde", "17:00"), ("Noche", "21:00")]

app = Flask(__name__)
_LOCK = threading.Lock()   # matplotlib y tracemalloc no son thread-safe


@dataclass(frozen=True)
class Params:
    origen: str = ""
    destino: str = ""
    alg: str = "astar"
    hora: str = "08:00"
    peso: float = 0.5
    sim: bool = False
    semilla: int = 1
    obs: str = "global"   # el agente siempre ve el trafico de toda la red
    todos: bool = False   # True = ejecutar los algoritmos y armar la tabla comparativa

    @property
    def t_min(self) -> int:
        return int(parse_hora(self.hora) // 60)


def _params() -> Params:
    a = request.args
    alg = a.get("alg", "astar")
    hora = a.get("hora", "08:00")
    try:
        parse_hora(hora)
    except (ValueError, IndexError):
        hora = "08:00"
    try:
        peso = round(min(1.0, max(0.0, float(a.get("peso", 0.5)))), 2)
    except ValueError:
        peso = 0.5
    try:
        semilla = int(a.get("semilla", 1))
    except ValueError:
        semilla = 1
    return Params((a.get("origen") or "").strip(), (a.get("destino") or "").strip(),
                  alg if alg in ALGOS else "astar", hora, peso, a.get("sim") == "1",
                  semilla, todos=a.get("todos") == "1")


# ---------------------------------------------------------------------------
# Calculo (con cache: la pagina y la imagen comparten exactamente el mismo resultado)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _modelo(peso: float) -> ModeloTrafico:
    return ModeloTrafico(G, cfg=ConfigTrafico(peso=peso), semilla=0)


@lru_cache(maxsize=64)
def _mult(t_min: int, peso: float) -> dict | None:
    """Snapshot del trafico tipico a esa hora (None si el trafico esta desactivado)."""
    return _modelo(peso).mult_esperado(t_min * 60.0) if peso else None


@lru_cache(maxsize=512)
def _buscar(oid: str, did: str, alg: str, t_min: int, peso: float) -> SearchResult:
    with _LOCK:
        return search(G, oid, did, alg, mult=_mult(t_min, peso), peso_trafico=peso,
                      v_libre_ms=_modelo(peso).v_libre_ms)


def _txt_trafico(t_min: int, peso: float) -> str | None:
    return f"{fmt_hora(t_min * 60)}, peso {peso:g}" if peso else None


@lru_cache(maxsize=64)
def _png(oid: str, did: str, alg: str, t_min: int, peso: float) -> bytes:
    res = _buscar(oid, did, alg, t_min, peso)
    with _LOCK:
        tmp = Path(tempfile.gettempdir()) / f"ruta_web_{alg}_{abs(hash((oid, did, t_min, peso)))}.png"
        try:
            plot_result(G, res, out_path=tmp, origen=G.name(oid), destino=G.name(did),
                        mult=_mult(t_min, peso), trafico_txt=_txt_trafico(t_min, peso))
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)


def _benchmark(oid: str, did: str, t_min: int, peso: float):
    res = {a: _buscar(oid, did, a, t_min, peso) for a in ALGOS}
    opt = res["ucs"].path_cost_m   # UCS es optimo en coste (con el mismo trafico)
    des = {a: (100.0 * opt / r.path_cost_m if r.found and r.path_cost_m else 0.0)
           for a, r in res.items()}
    return res, des


@lru_cache(maxsize=32)
def _simular(oid: str, did: str, alg: str, t_min: int, peso: float, semilla: int,
             obs: str) -> dict[str, ViajeSimulado]:
    with _LOCK:   # simular_viaje usa search() (tracemalloc): una a la vez
        modelo = _modelo(peso)
        return {pol: simular_viaje(G, modelo, oid, did, alg, pol, obs, t_min * 60.0,
                                   True, semilla) for pol in POLITICAS}


# ---------------------------------------------------------------------------
# Estilos y script
# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#f4f6fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;
  --brand:#2563eb;--brand2:#1d4ed8;--chip:#eff6ff;--ok:#15803d;--okbg:#dcfce7;
  --warn:#b45309;--warnbg:#fef3c7;--bad:#b91c1c;--badbg:#fee2e2;--r:14px}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 -apple-system,"Segoe UI",system-ui,Roboto,sans-serif}
header{background:linear-gradient(120deg,#1e3a8a,#2563eb 60%,#0ea5e9);color:#fff;
  padding:26px 20px 70px}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header h1{margin:0;font-size:24px;letter-spacing:-.01em}
header p{margin:6px 0 0;opacity:.85;font-size:14px}
main{margin-top:-48px;padding-bottom:40px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:0 1px 2px rgba(15,23,42,.04),0 8px 24px rgba(15,23,42,.06);padding:20px;margin-bottom:20px}
.card h2{margin:0 0 4px;font-size:17px}
.card .sub{color:var(--muted);font-size:13px;margin:0 0 14px}
form .row{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:end}
label.f{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600;
  color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
input[type=text],input[type=time],input[type=number],select.f{font:inherit;font-size:15px;font-weight:400;
  text-transform:none;letter-spacing:0;padding:11px 12px;border:1px solid #cbd5e1;border-radius:10px;
  background:#fff;color:var(--ink);width:100%}
input[type=text]:focus,input[type=time]:focus,input[type=number]:focus,select.f:focus{
  outline:2px solid #93c5fd;border-color:var(--brand)}
input[type=range]{width:100%;accent-color:var(--brand);margin:8px 0 4px}
.row2{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px 14px;
  margin-top:16px;align-items:end;padding-top:16px;border-top:1px dashed var(--line)}
label.chk{display:flex;gap:8px;align-items:center;font-size:14px;font-weight:500;color:var(--ink);
  text-transform:none;letter-spacing:0;padding-bottom:11px;cursor:pointer}
.hrs{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:10px;font-size:12.5px;color:var(--muted)}
.hrs button{font:inherit;font-size:12.5px;padding:3px 11px;border-radius:999px;border:1px solid #cbd5e1;
  background:#fff;cursor:pointer;color:var(--ink)}
.hrs button:hover{background:var(--chip);border-color:var(--brand)}
.swap{border:1px solid #cbd5e1;background:#fff;border-radius:10px;width:42px;height:42px;
  cursor:pointer;font-size:18px;color:var(--brand)}
.swap:hover{background:var(--chip)}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px}
.pills input{position:absolute;opacity:0;pointer-events:none}
.pills label{padding:8px 14px;border:1px solid #cbd5e1;border-radius:999px;cursor:pointer;
  font-size:14px;background:#fff;transition:.12s}
.pills label:hover{border-color:var(--brand)}
.pills input:checked+label{background:var(--brand);color:#fff;border-color:var(--brand);font-weight:600}
.pills input:focus-visible+label{outline:2px solid #93c5fd}
.actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px}
button.go{font:inherit;font-weight:600;background:var(--brand);color:#fff;border:0;border-radius:10px;
  padding:11px 22px;cursor:pointer}
button.go:hover{background:var(--brand2)} button.go:disabled{opacity:.6;cursor:wait}
button.alt{font:inherit;background:#fff;color:var(--brand);border:1px solid #bfdbfe;border-radius:10px;
  padding:10px 16px;cursor:pointer}
button.alt:hover{background:var(--chip)}
.ej{margin-top:12px;font-size:13px;color:var(--muted)}
.ej a{display:inline-block;margin:4px 6px 0 0;padding:4px 10px;border-radius:999px;
  background:var(--chip);color:var(--brand2);text-decoration:none;font-size:12.5px}
.ej a:hover{background:#dbeafe}
.err{background:var(--badbg);color:var(--bad);border:1px solid #fecaca;padding:12px 16px;
  border-radius:10px;font-weight:500;margin-bottom:20px}
.head{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin-bottom:14px}
.head h2{font-size:19px;margin:0}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12.5px;font-weight:600}
.b-alg{background:var(--chip);color:var(--brand2)}
.b-ok{background:var(--okbg);color:var(--ok)} .b-warn{background:var(--warnbg);color:var(--warn)}
.b-bad{background:var(--badbg);color:var(--bad)} .b-traf{background:#f1f5f9;color:#334155}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.kpi{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:#fbfdff}
.kpi .v{font-size:23px;font-weight:700;letter-spacing:-.02em;line-height:1.15}
.kpi .v small{font-size:12px;font-weight:500;color:var(--muted);margin-left:3px}
.kpi .l{font-size:12px;color:var(--muted);margin-top:3px}
.kpi.ok .v{color:var(--ok)} .kpi.warn .v{color:var(--warn)} .kpi.bad .v{color:var(--bad)}
.tablewrap{overflow-x:auto}
table.bm{border-collapse:collapse;width:100%;font-size:13.5px;min-width:820px}
table.bm th,table.bm td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
table.bm th:first-child,table.bm td:first-child{text-align:left}
table.bm thead th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
  background:#f8fafc;position:sticky;top:0}
table.bm tr.sel td{background:#eff6ff;font-weight:600}
table.bm td.best{color:var(--ok);font-weight:700}
table.bm td.best::after{content:" ●";font-size:8px;vertical-align:middle}
.bar{display:inline-block;vertical-align:middle;width:54px;height:7px;background:#e2e8f0;
  border-radius:99px;margin-right:7px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--brand)}
.note{color:var(--muted);font-size:12.5px;margin:10px 0 0}
img.plot{width:100%;border:1px solid var(--line);border-radius:12px;display:block}
.route{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px;margin-bottom:6px}
.route span{background:#f1f5f9;border-radius:6px;padding:2px 8px}
.route em{font-style:normal;color:var(--muted)}
ol.steps{list-style:none;margin:0;padding:0;max-height:380px;overflow:auto;
  border:1px solid var(--line);border-radius:12px}
ol.steps li{display:flex;gap:10px;align-items:baseline;padding:8px 14px;border-bottom:1px solid var(--line);
  font-size:13.5px}
ol.steps li:last-child{border-bottom:0}
ol.steps li.tr{background:#fffbeb}
ol.steps li.rp{background:#eff6ff}
ol.steps .n{width:28px;color:var(--muted);text-align:right;flex:none}
ol.steps .bus{font-weight:700;color:var(--brand2);font-family:ui-monospace,Consolas,monospace}
ol.steps .d{margin-left:auto;color:var(--muted);flex:none;text-align:right}
.tag{font-size:11px;font-weight:700;padding:1px 7px;border-radius:99px;flex:none}
.t-go{background:#dbeafe;color:#1e40af} .t-keep{background:#f1f5f9;color:#475569}
.t-tr{background:#fde68a;color:#92400e} .t-rp{background:#c7d2fe;color:#3730a3}
details{margin-top:14px} summary{cursor:pointer;color:var(--brand2);font-weight:600;font-size:13.5px}
details[open] summary{margin-bottom:10px}
.gwrap{border:1px solid var(--line);border-radius:12px;overflow:hidden}
#mapa{height:72vh;min-height:420px;width:100%}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:6px 0 30px}
@media(max-width:720px){form .row{grid-template-columns:1fr}.swap{width:100%}}
"""

_JS = """
const NOMBRES = __NOMBRES__;
const $ = s => document.querySelector(s);
$('#swap').addEventListener('click', () => {
  const a = $('#origen'), b = $('#destino'); [a.value, b.value] = [b.value, a.value];
});
$('#azar').addEventListener('click', () => {
  const i = Math.floor(Math.random() * NOMBRES.length);
  let j = Math.floor(Math.random() * NOMBRES.length);
  while (j === i) j = Math.floor(Math.random() * NOMBRES.length);
  $('#origen').value = NOMBRES[i]; $('#destino').value = NOMBRES[j];
});
document.querySelectorAll('[data-h]').forEach(b =>
  b.addEventListener('click', () => { $('#hora').value = b.dataset.h; }));
$('#peso').addEventListener('input', e => { $('#pv').textContent = e.target.value; });
(function(){
  const el = $('#mapa'); if (!el) return;
  if (typeof L === 'undefined') { el.innerHTML = '<p style="padding:20px">No se pudo cargar el mapa (sin conexión a internet).</p>'; return; }
  const D = JSON.parse($('#gdata').textContent);
  const map = L.map(el, {preferCanvas: false});
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; colaboradores de <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }).addTo(map);
  const pts = D.nodos.map(n => [n[0], n[1]]);
  const vista = () => map.fitBounds(L.latLngBounds(pts), {padding: [20, 20]});
  vista();
  D.tramos.forEach(t => { if (!t[3]) L.polyline([pts[t[0]], pts[t[1]]],
    {color: t[2], weight: 3, opacity: D.hay_ruta ? 0.55 : 0.85}).addTo(map); });
  D.tramos.forEach(t => { if (t[3]) L.polyline([pts[t[0]], pts[t[1]]],
    {color: '#1b2631', weight: 11, opacity: 1, lineCap: 'round'}).addTo(map); });
  D.tramos.forEach(t => { if (t[3]) L.polyline([pts[t[0]], pts[t[1]]],
    {color: '#ffd60a', weight: 6, opacity: 1, lineCap: 'round'}).addTo(map); });
  const estilo = {0: ['#ffffff', '#475569', 4], 1: ['#ffd60a', '#1b2631', 6.5], 2: ['#22c55e', '#1b2631', 10], 3: ['#ef4444', '#1b2631', 10]};
  const orden = D.nodos.map((n, i) => i).sort((a, b) => D.nodos[a][3] - D.nodos[b][3]);
  orden.forEach(i => {
    const n = D.nodos[i], e = estilo[n[3]];
    const m = L.circleMarker(pts[i], {radius: e[2], fillColor: e[0], fillOpacity: 1, color: e[1], weight: n[3] ? 2.5 : 1.5}).addTo(map);
    m.bindTooltip(n[2], {sticky: true, direction: 'top'});
    m.on('click', () => {
      const o = $('#origen'), d = $('#destino');
      if (!o.value || (o.value && d.value)) { o.value = n[2]; d.value = ''; d.focus(); }
      else d.value = n[2];
      window.scrollTo({top: 0, behavior: 'smooth'});
    });
  });
  $('#zreset').addEventListener('click', e => { e.preventDefault(); vista(); });
})();
$('#f').addEventListener('submit', e => {
  const b = e.submitter || $('#go'); b.textContent = 'Buscando…';
  document.querySelectorAll('#f .actions button').forEach(x => { x.style.pointerEvents = 'none'; x.style.opacity = '.6'; });
});
"""


# ---------------------------------------------------------------------------
# Piezas de HTML
# ---------------------------------------------------------------------------
def _form(p: Params) -> str:
    datalist = "".join(f'<option value="{escape(n)}">' for n in NOMBRES)
    pills = "".join(
        f'<input type="radio" id="a-{k}" name="alg" value="{k}"{" checked" if k == p.alg else ""}>'
        f'<label for="a-{k}">{escape(v[0])}</label>'
        for k, v in ALGOS.items()
    )
    ejemplos = "".join(
        f'<a href="/?{escape(urlencode({"origen": o, "destino": d, "alg": p.alg, "hora": p.hora, "peso": p.peso}))}">'
        f'{escape(o)} → {escape(d)}</a>'
        for o, d in EJEMPLOS
    )
    horas = "".join(f'<button type="button" data-h="{h}">{escape(n)} {h}</button>'
                    for n, h in HORAS_RAPIDAS)
    return f"""
<section class="card">
  <h2>¿A dónde quieres ir?</h2>
  <p class="sub">Escribe para buscar entre las {len(NOMBRES)} paradas, o usa «Al azar».</p>
  <form id="f" method="get" action="/" autocomplete="off">
    <div class="row">
      <label class="f">Origen
        <input id="origen" name="origen" type="text" list="paradas" value="{escape(p.origen)}"
               placeholder="Ej. Br. Chicó Norte III Sector" required></label>
      <button type="button" id="swap" class="swap" title="Intercambiar origen y destino">⇄</button>
      <label class="f">Destino
        <input id="destino" name="destino" type="text" list="paradas" value="{escape(p.destino)}"
               placeholder="Ej. Br. Sucre" required></label>
    </div>
    <datalist id="paradas">{datalist}</datalist>
    <div class="pills" role="radiogroup" aria-label="Algoritmo">{pills}</div>

    <div class="row2">
      <label class="f">Hora de salida
        <input id="hora" name="hora" type="time" value="{escape(p.hora)}" required></label>
      <label class="f">Peso del tráfico en el coste: <b id="pv">{p.peso:g}</b>
        <input id="peso" name="peso" type="range" min="0" max="1" step="0.1" value="{p.peso:g}"></label>
      <label class="chk"><input type="checkbox" name="sim" value="1"{" checked" if p.sim else ""}>
        Simular tráfico cambiante</label>
      <label class="f" title="Número que fija el tráfico «real» simulado: misma semilla = mismo tráfico (repetible); otra semilla = otro escenario.">Semilla (realidad) ⓘ
        <input name="semilla" type="number" min="0" value="{p.semilla}"></label>
    </div>
    <div class="hrs">Hora rápida: {horas}
      <span>· peso 0 = solo distancia · peso 1 = distancia equivalente en tiempo</span></div>

    <div class="actions">
      <button id="go" class="go" type="submit">Buscar ruta</button>
      <button id="todos" class="alt" type="submit" name="todos" value="1"
              title="Ejecuta todos los algoritmos y los compara en una tabla">Ejecutar todos</button>
      <button id="azar" class="alt" type="button">🎲 Al azar</button>
    </div>
  </form>
  <div class="ej">Probar con: {ejemplos}</div>
</section>"""


def _kpis(r: SearchResult, desempeno: float | None) -> str:
    items = [
        ("Tramos", f"{r.path_edges}", "", ""),
        ("Transbordos", f"{r.transbordos}", "", "ok" if r.transbordos == 0 else ""),
        ("Distancia", f"{r.path_dist_m:,.0f}", "m", ""),
        ("Coste (dist. + tráfico + transb.)", f"{r.path_cost_m:,.0f}", "m", ""),
        ("Tiempo estimado de viaje", f"{r.path_time_s / 60:,.1f}", "min", ""),
        ("Desempeño", "" if desempeno is None else f"{desempeno:.1f}", "% del óptimo",
         "" if desempeno is None else "ok" if desempeno >= 99.95 else
         "warn" if desempeno >= 60 else "bad"),
        ("Nodos expandidos", f"{r.nodes_expanded:,}", "", ""),
        ("Nodos generados", f"{r.nodes_generated:,}", "", ""),
        ("Frontera máxima", f"{r.max_frontier:,}", "", ""),
        ("Tiempo de cómputo", f"{r.elapsed_ms:,.2f}", "ms", ""),
        ("Memoria pico", f"{r.peak_kib:,.1f}", "KiB", ""),
    ]
    if desempeno is None:   # sin benchmark no hay optimo con que comparar
        items = [i for i in items if i[0] != "Desempeño"]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi {c}"><div class="v">{v}<small>{u}</small></div><div class="l">{l}</div></div>'
        for l, v, u, c in items
    ) + "</div>"


# columna -> (titulo, extractor, mejor = min|max, formato)
_COLS = [
    ("Tramos", lambda r, d: r.path_edges, "min", "{:,.0f}"),
    ("Transb.", lambda r, d: r.transbordos, "min", "{:,.0f}"),
    ("Dist. (m)", lambda r, d: r.path_dist_m, "min", "{:,.0f}"),
    ("Coste (m)", lambda r, d: r.path_cost_m, "min", "{:,.0f}"),
    ("Viaje (min)", lambda r, d: r.path_time_s / 60, "min", "{:,.1f}"),
    ("Expandidos", lambda r, d: r.nodes_expanded, "min", "{:,.0f}"),
    ("Generados", lambda r, d: r.nodes_generated, "min", "{:,.0f}"),
    ("Frontera", lambda r, d: r.max_frontier, "min", "{:,.0f}"),
    ("Cómputo (ms)", lambda r, d: r.elapsed_ms, "min", "{:,.2f}"),
    ("Memoria (KiB)", lambda r, d: r.peak_kib, "min", "{:,.1f}"),
]


def _tabla_benchmark(res: dict, des: dict, sel: str) -> str:
    valores = {a: [f(res[a], des.get(a)) for _, f, _, _ in _COLS] for a in res}
    mejores = []
    for j, (_, _, modo, _) in enumerate(_COLS):
        col = [valores[a][j] for a in res]
        mejor = min(col) if modo == "min" else max(col)
        mejores.append(mejor if min(col) != max(col) else None)  # sin resaltar si todo igual

    head = "<th>Algoritmo</th>" + "".join(f"<th>{t}</th>" for t, *_ in _COLS)
    filas = []
    for a in res:
        celdas = []
        for j, (_, _, _, fmt) in enumerate(_COLS):
            v = valores[a][j]
            txt = fmt.format(v)
            cls = ' class="best"' if mejores[j] is not None and v == mejores[j] else ""
            celdas.append(f"<td{cls}>{txt}</td>")
        filas.append(f'<tr class="{"sel" if a == sel else ""}"><td>{escape(ALGOS[a][1])}</td>'
                     + "".join(celdas) + "</tr>")
    return (f'<div class="tablewrap"><table class="bm"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table></div>')


def _pasos(log) -> str:
    nombres = log.plan.path_names[1:]
    lis = []
    for s, destino in zip(log.bitacora, nombres):
        etiqueta = ("t-tr", "transbordo") if s.transbordo else \
                   ("t-go", "tomar") if s.accion.startswith("tomar") else ("t-keep", "seguir")
        lis.append(
            f'<li class="{"tr" if s.transbordo else ""}"><span class="n">{s.paso}</span>'
            f'<span class="tag {etiqueta[0]}">{etiqueta[1]}</span>'
            f'<span>ruta <span class="bus">{escape(s.ruta_bus)}</span> → {escape(destino)}</span>'
            f'<span class="d">{s.coste_tramo_m:,.0f} m</span></li>')
    return f'<ol class="steps">{"".join(lis)}</ol>'


def _card_simulacion(p: Params, oid: str, did: str) -> str:
    # la simulacion replanifica muchas veces: con metaheuristicas (lentas) usa A*
    alg_sim = p.alg if p.alg in STRATEGIES else "astar"
    viajes = _simular(oid, did, alg_sim, p.t_min, p.peso, p.semilla, p.obs)
    mejor = min(v.tiempo_real_s for v in viajes.values())
    filas = []
    for pol, v in viajes.items():
        cls = ' class="best"' if v.tiempo_real_s == mejor and len({x.tiempo_real_s for x in viajes.values()}) > 1 else ""
        filas.append(
            f"<tr><td>{escape(POLITICAS_TXT[pol])}</td>"
            f"<td>{v.tiempo_planeado_s / 60:,.1f}</td><td{cls}>{v.tiempo_real_s / 60:,.1f}</td>"
            f"<td>{v.retraso_s / 60:+,.1f}</td><td>{v.replanificaciones}</td><td>{v.picos}</td>"
            f"<td>{v.transbordos}</td><td>{v.dist_real_m:,.0f}</td><td>{v.computo_ms:,.0f}</td></tr>")
    head = "".join(f"<th>{t}</th>" for t in (
        "Política", "Plan (min)", "Real (min)", "Retraso (min)", "Replanif.", "Picos de tráfico",
        "Transb.", "Dist. real (m)", "Cómputo (ms)"))
    v = viajes["umbral"]
    pasos = "".join(
        f'<li class="{"rp" if s.replanifico else "tr" if s.transbordo else ""}">'
        f'<span class="n">{s.paso}</span>'
        f'<span class="tag {"t-rp" if s.replanifico else "t-tr" if s.transbordo else "t-keep"}">'
        f'{"replanifica" if s.replanifico else "transbordo" if s.transbordo else "avanza"}</span>'
        f'<span>ruta <span class="bus">{escape(s.ruta_bus)}</span> · {escape(s.desde)} → {escape(s.hasta)}</span>'
        f'<span class="d">tráfico ×{s.m_real:.2f} (esperado ×{s.m_esperado:.2f}) · {s.dt_s:.0f} s</span></li>'
        for s in v.pasos)
    return f"""
<section class="card">
  <h2>Simulación con tráfico cambiante</h2>
  <p class="sub">El tráfico real cambia mientras el agente avanza (fluctuaciones y picos repentinos de
    congestión) y el agente percibe el tráfico de toda la red (observabilidad global).
    La realidad es la misma (semilla {p.semilla}) para las tres políticas · {escape(ALGOS[alg_sim][0])},
    salida {escape(p.hora)}.</p>
  <div class="tablewrap"><table class="bm" style="min-width:760px"><thead><tr>{head}</tr></thead>
    <tbody>{"".join(filas)}</tbody></table></div>
  <p class="note">Plan = tiempo estimado con el tráfico típico al salir · Real = tiempo vivido en la simulación ·
    ● = menor tiempo real. Cambia la semilla para ver otra «realidad».</p>
  <details><summary>Ver el viaje paso a paso (política «replanificar si empeora»)</summary>
    <ol class="steps">{pasos}</ol></details>
</section>"""


def _color_m(m: float) -> str:
    """Verde (flujo libre) -> amarillo -> rojo (congestionado)."""
    x = min(1.0, max(0.0, (m - 1.0) / 1.6))
    r, g = (int(510 * x), 200) if x < 0.5 else (255, int(200 * (2 - 2 * x)))
    return f"rgb({min(r, 255)},{g},60)"


def _datos_grafo(path: list[str] | None = None, mult: dict | None = None) -> str:
    """JSON con nodos, tramos y camino para dibujar la red sobre un mapa (Leaflet)."""
    ids = sorted(G.adj)
    idx = {n: i for i, n in enumerate(ids)}
    rol = {n: 1 for n in (path or [])}
    if path:
        rol[path[0]], rol[path[-1]] = 2, 3
    en_ruta = set(zip(path, path[1:])) | set(zip(path[1:], path)) if path else set()
    nodos = [[round(G.coords[n][0], 6), round(G.coords[n][1], 6), G.name(n), rol.get(n, 0)]
             for n in ids]
    tramos = []
    for a in ids:
        for b in G.adj[a]:
            if a < b:
                col = _color_m(max(mult.get((a, b), 1.0), mult.get((b, a), 1.0))) if mult else "#64748b"
                tramos.append([idx[a], idx[b], col, 1 if (a, b) in en_ruta else 0])
    return json.dumps({"nodos": nodos, "tramos": tramos, "hay_ruta": bool(path)},
                      ensure_ascii=False).replace("</", "<\\/")


def _card_grafo(path=None, mult=None) -> str:
    leyenda = ("camino encontrado en amarillo · verde = origen · rojo = destino"
               if path else "elige origen y destino para ver el camino resaltado")
    if mult:
        leyenda += " · color de los tramos = congestión (verde libre, rojo congestionado)"
    return f"""
<section class="card">
  <h2>Grafo sobre el mapa</h2>
  <p class="sub">{len(NOMBRES)} paradas y {G.number_of_edges()} tramos sobre el mapa de Bogotá. Pasa el cursor
    sobre un nodo para ver el nombre de la parada; haz clic para usarla como origen o destino; rueda del
    mouse = zoom, arrastrar = mover. {leyenda}. <a href="#" id="zreset">Restablecer vista</a></p>
  <div class="gwrap"><div id="mapa"></div></div>
  <script type="application/json" id="gdata">{_datos_grafo(path, mult)}</script>
</section>"""


def _pagina(p: Params, cuerpo: str = "") -> str:
    nombres_js = json.dumps(NOMBRES, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planificador de rutas · red zonal SITP</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🚌%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<style>{_CSS}</style></head><body>
<header><div class="wrap">
  <h1>Planificador de rutas - Chapinero, Teusaquillo, Usaquen Sur</h1>
</div></header>
<main class="wrap">
{_form(p)}
{cuerpo}
<footer>Coste = mezcla de distancia y congestión (según el peso) + 800 m por transbordo · Desempeño = 100 · coste óptimo (UCS) / coste de la ruta</footer>
</main>
<script>{_JS.replace("__NOMBRES__", nombres_js)}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Rutas HTTP
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    p = _params()
    if not (p.origen and p.destino):
        return _pagina(p, _card_grafo())
    try:
        oid, did = G.resolve(p.origen), G.resolve(p.destino)
    except KeyError as e:
        return _pagina(p, f'<div class="err">{escape(str(e.args[0]))}. '
                          f'Elige una parada de la lista.</div>')
    if oid == did:
        return _pagina(p, '<div class="err">El origen y el destino son la misma parada.</div>')

    if p.todos:
        res, des = _benchmark(oid, did, p.t_min, p.peso)
    else:   # solo el algoritmo elegido: sin optimo de referencia, sin desempeño
        res, des = {p.alg: _buscar(oid, did, p.alg, p.t_min, p.peso)}, {}
    r = res[p.alg]
    log = RoutePlannerAgent(G).run(oid, did, p.alg, plan=r)
    p = Params(G.name(oid), G.name(did), p.alg, p.hora, p.peso, p.sim, p.semilla, todos=p.todos)
    q = urlencode({"origen": p.origen, "destino": p.destino, "alg": p.alg,
                   "hora": p.hora, "peso": p.peso})
    d_sel = des.get(p.alg)
    if d_sel is None:
        badge_nivel = ""
    else:
        nivel_cls, nivel_txt = (("b-ok", "ruta óptima") if d_sel >= 99.95 else
                                ("b-warn", f"{d_sel:.0f} % del óptimo") if d_sel >= 60 else
                                ("b-bad", f"solo {d_sel:.0f} % del óptimo"))
        badge_nivel = f'<span class="badge {nivel_cls}">{nivel_txt}</span>'
    if p.peso:
        mult = _mult(p.t_min, p.peso)
        cm = sum(mult[(a, b)] for a, b in zip(r.path, r.path[1:])) / max(1, r.path_edges)
        traf = f"tráfico {fmt_hora(p.t_min * 60)} · peso {p.peso:g} · congestión media ×{cm:.2f}"
    else:
        traf = "sin tráfico (solo distancia)"
    ruta = ' <em>→</em> '.join(f"<span>{escape(n)}</span>" for n in r.path_names[:8])
    if len(r.path_names) > 8:
        ruta += f' <em>… {len(r.path_names) - 8} paradas más</em>'

    cuerpo = f"""
<section class="card">
  <div class="head">
    <h2>{escape(p.origen)} → {escape(p.destino)}</h2>
    <span class="badge b-alg">{escape(ALGOS[p.alg][0])}</span>
    {badge_nivel}
    <span class="badge b-traf">{traf}</span>
  </div>
  <div class="route">{ruta}</div>
  {_kpis(r, d_sel)}
</section>

<section class="card">
  <h2>{"Benchmark en este par" if p.todos else "Medidas del algoritmo"}</h2>
  <p class="sub">{f"Los {len(res)} algoritmos resolviendo la misma ruta ({traf})." if p.todos else
      f"Solo se ejecutó {escape(ALGOS[p.alg][0])} ({traf}). Usa «Ejecutar todos» para compararlo con el resto."}</p>
  {_tabla_benchmark(res, des, p.alg)}
</section>

{_card_grafo(r.path, _mult(p.t_min, p.peso))}

<section class="card">
  <h2>Ruta sobre el grafo (imagen)</h2>
  <p class="sub">Amarillo = camino elegido · verde = origen · rojo = destino{
      " · color de los tramos = congestión (verde libre, rojo congestionado)" if p.peso else ""}.
    <a href="/ruta.png?{q}" target="_blank">Abrir imagen completa ↗</a></p>
  <img class="plot" src="/ruta.png?{q}" alt="Ruta {escape(p.alg)}">
</section>

<section class="card">
  <h2>Ciclo percepción → acción del agente</h2>
  <p class="sub">{r.path_edges} tramos · {r.transbordos} transbordo(s). En amarillo, los cambios de bus.</p>
  {_pasos(log)}
</section>"""
    if p.sim:
        cuerpo += _card_simulacion(p, oid, did)
    return _pagina(p, cuerpo)


@app.get("/ruta.png")
def ruta_png():
    p = _params()
    if not (p.origen and p.destino):
        abort(400)
    try:
        oid, did = G.resolve(p.origen), G.resolve(p.destino)
    except KeyError:
        abort(404)
    return send_file(io.BytesIO(_png(oid, did, p.alg, p.t_min, p.peso)),
                     mimetype="image/png", max_age=0)


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Servidor en {url}  (Ctrl+C para parar)")
    app.run(host="127.0.0.1", port=5000, debug=False)
