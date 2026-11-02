# Planificación de rutas en la red zonal del SITP (buses azules) — Bogotá

Asignatura **Inteligencia Artificial** — Universidad Sergio Arboleda · **Corte 2** (algoritmos de búsqueda).

Un agente que planifica viajes sobre la red de **buses azules (componente zonal, SITP‑Urbano)** de
TransMilenio, a partir del **GTFS oficial**. El problema se modela como un grafo de paradas con rutas
fijas y se resuelve con cinco algoritmos de búsqueda (BFS, DFS, UCS, voraz, A\*) y cinco metaheurísticas
(ascenso de colinas, temple simulado, genético, hormigas, enjambre; ver más abajo), que se comparan en
tiempo, memoria, nodos expandidos, transbordos y calidad de la ruta. Incluye simulación de tráfico por
hora y un entorno dinámico en el que el agente puede replanificar.

## El grafo

| | |
|---|---|
| Zona | Chapinero y barrios vecinos (Teusaquillo, Chicó, Usaquén sur) |
| Nodos | **300 paradas reales** (no postes) |
| Aristas | **557 tramos** entre paradas consecutivas |
| Conexo | sí |
| Fuente | GTFS SIMUR / Transmilenio S.A. `2026-04-29`, agencia 3 «Zonal‑Urbano» |

**Cómo se construye** (`src/gtfs_graph.py`, `_build_from_gtfs`), a partir de `stops.txt`, `routes.txt`,
`trips.txt` y `stop_times.txt` del zip de `data/`:

1. Se toman los viajes de las rutas zonales y, en cada uno, los pares de paradas consecutivas. Un tramo
   cuenta si lo recorren **≥ 8 viajes** y sus dos extremos caen dentro del recuadro (`BBOX`).
2. **Postes → paradas.** El GTFS trae un `stop_id` por poste (uno por sentido). Se fusionan los que
   comparten `parent_station` o el mismo nombre normalizado (sin tildes, abreviaturas expandidas). Si
   los postes de una parada quedan a más de 600 m, se vuelve a partir por cercanía.
3. Cada parada guarda el **centroide** de sus postes; cada tramo guarda su `dist_m` (distancia
   euclidiana, sin redondear), el nº de viajes y la lista de **rutas** que lo recorren.
4. Se recorta a `TARGET_NODES = 300` con un crecimiento tipo BFS ponderado por nº de viajes.
5. Los nombres repetidos se desambiguan (`Br. Pasadena (junto a Br. La Castellana)`).

La construcción es **determinista** (mismo grafo en cada ejecución, verificado con distintas semillas de
hash) y se cachea en `data/red_zonal.json`. Parámetros: `BBOX`, `MIN_TRIPS_EDGE`, `TARGET_NODES`.

## Modelo del problema

- **Rutas fijas:** un bus sigue su ruta; cambiar de ruta exige bajarse y esperar otro bus. Por eso el
  estado de la búsqueda es **(parada, ruta en la que vienes)**.
- **Coste** de un paso = distancia del tramo (+ `TRANSFER_PENALTY_M = 800 m` si cambias de ruta; subir
  al primer bus no cuenta). Coste total = `distancia + 800 · transbordos`.
- **Heurística** `h` = distancia euclidiana en línea recta a la meta: admisible y consistente.
- **Medida de desempeño (PEAS)** = `100 · coste_óptimo / coste_obtenido` (el óptimo es el de UCS).
  100 % = ruta óptima.

| Algoritmo | Prioridad | Óptimo | Nota |
|---|---|---|---|
| `bfs` | nivel (FIFO) | en nº de tramos | ignora el coste y los transbordos |
| `dfs` | LIFO | no | rutas larguísimas |
| `ucs` | `g` | en coste | Dijkstra |
| `greedy` | `h` | no | expande muy pocos nodos |
| `astar` | `g + h` | en coste | mismo resultado que UCS, expandiendo menos |

### El agente (`src/agent.py`)

`RoutePlannerAgent` implementa el modelo PEAS: **delibera una sola vez** con el algoritmo elegido
(`plan`) y luego **ejecuta** la ruta con el ciclo percepción → acción (`perceive` / avanzar a una parada
adyacente), registrando qué bus tomar en cada tramo y cuándo hay transbordo. El entorno es estático y
determinista, por lo que no replanifica.

## Estructura

```
Proyecto_IA/
├── data/        red_zonal.json (caché), perfil_trafico.json, benchmark_50.xlsx, paradas.xlsx, README_dataset.md
│                (el zip del GTFS no está en el repo: pesa 120 MB; ver «Dataset»)
├── src/
│   ├── gtfs_graph.py   modelo del grafo (TransitGraph) y su construcción desde el GTFS
│   ├── search.py       los 5 algoritmos + modelo de rutas fijas
│   ├── metaheuristics.py  hc, sa, ga, aco, pso
│   ├── traffic.py      perfil horario y coste combinado con tráfico
│   ├── simulacion.py   entorno dinámico y replanificación del agente
│   ├── agent.py        agente PEAS (percepción → acción)
│   ├── plot_search.py  dibuja la ruta sobre el grafo + panel de métricas
│   ├── demo.py         ejecuta un algoritmo (gráfica + resumen); par por defecto
│   ├── benchmark.py    comparación de los 5 algoritmos sobre varios pares
│   └── geo.py          distancia euclidiana
├── scripts/     run_<algoritmo>.py (10), run_todos.py, benchmark_50.py, simular_viaje.py,
│                mapa_trafico.py, calibrar_trafico.py, listar_paradas.py, build_red_zonal.py, parse_gtfs.py
├── webapp/      app.py  (interfaz web con Flask)
├── tests/       171 pruebas: algoritmos, metaheurísticas, tráfico, simulación, grafo
├── output/      red del grafo, una gráfica por algoritmo, comparación
└── docs/        Informe_proyecto_IA.pdf
```

## Uso

Requiere Python 3.10+ y (el zip del GTFS debe estar en `data/` solo para reconstruir el grafo): `pip install networkx matplotlib openpyxl flask pytest`

```bash
python -m src.gtfs_graph                 # (re)construye el grafo y su caché (~1 min)
python scripts/build_red_zonal.py        # dibuja la red -> output/red_zonal_chapinero.png

python scripts/run_astar.py              # un algoritmo -> output/ruta_astar.png + resumen en consola
python scripts/run_bfs.py "Br. Colombia" "Universidad De La Salle"
python scripts/run_todos.py              # los 5 + comparación global

python scripts/benchmark_50.py                                    # 50 corridas, pares aleatorios -> Excel
python scripts/benchmark_50.py -n 100 -s 7                        # 100 corridas, otra semilla
python scripts/benchmark_50.py -o "Br. Colombia" -d "Universidad EAN"   # par fijo
python scripts/benchmark_50.py --listar                           # lista las paradas
python scripts/listar_paradas.py                                  # data/paradas.xlsx

python webapp/app.py                     # interfaz web en http://127.0.0.1:5000
python -m pytest -q                      # 171 pruebas
```

- **Par por defecto** de los `run_*.py`: las constantes `ORIGEN_DEF` / `DESTINO_DEF` de `src/demo.py`
  (hoy `Br. Chicó Norte III Sector → Br. Sucre`). Si pasas origen y destino por argumentos, ganan ellos.
- Los nombres aceptan coincidencia parcial (`-o "colombia"`).
- `benchmark_50.py` sortea un par origen‑destino por corrida (el mismo para los 5 algoritmos) y escribe
  `data/benchmark_50.xlsx`: hoja `Resumen` + una hoja por algoritmo con las corridas y su fila `PROMEDIO`.

### Interfaz web

Elige origen, destino (se escribe para buscar entre las 300 paradas) y algoritmo. Muestra las medidas del
algoritmo elegido, el **benchmark de los 5 algoritmos sobre ese mismo par** (con el mejor valor de cada
columna resaltado), la ruta dibujada sobre el grafo y el ciclo percepción → acción del agente.

## Resultados (50 pares aleatorios, semilla 42, 300 paradas)

| Algoritmo | Desempeño | Transbordos | Nodos expandidos | Tiempo | Memoria pico |
|---|---|---|---|---|---|
| UCS | **100 %** | 1.0 | 1596 | 165 ms | 404 KiB |
| A\* | **100 %** | 1.0 | 485 | 58 ms | 193 KiB |
| BFS | 66.1 % | 4.8 | 1241 | 55 ms | 97 KiB |
| Voraz | 63.8 % | 5.1 | 46 | 6 ms | 42 KiB |
| DFS | 9.9 % | 46.3 | 1037 | 55 ms | 162 KiB |

- **UCS y A\*** siempre dan la ruta óptima; A\* expande ~⅓ de los nodos.
- **BFS** minimiza tramos pero no transbordos; el **voraz** es rapidísimo pero subóptimo.
- **DFS** recorre media red: en `Chicó Norte III Sector → Sucre` da 100 tramos y 66 transbordos, frente a
  14 tramos y 1 transbordo de A\*.
- BFS usa menos memoria que UCS/A\* porque solo guarda un diccionario de padres y una cola que se
  drena; UCS/A\* añaden la tabla de costes, el conjunto cerrado y un heap con entradas obsoletas.

## Limitaciones

- Entorno **estático**: no hay horarios, frecuencias, tráfico ni tramos cerrados; el agente no replanifica.
- La penalización de 800 m por transbordo es una suposición del modelo, no un dato del GTFS.
- Las paradas se agrupan por nombre/estación: nombres distintos para un mismo sitio (p. ej. `U. Pedagogica`
  y `U. Pedagógica Nacional`) quedan como paradas separadas.
- Tiempo y memoria de una sola ejecución son ruidosos; para comparar usa el promedio de `benchmark_50`.

## Dataset

Feed **SIMUR / Transmilenio S.A.** (Mobility Database `mdb-3358`):
<https://storage.googleapis.com/gtfs-estaticos/GTFS-2026-04-29.zip> — catálogo:
<https://mobilitydatabase.org/feeds/mdb-3358>. Más detalle en [`data/README_dataset.md`](data/README_dataset.md).
Licencia: CC BY‑SA 4.0 / <http://www.simur.gov.co/gtfs>.

## Simulación de tráfico

- **Perfil horario** (`data/perfil_trafico.json`, generado con `scripts/calibrar_trafico.py`): velocidades programadas del GTFS por hora → multiplicador de congestión `m(h)` (≈1.0 de madrugada, ≈1.89 a las 16–17 h). Es tráfico *programado*, no medido.
- **Coste combinado**: `dist · (1 + peso · (m − 1))` + 800 m por transbordo. `peso=0` = solo distancia; `peso=1` = distancia equivalente en tiempo (defecto 0.5). Como `m ≥ 1`, la heurística euclidiana sigue siendo admisible y UCS/A* siguen siendo óptimos sobre la foto de tráfico al planear.
- **Factor espacial** por tramo: sintético (lognormal con semilla), es un supuesto, no un dato.
- **Entorno dinámico** (`src/simulacion.py`): el tráfico real cambia mientras el agente avanza (fluctuación + picos repentinos). El agente solo percibe los tramos adyacentes (`--observabilidad local|global`) y puede replanificar: `nunca`, `umbral`, `cada_parada`. Sin incidentes ni cierres.
- Uso: `python scripts/simular_viaje.py ORIGEN DESTINO --hora 17:00 --algoritmo astar --peso 0.5 --semilla 1`, `python scripts/mapa_trafico.py`, y `--hora/--peso` en `scripts/run_*.py`. La web tiene hora, peso y «Simular tráfico cambiante».
- Pruebas: 171 (`python -m pytest -q`).

## Metaheurísticas (`src/metaheuristics.py`)

Resuelven el **mismo problema y el mismo coste** que los algoritmos de búsqueda (distancia combinada con tráfico + 800 m por transbordo). Una solución es un camino simple de paradas; las rutas de bus de cada tramo se reparten con programación dinámica minimizando transbordos. Son estocásticas (misma `semilla` = mismo resultado) y **no garantizan el óptimo**.

| Clave | Algoritmo | Idea |
|---|---|---|
| `hc` | Ascenso de colinas | Mejor de 25 mutaciones por paso; reinicios aleatorios al estancarse |
| `sa` | Temple simulado | Acepta peores caminos con prob. exp(−Δ/T); T baja geométricamente |
| `ga` | Algoritmo genético | Población de caminos, cruce por parada común, mutación de segmento, elitismo |
| `aco` | Colonia de hormigas | Feromona por tramo + cercanía al destino (MAX-MIN Ant System) |
| `pso` | Enjambre de partículas | Vector de prioridades por parada, decodificado a camino con DFS ordenado |

Ninguna usa Dijkstra ni el costo óptimo para construir soluciones. En las métricas, «expandidos» = evaluaciones del coste, «generados» = soluciones candidatas construidas, «frontera» = tamaño de población/colonia/enjambre (1 en `hc` y `sa`).

Uso: `python scripts/run_aco.py` (también `run_hc`, `run_sa`, `run_ga`, `run_pso`), `search(g, o, d, "aco", semilla=1)`, y los 10 algoritmos aparecen en la web y en `scripts/benchmark_50.py`. Pruebas: `tests/test_metaheuristicas.py`.
