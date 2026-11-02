# Dataset GTFS — SITP / TransMilenio (Bogotá)

## Origen

| | |
|---|---|
| Catálogo | **Mobility Database**, feed `mdb-3358` — <https://mobilitydatabase.org/feeds/mdb-3358> |
| Nombre en el catálogo | *SIMUR Bogota* (Official Feed, verificado) |
| Productor | **Transmilenio S.A.** (`feed_info.feed_publisher_name`) |
| URL de descarga | <https://storage.googleapis.com/gtfs-estaticos/GTFS-2026-04-29.zip> |
| Licencia | <http://www.simur.gov.co/gtfs> |
| Vigencia del servicio | 2026-01-01 a 2026-12-31 |
| Archivo local | `data/GTFS-SIMUR-2026-04-29.zip` (≈ 115 MB comprimido, ≈ 520 MB sin comprimir) |
| Descargado | 2026-09-08 |

Se eligió este feed sobre `GTFS-2025-07-09.zip` (mismo bucket, versión anterior)
porque aquel **solo contenía la capa zonal (SITP)**, mientras que este incluye
además el **componente troncal de TransMilenio** con estaciones nombradas.

Nota: todas las fuentes oficiales colombianas (datosabiertos.bogota.gov.co de la
Alcaldía, datos.movilidadbogota.gov.co de la Secretaría de Movilidad, el Portal de
Datos Abiertos de TransMilenio) distribuyen el mismo GTFS producido por
Transmilenio S.A. desde el bucket `storage.googleapis.com/gtfs-estaticos/`, que
además publica un `GTFS_YYYYMMDD.zip` diario. Este proyecto fija la versión
`GTFS-2026-04-29.zip`.

## Contenido (parseado con `scripts/parse_gtfs.py`)

| Tabla | Volumen |
|---|---|
| `agency.txt` | 7 agencias (ver abajo) |
| `routes.txt` | **1.024 rutas** |
| `stops.txt` | 8.309 paradas — **152 estaciones/portales** (`location_type=1`) + 8.157 plataformas |
| `trips.txt` | 181.543 viajes |
| `stop_times.txt` | 9.600.552 registros |
| `shapes.txt`, `calendar*.txt`, `frequencies.txt`, `fare_attributes.txt`, `feed_info.txt` | presentes |

### Rutas por agencia

| id | agencia | rutas |
|---|---|---|
| 1 | Transmilenio-Troncal | 109 |
| 2 | Transmilenio-Alimentadores | 158 |
| 6 | Transmilenio-Dual | 26 |
| 3 | Zonal-Urbano | 717 |
| 4 | Zonal-Complementario | 9 |
| 5 | Zonal-Especial | 4 |
| 7 | TransMiCable | 1 |

`route_type`: 1.023 bus (`3`) + 1 cable (`6`).

## Archivos derivados (generados por `scripts/parse_gtfs.py`)

- `data/stations_troncal.csv` — 152 estaciones troncales (`stop_id`, nombre, lat, lon).
- `data/routes_troncal.csv` — 109 rutas troncales (`route_id`, código, destino).

## Notas de estructura

- `stop_times.txt` referencia **plataformas** (`location_type=0`); cada una apunta
  a su estación con `parent_station`. Para razonar a nivel de estación hay que
  agrupar por `parent_station`.
- No hay `direction_id` ni `route_desc`; el sentido se infiere de `trip_headsign`
  o del orden de `stop_sequence`.
- Servicio de cable (`frequencies.txt`) modelado por headway, no por viajes fijos.
