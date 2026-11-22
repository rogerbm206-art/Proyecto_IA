"""Agrupamiento con K-Medias: tramos con perfiles horarios de velocidad parecidos.

Cada tramo se describe con su perfil de velocidad: la mediana de `vel_kmh` en cada hora
de 5 a 23 h de un dia habil (19 valores, todos en km/h, sin normalizar). Los tramos con
menos de 15 de esas 19 horas se descartan, y las horas que faltan en los demas se
interpolan linealmente entre las horas vecinas del mismo tramo.

K-Medias asigna cada tramo al centroide mas cercano (distancia euclidiana entre perfiles)
y mueve cada centroide al promedio de su grupo, hasta que los grupos dejan de cambiar. El
numero de grupos k se elige comparando la inercia (metodo del codo) y el coeficiente de
silueta. Los grupos se numeran de menor a mayor velocidad media, para que el numero tenga
significado y no dependa de la semilla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

HORAS = list(range(5, 24))
MIN_HORAS = 15
SEMILLA = 0
N_INIT = 10


def perfiles(df: pd.DataFrame, tipo_dia: str = "habil") -> pd.DataFrame:
    """Matriz tramo x hora de velocidades medianas (km/h); indice = (u, v)."""
    d = df[df["tipo_dia"] == tipo_dia]
    m = d.groupby(["u", "v", "hora"])["vel_kmh"].median().unstack("hora").reindex(columns=HORAS)
    m = m[m.notna().sum(axis=1) >= MIN_HORAS]
    return m.interpolate(axis=1, limit_direction="both")


def evaluar_k(P: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Inercia y silueta para cada k."""
    filas = []
    for k in ks:
        km = KMeans(k, n_init=N_INIT, random_state=SEMILLA).fit(P.values)
        filas.append({"k": k, "inercia": float(km.inertia_),
                      "silueta": float(silhouette_score(P.values, km.labels_))})
    return pd.DataFrame(filas)


def ajustar(P: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(etiquetas, centroides) con los grupos ordenados de menor a mayor velocidad media."""
    km = KMeans(k, n_init=N_INIT, random_state=SEMILLA).fit(P.values)
    orden = np.argsort(km.cluster_centers_.mean(axis=1))
    nuevo = np.empty(k, dtype=int)
    nuevo[orden] = np.arange(k)
    return nuevo[km.labels_], km.cluster_centers_[orden]


def estabilidad(P: pd.DataFrame, k: int, repeticiones: int = 20, fraccion: float = 0.8) -> dict:
    """Indice de Rand ajustado entre el agrupamiento completo y el de submuestras al azar.

    Cerca de 1 = los grupos casi no cambian al quitar tramos.
    """
    base, _ = ajustar(P, k)
    rng = np.random.default_rng(SEMILLA)
    aris = []
    for r in range(repeticiones):
        idx = np.sort(rng.choice(len(P), size=int(fraccion * len(P)), replace=False))
        km = KMeans(k, n_init=N_INIT, random_state=r).fit(P.values[idx])
        aris.append(adjusted_rand_score(base[idx], km.labels_))
    return {"ari_medio": float(np.mean(aris)), "ari_min": float(np.min(aris))}


def asignar(P: pd.DataFrame, centroides: np.ndarray) -> np.ndarray:
    """Grupo (centroide mas cercano) de nuevos perfiles."""
    d = ((P.values[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)
