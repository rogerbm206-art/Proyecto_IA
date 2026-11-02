"""Distancia euclidiana (en linea recta) entre coordenadas (lat, lon) -> metros.

Las coordenadas se proyectan a un plano local (equirectangular centrado en la
latitud media) para poder medir en metros, y luego se aplica la distancia
euclidiana d = sqrt(dx^2 + dy^2). Para un area pequeña como Chapinero la
diferencia con la distancia sobre la esfera es despreciable (< 0,1 %).
"""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def euclidean_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia euclidiana en metros entre dos puntos (lat, lon) en grados."""
    lat0 = math.radians((lat1 + lat2) / 2.0)
    dx = math.radians(lon2 - lon1) * math.cos(lat0) * EARTH_RADIUS_M
    dy = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    return math.hypot(dx, dy)
