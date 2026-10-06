"""Contrato RS4-Z-2 de Real Soccer ONE: medidas, reglas y mecanismos del script de la sala.

Fuentes:
* v1 (rs_one_v1): `reports/rs4_b1/contract.json`, reconstruido de 61 grabaciones únicas.
* v2 (este contrato): auditoría RS4-Z sobre las mismas grabaciones (`data/rs4_jsonl`, motor
  original). En vez de aproximar las restricciones, v2 reproduce los mecanismos que el script
  aplica con `setDiscProperties` (ver `reports/rs4z/contract.json`):

  - F1: el motor restaura la masa del mapa (invMass 0,5) en cada reposicionamiento; el script
    fija 0,3 con la primera patada de un saque (lateral) o al liberar córner/saque de arco.
    En 72 archivos: 431 cambios a 0,3, el 99% a 0–3 ticks de una patada; mediana de 1218 ticks
    jugados con 0,5 tras cada saque inicial.
  - F2: el reloj no corre durante la espera del saque inicial (322 490 de 323 036 ticks).
  - F5: la barrera del área en el saque de arco son los segmentos c0 del mapa (|x|=840,
    |y|=320); el script agrega c0 al cGroup de los rivales.
  - Lateral: segmentos c1 del mapa (y=±555); el script agrega c1 al cGroup de los rivales.
  - F22: córner: disco de radio 445 en (±1150, ±740) que sólo choca con el equipo que defiende
    ese arco (disco 1 = rojo a la izquierda, disco 2 = azul a la derecha; 510 de 510 córners).
  - Saque de arco: disco 3 (choca con ambos equipos) de radio 18 en el punto durante 180 ticks
    desde la colocación de la pelota. Bloquea también al ejecutor (F6 no era un error).
  - F7: no hay franja muerta para cobrar salidas en las esquinas.

Lo que v2 no reproduce (documentado): demora del cobro de la sala (1–9 ticks en laterales,
~20–40 en córners y saques de arco), 3% de laterales asignados al otro equipo, la curva del
córner (ajuste aproximado, residuo p90 0,04), duración variable de partidos (~605–681 s).
"""
from __future__ import annotations

import numpy as np

VERSION = "RS4-Z-2"

# ----------------------------------------------------------------------------- medidas
LINE_W, LINE_H = 1150.0, 670.0          # líneas de la cancha (bg del .hbs)
GOAL_X, GOAL_HALF_H = 1162.0, 124.0     # línea de gol y medio ancho del arco
POST_X, POST_Y = 1150.0, 124.0          # postes (r=5)
SPAWN_X, SPAWN_DY = 560.0, 55.0         # spawnDistance y separación vertical de HaxBall
PLAYER_RADIUS = 15.0

# ----------------------------------------------------------------------------- saques
LATERAL, CORNER, GOAL_KICK = 1, 2, 3
KIND_NAMES = {0: "none", LATERAL: "lateral", CORNER: "corner", GOAL_KICK: "goal_kick"}

# Parámetros heredados de rs_one_v1 (mismos valores y significado).
V1 = dict(
    line_half_h=670.0, lateral_ball_y=688.0, lateral_x_margin=10.0, lateral_release_y=682.0,
    spot_release_distance=3.0, corner_x=1140.0, corner_y=660.0, goal_kick_x=1030.0, goal_kick_y=180.0,
    barrier_y=555.0, push_y=540.0, push_dx=270.0, box_front=840.0, box_half_h=320.0, box_push_x=825.0,
    play_inv_mass=0.3, piece_inv_mass=100000.0,
    corner_boost=1.98, corner_boost_delay=2.0, goal_kick_boost=2.71, goal_kick_boost_delay=3.0,
    corner_gravity_along=0.0438, corner_gravity_along_per_speed=-0.0173, corner_gravity_perp=-0.0169,
    goal_kick_gravity=-0.03, gravity_decay=0.97, gravity_ticks=143.0,
    safety_ticks=900.0, corner_rival_clearance=100.0, spot_clearance=33.0, goal_kick_hold_ticks=180.0,
)

# Mecanismos v2.
V2 = dict(
    map_inv_mass=0.5,            # masa del mapa tras cada reposicionamiento (F1)
    corner_disc_x=1150.0, corner_disc_y=740.0, corner_disc_radius=445.0,
    spot_disc_radius=18.0,
    lateral_run_distance=270.0, # Host Publico: throwinDistance, respecto al punto original
    lateral_timeout_ticks=420.0, # Host Publico: throwTimeOut (7 s)
)

# Índices del vector de parámetros que recibe el kernel (orden fijo).
PARAM_NAMES = tuple(V1) + tuple(V2)
PI = {name: i for i, name in enumerate(PARAM_NAMES)}


# Mapas. HAXARG 2K23 (script HaxArg Lite 2026, mapa exportado de la sala: `stadiums/haxarg_2k23.hbs`) es el
# mismo diseño que Real Soccer ONE con la cancha 70 px más angosta (línea lateral 600): banderines,
# barrera del lateral (c1 en ±485) y segmentos c0 de las esquinas corridos 70 px; mismos discos del
# script, área (840/320) y arcos. Puntos de saque medidos con el bot en la sala (2026-10-06): lateral
# y=±618, córner (±1140, ±590), saque de arco (±1060, 0). Pelota 9 y patada 5,65 salen del mapa.
# Supuesto sin verificar todavía con grabaciones: impulsos, curva, plazos y disco del córner (±1150, ±670)
# iguales a RS ONE corridos 70 px.
MAPS = {
    "rs_one": dict(stadium="rs_one", overrides={}, kick_strengths=(5.85, 5.75), ball_radii=(8.325, 8.0)),
    "haxarg_2k23": dict(stadium="haxarg_2k23", overrides=dict(
        line_half_h=600.0, lateral_ball_y=618.0, lateral_release_y=612.0, corner_y=590.0,
        goal_kick_x=1060.0, goal_kick_y=0.0, barrier_y=485.0, push_y=470.0, corner_disc_y=670.0),
        kick_strengths=(5.65,), ball_radii=(9.0,)),
}


def params(map_name: str = "rs_one") -> np.ndarray:
    values = {**V1, **V2, **MAPS[map_name]["overrides"]}
    return np.array([float(values[name]) for name in PARAM_NAMES], dtype=np.float64)


# Banderas de corrección. v1 = 0 reproduce exactamente rs_one_v1 (paridad); v2 = todas.
FIX_MASS = 1          # F1: masa 0,5 tras reposicionar, 0,3 con la patada del saque
FIX_CLOCK = 2         # F2: reloj congelado durante el saque inicial
FIX_ENGINE = 4        # F5/F22: barreras, discos de exclusión y del punto con el motor de colisiones
FIX_STRIP = 8         # F7: sin franja muerta en las esquinas
FIX_LATERAL = 16      # variante Host Publico: rango, patada obligatoria y vencimiento
FLAGS = {"v1": 0, "v2": FIX_MASS | FIX_CLOCK | FIX_ENGINE | FIX_STRIP}
FLAGS["v2_lateral"] = FLAGS["v2"] | FIX_LATERAL

# ----------------------------------------------------------------------------- entrenamiento
# Plazo sólo de entrenamiento (la sala no vence saques). A los 600 ticks sin ejecutar, el saque
# pasa al rival (lateral → lateral rival; córner → saque de arco del defensor; saque de arco →
# córner del atacante; saque inicial → saque inicial del rival). Sin multa ajustable.
TRAINING_DEADLINE = 600
# Seguridad de evaluación del saque inicial: la sala no lo vence y el reloj está congelado, así
# que una política que nunca saca congelaría el partido. Se registra como evento de detector.
KICKOFF_SAFETY = 1800

# ----------------------------------------------------------------------------- variantes
# Variantes reales del mapa (contrato v1: kickStrength 5,75 en 9/61 grabaciones; radio 8 en 5/61).
KICK_STRENGTHS = (5.85, 5.75)
BALL_RADII = (8.325, 8.0)

# Índices de los discos del script dentro del layout RS4-Z: [pelota, s1, s2, s3, mapa..., jugadores].
SD_RED, SD_BLUE, SD_BOTH = 1, 2, 3
SD_HOME = ((-1311.0, -19.0), (-1310.0, 29.0), (-1308.0, 62.0))
