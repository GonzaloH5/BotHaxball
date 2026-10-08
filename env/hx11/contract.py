"""Contrato HX11-1: reglas del script de la sala HaxEleven, versión del estadio "HaxEleven RSx11 [1800, 1200]".

El script de la sala está ofuscado y no se ejecuta: cada regla se midió en las grabaciones (S0–S2 de
`docs/PLAN_HAXELEVEN.md`). Fuentes: `reports/hx11/inventario.md`, `s1_fisica.md`, `s2_script.md` y las mediciones de
S3 anotadas abajo (31 partidos de la versión actual, del 27 de septiembre al 5 de octubre de 2026). Las tablas medidas
(forma de onda de la barra, rampa del powershot) están en `env/hx11/tables.json` (`python -m tools.hx11_script tablas`).

Cada constante dice de dónde sale: [medido: n eventos] o [supuesto] cuando las grabaciones no alcanzan.
Tiempos en ticks de 60/s. Coordenadas del mapa (y hacia abajo). Equipos: 0 rojo (defiende x < 0), 1 azul.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VERSION = "HX11-1"
STADIUM = "haxeleven_x11"

# ----------------------------------------------------------------------------- cancha (S1)
LINE_W, LINE_H = 1800.0, 1200.0       # líneas (sólo dibujo: la salida la cobra el script)
GOAL_X, GOAL_HALF_H = 1810.0, 170.0
BOX_X, BOX_HALF_H = 1535.0, 350.0     # área grande: |x| >= 1535 y |y| <= 350 del lado del arco
PENALTY_X = 1275.0                    # punto de penal (dibujo del mapa)
BALL_R, PLAYER_R = 8.0, 15.0
BASE_BALL_INV = 1.65                  # invMass de la pelota en juego (mapa)
PLAYER_INV = 0.69                     # invMass del jugador (mapa)
GHOST_INV = 999999.0                  # jugadores sin masa en córner, tiro libre y saque de arco [medido: 1.175 saques]

# ----------------------------------------------------------------------------- tipos de reinicio
NONE, LATERAL, CORNER, GOAL_KICK, FREE_KICK, PENALTY = 0, 1, 2, 3, 4, 5
KIND_NAMES = {NONE: "ninguno", LATERAL: "lateral", CORNER: "corner", GOAL_KICK: "saque_arco",
              FREE_KICK: "tiro_libre", PENALTY: "penal"}

# ----------------------------------------------------------------------------- salida de la pelota (S3)
# La pelota sale cuando queda entera fuera de la línea (|y| > 1200 + 8 o |x| > 1800 + 8 sin gol).
# [medido: 289 laterales, 94 córners, 62 saques de arco en 12 partidos]
OUT_MARGIN = BALL_R
LATERAL_DELAY = 6          # colocación: p10 5, p50 6, p90 8 ticks después de salir entera
END_LINE_DELAY = 15        # córner y saque de arco: p10 14, p50 15, p90 17
LATERAL_Y = 1230.0         # lateral en (x donde el centro cruzó y = ±1200, ±1230): diferencia p50 0 px
CORNER_SPOT = (1765.0, 1165.0)
GOAL_KICK_SPOT = (1572.0, 0.0)  # el arquero puede moverlo con !lado / !centro [no modelado]
# saca el rival del último que tocó (280/289, 93/94, 53/62); córner si el último toque fue del que defiende ese arco
# (93/94), saque de arco si fue del atacante (58/62)

# ----------------------------------------------------------------------------- pelota parada (S2–S3)
LATERAL_INV = 1.8          # masa inversa fija del lateral [medido: 893/893]
PENALTY_INV = 2.7          # [medido: 2 penales; poco]
BAR_PERIOD = 150           # barra: onda de 150 ticks después de un valor fijo de 7 [medido: 649 saques]
# disco 1 (barrera contra el equipo rival, sólo choca con ese equipo): radio y duración por tipo
BARRIER_R = {LATERAL: 373.0, CORNER: 750.0, GOAL_KICK: 373.0, FREE_KICK: 373.0, PENALTY: 373.0}
BARRIER_OFF_AFTER_KICK = {LATERAL: 3, CORNER: 1, FREE_KICK: 1, PENALTY: 1}  # ticks después de la patada [medido]
GOAL_KICK_BARRIER_TICKS = 121    # en el saque de arco la barrera va al punto de penal y dura 121 ticks fijos
GOAL_KICK_BARRIER_DX = 297.0     # disco 1 a 297 px de la pelota hacia el centro (= punto de penal)
# disco 2 (radio 17,5, choca con los dos equipos) en el punto: nadie llega a patear mientras está (deja a los jugadores
# a >= 32,5 px del centro de la pelota; el alcance de patada es 27)
SPOT_DISC_TICKS = {LATERAL: 30, CORNER: 120, GOAL_KICK: 120, FREE_KICK: 120, PENALTY: 120}  # [medido; penal supuesto]
GHOST_RESTORE_AFTER_KICK = 1     # masas y grupos vuelven en el tick siguiente a la patada [medido]
BALL_RESTORE_AFTER_KICK = {LATERAL: 5, CORNER: 1, GOAL_KICK: 1, FREE_KICK: 1, PENALTY: 1}  # masa de la pelota a 1,65
BAD_LATERAL_TICKS = 120          # [supuesto] lateral que no entra a la cancha en 2 s: se repite para el rival
PIECE_TIMEOUT = 600              # [supuesto] "¡SE ACABA EL TIEMPO…!" a ~510–600; cesión al rival "por hacer tiempo"
                                 # (7 + 5 + 4 casos en el banco): se entrega al rival en el mismo punto
# saque de arco: los rivales quedan con cGroup = c0 (saca el rojo, zona x <= -1050) o c1 (saca el azul): no chocan con
# nadie ni con la pelota y no pueden entrar a la zona [medido: 68/68]

# ----------------------------------------------------------------------------- faltas (S2)
FOUL_CONTACT_TOL = 2.5     # rival "pegado": centros a <= 30 + 2,5 px [ajuste: 2–3 px igual de bueno]
FOUL_GAP = 6               # cortes de hasta 6 ticks no reinician la cuenta
FOUL_BALL_DIST = 25.0      # el poseedor tiene la pelota a <= 25 px de su borde (si no, se reinicia)
FOUL_TICKS = 113           # falta a los 113 ticks (1,9 s) [ajuste: 38/50 faltas, 5 disparos falsos]
FREEZE_TICKS = 14          # la pelota queda congelada 14 ticks antes del tiro libre [medido: 78, p10 14, p90 15]
TOUCH_EPS = 0.5            # toque = pelota a <= 0,5 px del borde del jugador, o patada

# ----------------------------------------------------------------------------- offside (S2)
# Patada de un compañero; en ese tick el receptor está por delante del anteúltimo rival (arquero incluido), en campo
# rival y por delante de la pelota; se cobra cuando toca la pelota. Un toque de un rival sin patear no lo anula; una
# patada de cualquiera lo corta. Exentas las ejecuciones de lateral, córner y saque de arco.
# [medido: 120/138 cobros explicados, 91% de precisión]

# ----------------------------------------------------------------------------- powershot (S2)
PS_CONTACT_EPS = 2.0       # contacto para cargar: pelota a <= 2 px del borde del poseedor
PS_DELAY = 90              # la carga empieza a los 90 ticks de contacto continuo [medido: 1.113 cargas, p50 91]
PS_LEVELS = (2.43, 3.21, 4.0)   # niveles (33/66/100%) a +24, +209 y +359 ticks del inicio (tabla `ramp`)
PS_RESET_AFTER_KICK = 1    # la masa vuelve a 1,65 en el tick siguiente a la patada (p50)
# comba de la patada cargada: gravedad en y = -0,0171 × velocidad vertical del jugador, desde el tick siguiente, ×0,982
# por tick, hasta 150 ticks [medido: 894 patadas]
PS_CURVE_K = 0.0171
PS_CURVE_DECAY = 0.982
PS_CURVE_TICKS = 150
# comba de la pelota parada [aproximado: falta la regresión de S2]: tiro libre y saque de arco igual que el powershot
# con k = 0,024; córner con gravedad en x = -0,021 × vx del jugador y 0,01 en y hacia la cancha
SP_CURVE_K = 0.024
CORNER_CURVE_KX = 0.021
CORNER_CURVE_Y = 0.01

# ----------------------------------------------------------------------------- saque inicial y gol (S3)
GOAL_RESET_TICKS = 150     # el reposicionamiento llega 150 ticks después del gol [medido: 31/31]
GK_KICKOFF = (1760.0, 0.0)  # en cada saque inicial el script lleva al arquero a (±1760, 0) [medido: 16/16 por equipo]


def load_tables(path: str | Path | None = None) -> dict:
    """Tablas medidas: barra (onda de 150 ticks, valor fijo y su duración, por tipo) y rampa del powershot."""
    p = Path(path) if path else Path(__file__).with_name("tables.json")
    t = json.loads(p.read_text(encoding="utf-8"))
    bar = np.zeros((6, BAR_PERIOD))
    hold = np.zeros(6)
    hold_ticks = np.zeros(6, np.int64)
    for kind, name in ((CORNER, "corner"), (FREE_KICK, "tiro_libre"), (GOAL_KICK, "saque_arco")):
        b = t["bar"][name]
        bar[kind] = b["wave"]
        hold[kind] = b["hold_value"]
        hold_ticks[kind] = int(b["hold_ticks"])
    bar[PENALTY] = PENALTY_INV
    hold[PENALTY] = PENALTY_INV
    bar[LATERAL] = LATERAL_INV
    hold[LATERAL] = LATERAL_INV
    ramp = np.array(t["ramp"]["values"], dtype=np.float64)
    return dict(bar=bar, hold=hold, hold_ticks=hold_ticks, ramp=ramp)
