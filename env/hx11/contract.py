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
LATERAL_OUT_Y = 1210.0     # el lateral se cobra cuando el centro pasa |y| > 1210, y su x es la de ese tick [S4: 97%
                           # exacto en 219 laterales; con 1208, 42%]
LATERAL_DELAY = 6          # colocación: p10 5, p50 6, p90 8 ticks después de salir entera
END_LINE_DELAY = 15        # córner y saque de arco: p10 14, p50 15, p90 17
LATERAL_Y = 1230.0         # lateral en (x de la detección, ±1230)
CORNER_SPOT = (1765.0, 1165.0)
GOAL_KICK_SPOT = (1572.0, 0.0)  # el arquero puede moverlo con !lado / !centro [no modelado]
# saca el rival del último que tocó (280/289, 93/94, 53/62); córner si el último toque fue del que defiende ese arco
# (93/94), saque de arco si fue del atacante (58/62)

# ----------------------------------------------------------------------------- pelota parada (S2–S3)
LATERAL_INV = 1.8          # masa inversa fija del lateral [medido: 893/893]
PENALTY_INV = 2.7          # [medido: 2 penales; poco]
HOLD_MAX = 15
BAR_PERIOD = 150           # barra: onda de 150 ticks después de un valor fijo de 7 [medido: 649 saques]
# disco 1 (barrera contra el equipo rival, sólo choca con ese equipo): radio y duración por tipo
BARRIER_R = {LATERAL: 373.0, CORNER: 750.0, GOAL_KICK: 373.0, FREE_KICK: 373.0, PENALTY: 373.0}
BARRIER_OFF_AFTER_KICK = {CORNER: 1, FREE_KICK: 1, PENALTY: 1}  # ticks después de la patada [medido]; el lateral
# termina cuando la pelota vuelve a |y| < 1210 (barrera y masa 1,8 se van ahí) [S4]
GOAL_KICK_BARRIER_TICKS = 121    # en el saque de arco la barrera va al punto de penal y dura 121 ticks fijos
GOAL_KICK_BARRIER_DX = 297.0     # disco 1 a 297 px de la pelota hacia el centro (= punto de penal)
# disco 2 (radio 17,5, choca con los dos equipos) en el punto: nadie llega a patear mientras está (deja a los jugadores
# a >= 32,5 px del centro de la pelota; el alcance de patada es 27)
SPOT_DISC_TICKS = {LATERAL: 30, CORNER: 120, GOAL_KICK: 120, FREE_KICK: 120, PENALTY: 120}  # [medido; penal supuesto]
PENALTY_DISC_TICKS = 76          # penal: la barrera (contra los compañeros del que patea: quedan afuera del área) y el
                                 # disco del punto duran 76 ticks [medido en 1 penal; débil]
GHOST_RESTORE_AFTER_KICK = 1     # masas y grupos vuelven en el tick siguiente a la patada [medido]
BALL_RESTORE_AFTER_KICK = {CORNER: 1, GOAL_KICK: 1, FREE_KICK: 1, PENALTY: 1}  # masa de la pelota a 1,65
# Lateral mal sacado (2026-10-09, banco de prueba y desarrollo + los 9 anuncios "COMO NO VAS A SABER SACAR UN LATERAL"
# de la sala): la pelota no entra a la cancha (|y| < LATERAL_OUT_Y) después de la patada del lateral. Se repite para el
# rival del ÚLTIMO que la tocó (si la tocó un rival afuera, vuelve a sacar el mismo equipo: 6 de 8 en el banco; si la
# tocó un compañero y no entró, saca el otro equipo: 3 de 3).
# * Los legales que nadie toca entran p50 4, p99 28 ticks después de la patada (6 de 2437 después de 45).
# * Los que no entran y nadie toca: la sala los repite para el rival en el punto original, p50 51 ticks después de la
#   patada (p10 41, p90 100) = 45 + LATERAL_DELAY.
# * Llevar la pelota por la línea antes de patear es legal (122 px de mediana en el banco).
BAD_LATERAL_TICKS = 45
# Después de la patada, antes de que entre:
# * si la toca un RIVAL afuera, la sala corta enseguida (2–29 ticks) y el lateral vuelve al equipo que sacaba, en el
#   mismo punto (34 casos del banco);
# * si la vuelve a tocar el que sacó, o un compañero, el lateral ya cuenta como sacado: la pelota puede seguir por la
#   línea y entrar hasta 418 ticks después de la patada (el mismo: 370 casos, p50 106; compañero: 35, p50 129).
#   Plazo largo de seguridad:
BAD_LATERAL_TICKS_TOUCHED = 480
# Lateral pateado con la pelota ya adentro (la metieron empujando sin patear): mal sacado, se repite para el rival en el
# punto original. En el banco (6603 laterales, 2026-10-09): 36 de 37 laterales pateados con |y| < LATERAL_OUT_Y se
# repitieron para el rival; meterla un poco y volver a sacarla antes de patear es legal (24 de 25), y llevarla por la
# línea también (122 px de mediana).
# Metida empujando y sin patear: mal sacado si queda adentro más de LATERAL_INSIDE_TICKS seguidos [usuario: "si la
# empujan sin patear es mal sacado"; en el banco los humanos casi nunca lo hacen (2 casos), y los legales que entraron y
# volvieron a salir antes de patear estuvieron adentro hasta 66 ticks seguidos].
LATERAL_INSIDE_TICKS = 90
# saque de arco: la sala saca a los rivales del área grande (|x| > 1050, |y| < 675) a |x| = 1035, con su misma y, un tick
# después de la colocación (328 de 328 rivales en 30 partidos de prueba; 2026-10-09)
BOX_X, BOX_Y = 1050.0, 675.0
GK_CLEAR_X = 1035.0
# "¡SE ACABA EL TIEMPO PARA SACAR EL …!" a los 600 ticks es solo un aviso: en ~10.600 pelotas paradas del banco
# (2026-10-09) la primera patada fue siempre del equipo que sacaba, con esperas de hasta 1366 ticks (córner). No se
# observó ninguna cesión; 1800 es una red de seguridad del simulador para que una pelota parada no quede trabada.
PIECE_TIMEOUT = 1800
                                 # (7 + 5 + 4 casos en el banco): se entrega al rival en el mismo punto
# saque de arco: los rivales quedan con cGroup = c0 (saca el rojo, zona x <= -1050) o c1 (saca el azul): no chocan con
# nadie ni con la pelota y no pueden entrar a la zona [medido: 68/68]

# ----------------------------------------------------------------------------- faltas (S2)
FOUL_CONTACT_TOL = 2.5     # rival "pegado": centros a <= 30 + 2,5 px [ajuste: 2–3 px igual de bueno]
FOUL_GAP = 6               # cortes de hasta 6 ticks no reinician la cuenta
FOUL_BALL_DIST = 25.0      # el poseedor tiene la pelota a <= 25 px de su borde (si no, se reinicia)
FOUL_TICKS = 115           # falta a los 115 ticks (1,9 s) [ajuste S2: 113, 38/50 faltas; S4: el árbitro en sombra cobraba 2 ticks antes]
FREEZE_TICKS = 14          # la pelota queda congelada 14 ticks antes del tiro libre [medido: 78, p10 14, p90 15]
# punto del tiro libre [S4]: falta, en la posición del jugador que la recibió (82% exacto en 11); offside, en la del
# adelantado + 15 px hacia el arco rival (+x el rojo, -x el azul; exacto en 44); penal: la pelota va directo al punto
OFFSIDE_SPOT_DX = 15.0
TOUCH_EPS = 0.5            # toque = pelota a <= 0,5 px del borde del jugador, o patada

# ----------------------------------------------------------------------------- offside (S2)
# Patada de un compañero; en ese tick el receptor está por delante del anteúltimo rival (arquero incluido), en campo
# rival y por delante de la pelota; se cobra cuando toca la pelota. Un toque de un rival sin patear no lo anula; una
# patada de cualquiera lo corta. Exentas las ejecuciones de lateral, córner y saque de arco.
# [medido: 120/138 cobros explicados, 91% de precisión]

# ----------------------------------------------------------------------------- powershot (S2)
PS_CONTACT_EPS = 3.0       # contacto para cargar: pelota a <= 3 px del borde del poseedor
PS_DELAY = 92              # la carga empieza en el tick 92 de contacto continuo: 91 seguidos antes del inicio [S4: con
                           # 3 px, p10 = p50 = p90 = 91 en 344 cargas, 98,5% en 88–92; con 2 px, p10 84]
PS_LEVELS = (2.43, 3.21, 4.0)   # niveles (33/66/100%) a +24, +209 y +359 ticks del inicio (tabla `ramp`)
PS_RESET_AFTER_KICK = 1    # la masa vuelve a 1,65 en el tick siguiente a la patada (p50)
# comba de la patada cargada: gravedad en y = -0,0171 × velocidad vertical del jugador, desde el tick siguiente, ×0,982
# por tick, hasta 150 ticks [medido: 894 patadas]
PS_CURVE_K = 0.0171
# Perfil de toda comba (powershot, córner, tiro libre, saque de arco; 2026-10-09, 7.800 patadas del banco): entera los
# primeros CURVE_HOLD_TICKS ticks y después ×0,981 por tick (+8: 0,961; +10: 0,923; +20: 0,767; +40: 0,52 en los cuatro).
CURVE_HOLD_TICKS = 6
PS_CURVE_DECAY = 0.981
PS_CURVE_TICKS = 150
# comba de la pelota parada [aproximado: falta la regresión de S2]: tiro libre y saque de arco igual que el powershot
# con k = 0,024; córner con gravedad en x = -0,021 × vx del jugador y 0,01 en y hacia la cancha
SP_CURVE_K = 0.024              # (antes, para los dos) tiro libre y saque de arco: ver FK/GK
FK_CURVE_K = 0.0215             # tiro libre: gy = −0,0215 · vy del que patea (698 tiros libres, 2026-10-09)
GK_CURVE_K = 0.025              # saque de arco: gy = −0,025 · vy (766)
# córner (2465, 2026-10-09; marco normalizado con la esquina en (+x, +y)): gx = −0,0241·vx + 0,0015·vy − 0,0044
# (R² 0,94) y gy = −0,01 fija hacia la cancha. vx, vy: velocidad del que patea.
CORNER_CURVE_KX = 0.0241
CORNER_CURVE_KY = 0.0015
CORNER_CURVE_X0 = 0.0044
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
    # duración del valor fijo, sorteada en cada saque con la distribución medida (6–9 ticks; la sala usa temporizadores
    # de reloj y varía ±1 tick): tabla acumulada por tipo, índice = ticks
    hold_cdf = np.ones((6, HOLD_MAX + 1))
    for kind, name in ((CORNER, "corner"), (FREE_KICK, "tiro_libre"), (GOAL_KICK, "saque_arco")):
        b = t["bar"][name]
        bar[kind] = b["wave"]
        hold[kind] = b["hold_value"]
        hold_ticks[kind] = int(b["hold_ticks"])
        dist = {int(h): f for h, f in b.get("hold_dist", {str(hold_ticks[kind]): 1.0}).items() if 6 <= int(h) <= 9}
        tot = sum(dist.values())
        acc = 0.0
        for h in range(HOLD_MAX + 1):
            acc += dist.get(h, 0.0) / tot
            hold_cdf[kind, h] = min(1.0, acc)
    bar[PENALTY] = PENALTY_INV
    hold[PENALTY] = PENALTY_INV
    bar[LATERAL] = LATERAL_INV
    hold[LATERAL] = LATERAL_INV
    ramp = np.array(t["ramp"]["values"], dtype=np.float64)
    return dict(bar=bar, hold=hold, hold_ticks=hold_ticks, hold_cdf=hold_cdf, ramp=ramp)


def fixed_hold_cdf(hold_ticks):
    """Tabla acumulada con toda la probabilidad en la duración típica (para comparar contra grabaciones)."""
    cdf = np.zeros((6, HOLD_MAX + 1))
    for kind in range(6):
        cdf[kind, int(hold_ticks[kind]):] = 1.0
    return cdf
