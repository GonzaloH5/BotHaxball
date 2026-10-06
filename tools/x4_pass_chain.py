"""Cadena del juego de pase, medida igual en grabaciones humanas y en simulación, con gate contra humanos.

posicionarse → generar línea de pase → desmarcarse → elegir el pase → ejecutarlo → anticipar la recepción →
controlar → orientarse → aprovechar la ventaja → seguir la jugada.

Entrada: tramos `tools.x4_metrics.Episode` (un muestreo cada 3 ticks = 0,05 s) y el modelo de valor `learn.x4_epv`
(φ = P(marca el rojo en 10 s) − P(marca el azul), aprendido de humanos). Todo se mide en el marco del equipo que tiene
la pelota (ataca hacia +x). Definiciones, todas con constantes de este módulo:

* Toques: dueño de la pelota en cada muestreo (`x4_metrics._touch_owner`: patada en la ventana o contacto). Una
  *corrida* es una serie de toques del mismo jugador sin toques de otro en el medio. Entre dos corridas de jugadores
  distintos: mismo equipo y la pelota recorrió ≥ MIN_PASS en juego abierto → pase completado (misma definición que
  `x4_metrics.possession_sequence`); otro equipo → pérdida.
* Intento de pase: corrida que termina con patada y la pelota sale dirigida (≤ AIM_DEG) a un compañero a distancia de
  pase. Si el siguiente toque es rival o la pelota sale de la cancha, es un pase fallido. Precisión = completados /
  (completados + fallidos). Patada dirigida al arco rival desde ≤ SHOT_DIST → tiro (no cuenta como pase).
* Línea de pase abierta (cono de intercepción): ningún rival tiene margen = d(rival, segmento) − (R + KAPPA·s) ≤ 0,
  donde s es la distancia desde la pelota al punto del segmento más cercano al rival y KAPPA ≈ velocidad del jugador /
  velocidad de la pelota pateada (el rival que está más lejos sobre la línea tiene más tiempo para cortarla).
  Compañero disponible = línea abierta y distancia entre MIN_PASS y MAX_PASS.
* Tipos de pase (no excluyentes): progresivo (acerca la pelota ≥ PROG_FRAC al arco rival y avanza ≥ PROG_MIN), rompe
  líneas (deja atrás ≥ 1 rival que estaba entre la pelota y su arco), atrás (descarga/apoyo, avanza ≤ −40), cambio de
  orientación (|Δy| ≥ SWITCH_DY y mayor que |Δx|), al espacio (el receptor corrió ≥ SPACE_RUN para llegar y el pase
  avanza), pared (A→B→A con B de primera), salida de presión (rival a ≤ PRESS_R del pasador al soltar y el equipo
  retiene RET_S), asistencia (último pase de una posesión que termina en gol a ≤ ASSIST_S de la recepción).
* Recepción: retención (el equipo no pierde la pelota en RET_S después de recibir), bajo presión (rival a ≤ PRESS_R
  del receptor al recibir), de primera (el receptor la suelta en ≤ FIRST_TIME_S con pase o tiro), control orientado
  (en 1 s la pelota avanza ≥ 30 px sin perderla), anticipación (componente de la velocidad del receptor hacia el punto
  de recepción durante el vuelo, en unidades de la velocidad máxima).
* Aprovechar la ventaja: después de recibir un pase progresivo o que rompe líneas, avance máximo de la pelota en
  EXPLOIT_S mientras el equipo la conserva, y si hay tiro en ASSIST_S.
* Posesión: corridas seguidas del mismo equipo. Valor = φ al final − φ al principio (1 si termina en gol). Circulación
  inútil: ≥ 3 pases, avance < 100 px y valor ≤ 0.
* Estado (cada STATE_EVERY muestreos, juego abierto, el equipo controla la pelota): compañeros disponibles, línea
  progresiva disponible, apoyo (disponible a ≤ 300 px y no adelantado), separación de la marca, control del espacio
  (fracción de la cancha más cerca de un atacante que de un defensor) y presión sobre el portador.
* Desmarque: un compañero sin pelota pasa de no disponible (≥ 0,5 s) a disponible mientras corre; útil si recibe un
  pase en 2 s. Reofrecerse: el pasador vuelve a estar disponible (o avanza 50 px) en 1,5 s después de pasar.
* Decisión: al empezar cada corrida con control, contexto (presión, línea progresiva, tercio) y lo que hizo: pase,
  tiro, conducción (≥ CARRY_MIN), aguantar o perderla; valor = Δφ hasta 1 s después de terminar la corrida.

Las tasas se agregan como Σ eventos / Σ minutos (o Σ / Σ) y el intervalo del 90% sale de un bootstrap por unidad
(grabación humana o partido simulado). El gate compara contra la referencia humana: las métricas de eficacia tienen
que llegar al promedio humano; las de estilo, quedar dentro de la dispersión humana (p10–p90 entre unidades).

  python -m tools.x4_pass_chain --human --out reports/x4/pass_chain_human.json
  python -m tools.x4_pass_chain --policy runs/x4_bc/final_sangu_rsone/best.pt --matches 32 --minutes 3 \\
      --out reports/x4/pass_chain_bc_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools import x4_metrics as XM

ROOT = Path(__file__).resolve().parent.parent
DT = 3.0 / 60.0               # s por muestreo
SX, SY = 1150.0, 670.0
GOAL_HH = 124.0
MIN_PASS = XM.MIN_PASS        # 40 px
MAX_PASS = 900.0
KAPPA = 0.3
PRESS_R = 75.0                # ~0,5 s de carrera de un rival (velocidad máxima ≈ 2,5 px/tick)
VMAX = 2.5
AIM_DEG = 20.0
SHOT_DIST = 800.0
PROG_FRAC, PROG_MIN = 0.25, 60.0
SWITCH_DY = 400.0
SPACE_RUN = 80.0
CARRY_MIN = 100.0
FIRST_TIME_S = 0.35
RET_S = 2.0
EXPLOIT_S = 3.0
ASSIST_S = 5.0
STATE_EVERY = 5
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
GRID = np.stack(np.meshgrid(np.linspace(-SX, SX, 21), np.linspace(-SY, SY, 13)), -1).reshape(-1, 2)

# ------------------------------------------------------------------------------- catálogo de métricas
# (nombre, etapa, tipo): "mas" = eficacia, el agente tiene que llegar al promedio humano; "menos" = al revés;
# "banda" = estilo, dentro de p10–p90 de las unidades humanas.
METRICS = [
    ("lineas_disponibles", "posicionamiento y líneas", "mas"),
    ("linea_progresiva_frac", "posicionamiento y líneas", "mas"),
    ("apoyo_frac", "posicionamiento y líneas", "mas"),
    ("control_espacio", "posicionamiento y líneas", "mas"),
    ("separacion_media", "posicionamiento y líneas", "banda"),
    ("desmarques_por_min_posesion", "desmarques y apoyo", "mas"),
    ("desmarques_utiles_frac", "desmarques y apoyo", "mas"),
    ("reofrece_tras_pasar_frac", "desmarques y apoyo", "mas"),
    ("pases_por_min", "elección y ejecución", "mas"),
    ("precision_pase", "elección y ejecución", "mas"),
    ("precision_bajo_presion", "elección y ejecución", "mas"),
    ("progresivos_por_min", "elección y ejecución", "mas"),
    ("rompe_lineas_por_min", "elección y ejecución", "mas"),
    ("al_espacio_por_min", "elección y ejecución", "mas"),
    ("cambios_orientacion_por_min", "elección y ejecución", "mas"),
    ("paredes_por_min", "elección y ejecución", "mas"),
    ("salidas_presion_por_min", "elección y ejecución", "mas"),
    ("asistencias_por_min", "elección y ejecución", "mas"),
    ("epv_por_intento", "elección y ejecución", "mas"),
    ("atras_frac", "elección y ejecución", "banda"),
    ("largo_pase_p50", "elección y ejecución", "banda"),
    ("velocidad_pase_p50", "elección y ejecución", "banda"),
    ("retencion_tras_recibir", "recepción y control", "mas"),
    ("retencion_bajo_presion", "recepción y control", "mas"),
    ("control_orientado_frac", "recepción y control", "mas"),
    ("anticipacion_receptor", "recepción y control", "mas"),
    ("de_primera_frac", "recepción y control", "banda"),
    ("tiempo_con_pelota_p50", "recepción y control", "banda"),
    ("progreso_tras_pase_avance", "aprovechar la ventaja", "mas"),
    ("tiro_tras_pase_avance_frac", "aprovechar la ventaja", "mas"),
    ("devolucion_inmediata_frac", "aprovechar la ventaja", "banda"),
    ("epv_por_posesion", "posesión y valor", "mas"),
    ("pases_por_posesion", "posesión y valor", "mas"),
    ("circulacion_inutil_frac", "posesión y valor", "menos"),
    ("tiros_por_min", "posesión y valor", "mas"),
    ("epv_por_decision", "decisión", "mas"),
    ("perdida_sin_pase_bajo_presion", "decisión", "menos"),
    ("pase_con_linea_progresiva", "decisión", "banda"),
    ("presion_al_portador_p50", "contexto defensivo", "banda"),
]
METRIC_KIND = {m: k for m, _, k in METRICS}
METRIC_STAGE = {m: s for m, s, _ in METRICS}


# ------------------------------------------------------------------------------- geometría
def lane_margin(ball, mate, opps, r_int):
    """Margen mínimo de los rivales sobre la línea pelota→compañero (cono de intercepción). ball (N, 2), mate (N, 2),
    opps (N, K, 2) → (N,). > 0 = línea abierta."""
    seg = mate - ball
    L2 = np.maximum((seg ** 2).sum(-1), 1e-9)
    rel = opps - ball[:, None]
    u = np.clip((rel * seg[:, None]).sum(-1) / L2[:, None], 0.0, 1.0)
    closest = ball[:, None] + u[..., None] * seg[:, None]
    d = np.hypot(*(opps - closest).transpose(2, 0, 1))
    s = u * np.sqrt(L2)[:, None]
    return (d - (r_int + KAPPA * s)).min(-1)


def availability(ep, t_idx, team):
    """(n, 4) compañeros del equipo `team` disponibles para un pase desde la pelota en los muestreos t_idx."""
    sl = slice(0, 4) if team == 0 else slice(4, 8)
    osl = slice(4, 8) if team == 0 else slice(0, 4)
    ball = ep.ball[t_idx, :2]
    mates = ep.pos[t_idx, sl]
    opps = ep.pos[t_idx, osl]
    r_int = XM.PLAYER_R + ep.ball_r
    out = np.zeros((len(t_idx), 4), bool)
    for q in range(4):
        dist = np.hypot(*(mates[:, q] - ball).T)
        m = lane_margin(ball, mates[:, q], opps, r_int)
        out[:, q] = (m > 0) & (dist >= MIN_PASS) & (dist <= MAX_PASS)
    return out


# ------------------------------------------------------------------------------- eventos
def runs_of(owner):
    """Corridas (inicio, fin, lugar) de toques limpios; los muestreos sin dueño no cortan la corrida."""
    runs = []
    for i in np.flatnonzero(owner >= 0):
        o = int(owner[i])
        if runs and runs[-1][2] == o:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i), o])
    return runs


def _aim(ep, r, a):
    """(compañero apuntado o -1, es_tiro) para la patada de `a` que suelta la pelota en el muestreo r."""
    T = len(ep.ball)
    team = TEAM[a]
    s = 1.0 if team == 0 else -1.0
    j = min(r + 1, T - 1)
    v = ep.ball[j, 2:4] * s
    if np.hypot(*v) < 1.0:
        return -1, False
    b = ep.ball[r, :2] * s
    if v[0] > 0:
        y_hit = b[1] + v[1] * (SX - b[0]) / v[0]
        if abs(y_hit) <= GOAL_HH + 10.0 and np.hypot(SX - b[0], b[1]) <= SHOT_DIST:
            return -1, True
    best, best_ang = -1, AIM_DEG
    for q in range(8):
        if TEAM[q] != team or q == a:
            continue
        w = ep.pos[r, q] * s - b
        d = np.hypot(*w)
        if d < MIN_PASS or d > MAX_PASS:
            continue
        ang = np.degrees(np.arccos(np.clip((w @ v) / (d * np.hypot(*v)), -1.0, 1.0)))
        if ang <= best_ang:
            best, best_ang = q, ang
    return best, False


def chain_events(ep, phi=None):
    """Eventos de la cadena de un tramo. `phi`: φ por muestreo (rojo); si falta, las métricas de valor quedan en 0."""
    T = len(ep.ball)
    owner, dist = XM._touch_owner(ep)
    phi = np.zeros(T) if phi is None else np.asarray(phi, float)
    open_ = ep.open_play
    goal_ev = ep.goal_ev if ep.goal_ev is not None else np.zeros(T, np.int8)
    rteam = ep.restart_team if ep.restart_team is not None else np.full(T, -1)
    runs = runs_of(owner)
    sgn = lambda team: 1.0 if team == 0 else -1.0

    # momento en que el equipo `team` pierde la pelota después de t: toque rival, saque para el rival o gol en contra
    opp_touch = {0: np.flatnonzero((owner >= 4)), 1: np.flatnonzero((owner >= 0) & (owner < 4))}
    rival_restart = {0: np.flatnonzero(rteam == 1), 1: np.flatnonzero(rteam == 0)}
    goal_against = {0: np.flatnonzero(goal_ev < 0), 1: np.flatnonzero(goal_ev > 0)}
    goal_for = {0: np.flatnonzero(goal_ev > 0), 1: np.flatnonzero(goal_ev < 0)}

    def next_after(arr, t):
        k = np.searchsorted(arr, t, side="right")
        return int(arr[k]) if k < len(arr) else T

    def loss_time(team, t):
        return min(next_after(opp_touch[team], t), next_after(rival_restart[team], t), next_after(goal_against[team], t))

    passes, attempts_failed, shots = [], [], []
    run_info = []
    for k, (a0, b0, o) in enumerate(runs):
        nxt = runs[k + 1] if k + 1 < len(runs) else None
        run_info.append(dict(a=a0, b=b0, o=o, next=nxt))
        if nxt is None:
            continue
        a1, b1, o1 = nxt
        team = TEAM[o]
        kicked = ep.kicked[max(a0, b0 - 1):b0 + 1, o].any()
        aim, is_shot = _aim(ep, b0, o) if kicked else (-1, False)
        travel = float(np.hypot(*(ep.ball[a1, :2] - ep.ball[b0, :2])))
        if TEAM[o1] == team and o1 != o and travel >= MIN_PASS and open_[b0:a1 + 1].all():
            passes.append(dict(r=b0, c=a1, a=o, b=o1, team=team, run_b=(a1, b1), next_run=k + 1, aimed=aim >= 0,
                               shot_like=is_shot, kicked=bool(kicked)))
        elif TEAM[o1] != team:
            if is_shot:
                shots.append(dict(r=b0, team=team, a=o))
            elif aim >= 0:
                attempts_failed.append(dict(r=b0, a=o, team=team, end=a1, aim=aim))
        elif is_shot:
            shots.append(dict(r=b0, team=team, a=o))
    # tiros que terminan en gol sin otro toque (la corrida siguiente no existe o es después del gol)
    for k, (a0, b0, o) in enumerate(runs):
        if runs[k + 1:k + 2]:
            continue
        team = TEAM[o]
        if ep.kicked[max(a0, b0 - 1):b0 + 1, o].any():
            aim, is_shot = _aim(ep, b0, o)
            if is_shot:
                shots.append(dict(r=b0, team=team, a=o))
    return dict(runs=run_info, passes=passes, failed=attempts_failed, shots=shots, owner=owner, dist=dist, phi=phi,
                loss_time=loss_time, goal_for=goal_for, next_after=next_after, sgn=sgn)


# ------------------------------------------------------------------------------- conteos por unidad
def _new_counts():
    return dict(minutes=0.0, open_minutes=0.0, ctrl_samples=0, state_samples=0,
                avail_sum=0.0, prog_lane=0, support=0, space_sum=0.0, sep_sum=0.0, sep_n=0, press_carrier=[],
                desm=0, desm_util=0, reofrece=0, reofrece_n=0,
                passes=0, failed=0, press_att=0, press_ok=0, prog=0, lines=0, space=0, switch=0, back=0, wall=0,
                salida=0, assist=0, epv_att_sum=0.0, pass_len=[], pass_speed=[],
                rec=0, rec_ret=0, rec_press=0, rec_press_ret=0, oriented=0, first_time=0, antic_sum=0.0, hold_s=[],
                adv_n=0, adv_prog_sum=0.0, adv_shot=0, quick_back=0, next_pass_n=0,
                poss=0, poss_val_sum=0.0, poss_passes=0, useless=0, poss3=0, shots=0, goals=0,
                dec=0, dec_val_sum=0.0, dec_press=0, dec_press_lost=0, dec_prog=0, dec_prog_pass=0)


def unit_counts(episodes, epv=None, team=None):
    """Totales de un conjunto de tramos (una unidad de bootstrap). `team` (0 rojo, 1 azul): sólo los eventos de ese
    equipo (para medir a una política contra un rival fijo); None = los dos."""
    c = _new_counts()
    for ep in episodes:
        _add_episode(c, ep, epv, team)
    return c


def _add_episode(c, ep, epv, only=None):
    T = len(ep.ball)
    if T < 2:
        return
    phi = epv.phi_episode(ep) if epv is not None else np.zeros(T)
    E = chain_events(ep, phi)
    owner, dist = E["owner"], E["dist"]
    loss_time, sgn = E["loss_time"], E["sgn"]
    open_ = ep.open_play
    # con un solo equipo, las tasas por minuto se cuentan por "medio partido" (comparables con las de los dos equipos)
    share = 1.0 if only is None else 0.5
    c["minutes"] += share * T * DT / 60.0
    c["open_minutes"] += share * float(open_.sum()) * DT / 60.0
    if ep.goal_ev is None:
        c["goals"] += int(ep.goals)
    elif only is None:
        c["goals"] += int(np.abs(ep.goal_ev).sum())
    else:
        c["goals"] += int((ep.goal_ev == (1 if only == 0 else -1)).sum())
    r_ctrl = XM.PLAYER_R + ep.ball_r + 15.0

    # --- posesión controlada por muestreo: equipo y portador de la última corrida
    carrier = np.full(T, -1)
    last = -1
    for i in range(T):
        if owner[i] >= 0:
            last = int(owner[i])
        carrier[i] = last
    ctrl = (carrier >= 0) & open_
    ctrl &= dist[np.arange(T), np.maximum(carrier, 0)] <= r_ctrl
    c["ctrl_samples"] += int(ctrl.sum()) if only is None else int((ctrl & (TEAM[np.maximum(carrier, 0)] == only)).sum())

    # --- disponibilidad de compañeros en cada muestreo con control (para desmarques) y muestras de estado
    avail = np.zeros((T, 8), bool)
    for team in (0, 1):
        idx = np.flatnonzero(ctrl & (TEAM[np.maximum(carrier, 0)] == team))
        if len(idx):
            sl = slice(0, 4) if team == 0 else slice(4, 8)
            av = availability(ep, idx, team)
            avail[idx, sl] = av
            avail[idx, carrier[idx]] = False
    for team in (0, 1):
        if only is not None and team != only:
            continue
        idx = np.flatnonzero(ctrl & (TEAM[np.maximum(carrier, 0)] == team))[::STATE_EVERY]
        if not len(idx):
            continue
        s = sgn(team)
        sl = slice(0, 4) if team == 0 else slice(4, 8)
        osl = slice(4, 8) if team == 0 else slice(0, 4)
        av = avail[idx, sl]
        bx = ep.ball[idx, 0] * s
        mx = ep.pos[idx, sl, 0] * s
        dball = np.hypot(ep.pos[idx, sl, 0] - ep.ball[idx, None, 0], ep.pos[idx, sl, 1] - ep.ball[idx, None, 1])
        c["state_samples"] += len(idx)
        c["avail_sum"] += float(av.sum())
        c["prog_lane"] += int((av & (mx > bx[:, None] + 50.0)).any(1).sum())
        c["support"] += int((av & (dball <= 300.0) & (mx <= bx[:, None] + 50.0)).any(1).sum())
        mates, opps = ep.pos[idx, sl], ep.pos[idx, osl]
        dmo = np.hypot(mates[:, :, None, 0] - opps[:, None, :, 0], mates[:, :, None, 1] - opps[:, None, :, 1]).min(-1)
        offb = np.ones_like(av)
        offb[np.arange(len(idx)), carrier[idx] - sl.start] = False
        c["sep_sum"] += float(dmo[offb].sum())
        c["sep_n"] += int(offb.sum())
        car = ep.pos[idx, carrier[idx]]
        c["press_carrier"] += list(np.hypot(opps[..., 0] - car[:, None, 0], opps[..., 1] - car[:, None, 1]).min(1))
        dg_m = np.hypot(GRID[None, :, None, 0] - mates[:, None, :, 0], GRID[None, :, None, 1] - mates[:, None, :, 1]).min(-1)
        dg_o = np.hypot(GRID[None, :, None, 0] - opps[:, None, :, 0], GRID[None, :, None, 1] - opps[:, None, :, 1]).min(-1)
        c["space_sum"] += float((dg_m < dg_o).mean(1).sum())

    # --- desmarques: compañero sin pelota que pasa a disponible (tras ≥ 0,5 s sin estarlo) mientras corre
    received_at = {}
    for p in E["passes"]:
        received_at.setdefault(p["b"], []).append(p["c"])
    speed = np.hypot(ep.vel[..., 0], ep.vel[..., 1])
    for q in range(8):
        if only is not None and TEAM[q] != only:
            continue
        closed_for = 0
        for i in range(1, T):
            if not ctrl[i] or TEAM[carrier[i]] != TEAM[q] or carrier[i] == q:
                closed_for = 0
                continue
            if avail[i, q]:
                if closed_for >= 10 and speed[i, q] >= 1.2:
                    c["desm"] += 1
                    if any(i < t <= i + int(2.0 / DT) for t in received_at.get(q, ())):
                        c["desm_util"] += 1
                closed_for = 0
            else:
                closed_for += 1

    # --- pases completados
    passes = E["passes"]
    for k, p in enumerate(passes):
        r, cc, a, b, team = p["r"], p["c"], p["a"], p["b"], p["team"]
        if only is not None and team != only:
            continue
        s = sgn(team)
        osl = slice(4, 8) if team == 0 else slice(0, 4)
        br, bc = ep.ball[r, :2] * s, ep.ball[cc, :2] * s
        dx, dy = bc - br
        length = float(np.hypot(dx, dy))
        dg_r, dg_c = np.hypot(SX - br[0], br[1]), np.hypot(SX - bc[0], bc[1])
        prog = dg_c <= (1 - PROG_FRAC) * dg_r and dx >= PROG_MIN
        before = int((ep.pos[r, osl, 0] * s > br[0]).sum())
        after = int((ep.pos[cc, osl, 0] * s > bc[0]).sum())
        lines = before - after >= 1
        run_b = float(np.hypot(*(ep.pos[cc, b] - ep.pos[r, b])))
        press_a = float(np.hypot(*(ep.pos[r, osl] - ep.pos[r, a]).T).min())
        press_b = float(np.hypot(*(ep.pos[cc, osl] - ep.pos[cc, b]).T).min())
        lt = loss_time(team, cc)
        retained = lt - cc > int(RET_S / DT)
        c["passes"] += 1
        c["pass_len"].append(length)
        c["pass_speed"].append(float(np.hypot(*ep.ball[r:min(r + 3, cc + 1), 2:4].T).max()))
        c["prog"] += prog
        c["lines"] += lines
        c["back"] += dx <= -40.0
        c["switch"] += abs(dy) >= SWITCH_DY and abs(dy) > abs(dx)
        c["space"] += run_b >= SPACE_RUN and dx > 0
        if press_a <= PRESS_R:
            c["press_att"] += 1
            c["press_ok"] += 1
            c["salida"] += retained
        c["epv_att_sum"] += s * (E["phi"][cc] - E["phi"][r])
        # recepción
        c["rec"] += 1
        c["rec_ret"] += retained
        if press_b <= PRESS_R:
            c["rec_press"] += 1
            c["rec_press_ret"] += retained
        a1, b1 = p["run_b"]
        hold = (b1 - a1) * DT
        c["hold_s"].append(hold)
        nxt = next((q for q in passes[k + 1:k + 2] if q["r"] == b1), None)
        is_shot_next = any(sh["r"] == b1 for sh in E["shots"])
        if hold <= FIRST_TIME_S and (nxt is not None or is_shot_next or any(f["r"] == b1 for f in E["failed"])):
            c["first_time"] += 1
        w = min(cc + int(1.0 / DT), T - 1)
        if lt > w and (ep.ball[w, 0] - ep.ball[cc, 0]) * s >= 30.0:
            c["oriented"] += 1
        if cc > r:
            fl = np.arange(r, cc)
            to = ep.ball[cc, :2] - ep.pos[fl, b]
            n = np.maximum(np.hypot(*to.T), 1e-6)
            c["antic_sum"] += float(np.mean((ep.vel[fl, b] * to).sum(-1) / n) / VMAX)
        # devolución inmediata: el siguiente pase sale en ≤ 1 s y va hacia atrás
        if nxt is not None:
            c["next_pass_n"] += 1
            if (nxt["r"] - cc) * DT <= 1.0 and (ep.ball[nxt["c"], 0] - ep.ball[nxt["r"], 0]) * s <= -40.0:
                c["quick_back"] += 1
            # pared: A→B y B→A con B de primera
            if nxt["b"] == a and hold <= 0.5 and (nxt["c"] - r) * DT <= 3.0:
                c["wall"] += 1
        # aprovechar la ventaja
        if prog or lines:
            c["adv_n"] += 1
            end = min(cc + int(EXPLOIT_S / DT), lt, T - 1)
            if end > cc:
                c["adv_prog_sum"] += max(0.0, float((ep.ball[cc + 1:end + 1, 0] * s).max() - bc[0]))
            if any(cc <= sh["r"] <= cc + int(ASSIST_S / DT) and sh["team"] == team for sh in E["shots"]):
                c["adv_shot"] += 1
        # reofrecerse: el pasador vuelve a estar disponible o avanza 50 px en 1,5 s
        w2 = min(r + int(1.5 / DT), T - 1)
        c["reofrece_n"] += 1
        if avail[r + 1:w2 + 1, a].any() or (ep.pos[w2, a, 0] - ep.pos[r, a, 0]) * s >= 50.0:
            c["reofrece"] += 1
        # asistencia: el equipo marca en ≤ ASSIST_S sin perderla y no hay otro pase completado antes del gol
        g = E["next_after"](E["goal_for"][team], cc)
        if g < T and g - cc <= int(ASSIST_S / DT) and lt > g:
            later = [q for q in passes[k + 1:] if q["team"] == team and cc < q["c"] <= g]
            if not later:
                c["assist"] += 1
    # pases fallidos
    for f in E["failed"]:
        team = f["team"]
        if only is not None and team != only:
            continue
        s = sgn(team)
        c["failed"] += 1
        osl = slice(4, 8) if team == 0 else slice(0, 4)
        if np.hypot(*(ep.pos[f["r"], osl] - ep.pos[f["r"], f["a"]]).T).min() <= PRESS_R:
            c["press_att"] += 1
        end = min(f["end"], T - 1)
        c["epv_att_sum"] += s * (E["phi"][end] - E["phi"][f["r"]])
    c["shots"] += sum(1 for sh in E["shots"] if only is None or sh["team"] == only)

    # --- posesiones (corridas seguidas del mismo equipo)
    runs = E["runs"]
    i = 0
    while i < len(runs):
        team = TEAM[runs[i]["o"]]
        j = i
        while j + 1 < len(runs) and TEAM[runs[j + 1]["o"]] == team:
            j += 1
        start = runs[i]["a"]
        if only is not None and team != only:
            i = j + 1
            continue
        lt = loss_time(team, runs[j]["b"])
        g = E["next_after"](E["goal_for"][team], start)
        scored = g < T and g <= lt
        if j + 1 < len(runs) or scored:
            s = sgn(team)
            end = min(g if scored else lt, T - 1)
            val = 1.0 - s * E["phi"][start] if scored else s * (E["phi"][end] - E["phi"][start])
            n_p = sum(1 for p in passes if p["team"] == team and start <= p["r"] and p["c"] <= runs[j]["b"])
            seg = ep.ball[start:end + 1, 0] * s
            prog = float(seg.max() - seg[0]) if len(seg) else 0.0
            c["poss"] += 1
            c["poss_val_sum"] += val
            c["poss_passes"] += n_p
            if n_p >= 3:
                c["poss3"] += 1
                if prog < 100.0 and val <= 0 and not scored:
                    c["useless"] += 1
        i = j + 1

    # --- decisiones al empezar cada corrida con control
    pass_by_r = {p["r"]: p for p in passes}
    fail_by_r = {f["r"]: f for f in E["failed"]}
    shot_r = {sh["r"] for sh in E["shots"]}
    for ri in runs:
        a0, b0, o = ri["a"], ri["b"], ri["o"]
        if not (ctrl[a0] or (a0 + 1 < T and ctrl[a0 + 1])) or not open_[a0]:
            continue
        team = TEAM[o]
        if only is not None and team != only:
            continue
        s = sgn(team)
        osl = slice(4, 8) if team == 0 else slice(0, 4)
        sl = slice(0, 4) if team == 0 else slice(4, 8)
        pressed = np.hypot(*(ep.pos[a0, osl] - ep.pos[a0, o]).T).min() <= PRESS_R
        mx = ep.pos[a0, sl, 0] * s
        prog_lane = bool((avail[a0, sl] & (mx > ep.ball[a0, 0] * s + 50.0)).any())
        if b0 in pass_by_r or b0 in fail_by_r:
            dec = "pase"
        elif b0 in shot_r:
            dec = "tiro"
        elif ri["next"] is not None and TEAM[ri["next"][2]] != team:
            dec = "perdida"
        elif np.hypot(*(ep.ball[b0, :2] - ep.ball[a0, :2])) >= CARRY_MIN:
            dec = "conduce"
        else:
            dec = "aguanta"
        end = min(b0 + int(1.0 / DT), T - 1)
        c["dec"] += 1
        c["dec_val_sum"] += s * (E["phi"][end] - E["phi"][a0])
        if pressed:
            c["dec_press"] += 1
            c["dec_press_lost"] += dec == "perdida"
        if prog_lane:
            c["dec_prog"] += 1
            c["dec_prog_pass"] += dec == "pase"


# ------------------------------------------------------------------------------- tasas, bootstrap y gate
def rates(units):
    S = lambda k: sum(u[k] for u in units)
    L = lambda k: [x for u in units for x in u[k]]
    div = lambda a, b: a / b if b else float("nan")
    om = S("open_minutes")
    poss_min = S("ctrl_samples") * DT / 60.0
    lens, speeds, holds, press = L("pass_len"), L("pass_speed"), L("hold_s"), L("press_carrier")
    att = S("passes") + S("failed")
    return dict(
        lineas_disponibles=div(S("avail_sum"), S("state_samples")),
        linea_progresiva_frac=div(S("prog_lane"), S("state_samples")),
        apoyo_frac=div(S("support"), S("state_samples")),
        control_espacio=div(S("space_sum"), S("state_samples")),
        separacion_media=div(S("sep_sum"), S("sep_n")),
        desmarques_por_min_posesion=div(S("desm"), poss_min),
        desmarques_utiles_frac=div(S("desm_util"), S("desm")),
        reofrece_tras_pasar_frac=div(S("reofrece"), S("reofrece_n")),
        pases_por_min=div(S("passes"), om),
        precision_pase=div(S("passes"), att),
        precision_bajo_presion=div(S("press_ok"), S("press_att")),
        progresivos_por_min=div(S("prog"), om),
        rompe_lineas_por_min=div(S("lines"), om),
        al_espacio_por_min=div(S("space"), om),
        cambios_orientacion_por_min=div(S("switch"), om),
        paredes_por_min=div(S("wall"), om),
        salidas_presion_por_min=div(S("salida"), om),
        asistencias_por_min=div(S("assist"), om),
        epv_por_intento=div(S("epv_att_sum"), att),
        atras_frac=div(S("back"), S("passes")),
        largo_pase_p50=float(np.median(lens)) if lens else float("nan"),
        velocidad_pase_p50=float(np.median(speeds)) if speeds else float("nan"),
        retencion_tras_recibir=div(S("rec_ret"), S("rec")),
        retencion_bajo_presion=div(S("rec_press_ret"), S("rec_press")),
        control_orientado_frac=div(S("oriented"), S("rec")),
        anticipacion_receptor=div(S("antic_sum"), S("rec")),
        de_primera_frac=div(S("first_time"), S("rec")),
        tiempo_con_pelota_p50=float(np.median(holds)) if holds else float("nan"),
        progreso_tras_pase_avance=div(S("adv_prog_sum"), S("adv_n")),
        tiro_tras_pase_avance_frac=div(S("adv_shot"), S("adv_n")),
        devolucion_inmediata_frac=div(S("quick_back"), S("next_pass_n")),
        epv_por_posesion=div(S("poss_val_sum"), S("poss")),
        pases_por_posesion=div(S("poss_passes"), S("poss")),
        circulacion_inutil_frac=div(S("useless"), S("poss3")),
        tiros_por_min=div(S("shots"), om),
        epv_por_decision=div(S("dec_val_sum"), S("dec")),
        perdida_sin_pase_bajo_presion=div(S("dec_press_lost"), S("dec_press")),
        pase_con_linea_progresiva=div(S("dec_prog_pass"), S("dec_prog")),
        presion_al_portador_p50=float(np.median(press)) if press else float("nan"),
        # contexto (no entran en el gate)
        minutos=S("minutes"), minutos_juego_abierto=om, pases=S("passes"), intentos=att, goles_por_min=div(S("goals"), om),
    )


# soporte de cada métrica: ("tasa", exposición) para eventos por minuto (se exige que con el ritmo humano se esperen
# ≥ N_MIN eventos en esa exposición: así un 0 del agente sí cuenta como déficit) o ("n", denominador) para fracciones,
# medias y medianas (≥ N_MIN observaciones). Las métricas de estado muestrean cada 0,25 s: se cuenta 1 de cada 4.
N_MIN = 20
_RATE_OPEN = ("pases_por_min", "progresivos_por_min", "rompe_lineas_por_min", "al_espacio_por_min",
              "cambios_orientacion_por_min", "paredes_por_min", "salidas_presion_por_min", "asistencias_por_min",
              "tiros_por_min")
_DEN = dict(lineas_disponibles="state4", linea_progresiva_frac="state4", apoyo_frac="state4", control_espacio="state4",
            separacion_media="state4", desmarques_utiles_frac="desm", reofrece_tras_pasar_frac="reofrece_n",
            precision_pase="attempts", precision_bajo_presion="press_att", epv_por_intento="attempts",
            atras_frac="passes", largo_pase_p50="passes", velocidad_pase_p50="passes", retencion_tras_recibir="rec",
            retencion_bajo_presion="rec_press", control_orientado_frac="rec", anticipacion_receptor="rec",
            de_primera_frac="rec", tiempo_con_pelota_p50="rec", progreso_tras_pase_avance="adv_n",
            tiro_tras_pase_avance_frac="adv_n", devolucion_inmediata_frac="next_pass_n", epv_por_posesion="poss",
            pases_por_posesion="poss", circulacion_inutil_frac="poss3", epv_por_decision="dec",
            perdida_sin_pase_bajo_presion="dec_press", pase_con_linea_progresiva="dec_prog",
            presion_al_portador_p50="state4")


def support(units):
    """Exposición (minutos de juego abierto y de posesión) y denominadores de cada métrica."""
    S = lambda k: sum(u[k] for u in units)
    den = dict(state4=S("state_samples") / 4.0, attempts=S("passes") + S("failed"))
    for k in ("desm", "reofrece_n", "press_att", "passes", "rec", "rec_press", "adv_n", "next_pass_n", "poss", "poss3",
              "dec", "dec_press", "dec_prog"):
        den[k] = S(k)
    return dict(open_minutes=S("open_minutes"), poss_minutes=S("ctrl_samples") * DT / 60.0, den=den)


def reliable(metric, sup, human_value):
    """¿Hay datos suficientes para juzgar la métrica?"""
    if sup is None:
        return True
    if metric in _RATE_OPEN:
        return human_value is not None and human_value * sup["open_minutes"] >= N_MIN
    if metric == "desmarques_por_min_posesion":
        return human_value is not None and human_value * sup["poss_minutes"] >= N_MIN
    return sup["den"].get(_DEN.get(metric), 0) >= N_MIN


def bootstrap(units, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    point = rates(units)
    draws = {k: [] for k in point}
    for _ in range(n):
        r = rates([units[i] for i in rng.integers(0, len(units), len(units))])
        for k, v in r.items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        d = np.asarray(draws[k], float)
        d = d[np.isfinite(d)]
        out[k] = dict(valor=_r(v), ic90=[_r(np.percentile(d, 5)), _r(np.percentile(d, 95))] if len(d) else [None, None])
    return out


def unit_spread(units):
    """p10 y p90 de cada métrica entre unidades (para las métricas de estilo)."""
    per = [rates([u]) for u in units]
    out = {}
    for m, _, _ in METRICS:
        v = np.array([p[m] for p in per], float)
        v = v[np.isfinite(v)]
        out[m] = [_r(np.percentile(v, 10)), _r(np.percentile(v, 90))] if len(v) else [None, None]
    return out


def _r(x):
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None


def _ratio(kind, a, h):
    if a is None or h in (None, 0):
        return None
    r = a / h if kind == "mas" else h / a if a else 2.0
    return float(np.clip(r, 0.25, 2.0))


def chain_index(values, human, sup=None):
    """Índice de la cadena: media geométrica de agente/humano (invertida en las "menos") sobre las métricas de eficacia
    con datos suficientes, cada razón en [0,25; 2]. 1 = promedio humano."""
    logs = []
    for m, _, kind in METRICS:
        if kind == "banda" or not reliable(m, sup, human[m]["valor"]):
            continue
        r = _ratio(kind, values.get(m), human[m]["valor"])
        if r is not None:
            logs.append(np.log(r))
    return float(np.exp(np.mean(logs))) if logs else None


def index_ci(units, human, n=300, seed=0):
    """(índice, p5, p95) por bootstrap de unidades."""
    rng = np.random.default_rng(seed)
    sup = support(units)
    point = chain_index(rates(units), human, sup)
    draws = []
    for _ in range(n):
        pick = [units[i] for i in rng.integers(0, len(units), len(units))]
        v = chain_index(rates(pick), human, sup)
        if v is not None:
            draws.append(v)
    if point is None or not draws:
        return None, None, None
    return round(point, 4), round(float(np.percentile(draws, 5)), 4), round(float(np.percentile(draws, 95)), 4)


def gate(agent, human, spread, sup=None, strict=False):
    """Veredicto por métrica y global. `agent` y `human`: salidas de `bootstrap`; `spread`: `unit_spread` humano;
    `sup`: `support` de las unidades del agente. Eficacia: valor del agente ≥ promedio humano ("mas") o ≤ ("menos").
    Estilo: dentro de p10–p90 humano. Una métrica sin datos suficientes no se aprueba; `strict` exige que todas tengan
    datos (certificación); si no, alcanza con el 80% (evaluaciones del entrenamiento, más chicas)."""
    rows, by_stage = {}, {}
    for m, stage, kind in METRICS:
        a, h = agent[m]["valor"], human[m]["valor"]
        enough = reliable(m, sup, h)
        if a is None or h is None:
            ok = False
        elif kind == "mas":
            ok = a >= h
        elif kind == "menos":
            ok = a <= h
        else:
            lo, hi = spread[m]
            ok = lo is not None and lo <= a <= hi
        rows[m] = dict(etapa=stage, tipo=kind, agente=a, agente_ic90=agent[m]["ic90"], humano=h,
                       humano_ic90=human[m]["ic90"], banda_humana=spread[m] if kind == "banda" else None,
                       razon=_r(a / h) if a is not None and h not in (None, 0) else None, ok=bool(ok and enough),
                       datos_suficientes=bool(enough))
        st = by_stage.setdefault(stage, [0, 0])
        st[0] += bool(ok and enough)
        st[1] += 1
    judged = [m for m in rows if rows[m]["datos_suficientes"]]
    frac_rel = len(judged) / len(rows)
    ok_judged = all(rows[m]["ok"] for m in judged)
    aprobado = ok_judged and (frac_rel == 1.0 if strict else frac_rel >= 0.8)
    eff = [m for m, _, k in METRICS if k != "banda"]
    frac = sum(rows[m]["ok"] for m in eff) / len(eff)
    idx = chain_index({m: rows[m]["agente"] for m in rows}, human, sup)
    return dict(aprobado=bool(aprobado), fraccion_eficacia_ok=round(frac, 3), fraccion_con_datos=round(frac_rel, 3),
                indice_cadena=round(idx, 4) if idx is not None else None,
                por_etapa={k: f"{v[0]}/{v[1]}" for k, v in by_stage.items()}, metricas=rows)


# ------------------------------------------------------------------------------- humanos y políticas
def human_units(split, map_name="sanguchito_rs_x4", epv=None, family=None, limit=None):
    from tools.x4_ticks import MAP_IDS
    rows = json.loads((ROOT / "reports" / "x4" / "splits.json").read_text(encoding="utf-8"))["recordings"]
    index = json.loads((ROOT / "reports" / "x4" / "index.json").read_text(encoding="utf-8"))
    splits = split.split("+")
    units = []
    for name, r in sorted(rows.items()):
        if r["split"] not in splits or map_name not in (r.get("maps") or {}):
            continue
        if family and index.get(name, {}).get("family") != family:
            continue
        path = ROOT / "data" / "x4_ticks" / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        eps = XM.episodes_from_ticks(str(path), map_id=MAP_IDS[map_name])
        if eps:
            units.append(unit_counts(eps, epv))
        if limit and len(units) >= limit:
            break
    return units


def policy_units(red, blue=None, matches=32, minutes=3.0, seed=11, epv=None, map_name="sanguchito_rs_x4",
                 delays=(8, 9, 10, 11), obs_delay=None):
    """Partidos simulados (self-play si `blue` es None) → unidades por partido."""
    from learn.x4_eval import play
    r = play(red, blue or red, map_name=map_name, matches=matches, minutes=minutes, seed=seed, record=matches,
             delays=delays, obs_delay=obs_delay)
    return [unit_counts([ep], epv) for ep in r["episodes"]], r


def load_reference(path=None):
    p = Path(path or ROOT / "reports" / "x4" / "pass_chain_human.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", action="store_true", help="referencias humanas (Sanguchito train/test y liga RS ONE)")
    ap.add_argument("--policy", action="append", default=[], help="checkpoint (self-play); se puede repetir")
    ap.add_argument("--vs", default="", help="rival fijo para las políticas (p. ej. la BC); por defecto self-play")
    ap.add_argument("--matches", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--epv", default=str(ROOT / "runs" / "x4_epv" / "epv.pt"))
    ap.add_argument("--reference", default=str(ROOT / "reports" / "x4" / "pass_chain_human.json"))
    ap.add_argument("--gate-ref", default="sanguchito_test", help="referencia del gate dentro del archivo humano")
    ap.add_argument("--delays", default="8,9,10,11", help="retardos de sala de la simulación (ticks)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from learn.x4_epv import EPV
    epv = EPV(a.epv) if Path(a.epv).exists() else None
    report = {}
    if a.human:
        refs = dict(sanguchito_train=("train", "sanguchito_rs_x4", None), sanguchito_test=("test", "sanguchito_rs_x4", None),
                    liga_rs_one=("train+dev+test", "rs_one", "haxarg"))
        for key, (sp, m, fam) in refs.items():
            u = human_units(sp, m, epv, fam)
            report[key] = dict(unidades=len(u), split=sp, mapa=m, familia=fam, metricas=bootstrap(u), banda=unit_spread(u))
            print(key, len(u), "unidades", flush=True)
    else:
        ref = load_reference(a.reference)
        from learn.x4_eval import Policy
        vs = Policy(a.vs) if a.vs else None
        for spec in a.policy:
            pol = Policy(spec)
            delays = tuple(int(x) for x in a.delays.split(","))
            u, r = policy_units(pol, vs, a.matches, a.minutes, epv=epv, delays=delays)
            m = bootstrap(u)
            row = dict(unidades=len(u), rival=a.vs or "self-play", retardos=list(delays), safety=int(r["safety"].sum()),
                       metricas=m)
            if ref:
                h = ref[a.gate_ref]
                row["gate"] = gate(m, h["metricas"], h["banda"], support(u), strict=True)
                row["indice_ic90"] = index_ci(u, h["metricas"])
            report[spec] = row
            print(spec, json.dumps({k: v for k, v in row.get("gate", {}).items() if k != "metricas"}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
