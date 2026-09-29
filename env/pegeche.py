"""Reglas del script de la sala Pegeche (Real Soccer X6), sobre BatchSim.

Réplica de `Pegeche Aureus/Script Pegeche` (ref.js, setpiece.js, slide.js, foul.js, tick.js y
onPlayerBallKick de pegeche-room-config.js). Como en tick.js, la lógica del script corre cada 3 ticks;
los setTimeout del script se pasan a ticks (60/s). Las coordenadas del script son las del x6
(1150 x 600) y se escalan con el ancho de la cancha (x6_half = 0.5); lo que depende del radio del
jugador (umbrales de contacto de las faltas) no se escala.

Mecánicas:
- Slide: mantener X (patear) 500 ms moviéndose. Impulso x5.5 (tope 5.5) y 4 s casi inmóvil
  (velocidad tope 0.1); 25 s de cooldown. OJO: el script también cambia el damping del jugador, pero
  HaxBall lo reescribe en cada tick con el del mapa (node-haxball: `n.m = kicking ? kickingDamping :
  damping`), así que no tiene efecto y no se replica.
- Faltas: contacto de un slide (fase de impulso) con un rival, o choque de rivales cerca de la pelota
  con velocidad >= 1. La víctima la "pide" manteniendo X 500 ms dentro de 5 s -> tarjeta (20% nada,
  70% amarilla, 10% roja) y tiro libre en el lugar, o penal si fue en el área del infractor. Si no la
  pide, la tarjeta se aplica en la próxima salida y hasta entonces no se cobran otras faltas.
  2 amarillas o roja = expulsado (acá: fuera de la cancha hasta que termine el episodio). La lesión
  del script (slowdown 0.12) no tiene efecto real (slowMaxSpeed queda en 0) y no se replica.
- Pelotas paradas (laterales, córners, saques de arco, tiros libres, penales) con sus barreras,
  "congelamiento" (invMass 100000: el que no patea no mueve la pelota), potencia (invMass de la
  pelota al patear), curva, timeouts de 7 s y validez del lateral.
- No se simulan las pausas del script (pedir falta pausa 1.5 s): se descuentan de los tiempos.
"""
from __future__ import annotations

import math

import numpy as np

from sim.physics import PS_CURVE, PS_GRAV, PS_MAX_LAT

# estados de pelota parada ("outStatus")
NONE, THROW, CORNER, GOALKICK, FREEKICK, PENALTY = range(6)
N_RULE_FEATS = 15

FROZEN_INV = 100000.0
EVERY = 3                       # tick.js: la lógica del script corre cada 3 ticks
FAR = 1e9

# tiempos en ticks (60/s)
HOLD = 30                       # SLIDE_CONFIG.HOLD_TIME_MS 500
SLIDE_BURST = 48                # BURST_DURATION_MS 800
SLIDE_SLOW = 240                # SLOW_DURATION_MS 4000
SLIDE_COOLDOWN = 1500           # COOLDOWN_MS 25000
PAIR_COOLDOWN = 180             # misma pareja: 3 s entre faltas
FOUL_CLAIM = 300                # 5 s para pedir la falta
EXPEL_DELAY = 120               # expulsión 2 s después de la tarjeta
SP_TIMEOUT = 420                # 7 s
ANIM = 40                       # throwRealBall: 60 x sleep(10 ms) ~ 0.65 s
BALL_RESTORE = 7                # invMass de la pelota vuelve a 1.5 a los 120 ms del tiro
FK_POST = 48                    # tiro libre / penal: estado se limpia 800 ms después del tiro
THROW_RETRY = 30                # lateral vencido: pasa al otro equipo a los 500 ms
FK_INDICATOR = 84               # disco 26 (bloquea a todos) 3 s desde el setup, 1.6 s en pausa
FK_TIMEOUT = 330                # 7 s desde el setup, 1.5 s en pausa
PEN_PROTECT = 210               # 5 s desde el setup, 1.5 s en pausa
GK_INDICATOR = 180              # disco 26 en saque de arco: 3 s
CATAPULT_TICKS = (0, 1, 3, 6)   # anti-catapulta a los 0/20/50/100 ms

BURST_MULT, BURST_MAX, MIN_SLIDE_SPEED, SLOW_MAX = 5.5, 5.5, 0.15, 0.1
FOUL_MIN_SPEED, SLIDE_FOUL_MIN_SPEED, BALL_PROX, VICTIM_MAX = 1.0, 1.5, 80.0, 2.5
THROW_KICK_SPEED, THROW_SOFT_SPEED = 4.25, 2.7
CARD_P = (0.2, 0.7, 0.1)        # ninguna / amarilla / roja
INV_FK, INV_CORNER, INV_GK = 2.4, 2.5, 3.0
SLIDE_BLOCKED = (THROW, CORNER, GOALKICK, FREEKICK)   # isSetPieceActive (el penal no cuenta)


class PegecheRules:
    def __init__(self, sim, rng: np.random.Generator):
        if not sim.ps_on:
            raise ValueError("las reglas Pegeche requieren powershot=True (mismo script)")
        self.sim, self.rng = sim, rng
        st = sim.st
        N, P = sim.N, sim.P
        self.N, self.P = N, P
        self.team = sim.player_team
        self.fp = sim.first_player
        self.s = st.field_half_w / 1150.0
        s = self.s
        self.W, self.H, self.GH = st.field_half_w, st.field_half_h, st.goal_half_height
        self.rb = float(st.ball["radius"])
        self.rp = float(st.player["radius"])
        self.base_inv_p = float(st.player["invMass"])
        self.base_inv_b = float(st.ball["invMass"])
        # geometría del script (x6) escalada
        self.box_x, self.box_y = 840 * s, 320 * s
        self.throw_y = 618 * s
        self.c1_y = 500 * s
        self.pen_spot, self.gk_ball_x = 935 * s, 1060 * s
        self.corner_ball = (1140 * s, 590 * s)
        self.corner_barrier = (1150 * s, 670 * s, 420 * s)
        self.prot_r = 260 * s          # rivales a >= 260 (se los manda a 280) en córner / tiro libre
        self.slide_foul_d = 2 * self.rp + 1   # 29 con radio 14
        self.foul_d = 2 * self.rp + 7         # 35

        i8 = lambda v: np.full(N, v, dtype=np.int64)  # noqa: E731
        self.clock = i8(0)
        self.status, self.sp_team, self.sp_t0 = i8(NONE), i8(-1), i8(0)
        self.anim_until, self.sp_end_at, self.ball_restore_at = i8(-1), i8(-1), i8(-1)
        self.sp_waiting = np.zeros(N, bool)     # rsReady && !rsActive: esperando el tiro
        self.sp_pos = np.zeros((N, 2))
        self.throw_kicked = np.zeros(N, bool)
        self.throw_origin = np.zeros((N, 2))
        self.throw_retry_at, self.throw_retry_team = i8(-1), i8(-1)
        self.gk_barriers_off = np.zeros(N, bool)
        self.in_play = np.ones(N, bool)
        self.pen_gk, self.pen_kicker, self.pen_prot_until = i8(-1), i8(-1), i8(-1)
        # círculos que bloquean jugadores: slot 0 = barrera (discos 24/25), slot 1 = indicador (26)
        self.circ_c = np.zeros((N, 2, 2))
        self.circ_r = np.zeros((N, 2))
        self.circ_team = np.full((N, 2), -1, dtype=np.int64)  # equipo bloqueado, 2 = ambos
        self.circ_until = np.full((N, 2), -1, dtype=np.int64)
        self.c1 = np.zeros((N, P), bool)        # cGroup c1: líneas y = ±500 (lateral)
        self.c0 = np.zeros((N, P), bool)        # cGroup c0: paredes de las áreas
        self.last_touch = i8(-1)
        # slide
        self.sl_phase = np.zeros((N, P), np.int64)     # 0 nada, 1 impulso, 2 frenado
        self.sl_t = np.zeros((N, P), np.int64)
        self.sl_cd_until = np.full((N, P), -1, np.int64)
        self.sl_fouled = np.zeros((N, P), bool)
        self.hold_start = np.full((N, P), -1, np.int64)
        self.hold_conf = np.zeros((N, P), bool)
        # faltas
        self.pf_active, self.pf_expired, self.pf_pen, self.pf_card_done = (np.zeros(N, bool) for _ in range(4))
        self.pf_victim, self.pf_fouler, self.pf_card, self.pf_t = i8(-1), i8(-1), i8(0), i8(0)
        self.pf_pos = np.zeros((N, 2))
        self.pair_last = np.full((N, P, P), -10 ** 9, np.int64)
        self.cat_victim, self.cat_t0 = i8(-1), i8(-1)
        # tarjetas
        self.yellow = np.zeros((N, P), np.int64)
        self.expelled = np.zeros((N, P), bool)
        self.expel_at = np.full((N, P), -1, np.int64)
        # eventos del último env.step (para info / premios)
        self.ev = {k: np.zeros(N, np.int64) for k in ("out", "foul", "claim", "card", "slide", "red")}
        self.out_loser = i8(-1)

    # ------------------------------------------------------------------ reset
    def reset(self, idx) -> None:
        idx = np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return
        for a in (self.clock, self.sp_t0):
            a[idx] = 0
        for a in (self.sp_team, self.anim_until, self.sp_end_at, self.ball_restore_at, self.throw_retry_at,
                  self.throw_retry_team, self.pen_gk, self.pen_kicker, self.pen_prot_until, self.last_touch,
                  self.pf_victim, self.pf_fouler, self.cat_victim, self.cat_t0):
            a[idx] = -1
        self.status[idx] = NONE
        for a in (self.sp_waiting, self.throw_kicked, self.gk_barriers_off, self.pf_active, self.pf_expired,
                  self.pf_pen, self.pf_card_done):
            a[idx] = False
        self.in_play[idx] = True
        self.circ_r[idx] = 0.0
        self.circ_team[idx] = -1
        self.circ_until[idx] = -1
        self.c1[idx] = False
        self.c0[idx] = False
        self.sl_phase[idx] = 0
        self.sl_cd_until[idx] = -1
        self.sl_fouled[idx] = False
        self.hold_start[idx] = -1
        self.hold_conf[idx] = False
        self.pair_last[idx] = -10 ** 9
        self.yellow[idx] = 0
        self.expelled[idx] = False
        self.expel_at[idx] = -1
        self.sim.inv_env[idx, self.fp:] = self.base_inv_p

    def begin_step(self) -> None:
        for v in self.ev.values():
            v[:] = 0
        self.out_loser[:] = -1

    # ------------------------------------------------------------------ helpers
    def _freeze(self, n, players=None, frozen=True):
        ps = range(self.P) if players is None else players
        for p in ps:
            if not self.expelled[n, p]:
                self.sim.inv_env[n, self.fp + p] = FROZEN_INV if frozen else self.base_inv_p

    def _restore_players(self, n):
        """restoreAllPlayersAfterSetPiece: invMass y cGroup normales."""
        self._freeze(n, frozen=False)
        self.c0[n] = False
        self.c1[n] = False

    def _hide_circles(self, n, slots=(0, 1)):
        for c in slots:
            self.circ_r[n, c] = 0.0
            self.circ_team[n, c] = -1

    def _circle(self, n, slot, cx, cy, r, team, until=-1):
        self.circ_c[n, slot] = (cx, cy)
        self.circ_r[n, slot] = r
        self.circ_team[n, slot] = team
        self.circ_until[n, slot] = until

    def _place_ball(self, n, x, y, inv, collide=True):
        sim = self.sim
        sim._reset_ball_state([n])          # corta powershot / curva en curso
        sim.pos[n, 0] = (x, y)
        sim.vel[n, 0] = 0.0
        sim.inv_env[n, 0] = inv
        sim.mask[n, 0] = sim.base_mask[0] if collide else 0

    def _ball_normal(self, n):
        self.sim.inv_env[n, 0] = self.base_inv_b
        self.sim.mask[n, 0] = self.sim.base_mask[0]
        self.sim.ball_grav[n] = 0.0

    def _ppos(self, n, p):
        return self.sim.pos[n, self.fp + p]

    def _active_players(self, n):
        return [p for p in range(self.P) if not self.expelled[n, p]]

    def _clear_setpiece(self, n):
        self.status[n] = NONE
        self.sp_team[n] = -1
        self.sp_waiting[n] = False
        self.anim_until[n] = -1
        self.sp_end_at[n] = -1
        self.pen_gk[n] = -1
        self.pen_kicker[n] = -1
        self.pen_prot_until[n] = -1
        self.throw_kicked[n] = False
        self.gk_barriers_off[n] = False

    def _start(self, n, status, team):
        self._clear_setpiece(n)
        self.throw_retry_at[n] = -1
        self.status[n] = status
        self.sp_team[n] = team
        self.sp_t0[n] = self.clock[n]
        self.sp_waiting[n] = status != THROW
        self._hide_circles(n)

    # ------------------------------------------------------------------ pelotas paradas
    def _throw_in(self, n, team, pos):
        """ref.js throwIn: la pelota vuelve desde afuera hasta y = ±618 (fuera de la línea)."""
        s = self.s
        self._start(n, THROW, team)
        self.throw_origin[n] = pos
        sy = 1.0 if pos[1] > 0 else -1.0
        tx = float(np.clip(pos[0], -self.W + 20 * s, self.W - 20 * s))
        ty = sy * self.throw_y
        self.sp_pos[n] = (tx, ty)
        self.in_play[n] = False
        self.anim_until[n] = self.clock[n] + ANIM
        self._circle(n, 0, tx, ty, 270 * s, 1 - team)
        self._place_ball(n, tx, ty, 0.0, collide=False)

    def _throw_in_ready(self, n):
        """Fin de la animación: pelota normal y rivales congelados detrás de las líneas c1."""
        team = self.sp_team[n]
        tx, ty = self.sp_pos[n]
        self._place_ball(n, tx, ty, self.base_inv_b)
        for p in self._active_players(n):
            if self.team[p] == team:
                continue
            self.sim.inv_env[n, self.fp + p] = FROZEN_INV
            self.c1[n, p] = True
            y = self._ppos(n, p)[1]
            if ty < 0 and y < -450 * self.s:
                self._ppos(n, p)[1] = -440 * self.s
            elif ty > 0 and y > 450 * self.s:
                self._ppos(n, p)[1] = 440 * self.s

    def _corner(self, n, team, pos):
        cx, cy = self.corner_ball
        tx, ty = (cx if pos[0] > 0 else -cx), (cy if pos[1] > 0 else -cy)
        self._start(n, CORNER, team)
        self.sp_pos[n] = (tx, ty)
        self._freeze(n)
        self.anim_until[n] = self.clock[n] + ANIM
        self._circle(n, 1, tx, ty, 18 * self.s, 2)              # indicador: bloquea a todos
        self._place_ball(n, tx, ty, 0.0, collide=False)

    def _corner_ready(self, n):
        team = self.sp_team[n]
        tx, ty = self.sp_pos[n]
        self._place_ball(n, tx, ty, INV_CORNER)
        self._hide_circles(n, (1,))
        bx, by, br = self.corner_barrier
        self._circle(n, 0, math.copysign(bx, tx), math.copysign(by, ty), br, 1 - team)

    def _goal_kick(self, n, team, pos):
        x = self.gk_ball_x if pos[0] > 0 else -self.gk_ball_x
        self._start(n, GOALKICK, team)
        self.sp_pos[n] = (x, 0.0)
        self._place_ball(n, x, 0.0, INV_GK)
        self.sim.ball_grav[n] = 0.0
        self._freeze(n)
        self._circle(n, 1, x, 0.0, 18 * self.s, 2, until=self.clock[n] + GK_INDICATOR)

    def _free_kick(self, n, team, pos):
        s = self.s
        x = float(np.clip(pos[0], -self.W, self.W))
        y = float(np.clip(pos[1], -self.H, self.H))
        self._start(n, FREEKICK, team)
        self.sp_pos[n] = (x, y)
        self._freeze(n)
        self._circle(n, 1, x, y, 18 * s, 2, until=self.clock[n] + FK_INDICATOR)
        self._place_ball(n, x, y, INV_FK)
        self._circle(n, 0, x, y, 240 * s, 1 - team, until=self.clock[n] + FK_TIMEOUT)
        self._push_rivals(n, x, y, clamp=False)

    def _penalty(self, n, team, victim, fouler):
        """setupPenalty: pateador = víctima, arquero = defensor al azar (no el infractor)."""
        s = self.s
        side = 1.0 if team == 0 else -1.0             # el rojo ataca hacia +x
        spot = side * self.pen_spot
        self._start(n, PENALTY, team)
        self.sp_pos[n] = (spot, 0.0)
        self._freeze(n)
        defenders = [p for p in self._active_players(n) if self.team[p] != team]
        cands = [p for p in defenders if p != fouler] or defenders
        gk = int(self.rng.choice(cands)) if cands else -1
        self.pen_gk[n], self.pen_kicker[n] = gk, victim
        sim, fp = self.sim, self.fp
        if victim >= 0 and not self.expelled[n, victim]:
            sim.pos[n, fp + victim] = (spot - side * 60 * s, 0.0)
            sim.vel[n, fp + victim] = 0.0
        if gk >= 0:
            sim.pos[n, fp + gk] = (side * 1162 * s, 0.0)
            sim.vel[n, fp + gk] = 0.0
        for p in self._active_players(n):
            if p in (victim, gk):
                continue
            x, y = sim.pos[n, fp + p]
            if side * x > self.box_x and abs(y) < self.box_y:
                sim.pos[n, fp + p] = (side * (self.box_x - 50 * s), float(np.clip(y, -self.box_y, self.box_y)))
                sim.vel[n, fp + p] = 0.0
        self._place_ball(n, spot, 0.0, self.base_inv_b)
        self._circle(n, 0, spot, 0.0, 170 * s, 1 - team)
        self.pen_prot_until[n] = self.clock[n] + PEN_PROTECT

    def _push_rivals(self, n, cx, cy, clamp):
        """Rivales del que saca a menos de 260 -> a 280 (blockFreeKickZone / blockCornerZone)."""
        s, sim, fp = self.s, self.sim, self.fp
        for p in self._active_players(n):
            if self.team[p] == self.sp_team[n]:
                continue
            dx, dy = sim.pos[n, fp + p, 0] - cx, sim.pos[n, fp + p, 1] - cy
            d = math.hypot(dx, dy)
            if d < self.prot_r:
                a = math.atan2(dy, dx)
                nx, ny = cx + math.cos(a) * (self.prot_r + 20 * s), cy + math.sin(a) * (self.prot_r + 20 * s)
                if clamp:
                    nx = float(np.clip(nx, -self.W + 30 * s, self.W - 30 * s))
                    ny = float(np.clip(ny, -self.H + 30 * s, self.H - 30 * s))
                sim.pos[n, fp + p] = (nx, ny)
                sim.vel[n, fp + p] = 0.0

    def _setpiece_curve(self, n, p):
        """applySetPieceWithCurve: gravedad perpendicular al tiro según la velocidad lateral."""
        sim = self.sim
        par = sim.ps_par
        bp, pp = sim.pos[n, 0], sim.pos[n, self.fp + p]
        tx, ty = bp - pp
        td = math.hypot(tx, ty)
        sim.ball_grav[n] = 0.0
        if td < 0.1:
            return
        pvx, pvy = sim.vel[n, self.fp + p]
        lat = (tx * pvy - ty * pvx) / td
        ci = max(-1.0, min(1.0, lat / par[PS_MAX_LAT]))
        sx, sy = sim.vel[n, 0]
        sp = math.hypot(sx, sy)
        if sp > 0.1:
            sim.ball_grav[n] = (sy / sp * ci * par[PS_GRAV], -sx / sp * ci * par[PS_GRAV])
            sim.ps_grav_left[n] = int(par[PS_CURVE])
            sim.ps_has_pending[n] = False

    def _on_kick(self, n, p):
        """onPlayerBallKick para las pelotas paradas."""
        st = self.status[n]
        if not self.sp_waiting[n] or self.anim_until[n] >= 0:
            return
        if st in (FREEKICK, PENALTY):
            if self.team[p] != self.sp_team[n]:
                return
            # blockPenaltyZone (arquero en la línea) sigue hasta endPenalty, 800 ms después
            self.sp_waiting[n] = False
            self._hide_circles(n)
            self._setpiece_curve(n, p)
            self.ball_restore_at[n] = self.clock[n] + BALL_RESTORE
            self.sp_end_at[n] = self.clock[n] + FK_POST
        elif st in (CORNER, GOALKICK):
            self._setpiece_curve(n, p)
            self.ball_restore_at[n] = self.clock[n] + BALL_RESTORE
            self._hide_circles(n)
            self._restore_players(n)
            self._clear_setpiece(n)
            self.in_play[n] = True

    # ------------------------------------------------------------------ faltas
    def _card(self, n, p, card):
        if card == 0 or p < 0 or self.expelled[n, p]:
            return
        self.ev["card"][n] += 1
        if card == 1:
            self.yellow[n, p] += 1
        if card == 2 or self.yellow[n, p] >= 2:
            self.ev["red"][n] += 1
            if self.expel_at[n, p] < 0:
                self.expel_at[n, p] = self.clock[n] + EXPEL_DELAY

    def _clear_pending(self, n):
        self.pf_active[n] = False
        self.pf_expired[n] = False
        self.pf_card_done[n] = False
        self.pf_victim[n] = -1
        self.pf_fouler[n] = -1
        self.pair_last[n] = -10 ** 9
        self.sl_fouled[n] = False

    def _process_foul(self, n, fouler, victim, pos):
        self.ev["foul"][n] += 1
        self.cat_victim[n], self.cat_t0[n] = victim, self.clock[n]
        self._anti_catapult(n)
        u = self.rng.random()
        card = 0 if u < CARD_P[0] else (1 if u < CARD_P[0] + CARD_P[1] else 2)
        side = -1.0 if self.team[fouler] == 0 else 1.0      # área del infractor (el rojo defiende -x)
        s = self.s
        pen = side * pos[0] >= self.box_x and side * pos[0] <= 1150 * s and abs(pos[1]) <= self.box_y
        self.pf_active[n], self.pf_expired[n], self.pf_card_done[n] = True, False, False
        self.pf_victim[n], self.pf_fouler[n], self.pf_card[n] = victim, fouler, card
        self.pf_pos[n] = pos
        self.pf_pen[n] = pen
        self.pf_t[n] = self.clock[n]

    def _anti_catapult(self, n):
        v = self.cat_victim[n]
        if v < 0:
            return
        vel = self.sim.vel[n, self.fp + v]
        sp = math.hypot(*vel)
        if sp > VICTIM_MAX:
            vel *= VICTIM_MAX / sp

    def _ball_out_of_play(self, n):
        return self.status[n] in (THROW, CORNER, GOALKICK, FREEKICK)

    def _detect_fouls(self, n):
        if self.pf_active[n] or self._ball_out_of_play(n):
            return
        sim, fp, t = self.sim, self.fp, self.clock[n]
        act = self._active_players(n)
        if len(act) < 2:
            return
        pos, vel = sim.pos[n, fp:], sim.vel[n, fp:]
        # slide contra un rival (sin importar la pelota)
        for p in act:
            if self.sl_phase[n, p] != 1 or self.sl_fouled[n, p]:
                continue
            sp = math.hypot(*vel[p])
            if sp < SLIDE_FOUL_MIN_SPEED:
                continue
            for q in act:
                if q == p or self.team[q] == self.team[p]:
                    continue
                if math.hypot(*(pos[q] - pos[p])) >= self.slide_foul_d:
                    continue
                if t - self.pair_last[n, p, q] < PAIR_COOLDOWN:
                    continue
                self.sl_fouled[n, p] = True
                self.pair_last[n, p, q] = self.pair_last[n, q, p] = t
                self._process_foul(n, p, q, pos[q].copy())
                return
        # choque fuerte cerca de la pelota (sólo mira pares i < j con i en movimiento, como el script)
        ball = sim.pos[n, 0]
        for a, i in enumerate(act):
            s1 = math.hypot(*vel[i])
            if s1 < FOUL_MIN_SPEED:
                continue
            for j in act[a + 1:]:
                if self.team[i] == self.team[j]:
                    continue
                if math.hypot(*(pos[i] - pos[j])) >= self.foul_d:
                    continue
                if t - self.pair_last[n, i, j] < PAIR_COOLDOWN:
                    continue
                if math.hypot(*(pos[i] - ball)) > BALL_PROX and math.hypot(*(pos[j] - ball)) > BALL_PROX:
                    continue
                s2 = math.hypot(*vel[j])
                fouler, victim = (i, j) if s1 > s2 else (j, i)
                self.pair_last[n, i, j] = self.pair_last[n, j, i] = t
                self._process_foul(n, fouler, victim, (pos[i] + pos[j]) / 2)
                return

    def _claim(self, n, p):
        if not self.pf_active[n] or self.pf_victim[n] != p or self.pf_expired[n] or self._ball_out_of_play(n):
            return False
        self.ev["claim"][n] += 1
        if self.pf_card[n] != 0:
            self.pf_card_done[n] = True
            self._card(n, self.pf_fouler[n], self.pf_card[n])
        team = int(self.team[p])
        if self.pf_pen[n]:
            self._penalty(n, team, p, int(self.pf_fouler[n]))
        else:
            self._free_kick(n, team, self.pf_pos[n].copy())
        self._clear_pending(n)
        return True

    # ------------------------------------------------------------------ slide
    def _try_slide(self, n, p):
        """trySlidePlayer(force=true): True si se llegó a executeSlide (aunque no se mueva)."""
        if self.status[n] in SLIDE_BLOCKED or self.sl_phase[n, p] != 0:
            return False
        if self.clock[n] < self.sl_cd_until[n, p]:
            return False
        v = self.sim.vel[n, self.fp + p]
        sp = math.hypot(*v)
        if sp < MIN_SLIDE_SPEED:
            return True
        b = v * BURST_MULT
        bs = math.hypot(*b)
        if bs > BURST_MAX:
            b *= BURST_MAX / bs
        self.sim.vel[n, self.fp + p] = b
        self.sl_phase[n, p] = 1
        self.sl_t[n, p] = self.clock[n]
        self.sl_fouled[n, p] = False
        self.ev["slide"][n] += 1
        return True

    def _end_slide(self, n, p):
        v = self.sim.vel[n, self.fp + p]
        if math.hypot(*v) < 2:
            v[:] = 0.0
        if self.status[n] == PENALTY:   # isSetPieceActive no incluye el penal: restaura invMass
            self.sim.inv_env[n, self.fp + p] = self.base_inv_p
        self.sl_phase[n, p] = 0
        self.sl_cd_until[n, p] = self.clock[n] + SLIDE_COOLDOWN
        self.sl_fouled[n, p] = False

    # ------------------------------------------------------------------ árbitro
    def _ref(self, n):
        """realSoccerRef: salida por la línea de fondo (córner / saque de arco) o lateral."""
        if not self.in_play[n] or self.anim_until[n] >= 0 or self.status[n] != NONE:
            return
        bx, by = self.sim.pos[n, 0]
        lt = self.last_touch[n]
        if lt < 0:
            d = np.linalg.norm(self.sim.pos[n, self.fp:] - self.sim.pos[n, 0], axis=1)
            d[self.expelled[n]] = np.inf
            lt = int(self.team[int(np.argmin(d))])
        if abs(bx) > self.W + self.rb and abs(by) > self.GH:
            kind = "end"
        elif abs(by) > self.H + self.rb and abs(bx) < self.W:
            kind = "side"
        else:
            return
        self._enforce_pending_on_out(n)
        self.ev["out"][n] += 1
        self.out_loser[n] = lt
        pos = (float(bx), float(by))
        if kind == "end":
            defender = 0 if bx < 0 else 1           # el rojo defiende el arco de -x
            if lt == defender:
                self._corner(n, 1 - defender, pos)
            else:
                self._goal_kick(n, defender, pos)
        else:
            self._throw_in(n, 1 - lt, pos)
        self.last_touch[n] = -1

    def _enforce_pending_on_out(self, n):
        if not self.pf_active[n]:
            return
        if self.pf_card[n] != 0 and not self.pf_card_done[n]:
            self._card(n, self.pf_fouler[n], self.pf_card[n])
        self._clear_pending(n)

    def _throw_restart(self, n, team):
        self._restore_players(n)
        self._hide_circles(n)
        self._throw_in(n, team, self.throw_origin[n].copy())

    def _handle_ball_in_play(self, n):
        """handleBallInPlay: validez del lateral."""
        if self.anim_until[n] >= 0 or self.status[n] != THROW:
            return
        s = self.s
        bx, by = self.sim.pos[n, 0]
        tx, ty = self.sp_pos[n]
        top = ty < 0
        sp = math.hypot(*self.sim.vel[n, 0])
        other = 1 - self.sp_team[n]
        if not self.throw_kicked[n] and sp >= THROW_KICK_SPEED:
            self.throw_kicked[n] = True
        entered = (by > -self.H) if top else (by < self.H)
        dist_x = abs(bx - tx)
        if self.throw_kicked[n]:
            if dist_x > 270 * s:
                self._throw_restart(n, other)
                return
            if entered:
                self._clear_setpiece(n)
                self.in_play[n] = True
                self._ball_normal(n)
                self._hide_circles(n)
                self._restore_players(n)
            elif (by < ty - 50 * s) if top else (by > ty + 50 * s):
                self._throw_restart(n, other)
        elif sp < THROW_SOFT_SPEED and (entered or dist_x > 270 * s):
            self._throw_restart(n, other)

    def _block_zones(self, n):
        """blockCornerZone / blockFreeKickZone / blockGoalKickZone / blockPenaltyZone."""
        st, sim, fp, s = self.status[n], self.sim, self.fp, self.s
        if st == CORNER and self.anim_until[n] < 0:
            self._push_rivals(n, *self.sp_pos[n], clamp=True)
        elif st == FREEKICK and self.sp_waiting[n]:
            self._freeze(n)
            self._push_rivals(n, *self.sp_pos[n], clamp=False)
        elif st == GOALKICK and not self.gk_barriers_off[n]:
            left = self.sp_pos[n, 0] < 0
            rival = self.sp_team[n] ^ 1
            for p in self._active_players(n):
                if self.team[p] != rival:
                    continue
                x, y = sim.pos[n, fp + p]
                if (x < 0) == left:
                    self.c0[n, p] = True
                    if abs(x) > self.box_x and abs(y) < self.box_y:
                        sim.pos[n, fp + p, 0] = math.copysign(self.box_x - 15 * s, x)
        elif st == PENALTY and self.pen_gk[n] >= 0:
            side = 1.0 if self.sp_team[n] == 0 else -1.0
            line = side * 1150 * s
            gk = self.pen_gk[n]
            x, y = sim.pos[n, fp + gk]
            y = float(np.clip(y, -self.GH, self.GH))
            if side * x < side * line:
                sim.pos[n, fp + gk] = (line, y)
                sim.vel[n, fp + gk, 0] = 0.0
            else:
                sim.pos[n, fp + gk, 1] = y
            if self.clock[n] < self.pen_prot_until[n]:
                for p in self._active_players(n):
                    if p in (gk, self.pen_kicker[n]):
                        continue
                    x, y = sim.pos[n, fp + p]
                    if side * x > self.box_x and abs(y) < self.box_y:
                        sim.pos[n, fp + p, 0] = side * (self.box_x - 30 * s)
                        sim.vel[n, fp + p] = 0.0
                    if self.team[p] != self.sp_team[n]:
                        self.c0[n, p] = True

    # ------------------------------------------------------------------ timers (setTimeout)
    def _timers(self, n):
        t = self.clock[n]
        st = self.status[n]
        if self.anim_until[n] >= 0 and t >= self.anim_until[n]:
            self.anim_until[n] = -1
            (self._throw_in_ready if st == THROW else self._corner_ready)(n)
        for c in (0, 1):
            if 0 <= self.circ_until[n, c] <= t:
                self.circ_r[n, c] = 0.0
                self.circ_until[n, c] = -1
                if c == 0 and st == FREEKICK and self.sp_waiting[n]:
                    self.sim.inv_env[n, 0] = self.base_inv_b      # FK vencido: pelota normal
        if self.ball_restore_at[n] >= 0 and t >= self.ball_restore_at[n]:
            self.ball_restore_at[n] = -1
            self.sim.inv_env[n, 0] = self.base_inv_b
            self.sim.mask[n, 0] = self.sim.base_mask[0]
        if self.sp_end_at[n] >= 0 and t >= self.sp_end_at[n]:
            if st == PENALTY:
                self._ball_normal(n)
            self._restore_players(n)
            self._clear_setpiece(n)
            self.in_play[n] = True
        if self.pen_prot_until[n] >= 0 and t >= self.pen_prot_until[n]:
            self.pen_prot_until[n] = -1
            self.c0[n] = False
            self.circ_r[n, 0] = 0.0
        if st != NONE and t - self.sp_t0[n] == SP_TIMEOUT:
            if st == THROW:
                self._restore_players(n)
                self._hide_circles(n)
                self._clear_setpiece(n)
                self.throw_retry_at[n] = t + THROW_RETRY
                self.throw_retry_team[n] = 1 - self.sp_team[n]
            elif st == CORNER and self.anim_until[n] < 0:
                self._ball_normal(n)
                self._hide_circles(n)
                self._restore_players(n)
                self._clear_setpiece(n)
                self.in_play[n] = True
            elif st == GOALKICK:
                self.gk_barriers_off[n] = True
                self.sim.inv_env[n, 0] = self.base_inv_b
                self._hide_circles(n)
                self.c0[n] = False
        if self.throw_retry_at[n] >= 0 and t >= self.throw_retry_at[n]:
            self._throw_in(n, self.throw_retry_team[n], self.throw_origin[n].copy())
        if self.pf_active[n] and not self.pf_expired[n] and t - self.pf_t[n] >= FOUL_CLAIM:
            self.pf_expired[n] = True
        for p in range(self.P):
            ph = self.sl_phase[n, p]
            if ph == 1 and t - self.sl_t[n, p] >= SLIDE_BURST:
                self.sl_phase[n, p] = 2
                self.sl_t[n, p] = t
            if 0 <= self.expel_at[n, p] <= t:
                self._expel(n, p)
        if self.cat_victim[n] >= 0:
            dt = t - self.cat_t0[n]
            if dt in CATAPULT_TICKS:
                self._anti_catapult(n)
            if dt > CATAPULT_TICKS[-1]:
                self.cat_victim[n] = -1

    def _expel(self, n, p):
        sim, k = self.sim, self.fp + p
        self.expelled[n, p] = True
        self.expel_at[n, p] = -1
        self.sl_phase[n, p] = 0
        sim.pos[n, k] = (0.0, -(self.H + 5000.0))
        sim.vel[n, k] = 0.0
        sim.mask[n, k] = 0
        if self.pen_gk[n] == p:
            self.pen_gk[n] = -1

    # ------------------------------------------------------------------ tick
    def pre_tick(self, world_act: np.ndarray) -> np.ndarray:
        if self.expelled.any():
            world_act = np.where(self.expelled, 0, world_act)
        return world_act

    def post_tick(self, world_act: np.ndarray, goal: np.ndarray) -> None:
        sim = self.sim
        self.clock += 1
        live = goal == 0
        # kicks (onPlayerBallKick): último que tocó + pelotas paradas
        kicked = sim.kicked & live[:, None]
        for n, p in zip(*np.nonzero(kicked)):
            self.last_touch[n] = self.team[p]
            self._on_kick(n, p)
        self._touch()
        waiting = self.sp_waiting & live
        if waiting.any():  # el script no deja cargar el powershot en una pelota parada (ver docstring)
            sim.ps_charge[waiting] = -1
            sim.ps_comba[waiting] = False
        busy = live & ((self.status != NONE) | (self.throw_retry_at >= 0) | (self.sl_phase != 0).any(axis=1)
                       | self.pf_active | (self.expel_at >= 0).any(axis=1) | (self.ball_restore_at >= 0)
                       | (self.cat_victim >= 0))
        for n in np.nonzero(busy)[0]:
            self._timers(n)
        self._barriers(live)
        if self.expelled.any():
            ex = self.expelled
            sim.vel[:, self.fp:][ex] = 0.0
        # lógica del script cada 3 ticks
        due = live & (self.clock % EVERY == 0)
        if not due.any():
            return
        self._slowdown(due)
        self._holds(due, world_act)
        # prefiltros vectorizados: el detalle (en el orden del script) sólo corre donde puede pasar algo
        foul_c = self._foul_candidates(due)
        b = sim.pos[:, 0]
        out_c = due & ((np.abs(b[:, 0]) > self.W + self.rb) | (np.abs(b[:, 1]) > self.H + self.rb))
        for n in np.nonzero(foul_c | out_c | (due & (self.status != NONE)))[0]:
            if foul_c[n]:
                self._detect_fouls(n)
            self._ref(n)
            self._handle_ball_in_play(n)
            if self.status[n] != NONE:
                self._block_zones(n)

    def _foul_candidates(self, due):
        """Partidos donde _detect_fouls puede cobrar algo (sin mirar el cooldown por pareja)."""
        ok = due & ~self.pf_active & ~np.isin(self.status, (THROW, CORNER, GOALKICK, FREEKICK))
        if not ok.any():
            return ok
        sim = self.sim
        pp, pv = sim.pos[:, self.fp:], sim.vel[:, self.fp:]
        d = np.linalg.norm(pp[:, :, None] - pp[:, None], axis=3)                    # (N, P, P)
        opp = (self.team[:, None] != self.team[None, :])[None] & ~self.expelled[:, :, None] & ~self.expelled[:, None]
        sp = np.linalg.norm(pv, axis=2)
        slide = ((self.sl_phase == 1) & ~self.sl_fouled & (sp >= SLIDE_FOUL_MIN_SPEED))[:, :, None]
        c1 = (slide & opp & (d < self.slide_foul_d)).any(axis=(1, 2))
        near = np.linalg.norm(pp - sim.pos[:, None, 0], axis=2) <= BALL_PROX
        c2 = ((sp >= FOUL_MIN_SPEED)[:, :, None] & opp & (d < self.foul_d)
              & (near[:, :, None] | near[:, None])).any(axis=(1, 2))
        return ok & (c1 | c2)

    def _touch(self):
        """_updateLastTouchFromContact: jugador más cercano en contacto (pelota en juego)."""
        sim = self.sim
        ok = self.in_play & (self.anim_until < 0) & ~self.sp_waiting
        if not ok.any():
            return
        d = np.linalg.norm(sim.pos[:, self.fp:] - sim.pos[:, None, 0], axis=2)
        d[self.expelled] = np.inf
        close = d <= self.rp + self.rb + 2
        hit = ok & close.any(axis=1)
        if hit.any():
            who = np.argmin(np.where(close, d, np.inf), axis=1)
            self.last_touch[hit] = self.team[who[hit]]

    def _slowdown(self, due):
        """applySlowdown: fase de frenado del slide (tope 0.1) y su final."""
        sim = self.sim
        slow = (self.sl_phase == 2) & due[:, None]
        if not slow.any():
            return
        ended = slow & (self.clock[:, None] - self.sl_t > SLIDE_SLOW)
        for n, p in zip(*np.nonzero(ended)):
            self._end_slide(n, p)
        cap = slow & ~ended
        if cap.any():
            v = sim.vel[:, self.fp:]
            sp = np.linalg.norm(v, axis=2)
            m = cap & (sp > SLOW_MAX)
            if m.any():
                v[m] *= (SLOW_MAX / sp[m])[:, None]

    def _holds(self, due, world_act):
        """checkAllPlayersHold: X mantenida 500 ms -> pedir falta o slide."""
        held = (world_act >= 9) & due[:, None] & ~self.expelled
        rel = due[:, None] & ~held
        self.hold_start[rel] = -1
        self.hold_conf[rel] = False
        if not held.any():
            return
        c = self.clock[:, None]
        start = held & (self.hold_start < 0)
        self.hold_start = np.where(start, c, self.hold_start)
        fire = held & ~self.hold_conf & (c - self.hold_start >= HOLD)
        for n, p in zip(*np.nonzero(fire)):
            self.hold_conf[n, p] = True
            if self.pf_active[n] and self.pf_victim[n] == p and not self.pf_expired[n] and self._claim(n, p):
                self.hold_start[n, p] = -1
                self.hold_conf[n, p] = False
                continue
            if self._try_slide(n, p):
                self.hold_start[n, p] = -1
                self.hold_conf[n, p] = False

    def _barriers(self, live):
        """Discos-barrera y paredes c0/c1 que el script activa (colisiones que el sim no tiene)."""
        sim, fp, r = self.sim, self.fp, self.rp
        pp, pv = sim.pos[:, fp:], sim.vel[:, fp:]
        for c in (0, 1):
            act = live & (self.circ_r[:, c] > 0)
            if not act.any():
                continue
            ct = self.circ_team[:, c]
            blocked = act[:, None] & ((ct[:, None] == 2) | (ct[:, None] == self.team[None, :])) & ~self.expelled
            if not blocked.any():
                continue
            d = pp - self.circ_c[:, None, c]
            dist = np.linalg.norm(d, axis=2)
            R = self.circ_r[:, c, None] + r
            m = blocked & (dist < R)
            if m.any():
                u = d[m] / np.maximum(dist[m], 1e-9)[:, None]
                pp[m] = self.circ_c[:, None, c].repeat(self.P, 1)[m] + u * R.repeat(self.P, 1)[m][:, None]
                vn = np.einsum("ij,ij->i", pv[m], u)
                pv[m] -= u * np.minimum(vn, 0.0)[:, None]
        if self.c1.any():
            m1 = self.c1 & live[:, None]
            y = pp[..., 1]
            L = self.c1_y
            inner = m1 & (np.abs(y) <= L) & (np.abs(y) > L - r)
            outer = m1 & (np.abs(y) > L) & (np.abs(y) < L + r)
            if inner.any():
                y[inner] = np.sign(y[inner]) * (L - r)
                pv[..., 1][inner] = 0.0
            if outer.any():
                y[outer] = np.sign(y[outer]) * (L + r)
                pv[..., 1][outer] = 0.0
        if self.c0.any():
            m0 = self.c0 & live[:, None]
            x, y = pp[..., 0], pp[..., 1]
            inside = m0 & (np.abs(x) > self.box_x - r) & (np.abs(y) < self.box_y + r)
            if inside.any():
                dx = np.abs(x) - (self.box_x - r)              # salir por el frente del área
                dy = (self.box_y + r) - np.abs(y)              # salir por arriba / abajo
                front = inside & (dx <= dy)
                side = inside & (dx > dy)
                x[front] = np.sign(x[front]) * (self.box_x - r)
                pv[..., 0][front] = 0.0
                y[side] = np.sign(y[side]) * (self.box_y + r)
                pv[..., 1][side] = 0.0

    # ------------------------------------------------------------------ obs
    def features(self) -> np.ndarray:
        """(N, P, N_RULE_FEATS) en el marco de cada agente (nada depende del signo de x)."""
        N, P = self.N, self.P
        f = np.zeros((N, P, N_RULE_FEATS))
        st = self.status
        for k in range(1, 6):
            f[:, :, k - 1] = (st == k)[:, None]
        has = st != NONE
        mine = self.sp_team[:, None] == self.team[None, :]
        f[:, :, 5] = np.where(has[:, None], np.where(mine, 1.0, -1.0), 0.0)
        block = (self.anim_until >= 0) | (self.circ_r[:, 1] > 0)
        f[:, :, 6] = (has & block)[:, None]
        f[:, :, 7] = np.where(has, np.minimum((self.clock - self.sp_t0) / SP_TIMEOUT, 1.5), 0.0)[:, None]
        f[:, :, 8] = self.sl_phase == 1
        f[:, :, 9] = self.sl_phase == 2
        f[:, :, 10] = np.clip((self.sl_cd_until - self.clock[:, None]) / SLIDE_COOLDOWN, 0.0, 1.0)
        f[:, :, 11] = np.where(self.hold_start >= 0,
                               np.minimum((self.clock[:, None] - self.hold_start) / HOLD, 1.0), 0.0)
        claim = self.pf_active & ~self.pf_expired
        vic = self.pf_victim[:, None] == np.arange(P)[None, :]
        f[:, :, 12] = claim[:, None] & vic
        vteam = np.where(self.pf_victim >= 0, self.team[np.maximum(self.pf_victim, 0)], -1)
        f[:, :, 13] = np.where(self.pf_active[:, None], np.where(vteam[:, None] == self.team[None, :], 1.0, -1.0), 0.0)
        f[:, :, 14] = np.minimum(self.yellow, 1)
        return f
