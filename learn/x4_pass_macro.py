"""Controlador de pase X4: "pasarle la pelota al compañero k" como una sola acción.

En HaxBall pasar es motricidad: ubicarse detrás de la pelota en la línea justa y patear en el momento justo, con 9–12
ticks de latencia de sala. El RL tendría que descubrir esa secuencia de teclas al azar; con controles crudos el pase
aparece recién a ~1e10 pasos (docs/PLAN.md §0.1). En Google Research Football y TiZero el pase es una acción del juego.
Este controlador la agrega: la política decide a quién y cuándo, el controlador cómo.

Física (env/rs4z/kernel.physics_tick, verificada contra la sala): con la tecla de patear apretada, la patada sale en el
primer tick en que la pelota está a menos de KICK_REACH (4 px) del borde del jugador; suma kickStrength · invMass de la
pelota (5,85 · 1,05 ≈ 6,1 px/tick) en la dirección jugador → pelota. La pelota frena ×0,99 por tick; el jugador acelera
0,12 (0,07 con la tecla apretada) y frena ×0,96 (tope ~3 px/tick).

El controlador (por decisión, cada 3 ticks, con el estado que ve la política):
1. Predice pelota y jugador `lag` ticks adelante (la decisión se aplica tarde).
2. Apunta al compañero adelantado según su velocidad y el tiempo de viaje de la pelota.
3. Elige la dirección de patada d tal que velocidad actual de la pelota + 6,1·d apunte al objetivo.
4. Va al punto detrás de la pelota sobre −d; si está del lado del objetivo, la rodea por el costado.
5. Aprieta patear cuando está alineado (el vector jugador → pelota casi igual a d) y cerca: la patada sale sola al
   entrar en alcance.
La versión predictiva (`act` con `hist`; `pass_table` en numba para el RL) reemplaza 1 por las teclas que ya están en
cola (exacto contra el kernel) y 4–5 por una búsqueda sobre las 18 acciones (`_plan`).

Prueba desde momentos humanos (tools/x4_pass_probe, 256 situaciones, 2026-10-07): patea en el 33%, completa el 68% de lo
que patea (humanos 0,69) con 5,5° de error; con el receptor a ≤ 45° de giro pasa el 46% (el agente solo, 25%), con más
de 90° pierde la pelota antes de patear (79%). Por eso en el RL sólo se ofrece con giro ≤ 90° y la pelota cerca.
"""
from __future__ import annotations

import math

import numpy as np

KICK_REACH = 4.0
PLAYER_R = 15.0
BALL_DAMP = 0.99
KICK_IMPULSE = 5.85 * 1.05
P_ACC, P_KACC, P_DAMP = 0.12, 0.07, 0.96
FRAME_SKIP = 3
PLAN_TICKS = 9                  # horizonte de la búsqueda de una acción (3 decisiones)
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# direcciones de movimiento del mundo (las del kernel): índice → (dx, dy)
MOVES = np.array([(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)], dtype=float)
UNIT = MOVES / np.maximum(np.linalg.norm(MOVES, axis=1, keepdims=True), 1e-12)
MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2])


def move_toward(dx, dy, dead=3.0):
    """Índice de movimiento del mundo (0..8) más alineado con (dx, dy); 0 si ya está ahí."""
    if dx * dx + dy * dy < dead * dead:
        return 0
    v = np.array([dx, dy]) / math.hypot(dx, dy)
    return int(np.argmax(UNIT[1:] @ v) + 1)


def own_action(world_move, kick, team):
    """Acción del marco propio (la que recibe RS4ZEnv.step): el azul ve el mundo espejado en x."""
    m = MIRROR_MOVE[world_move] if team == 1 else world_move
    return int(m + (9 if kick else 0))


def travel_ticks(dist, v0):
    """Ticks que tarda la pelota en recorrer `dist` saliendo a `v0` (frena ×0,99 por tick); inf si no llega."""
    reach = v0 / (1 - BALL_DAMP)
    if dist >= reach * 0.98:
        return math.inf
    return math.log(1 - dist / reach) / math.log(BALL_DAMP)


def kick_direction(ball_vel, target_dir, impulse=KICK_IMPULSE):
    """Dirección de patada d (unitaria) tal que ball_vel + impulse·d sea paralela a target_dir. Si no hay solución
    (la pelota ya viene muy rápido en otra dirección), patea hacia el objetivo."""
    u = target_dir / (np.linalg.norm(target_dir) + 1e-9)
    v = np.asarray(ball_vel, float)
    uv = float(u @ v)
    disc = uv * uv - float(v @ v) + impulse * impulse
    if disc < 0:
        return u
    lam = uv + math.sqrt(disc)
    d = (lam * u - v) / impulse
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else u


def predict(pos, vel, ticks, damp):
    """Posición después de `ticks` ticks con frenado geométrico (sin fuerzas nuevas)."""
    if ticks <= 0:
        return np.asarray(pos, float).copy()
    f = (1 - damp ** ticks) / (1 - damp)
    return np.asarray(pos, float) + np.asarray(vel, float) * f


def _hist_index(t, delay, frame_skip=FRAME_SKIP):
    """Para el tick t (0 = primer tick del próximo paso): cuántas decisiones antes de la de ahora se tomó la acción que
    aplica el kernel (decision_step: h = ⌈(delay − k)/frame_skip⌉ en el historial ya con la decisión nueva). ≤ 0 si
    ya aplica la de ahora o una posterior."""
    j, k = divmod(t, frame_skip)
    lag = delay - k
    h = -(-lag // frame_skip) if lag > 0 else 0
    return h - j


def queued_ticks(delay, frame_skip=FRAME_SKIP):
    """Ticks desde el próximo paso hasta que empieza a aplicarse la acción decidida ahora (delay 9 → 9, 10 → 10)."""
    t = 0
    while _hist_index(t, delay, frame_skip) > 0:
        t += 1
    return t


def roll_queue(pos, vel, hist, delay, frame_skip=FRAME_SKIP):
    """Jugador (sin choques) hasta que aplica la decisión de ahora, con las teclas que ya están en cola.
    `hist`: acciones del mundo de este jugador, hist[0] la última decidida (RS4ZEnv.act_hist[n, p])."""
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    T = queued_ticks(delay, frame_skip)
    for t in range(T):
        a = int(hist[min(_hist_index(t, delay, frame_skip), len(hist)) - 1])
        acc = P_KACC if a >= 9 else P_ACC
        vx += UNIT[a % 9, 0] * acc
        vy += UNIT[a % 9, 1] * acc
        x += vx
        y += vy
        vx *= P_DAMP
        vy *= P_DAMP
    return np.array([x, y]), np.array([vx, vy]), T


class PassMacro:
    """Estado del pase de un jugador: a quién, desde cuándo, si ya pateó. Una instancia por (partido, jugador)."""

    def __init__(self, receiver, timeout=40, align=0.94, ball_r=8.325, aim_tol_deg=8.0):
        self.receiver = int(receiver)
        self.timeout = timeout          # decisiones para patear (40 = 2 s); después se abandona
        self.align = align              # coseno mínimo entre jugador → pelota y la dirección de patada
        self.ball_r = ball_r
        self.aim_tol = math.cos(math.radians(aim_tol_deg))   # patada aceptada si sale a menos de esto del objetivo
        self.steps = 0
        self.kicked = False
        self.done = False
        self.aim = None                 # punto al que apuntó al patear (para medir la precisión)

    def act(self, me, ball, ball_vel, my_pos, my_vel, mates_pos, mates_vel, lag, hist=None, delay=None):
        """Acción del marco propio para esta decisión. `mates_pos/vel`: (8, 2) de todos los lugares (mundo); `lag`:
        ticks entre esta decisión y su aplicación (retardo D + 1). Con `hist` (RS4ZEnv.act_hist[n, me]) y `delay`
        (RS4ZEnv.delay[n, me]) usa las teclas en cola y elige la acción simulando (`_plan`)."""
        team = int(TEAM[me])
        self.steps += 1
        if self.steps > self.timeout:
            self.done = True
        if self.done or self.kicked:
            return own_action(0, False, team)
        if hist is not None:
            p, v, lag = roll_queue(my_pos, my_vel, hist, int(delay))
        else:
            p = predict(my_pos, my_vel, lag, 0.96)
            v = None
        b = predict(ball, ball_vel, lag, BALL_DAMP)
        bv = np.asarray(ball_vel, float) * BALL_DAMP ** lag
        # objetivo: el compañero adelantado según el tiempo de viaje
        r = self.receiver
        tgt = np.asarray(mates_pos[r], float)
        dist = float(np.linalg.norm(tgt - b))
        t = travel_ticks(dist, KICK_IMPULSE + float(np.linalg.norm(bv)))
        t = min(t if math.isfinite(t) else 60.0, 60.0)
        tgt = tgt + np.asarray(mates_vel[r], float) * t * 0.6
        d = kick_direction(bv, tgt - b)
        contact = PLAYER_R + self.ball_r
        if v is not None:
            return self._plan(team, p, v, b, bv, tgt, contact)
        behind = b - d * (contact + 2.0)
        rel = b - p                                     # jugador → pelota
        dist_pb = float(np.linalg.norm(rel))
        cos = float(rel @ d) / max(dist_pb, 1e-9)
        if cos > self.align and dist_pb < contact + KICK_REACH + 6.0:
            # alineado y casi en contacto: apretar patear yendo hacia la pelota (la patada sale al entrar en alcance)
            self.aim = tgt
            return own_action(move_toward(*d, dead=0.0), True, team)
        if cos < 0.3 and dist_pb < contact + 40.0:
            # del lado equivocado: rodear la pelota por el costado en que está el jugador
            side = np.array([-d[1], d[0]])
            if float((p - b) @ side) < 0:
                side = -side
            way = b + side * (contact + 18.0) - d * 10.0
            return own_action(move_toward(*(way - p)), False, team)
        return own_action(move_toward(*(behind - p), dead=2.0), False, team)

    def _plan(self, team, p, v, b, bv, tgt, contact):
        """Elige entre las 18 acciones simulando desde el momento en que se aplica la de ahora (jugador sin choques,
        pelota libre). Patear: la acción sostenida PLAN_TICKS ticks; vale si la patada sale dentro del horizonte a menos
        de `aim_tol_deg` del objetivo, y gana la que patea antes. Si ninguna, moverse: la acción 3 ticks y después sin teclas, y
        gana la que termina más cerca del punto de patada (detrás de la pelota sobre la línea, o rodeándola si está del
        lado equivocado), con castigo si empuja la pelota con el cuerpo."""
        best_kick, best_kick_t = -1, PLAN_TICKS
        best_move, best_cost = 0, math.inf
        # lado por el que rodear, fijo en la decisión (con el jugador de ahora)
        d0 = kick_direction(bv, tgt - b)
        side = np.array([-d0[1], d0[0]])
        if float((p - b) @ side) < 0:
            side = -side
        for m in range(9):
            ux, uy = UNIT[m]
            for kick in (True, False):
                x, y, vx, vy = float(p[0]), float(p[1]), float(v[0]), float(v[1])
                bx, by, bvx, bvy = float(b[0]), float(b[1]), float(bv[0]), float(bv[1])
                bump = 0.0
                for t in range(PLAN_TICKS):
                    dx, dy = bx - x, by - y
                    dist = math.hypot(dx, dy)
                    if kick and dist - contact < KICK_REACH and dist > 0:
                        ox, oy = bvx + dx / dist * KICK_IMPULSE, bvy + dy / dist * KICK_IMPULSE
                        wx, wy = tgt[0] - bx, tgt[1] - by
                        c = (ox * wx + oy * wy) / (math.hypot(ox, oy) * math.hypot(wx, wy) + 1e-9)
                        if c > self.aim_tol and t < best_kick_t:
                            best_kick, best_kick_t = m, t
                        break
                    if not kick and dist < contact + 1.0:
                        bump += 1.0
                    held = kick or t < FRAME_SKIP
                    acc = (P_KACC if kick else P_ACC) if held else 0.0
                    vx += ux * acc
                    vy += uy * acc
                    x += vx
                    y += vy
                    vx *= P_DAMP
                    vy *= P_DAMP
                    bx += bvx
                    by += bvy
                    bvx *= BALL_DAMP
                    bvy *= BALL_DAMP
                if kick:
                    continue
                bb = np.array([bx, by])
                d = kick_direction(np.array([bvx, bvy]), tgt - bb)
                rel = bb - np.array([x, y])
                dist = float(np.linalg.norm(rel))
                cos = float(rel @ d) / max(dist, 1e-9)
                if cos < 0.3 and dist < contact + 40.0:
                    point = bb + side * (contact + 18.0) - d * 10.0
                else:
                    point = bb - d * (contact + 2.0)
                cost = math.hypot(point[0] - x, point[1] - y) + 15.0 * bump * (1.0 if cos < self.align else 0.2)
                if cost < best_cost:
                    best_move, best_cost = m, cost
        if best_kick >= 0:
            self.aim = tgt
            return own_action(best_kick, True, team)
        return own_action(best_move, False, team)

    def saw_kick(self, kicked_now):
        """Llamar después de cada paso con si este jugador pateó."""
        if kicked_now and not self.kicked:
            self.kicked = True
            self.done = True


# ---------------------------------------------------------------------------------------------- versión numba (RL)
from numba import njit, prange  # noqa: E402

N_PASS = 3                      # compañeros en la observación v3 (obs_v3.N_MATES), en el mismo orden
MAX_GAP = 40.0                  # px entre bordes jugador–pelota para ofrecer el pase
MAX_TURN_COS = 0.0              # giro ≤ 90° entre jugador → pelota y pelota → compañero
_UNIT = np.ascontiguousarray(UNIT)
_MIRROR = np.ascontiguousarray(MIRROR_MOVE.astype(np.int64))


@njit(cache=True)
def _kick_dir_nb(bvx, bvy, wx, wy):
    L = math.sqrt(wx * wx + wy * wy) + 1e-9
    ux, uy = wx / L, wy / L
    uv = ux * bvx + uy * bvy
    disc = uv * uv - (bvx * bvx + bvy * bvy) + KICK_IMPULSE * KICK_IMPULSE
    if disc < 0:
        return ux, uy
    lam = uv + math.sqrt(disc)
    dx, dy = (lam * ux - bvx) / KICK_IMPULSE, (lam * uy - bvy) / KICK_IMPULSE
    n = math.sqrt(dx * dx + dy * dy)
    if n > 1e-9:
        return dx / n, dy / n
    return ux, uy


@njit(cache=True)
def _plan_nb(px, py, vx0, vy0, bx0, by0, bvx0, bvy0, tx, ty, contact, align, aim_tol, unit):
    """Igual que PassMacro._plan; devuelve (movimiento del mundo, patear)."""
    best_kick, best_kick_t = -1, PLAN_TICKS
    best_move, best_cost = 0, 1e18
    d0x, d0y = _kick_dir_nb(bvx0, bvy0, tx - bx0, ty - by0)
    sx, sy = -d0y, d0x
    if (px - bx0) * sx + (py - by0) * sy < 0:
        sx, sy = -sx, -sy
    for m in range(9):
        ux, uy = unit[m, 0], unit[m, 1]
        for kk in range(2):
            kick = kk == 0
            x, y, vx, vy = px, py, vx0, vy0
            bx, by, bvx, bvy = bx0, by0, bvx0, bvy0
            bump = 0.0
            for t in range(PLAN_TICKS):
                dx, dy = bx - x, by - y
                dist = math.sqrt(dx * dx + dy * dy)
                if kick and dist - contact < KICK_REACH and dist > 0:
                    ox, oy = bvx + dx / dist * KICK_IMPULSE, bvy + dy / dist * KICK_IMPULSE
                    wx, wy = tx - bx, ty - by
                    c = (ox * wx + oy * wy) / (math.sqrt(ox * ox + oy * oy) * math.sqrt(wx * wx + wy * wy) + 1e-9)
                    if c > aim_tol and t < best_kick_t:
                        best_kick, best_kick_t = m, t
                    break
                if not kick and dist < contact + 1.0:
                    bump += 1.0
                held = kick or t < FRAME_SKIP
                acc = 0.0
                if held:
                    acc = P_KACC if kick else P_ACC
                vx += ux * acc
                vy += uy * acc
                x += vx
                y += vy
                vx *= P_DAMP
                vy *= P_DAMP
                bx += bvx
                by += bvy
                bvx *= BALL_DAMP
                bvy *= BALL_DAMP
            if kick:
                continue
            ddx, ddy = _kick_dir_nb(bvx, bvy, tx - bx, ty - by)
            rx, ry = bx - x, by - y
            dist = math.sqrt(rx * rx + ry * ry)
            cos = (rx * ddx + ry * ddy) / max(dist, 1e-9)
            if cos < 0.3 and dist < contact + 40.0:
                qx = bx + sx * (contact + 18.0) - ddx * 10.0
                qy = by + sy * (contact + 18.0) - ddy * 10.0
            else:
                qx = bx - ddx * (contact + 2.0)
                qy = by - ddy * (contact + 2.0)
            cost = math.sqrt((qx - x) ** 2 + (qy - y) ** 2) + 15.0 * bump * (1.0 if cos < align else 0.2)
            if cost < best_cost:
                best_move, best_cost = m, cost
    if best_kick >= 0:
        return best_kick, True
    return best_move, False


@njit(cache=True, parallel=True)
def _pass_table_nb(ball_pos, ball_vel, ball_r, ppos, pvel, act_hist, delay, active, max_gap, max_turn_cos,
                   align, aim_tol, unit, mirror, out):
    N = ppos.shape[0]
    H = act_hist.shape[2]
    for n in prange(N):
        for p in range(8):
            for k in range(N_PASS):
                out[n, p, k] = -1
            if not active[n, p]:
                continue
            contact = PLAYER_R + ball_r[n]
            bx, by = ball_pos[n, 0], ball_pos[n, 1]
            rx, ry = bx - ppos[n, p, 0], by - ppos[n, p, 1]
            rd = math.sqrt(rx * rx + ry * ry)
            if rd - contact > max_gap:
                continue
            # jugador al momento en que aplica la decisión de ahora, con las teclas en cola (roll_queue)
            D = delay[n, p]
            x, y, vx, vy = ppos[n, p, 0], ppos[n, p, 1], pvel[n, p, 0], pvel[n, p, 1]
            t = 0
            while True:
                j = t // FRAME_SKIP
                kt = t % FRAME_SKIP
                lag = D - kt
                h = (lag + FRAME_SKIP - 1) // FRAME_SKIP if lag > 0 else 0
                idx = h - j
                if idx <= 0:
                    break
                a = act_hist[n, p, min(idx, H) - 1]
                acc = P_KACC if a >= 9 else P_ACC
                vx += unit[a % 9, 0] * acc
                vy += unit[a % 9, 1] * acc
                x += vx
                y += vy
                vx *= P_DAMP
                vy *= P_DAMP
                t += 1
            f = (1.0 - BALL_DAMP ** t) / (1.0 - BALL_DAMP) if t > 0 else 0.0
            pbx, pby = bx + ball_vel[n, 0] * f, by + ball_vel[n, 1] * f
            bvx, bvy = ball_vel[n, 0] * BALL_DAMP ** t, ball_vel[n, 1] * BALL_DAMP ** t
            team = 0 if p < 4 else 1
            e = 0
            for q in range(8):
                if q == p or not active[n, q] or (0 if q < 4 else 1) != team:
                    continue
                if e >= N_PASS:
                    break
                k = e
                e += 1
                wx, wy = ppos[n, q, 0] - bx, ppos[n, q, 1] - by
                if (rx * wx + ry * wy) / (rd * math.sqrt(wx * wx + wy * wy) + 1e-9) < max_turn_cos:
                    continue
                tx, ty = ppos[n, q, 0], ppos[n, q, 1]
                dist = math.sqrt((tx - pbx) ** 2 + (ty - pby) ** 2)
                v0 = KICK_IMPULSE + math.sqrt(bvx * bvx + bvy * bvy)
                reach = v0 / (1 - BALL_DAMP)
                tt = 60.0
                if dist < reach * 0.98:
                    tt = min(math.log(1 - dist / reach) / math.log(BALL_DAMP), 60.0)
                tx += pvel[n, q, 0] * tt * 0.6
                ty += pvel[n, q, 1] * tt * 0.6
                m, kick = _plan_nb(x, y, vx, vy, pbx, pby, bvx, bvy, tx, ty, contact, align, aim_tol, unit)
                if team == 1:
                    m = mirror[m]
                out[n, p, k] = m + (9 if kick else 0)


def pass_table(env, out=None, max_gap=MAX_GAP, max_turn_cos=MAX_TURN_COS, align=0.94, aim_tol_deg=8.0):
    """(N, 8, N_PASS) int64: acción del marco propio que ejecuta "pasarle al compañero k" (k en el orden de la
    observación v3), o −1 si no se ofrece (sin compañero, pelota lejos o giro > 90°)."""
    N = env.N
    if out is None:
        out = np.empty((N, 8, N_PASS), np.int64)
    _pass_table_nb(np.ascontiguousarray(env.pos[:, 0]), np.ascontiguousarray(env.vel[:, 0]),
                   np.ascontiguousarray(env.radius[:, 0]), env.player_pos, env.player_vel, env.act_hist, env.delay,
                   env.active, float(max_gap), float(max_turn_cos), float(align),
                   math.cos(math.radians(aim_tol_deg)), _UNIT, _MIRROR, out)
    return out


def to_primitive(acts, table):
    """Acciones con macros (0..17 teclas, 18+k pasarle al compañero k) → teclas (0..17). Un pase no ofrecido queda
    quieto (no debería muestrearse: la política lo enmascara)."""
    acts = np.asarray(acts)
    if acts.max(initial=0) < 18:
        return acts
    k = np.clip(acts - 18, 0, N_PASS - 1)
    mac = np.take_along_axis(table, k[..., None], -1)[..., 0]
    return np.where(acts >= 18, np.maximum(mac, 0), acts)


MACRO_TIMEOUT = 20              # decisiones (1 s) para patear; después se suelta


class MacroState:
    """Compromiso de las macro-acciones de pase de N partidos: elegido "pasarle a k", el controlador maneja al jugador
    hasta que patea, el pase deja de ofrecerse (pelota lejos, giro > 90°) o pasa MACRO_TIMEOUT. Como el pase en Google
    Research Football: una acción. Mientras dura, lo que muestrea la política para ese jugador se ignora."""

    def __init__(self, N):
        self.k = np.full((N, 8), -1, np.int64)
        self.steps = np.zeros((N, 8), np.int64)

    def apply(self, acts, table):
        """acts (N, 8) con macros → (teclas (N, 8), jugadores que ya venían comprometidos (N, 8))."""
        acts = np.asarray(acts)
        cur = np.take_along_axis(table, np.maximum(self.k, 0)[..., None], -1)[..., 0]
        self.k[(self.k >= 0) & (cur < 0)] = -1
        busy = self.k >= 0
        new = ~busy & (acts >= 18)
        if new.any():
            kk = np.clip(acts - 18, 0, N_PASS - 1)
            ok = np.take_along_axis(table, kk[..., None], -1)[..., 0] >= 0
            self.k[new & ok] = kk[new & ok]
            self.steps[new & ok] = 0
        run = self.k >= 0
        mac = np.take_along_axis(table, np.maximum(self.k, 0)[..., None], -1)[..., 0]
        prim = np.where(run, mac, np.where(acts >= 18, 0, acts))
        return prim, busy

    def after(self, kicked, reset_rows=()):
        """Después del paso: suelta a los que patearon o se pasaron de tiempo, y a los partidos que se reiniciaron."""
        run = self.k >= 0
        self.steps[run] += 1
        self.k[run & (np.asarray(kicked) | (self.steps >= MACRO_TIMEOUT))] = -1
        if len(reset_rows):
            self.k[reset_rows] = -1

    def running(self):
        return self.k >= 0
