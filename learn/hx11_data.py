"""Datos de imitación de HaxEleven (L0) desde el caché D0, con la observación armada al vuelo.

Una muestra es (partido, tick de la etiqueta t, lugar s, retardo d), con la convención de X4 (`learn/x4_data.py`):
el jugador decide con el estado del tick t − d − 1 y su etiqueta es la entrada que la sala aplicó en t. El retardo se
sortea por muestra en [D_MIN, D_MAX] (S6: la latencia de sala va de 8 a 36 ticks) y entra como observación.

**Condición de puesto (revisión de la auditoría, decisión P2):** no el rol de plantilla por tick de D2 (el jugador
está fuera de su rol dominante el 37% del tiempo y ese rol sale de la posición actual: filtraría la respuesta), sino
lo que el usuario fija con `--pos`, que es un puesto por tramo:
* el **ancla** del jugador en esa mitad del partido: su posición media relativa al centroide de los jugadores de campo
  de su equipo, con y sin la pelota (4 números, medidos, sin etiquetador);
* la **línea** de su rol dominante del partido en D2 (ARQ, DEF, MCD, MED, DEL), que coincide entre mitades en el 83%.

**Modos (E2):** offside prendido por tick (anuncios) y faltas por partido (`faltas_l0` de `tools.hx11_modos`).

Todo en coordenadas del equipo (ataca hacia +x): para el azul se rota 180° el estado y también la acción.
Sin marcador ni reloj (preferencia del usuario para el actor).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.hx11_d0 import KICKOFF, OPEN, OUT_PENDING, PIECE, apt_names

ROOT = Path(__file__).resolve().parent.parent
D0 = ROOT / "data" / "hx11" / "d0"
D1 = ROOT / "data" / "hx11" / "d1"
D2 = ROOT / "data" / "hx11" / "d2"
D_MIN, D_MAX = 8, 36
HIST = (3, 6, 9, 12, 15)                 # entradas propias ya enviadas (como X4: sin t−1 ni t−2, contra el copycat)
MAX_BACK = D_MAX + 1 + max(HIST)
LABEL_PHASES = (OPEN, OUT_PENDING, PIECE, KICKOFF)
LINES = ("ARQ", "DEF", "MCD", "MED", "DEL")
N_KIND = 8
MOVE_DX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
MOVE_DY = np.array([0, -1, -1, 0, 1, 1, 1, 0, -1])
ROT_MOVE = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])        # rotación de 180°: (dx, dy) → (−dx, −dy)
FAULT_MODES = ("prendido", "apagado", "desconocido")
CTX = 11
SELF_DIM = 69
EXTRA_DIM = 5          # observación v2: rasgos de "voy yo" (ver `engage`)
ENT_DIM = 9
N_MATES, N_ENT = 10, 21
# cabeza de duración (brazo del 2026-10-08 contra la vibración sin historia propia): cuántos ticks más sostiene el
# humano la tecla actual. Clases por largo restante: <=3, <=6, <=12, <=24, <=48, más; se ejecuta el representante.
HOLD_EDGES = np.array([3, 6, 12, 24, 48])
HOLD_REP = np.array([3, 6, 10, 18, 36, 60])


def move_of_input(inp):
    inp = inp.astype(np.int64)
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    out = np.zeros(inp.shape, np.int64)
    for i, (x, y) in enumerate(zip(MOVE_DX, MOVE_DY)):
        out[(dx == x) & (dy == y)] = i
    return out


def action_of_input(inp, blue):
    """Acción 0..17 (dirección + 9 × patada) en coordenadas del equipo."""
    mv = move_of_input(inp)
    mv = np.where(blue, ROT_MOVE[mv], mv)
    return mv + 9 * ((inp.astype(np.int64) >> 4) & 1)


def anchors(o, owner, gk):
    """Ancla por (id de jugador, mitad): posición media relativa al centroide de los jugadores de campo, con y sin la
    pelota. El arquero también tiene la suya (no entra en el centroide)."""
    T = len(owner)
    pid = o["slot_pid"]
    pos = o["pos"].astype(np.float32) / 10
    ticks = np.nonzero(o["phase"] == OPEN)[0][::6]
    half = (ticks >= T // 2).astype(np.int64)
    out = {}
    for team in (0, 1):
        sl = np.arange(team * 11, team * 11 + 11)
        att = 1.0 if team == 0 else -1.0
        p = pid[ticks][:, sl]
        q = pos[ticks][:, sl] * att
        field = (p >= 0) & (p != gk[ticks, team][:, None])
        n = np.maximum(field.sum(1), 1)
        cen = (q * field[..., None]).sum(1) / n[:, None]
        rel = q - cen[:, None]
        own = owner[ticks]
        wb = np.where(own < 0, -1, (own // 11) == team)
        is_gk = (p >= 0) & (p == gk[ticks, team][:, None])
        for h in (0, 1):
            for i in range(11):
                # el arquero no entra en el centroide, pero tiene su ancla respecto de él
                m = (field[:, i] | is_gk[:, i]) & (half == h)
                for pp in np.unique(p[m, i]):
                    mm = m & (p[:, i] == pp)
                    a = np.zeros(4, np.float32)
                    for k, flag in enumerate((1, 0)):
                        sel = mm & (wb == flag)
                        if sel.sum() >= 30:
                            a[2 * k:2 * k + 2] = rel[sel, i].mean(0)
                        elif mm.sum():
                            a[2 * k:2 * k + 2] = rel[mm, i].mean(0)
                    key = (int(pp), h)
                    if key not in out or mm.sum() > out[key][1]:
                        out[key] = (a, int(mm.sum()))
    return {k: v[0] for k, v in out.items()}


@dataclass
class Match:
    name: str
    T: int
    pos: np.ndarray        # (T, 22, 2) int16 ×10
    vel: np.ndarray        # (T, 22, 2) int16 ×1000
    ball: np.ndarray       # (T, 2) int16 ×10
    bvel: np.ndarray       # (T, 2) int16 ×1000
    binv: np.ndarray       # (T,) float16
    bg: np.ndarray         # (T, 2) float16
    inp: np.ndarray        # (T, 22) int8
    kick: np.ndarray       # (T, 22) int8: patada efectiva en ese tick
    score: np.ndarray      # (T, 2) int16 (solo para medir; el actor no lo ve)
    pid: np.ndarray        # (T, 22) int32
    gk: np.ndarray         # (T, 2) int32
    phase: np.ndarray
    pkind: np.ndarray
    pteam: np.ndarray
    pticks: np.ndarray
    spot: np.ndarray
    offside: np.ndarray
    fouls: int             # índice en FAULT_MODES
    anchor: dict           # (pid, mitad) → 4 floats
    line: dict             # pid → índice en LINES
    hold: np.ndarray = None    # (T, 22) int16: ticks seguidos con la misma tecla desde t (incluido)


def load_match(name, modos=None, hold=False):
    """Partido en memoria para la imitación. Dieta de RAM (2026-10-08: con 290 partidos en Windows se pasaba de 8 GB):
    ids de jugador en int16 y duraciones (`hold`) solo si se piden."""
    from tools.hx11_d0 import load_d0
    o, _ = load_d0(D0 / name)
    owner = np.load(D1 / f"{name}.npz")["owner"]
    gk = np.where(o["gk_pid"] >= 0, o["gk_pid"], o["gk_inferred"]).astype(np.int32)
    d2 = json.loads((D2 / f"{name}.json").read_text(encoding="utf-8"))
    line = {}
    for t in d2["equipos"]:
        for p, info in t["jugadores"].items():
            ln = info["nombre_rol"].split("-")[0]
            line[int(p)] = LINES.index(ln) if ln in LINES else 3
    fm = (modos or {}).get(name, {}).get("faltas", {}).get("modo_l0", "desconocido")
    return Match(name=name, T=len(owner), pos=o["pos"], vel=o["vel"], ball=o["ball_pos"], bvel=o["ball_vel"],
                 binv=o["ball_inv"], bg=o["ball_g"], inp=o["input"], kick=o["kick"], score=o["score"],
                 pid=o["slot_pid"].astype(np.int16) if o["slot_pid"].max() < 32767 else o["slot_pid"], gk=gk, phase=o["phase"],
                 pkind=o["piece_kind"], pteam=o["piece_team"], pticks=o["phase_ticks"], spot=o["spot"],
                 offside=o["offside_on"], fouls=FAULT_MODES.index(fm), anchor=anchors(o, owner, gk), line=line,
                 hold=hold_runs(o["input"]) if hold else None)


def hold_runs(inp):
    """Largo de la corrida de la misma tecla hacia adelante desde cada tick (tope 120)."""
    T = len(inp)
    out = np.ones(inp.shape, np.int16)
    for t in range(T - 2, -1, -1):
        out[t] = np.where(inp[t] == inp[t + 1], np.minimum(out[t + 1] + 1, 120), 1)
    return out


def hold_class(m: Match, t, s):
    return np.searchsorted(HOLD_EDGES, m.hold[t, s].astype(np.int64), side="left")


def valid_samples(m: Match, stride=1):
    """(t, s) válidos: fase con decisión, jugador presente y el mismo desde t − MAX_BACK, entrada conocida."""
    t = np.arange(MAX_BACK, m.T, stride)
    t = t[np.isin(m.phase[t], LABEL_PHASES)]
    ok = (m.pid[t] >= 0) & (m.pid[t] == m.pid[t - MAX_BACK]) & (m.inp[t] >= 0)
    tt, ss = np.nonzero(ok)
    return t[tt].astype(np.int32), ss.astype(np.int8)


def engage(me, ball, bvel, pos, on, mates_idx, rivs_idx, s):
    """Rasgos de "voy yo" (v2, 2026-10-08: en lazo cerrado los bots no van a la pelota; con los 22 imitando, el más
    cercano queda a 200–500 px contra 57 px entre humanos). Con media y máximo, el codificador de conjuntos no saca
    fácil un ranking:
    * mi puesto en la cercanía a la pelota entre los jugadores de mi equipo (0 = el más cercano) / 10;
    * soy el más cercano de mi equipo;
    * distancia del compañero más cercano − la mía, y del rival más cercano − la mía (/1000; > 0 = llego antes);
    * lo mismo hacia donde va a estar la pelota en ~20 ticks (frenado 0,99)."""
    B = len(s)
    ar = np.arange(B)
    proj = ball + bvel * (1 - 0.99 ** 20) / (1 - 0.99)
    out = np.zeros((B, EXTRA_DIM), np.float32)
    dm = np.hypot(*(me - ball).T)
    pm = pos[ar[:, None], mates_idx]
    dmates = np.where(on[ar[:, None], mates_idx] & (mates_idx != s[:, None]),
                      np.hypot(pm[..., 0] - ball[:, None, 0], pm[..., 1] - ball[:, None, 1]), np.inf)
    pr = pos[ar[:, None], rivs_idx]
    drivs = np.where(on[ar[:, None], rivs_idx], np.hypot(pr[..., 0] - ball[:, None, 0], pr[..., 1] - ball[:, None, 1]),
                     np.inf)
    out[:, 0] = (dmates < dm[:, None]).sum(1) / 10
    out[:, 1] = out[:, 0] == 0
    out[:, 2] = np.clip(dmates.min(1) - dm, -2000, 2000) / 1000
    out[:, 3] = np.clip(drivs.min(1) - dm, -2000, 2000) / 1000
    dmp = np.hypot(*(me - proj).T)
    drp = np.where(on[ar[:, None], rivs_idx], np.hypot(pr[..., 0] - proj[:, None, 0], pr[..., 1] - proj[:, None, 1]),
                   np.inf)
    out[:, 4] = np.clip(drp.min(1) - dmp, -2000, 2000) / 1000
    return out


def build(m: Match, t, s, d, extra=False):
    """Observación (propia (B, SELF_DIM [+ EXTRA_DIM]), entidades (B, N_ENT, ENT_DIM)), acción etiqueta (B,) y si
    cambió la tecla. `extra` agrega los rasgos de `engage` (observación v2)."""
    B = len(t)
    to = t - d - 1
    team = (s >= 11).astype(np.int64)
    att = np.where(team == 0, 1.0, -1.0)[:, None]
    pos = m.pos[to].astype(np.float32) / 10 * att[:, :, None]          # (B, 22, 2)
    vel = m.vel[to].astype(np.float32) / 1000 * att[:, :, None]
    ball = m.ball[to].astype(np.float32) / 10 * att
    bvel = m.bvel[to].astype(np.float32) / 1000 * att
    on = m.pid[to] >= 0
    ar = np.arange(B)
    me, mv = pos[ar, s], vel[ar, s]
    gk_team = m.gk[to, team]
    gk_riv = m.gk[to, 1 - team]
    pid_to = m.pid[to]
    is_gk = (pid_to == np.where(np.arange(22)[None] < 11, m.gk[to, 0][:, None], m.gk[to, 1][:, None])) & on
    kicking = ((m.inp[to].astype(np.int64) >> 4) & 1) * on
    own = np.zeros((B, SELF_DIM), np.float32)
    own[:, 0:2] = me / 1000
    own[:, 2:4] = mv / 3
    own[:, 4:6] = (ball - me) / 1000
    own[:, 6:8] = bvel / 10
    own[:, 8:10] = ball / 1000
    own[:, 10] = np.hypot(*(ball - me).T) / 1000
    own[:, 11] = (m.binv[to].astype(np.float32) - 1.65) / 2
    own[:, 12:14] = m.bg[to].astype(np.float32) * att * 100
    own[:, 14] = kicking[ar, s]
    blue = team == 1
    for k, h in enumerate(HIST):
        ih = m.inp[t - h, s]
        a = action_of_input(np.maximum(ih, 0), blue)
        mvh = a % 9
        own[:, 15 + 3 * k] = MOVE_DX[mvh]
        own[:, 16 + 3 * k] = MOVE_DY[mvh]
        own[:, 17 + 3 * k] = a // 9
    own[:, 30] = d / D_MAX
    ph = m.phase[to].astype(np.int64)
    own[ar, 31 + np.clip(ph, 0, 5)] = 1
    pk = np.clip(m.pkind[to].astype(np.int64), 0, N_KIND - 1)
    own[ar, 37 + pk] = 1
    pt = m.pteam[to].astype(np.int64)
    own[ar, 45 + np.where(pt < 0, 2, np.where(pt == team, 0, 1))] = 1
    own[:, 48] = np.minimum(m.pticks[to].astype(np.float32), 600) / 600
    own[:, 49:51] = (m.spot[to].astype(np.float32) / 10 * att - me) / 1000
    own[:, 51] = m.offside[to]
    own[:, 52 + m.fouls] = 1
    half = (t >= m.T // 2).astype(np.int64)
    pid_s = m.pid[t, s]
    # ancla y línea: búsqueda por valor único (en un lote hay pocos jugadores; antes era un bucle por muestra)
    uk, inv = np.unique(pid_s.astype(np.int64) * 2 + half, return_inverse=True)
    tab = np.array([m.anchor.get((int(k // 2), int(k % 2)), m.anchor.get((int(k // 2), 1 - int(k % 2)),
                                                                          np.zeros(4, np.float32))) for k in uk],
                   np.float32).reshape(-1, 4)
    own[:, 55:59] = tab[inv] / 1000
    up, invp = np.unique(pid_s, return_inverse=True)
    ln = np.array([m.line.get(int(p), 3) for p in up], np.int64)[invp]
    own[ar, 59 + ln] = 1
    mates = np.where(team[:, None] == 0, np.arange(11)[None], np.arange(11, 22)[None])     # (B, 11)
    rivs = np.where(team[:, None] == 0, np.arange(11, 22)[None], np.arange(11)[None])
    field_m = on[ar[:, None], mates] & ~is_gk[ar[:, None], mates]
    cm = (pos[ar[:, None], mates] * field_m[..., None]).sum(1) / np.maximum(field_m.sum(1), 1)[:, None]
    own[:, 64:66] = (cm - me) / 1000
    xr = np.where(on[ar[:, None], rivs], pos[ar[:, None], rivs, 0], -np.inf)
    own[:, 66] = np.sort(xr, 1)[:, -2].clip(-1800, 1800) / 1000        # línea de offside para mí (anteúltimo rival)
    xm = np.where(on[ar[:, None], mates], pos[ar[:, None], mates, 0], np.inf)
    own[:, 67] = np.sort(xm, 1)[:, 1].clip(-1800, 1800) / 1000          # mi línea de offside para los rivales
    own[:, 68] = is_gk[ar, s]
    # entidades: 10 compañeros (sin mí) y 11 rivales
    others_m = mates[mates != s[:, None]].reshape(B, N_MATES)                                     # (B, 10)
    idx = np.concatenate([others_m, rivs], 1)                                                         # (B, 21)
    ents = np.zeros((B, N_ENT, ENT_DIM), np.float32)
    pe = pos[ar[:, None], idx]
    ents[..., 0] = on[ar[:, None], idx]
    ents[..., 1:3] = (pe - me[:, None]) / 1000
    ents[..., 3:5] = vel[ar[:, None], idx] / 3
    ents[..., 5:7] = (pe - ball[:, None]) / 1000
    ents[..., 7] = is_gk[ar[:, None], idx]
    ents[..., 8] = kicking[ar[:, None], idx]
    ents[~on[ar[:, None], idx]] = 0.0
    if extra:
        own = np.concatenate([own, engage(me, ball, bvel, pos, on, mates, rivs, s)], 1)
    y = action_of_input(m.inp[t, s], blue)
    prev = action_of_input(np.maximum(m.inp[t - 1, s], 0), blue)
    return own, ents, y, y != prev


class Sampler:
    """Muestras al azar de varios partidos: (propia, entidades, acción, cambió, línea)."""

    def __init__(self, names, stride=2, seed=0, modos=None, extra=False, cerca_frac=0.0, cerca_px=150.0, hold=False,
                 por_lote=32):
        """`cerca_frac` > 0: esa fracción de cada lote sale de muestras con el jugador a <= `cerca_px` de la pelota en
        el tick de la etiqueta (sobremuestreo de los momentos con pelota; brazo del 2026-10-08)."""
        self.rng = np.random.default_rng(seed)
        self.extra = extra
        self.hold = hold
        self.por_lote = por_lote
        self.cerca_frac = cerca_frac
        self.matches = [load_match(n, modos, hold=hold) for n in names]
        self.idx = [valid_samples(m, stride) for m in self.matches]
        self.near = []
        for m, (t, s) in zip(self.matches, self.idx):
            if cerca_frac <= 0:
                self.near.append(np.zeros(0, np.int32))
                continue
            d = np.hypot(*(m.pos[t, s].astype(np.float32) - m.ball[t].astype(np.float32)).T) / 10
            self.near.append(np.nonzero(d <= cerca_px)[0].astype(np.int32))
        self.wn = np.array([len(n) for n in self.near], np.float64)
        self.wn /= max(self.wn.sum(), 1)
        self.w = np.array([len(t) for t, _ in self.idx], np.float64)
        self.w /= self.w.sum()

    def _draw(self, n, p, near):
        """n muestras de `por_lote` partidos sorteados (con el peso p): con 290 partidos, repartir cada lote entre
        todos hacía que el costo fijo por partido dominara (3 pasos/s contra 9 con 20 partidos, 2026-10-08)."""
        k = min(self.por_lote, len(self.matches))
        js = self.rng.choice(len(self.matches), size=k, p=p)
        cnt = self.rng.multinomial(n, np.full(k, 1.0 / k))
        out = []
        for j in np.unique(js):
            m_n = int(cnt[js == j].sum())
            if m_n == 0:
                continue
            t_all, s_all = self.idx[j]
            r = (self.near[j][self.rng.integers(0, len(self.near[j]), m_n)] if near
                 else self.rng.integers(0, len(t_all), m_n))
            d = self.rng.integers(D_MIN, D_MAX + 1, m_n)
            out.append(build(self.matches[j], t_all[r], s_all[r], d, self.extra)
                       + ((hold_class(self.matches[j], t_all[r], s_all[r]),) if self.hold else ()))
        return out

    def batch(self, B):
        nb = int(round(B * self.cerca_frac))
        parts = (self._draw(nb, self.wn, True) if nb else []) + self._draw(B - nb, self.w, False)
        return tuple(np.concatenate([q[i] for q in parts]) for i in range(len(parts[0])))

def l0_split(limit_train=None, limit_dev=None):
    """Partidos aptos por parte (S0): train, dev, test (incluye test_grupos)."""
    full = json.loads((D0 / "manifest.json").read_text(encoding="utf-8"))
    man = full["recordings"]
    out = {"train": [], "dev": [], "test": []}
    for n in apt_names(full):
        sp = man[n + ".hbr2"]["split"]
        if (D2 / f"{n}.json").exists():
            out["test" if sp.startswith("test") else sp].append(n)
    out["train"] = out["train"][:limit_train]
    out["dev"] = out["dev"][:limit_dev]
    return out
