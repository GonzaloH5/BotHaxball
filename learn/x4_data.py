"""Dataset de imitación X4 desde el caché `tools.x4_ticks`, con la observación v3 armada al vuelo.

Una muestra es (tick de la etiqueta t, lugar p, retardo d): el jugador decide con el estado del registro
t − d − 1 y su etiqueta es la entrada que el host aplicó en t (convención del kernel: con retardo d, la
decisión tomada en S_t se aplica en el tick que produce S_{t+d+1}; ver `obs_v3.build_samples`). Las acciones propias pendientes salen de las entradas en
t − 3, …, t − 15 (`env/rs4z/obs_v3.build_samples`). Elegir d por muestra permite:
* estimar el retardo efectivo de los humanos (el que maximiza la verosimilitud de sus acciones; plan E1);
* entrenar la política con la distribución de latencias de la sala (revisión §5).

Ticks válidos: tramo 4v4 estable (mismos 8 jugadores, frames consecutivos, saque inicial o en juego, mapa
conocido) desde t − MAX_BACK hasta t + 2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env.rs4z import obs_v3

ROOT = Path(__file__).resolve().parent.parent
MAX_BACK = 3 * obs_v3.N_HIST + 25      # ticks hacia atrás que necesita una muestra (retardo ≤ 24)
MOVE_DX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
MOVE_DY = np.array([0, -1, -1, 0, 1, 1, 1, 0, -1])
MIRROR_MOVE = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2])
STATE_OPEN, STATE_RESTART_OWN, STATE_RESTART_RIVAL, STATE_KICKOFF_OWN, STATE_KICKOFF_RIVAL = range(5)
STATE_NAMES = ("abierto", "saque_propio", "saque_rival", "inicial_propio", "inicial_rival")


def move_of_input(inp):
    inp = inp.astype(np.int64)
    dx = ((inp >> 3) & 1) - ((inp >> 2) & 1)
    dy = ((inp >> 1) & 1) - (inp & 1)
    out = np.zeros(inp.shape, np.int64)
    for i, (x, y) in enumerate(zip(MOVE_DX, MOVE_DY)):
        out[(dx == x) & (dy == y)] = i
    return out


@dataclass
class X4Data:
    """Grabaciones concatenadas (arrays por tick) e índices de ticks válidos para etiquetas."""
    names: list
    rec_of_tick: np.ndarray
    ball: np.ndarray
    ball_r: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    inp: np.ndarray
    kicking: np.ndarray
    state: np.ndarray
    rkind: np.ndarray
    rteam: np.ndarray
    rage: np.ndarray
    ko_team: np.ndarray
    ko_age: np.ndarray
    mass: np.ndarray
    kstr: np.ndarray
    map_idx: np.ndarray
    pid: np.ndarray
    label: np.ndarray        # (T, 8) acción 0..17 en el marco propio
    kick_ev: np.ndarray      # (T, 8) patada efectiva (evento del motor) en ese registro
    valid: np.ndarray        # índices de ticks válidos como etiqueta
    ctx: np.ndarray          # (T, 8) contexto del estado (STATE_*) para métricas

    @property
    def ticks(self):
        return len(self.state)

    def player_key(self, t, p):
        """(grabación, id del jugador) del lugar p en el tick t."""
        return int(self.rec_of_tick[t]), int(self.pid[t, p])


def load(names, ticks_dir=ROOT / "data" / "x4_ticks", maps=("sanguchito_rs_x4",), stride_label=1):
    """Carga y concatena las grabaciones `names` (claves del índice). Sólo tramos en `maps`."""
    from tools.x4_ticks import MAP_IDS
    want = {MAP_IDS[m] for m in maps}
    to_v3 = {MAP_IDS[m]: obs_v3.MAP_NAMES.index(m) for m in MAP_IDS}
    parts = {k: [] for k in ("ball", "ball_r", "pos", "vel", "inp", "kicking", "state", "rkind", "rteam", "rage",
                             "ko_team", "ko_age", "mass", "kstr", "map_idx", "pid", "label", "kick_ev", "ok", "rec")}
    used = []
    for name in names:
        path = Path(ticks_dir) / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        frame = d["frame"]
        keep = np.isin(d["map_id"], list(want))
        if not keep.any():
            continue
        r = len(used)
        used.append(name)
        T = len(frame)
        # el inicio de algunas grabaciones tiene posiciones nulas (el replay empieza antes del estado completo)
        finite = (np.isfinite(d["ball"][:, :5]).all(1) & np.isfinite(d["pos"]).all((1, 2))
                  & np.isfinite(d["vel"]).all((1, 2)))
        ok = (keep & (d["n_red"] == 4) & (d["n_blue"] == 4) & np.isin(d["state"], (0, 1))
              & np.isfinite(d["kick_strength"]) & finite)
        same = np.zeros(T, bool)
        same[1:] = (np.diff(frame) == 1) & np.all(d["pid"][1:] == d["pid"][:-1], axis=1)
        # longitud de la racha estable que termina en cada tick
        run = np.zeros(T, np.int64)
        c = 0
        for i in range(T):
            c = c + 1 if (ok[i] and (i > 0 and same[i] and ok[i - 1])) else (1 if ok[i] else 0)
            run[i] = c
        # válido: racha ≥ MAX_BACK + 1 en t y el tramo sigue hasta t + 2
        fwd_ok = np.zeros(T, bool)
        fwd_ok[:-2] = (run[2:] >= run[:-2] + 2)
        lab_ok = (run > MAX_BACK) & fwd_ok
        kick_ev = np.zeros((T, 8), bool)
        if len(d["kicks"]):
            idx = {int(f): i for i, f in enumerate(frame)}
            for f, p in d["kicks"]:
                i = idx.get(int(f))
                if i is not None:
                    hit = np.flatnonzero(d["pid"][i] == p)
                    if len(hit):
                        kick_ev[i, hit[0]] = True
        # etiqueta: dirección de la tecla en t; patada si la tecla está apretada o hubo patada en [t, t+3)
        mv = move_of_input(d["inp"])
        mv[:, 4:] = MIRROR_MOVE[mv[:, 4:]]
        kick = (d["inp"] & 16) != 0
        win = kick_ev.copy()
        win[:-1] |= kick_ev[1:]
        win[:-2] |= kick_ev[2:]
        label = (mv + 9 * (kick | win)).astype(np.int8)
        parts["ball"].append(np.nan_to_num(d["ball"][:, :4]))
        parts["ball_r"].append(np.nan_to_num(d["ball"][:, 4].astype(np.float64), nan=8.325))
        parts["pos"].append(np.nan_to_num(d["pos"]))
        parts["vel"].append(np.nan_to_num(d["vel"]))
        parts["inp"].append(d["inp"])
        parts["kicking"].append(d["kicking"])
        parts["state"].append(d["state"])
        parts["rkind"].append(d["restart_kind"])
        parts["rteam"].append(d["restart_team"])
        parts["rage"].append(d["restart_age"])
        parts["ko_team"].append(d["ko_team"])
        parts["ko_age"].append(d["ko_age"])
        parts["mass"].append(d["mass"])
        parts["kstr"].append(np.nan_to_num(d["kick_strength"], nan=5.85))
        parts["map_idx"].append(np.array([to_v3.get(int(m), -1) for m in range(3)], np.int8)[d["map_id"]])
        parts["pid"].append(d["pid"])
        parts["label"].append(label)
        parts["kick_ev"].append(kick_ev)
        parts["ok"].append(lab_ok)
        parts["rec"].append(np.full(T, r, np.int32))
    if not used:
        raise ValueError("ninguna grabación con tramos en los mapas pedidos")
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    ok = cat.pop("ok")
    rec = cat.pop("rec")
    valid = np.flatnonzero(ok)
    if stride_label > 1:
        valid = valid[valid % stride_label == 0]
    # contexto por jugador
    T = len(ok)
    ctx = np.zeros((T, 8), np.int8)
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ko = cat["state"] == 0
    rk = (cat["rkind"] > 0) & ~ko & (cat["rteam"] >= 0)
    own_r = rk[:, None] & (cat["rteam"][:, None] == team[None])
    riv_r = rk[:, None] & (cat["rteam"][:, None] != team[None])
    own_k = ko[:, None] & (cat["ko_team"][:, None] == team[None])
    riv_k = ko[:, None] & (cat["ko_team"][:, None] != team[None])
    ctx[own_r] = STATE_RESTART_OWN
    ctx[riv_r] = STATE_RESTART_RIVAL
    ctx[own_k] = STATE_KICKOFF_OWN
    ctx[riv_k] = STATE_KICKOFF_RIVAL
    return X4Data(names=used, rec_of_tick=rec, valid=valid, ctx=ctx, **cat)


def featurize(data: X4Data, t_lab, slot, delay, out=None):
    """Observaciones v3 (B, OBS_DIM) para etiquetas en `t_lab` con retardo `delay` (ticks)."""
    t_lab = np.asarray(t_lab, np.int64)
    slot = np.asarray(slot, np.int64)
    delay = np.asarray(delay, np.int64)
    if out is None:
        out = np.empty((len(t_lab), obs_v3.OBS_DIM), np.float32)
    obs_v3.build_samples(t_lab - delay - 1, t_lab, slot, delay, data.ball, data.ball_r, data.pos, data.vel, data.inp,
                         data.kicking, data.state, data.rkind, data.rteam, data.rage, data.ko_team, data.ko_age,
                         data.mass, data.kstr, data.map_idx, obs_v3.line_h_table(), 1150.0, out)
    return out


class Sampler:
    """Lotes aleatorios (obs, acción, contexto) con retardo por muestra.

    delay: entero fijo, tupla (lo, hi) uniforme en ticks, o función (t_lab, slot) → retardo.
    """

    def __init__(self, data: X4Data, delay=0, seed=0):
        self.data, self.delay = data, delay
        self.rng = np.random.default_rng(seed)

    def _delays(self, t, p):
        d = self.delay
        if callable(d):
            return np.asarray(d(t, p), np.int64)
        if isinstance(d, tuple):
            return self.rng.integers(d[0], d[1] + 1, size=len(t))
        return np.full(len(t), int(d), np.int64)

    def batch(self, n):
        t = self.data.valid[self.rng.integers(0, len(self.data.valid), size=n)]
        p = self.rng.integers(0, 8, size=n)
        d = self._delays(t, p)
        obs = featurize(self.data, t, p, d)
        return obs, self.data.label[t, p].astype(np.int64), self.data.ctx[t, p].astype(np.int64), d

    def fixed(self, n, seed=1234):
        """Conjunto fijo de evaluación (mismas muestras en cada llamada)."""
        rng = np.random.default_rng(seed)
        t = self.data.valid[rng.integers(0, len(self.data.valid), size=n)]
        p = rng.integers(0, 8, size=n)
        return t, p


def split_names(splits_path=ROOT / "reports" / "x4" / "splits.json", split="train", families=None, maps=None):
    rows = json.loads(Path(splits_path).read_text(encoding="utf-8"))["recordings"]
    out = []
    for name, r in sorted(rows.items()):
        if r["split"] != split:
            continue
        if families and r["family"] not in families:
            continue
        if maps and not (set(maps) & set((r.get("maps") or {}).keys())):
            continue
        out.append(name)
    return out
