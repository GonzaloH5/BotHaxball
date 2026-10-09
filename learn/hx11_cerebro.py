"""Cerebro del bot de sala de HaxEleven: recibe los ticks de la sala y decide las teclas del bot.

Corre como proceso hijo del cliente de node (`deploy/hx11/bot.js`), que le manda un JSON por línea y tick por stdin y
lee las decisiones por stdout. La observación sale de **la misma función del entrenamiento** (`hx11_data.build`) sobre
una ventana viva con el mismo formato que D0 (mismas cuantizaciones, lugares fijos por equipo), así que la paridad
con el entrenamiento es por construcción. La fase de juego sale de `hx11_fase.Fase` (causal; 97% de acuerdo con D0).

Tick (entrada):
  {"t": int, "state": int (0 = saque inicial), "score": [r, b],
   "ball": [x, y, vx, vy, invMass, gx, gy],
   "players": [[id, team (0 rojo / 1 azul), x, y, vx, vy, input], ...],
   "placement": [x, y] | null,   # el script colocó la pelota en este tick (setDiscProperties del disco 0)
   "kicks": [id, ...], "goal": bool, "gk": [id rojo | -1, id azul | -1] (de los !gk anunciados),
   "offside": bool, "me": id del bot, "delay": ticks (eco medido)}
Decisión (salida, cada `frame_skip` ticks): {"t": int, "input": bits de HaxBall}

El puesto lo fija el usuario con `--pos` (P2): ancla y línea del vocabulario (`reports/hx11/pos_vocab.json`).

  python -m learn.hx11_cerebro --model runs/hx11_l0/l0_sinhist_full/model.pt --pos LI [--temp 0.7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from learn.hx11_closed_loop import bits_of_action
from learn.hx11_data import D_MAX, D_MIN, FAULT_MODES, LINES, MAX_BACK, ROT_MOVE, Match, build
from learn.hx11_fase import Fase
from learn.hx11_policy import SetPolicy, sample

ROOT = Path(__file__).resolve().parent.parent
CAP = 4096


class Cerebro:
    def __init__(self, model, pos, vocab, temp=1.0, every=3):
        ck = torch.load(model, map_location="cpu", weights_only=False)
        self.net = SetPolicy(**ck["config"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.net.hold = False
        self.temp, self.every = temp, every
        self.anchor = np.array(vocab[pos]["ancla"], np.float32)
        self.line = int(vocab[pos]["linea_idx"])
        self.fase = Fase()
        self.slot_of = {}                       # id de jugador -> lugar fijo (0–10 rojo, 11–21 azul)
        self.n = 0                              # filas usadas de la ventana
        self.ticks = 0
        self.pending = []                       # (tick absoluto en que la sala la aplica, bits) de teclas ya enviadas
        z = lambda *shape, dt=np.int16: np.zeros(shape, dt)
        self.w = Match(name="sala", T=CAP, pos=z(CAP, 22, 2), vel=z(CAP, 22, 2), ball=z(CAP, 2), bvel=z(CAP, 2),
                       binv=np.full(CAP, 1.65, np.float16), bg=z(CAP, 2, dt=np.float16),
                       inp=np.full((CAP, 22), -1, np.int8), kick=z(CAP, 22, dt=np.int8), score=z(CAP, 2),
                       pid=np.full((CAP, 22), -1, np.int16), gk=np.full((CAP, 2), -1, np.int32),
                       phase=z(CAP, dt=np.int8), pkind=z(CAP, dt=np.int8), pteam=np.full(CAP, -1, np.int8),
                       pticks=z(CAP), spot=z(CAP, 2), offside=np.ones(CAP, bool), fouls=FAULT_MODES.index("prendido"),
                       anchor={}, line={})

    def _slots(self, players):
        present = {p[0] for p in players}
        for pid in [k for k in self.slot_of if k not in present]:
            del self.slot_of[pid]
        for p in players:
            pid, team = int(p[0]), int(p[1])
            if pid in self.slot_of and (self.slot_of[pid] >= 11) == (team == 1):
                continue
            used = set(self.slot_of.values())
            free = [s for s in range(team * 11, team * 11 + 11) if s not in used]
            if free:
                self.slot_of[pid] = free[0]

    def _compact(self):
        keep = MAX_BACK + D_MAX + 8
        w = self.w
        for f in ("pos", "vel", "ball", "bvel", "binv", "bg", "inp", "kick", "score", "pid", "gk", "phase", "pkind",
                  "pteam", "pticks", "spot", "offside"):
            a = getattr(w, f)
            a[:keep] = a[self.n - keep:self.n]
        self.n = keep

    def push(self, fr):
        if self.n >= CAP - D_MAX - 8:
            self._compact()
        w, i = self.w, self.n
        players = fr["players"]
        self._slots(players)
        w.pid[i] = -1
        w.inp[i] = -1
        w.kick[i] = 0
        for pid, team, x, y, vx, vy, inp in players:
            s = self.slot_of.get(int(pid))
            if s is None:
                continue
            w.pid[i, s] = pid
            w.pos[i, s] = np.clip(np.round([x * 10, y * 10]), -32767, 32767)
            w.vel[i, s] = np.clip(np.round([vx * 1000, vy * 1000]), -32767, 32767)
            w.inp[i, s] = inp
        for pid in fr.get("kicks", []):
            s = self.slot_of.get(int(pid))
            if s is not None:
                w.kick[i, s] = 1
        bx, by, bvx, bvy, binv, gx, gy = fr["ball"]
        w.ball[i] = np.clip(np.round([bx * 10, by * 10]), -32767, 32767)
        w.bvel[i] = np.clip(np.round([bvx * 1000, bvy * 1000]), -32767, 32767)
        w.binv[i] = binv
        w.bg[i] = (gx, gy)
        w.score[i] = fr.get("score", [0, 0])
        w.gk[i] = fr.get("gk", [-1, -1])
        w.offside[i] = bool(fr.get("offside", True))
        toucher = -1
        on = w.pid[i] >= 0
        if on.any():
            d = np.hypot(*(w.pos[i].astype(np.float64) / 10 - w.ball[i].astype(np.float64) / 10).T) - 23.0
            d[~on] = np.inf
            j = int(np.argmin(d))
            toucher = int(j >= 11) if d[j] <= 0.6 else -1
        kicks = sorted({int(s >= 11) for s in np.nonzero(w.kick[i])[0]})
        ph, kd, tm, pt, sx, sy = self.fase.step(int(fr.get("state", 1)), fr.get("placement"), kicks,
                                                bool(fr.get("goal")), toucher)
        w.phase[i], w.pkind[i], w.pteam[i], w.pticks[i] = ph, kd, tm, min(pt, 32767)
        w.spot[i] = np.round([sx * 10, sy * 10])
        self.n += 1
        self.ticks += 1

    def decide(self, me, delay):
        """Teclas del bot (bits de HaxBall), o None si no le toca decidir en este tick."""
        if self.ticks % self.every or self.n < MAX_BACK + 1:
            return None
        s = self.slot_of.get(int(me))
        if s is None:
            return None
        w = self.w
        d = int(np.clip(delay, D_MIN, D_MAX))
        L = self.n - 1                       # último tick visto
        t = L + d + 1                        # tick en que se aplica la decisión (convención del entrenamiento)
        # filas "futuras" que necesita build: el lugar sigue ocupado por el bot y sus teclas pendientes ya enviadas
        w.pid[L + 1:t + 1] = w.pid[L]
        w.inp[L + 1:t + 1, s] = w.inp[L, s]
        last_abs = self.ticks - 1                 # tick absoluto de la fila L
        for tick, bits in sorted(self.pending):
            k = L + (tick - last_abs)
            if L < k <= t:
                w.inp[k:t + 1, s] = bits
        pid = int(w.pid[L, s])
        w.anchor[(pid, 0)] = w.anchor[(pid, 1)] = self.anchor
        w.line[pid] = self.line
        w.T = 10 ** 9                        # la mitad del partido no importa: el ancla es la misma
        own, ents, _, _ = build(w, np.array([t]), np.array([s]), np.array([d]), self.net.extra)
        w.T = CAP
        with torch.no_grad():
            a = int(sample(self.net(torch.from_numpy(own), torch.from_numpy(ents)), self.temp)[0])
        if s >= 11:
            a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
        bits = int(bits_of_action(np.array([a]))[0])
        self.pending = [(tk, b) for tk, b in self.pending if tk > last_abs - 2] + [(last_abs + d + 1, bits)]
        return bits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(2)
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    if a.pos not in vocab:
        sys.exit(f"puesto desconocido {a.pos}; opciones: {', '.join(vocab)}")
    c = Cerebro(a.model, a.pos, vocab, a.temp, a.every)
    print(json.dumps({"listo": True, "pos": a.pos, "linea": LINES[c.line]}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        fr = json.loads(line)
        if "set_pos" in fr:
            # cambio de puesto desde la consola del usuario (lo elige solo él: P2 y §4 del plan)
            pos = fr["set_pos"]
            if pos in vocab:
                c.anchor = np.array(vocab[pos]["ancla"], np.float32)
                c.line = int(vocab[pos]["linea_idx"])
                print(json.dumps({"pos": pos, "linea": LINES[c.line]}), flush=True)
            else:
                print(json.dumps({"error": f"puesto desconocido {pos}"}), flush=True)
            continue
        c.push(fr)
        bits = c.decide(fr.get("me"), fr.get("delay", 20))
        if bits is not None:
            print(json.dumps({"t": fr.get("t"), "input": bits}), flush=True)


if __name__ == "__main__":
    main()
