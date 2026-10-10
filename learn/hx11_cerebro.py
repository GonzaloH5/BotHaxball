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
from learn.hx11_policy import (Ensemble, SetPolicy, StickyNoise, kick_rule_ok, near_rows, open_kick_blocked,
                               piece_rows, policy_logits, sample, seen_now)

ROOT = Path(__file__).resolve().parent.parent
CAP = 4096


def lookup(vocab, key):
    """Puesto del vocabulario (`LI`, `DC`, …) o rol de una formación (`4-3-3:DEF-CI`, para partidos con varios bots)."""
    if key in vocab and key != "roles":
        return vocab[key]
    return vocab.get("roles", {}).get(key)


class Cerebro:
    def __init__(self, model, pos, vocab, temp=1.0, every=3, hyst=0.0, ens_k=0, ens_m=0.0, renew=1.0, renew_kick=1.0,
                 sin_saques=(), regla_saques=False, regla_lateral=False, regla_juego=False):
        ck = torch.load(model, map_location="cpu", weights_only=False)
        self.net = SetPolicy(**ck["config"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.net.hold = False
        self.temp, self.every, self.hyst = temp, every, hyst
        self.prev = {}                          # id -> última acción (coordenadas del equipo), para la histéresis
        # acciones en bloque (ACT): ensamble temporal por jugador, si el modelo predice varias decisiones
        self.sin_saques = set(int(k) for k in sin_saques)   # tipos de pelota parada propia que el bot no patea
        self.regla_lateral = bool(regla_lateral)            # regla de patada en el lateral (ver kick_rule_ok)
        self.regla_juego = bool(regla_juego)                # regla de patada en juego abierto (ver open_kick_ok)
        self.regla_saques = bool(regla_saques)              # regla de patada (dirección y barra) en las que sí saca
        # ruido persistente (dirección y tecla de patear), por jugador
        self.renew, self.renew_kick = renew, renew_kick
        self.noise = StickyNoise(renew, 0, renew_kick) if (renew < 1.0 or renew_kick < 1.0) else None
        self.ens = (Ensemble(min(ens_k, self.net.chunk), ens_m, every)
                    if ens_k > 1 and getattr(self.net, "chunk", 1) > 1 else None)
        v = lookup(vocab, pos) if pos else None
        self.anchor = np.array(v["ancla"], np.float32) if v else None
        self.line = int(v["linea_idx"]) if v else 3
        self.bots = {}                          # varios bots: id -> (ancla, línea)
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

    def _noise(self, ids, slots=None):
        """Tiradas persistentes para `sample` (vacío si no hay ruido persistente). La dirección va con sorteo
        independiente cerca de la pelota (último tick visto)."""
        if self.noise is None:
            return {}
        fresh = None
        if self.renew < 1.0 and slots is not None and self.n >= 1:
            L = self.n - 1
            fresh = near_rows([self.w.pos[L, s].astype(np.float64) / 10 for s in slots],
                              [self.w.ball[L].astype(np.float64) / 10 for _ in slots])
        return dict(gumbel=self.noise.get(ids, fresh) if self.renew < 1.0 else None,
                    kick_u=self.noise.get_kick(ids, piece_rows([self.w.phase[self.n - 1]] * len(ids)))
                    if self.renew_kick < 1.0 else None)

    def _kick_blocked(self, s, d=0):
        """Regla de patada en las pelotas paradas propias, si está activada (`kick_rule_ok`): en el córner, el tiro libre
        y el saque de arco, solo desde atrás de la pelota y con la barra alta (o pasados 10 s de espera); en el lateral,
        desde afuera con ángulo hacia la cancha. Medida en el simulador (learn/hx11_corner.py, 2026-10-09: sin córners
        italianos y con la potencia humana; learn/hx11_selfplay.py, 2026-10-10: laterales). En juego abierto
        (`regla_juego`, `open_kick_ok`): no hacia afuera pegado a la banda ni hacia atrás en campo propio sin un compañero
        en esa dirección (los dos errores del partido real del 2026-10-09; la puerta P3 pasa completa con la regla)."""
        if self.n < 1:
            return False
        L = self.n - 1
        if self.regla_juego and int(self.w.phase[L]) == 0:
            return open_kick_blocked(self.w, L, s, d)
        if not (self.regla_saques or self.regla_lateral):
            return False
        team = 1 if s >= 11 else 0
        kind = int(self.w.pkind[L])
        if int(self.w.phase[L]) != 2 or int(self.w.pteam[L]) != team:
            return False
        if (kind == 1 and not self.regla_lateral) or (kind != 1 and not self.regla_saques):
            return False
        me, ball = seen_now(self.w, L, s, d)
        return not kick_rule_ok(me, ball, kind, team, float(self.w.binv[L]), int(self.w.pticks[L]))

    def _no_kick(self, s):
        """True si el bot del lugar s no debe patear ahora: pelota parada de su equipo de un tipo en `self.sin_saques`
        (piloto con humanos, 2026-10-09: los córners del bot salían por la línea de fondo en el 55%; que los saquen los
        humanos). Usa la fase causal del último tick visto, la misma que ve la red."""
        if not self.sin_saques or self.n < 1:
            return False
        L = self.n - 1
        team = 1 if s >= 11 else 0
        return int(self.w.phase[L]) == 2 and int(self.w.pteam[L]) == team and int(self.w.pkind[L]) in self.sin_saques

    def decide_many(self, bots):
        """Varios bots a la vez: bots = [(id, retardo)] registrados con `self.bots`. Devuelve {id: bits}."""
        if self.ticks % self.every or self.n < MAX_BACK + 1:
            return {}
        w = self.w
        L = self.n - 1
        rows = [(int(me), self.slot_of.get(int(me)), int(np.clip(dl, D_MIN, D_MAX))) for me, dl in bots
                if int(me) in self.bots and self.slot_of.get(int(me)) is not None]
        if not rows:
            return {}
        tmax = L + max(d for _, _, d in rows) + 1
        w.pid[L + 1:tmax + 1] = w.pid[L]
        for me, s, d in rows:
            w.inp[L + 1:tmax + 1, s] = w.inp[L, s]
            anc, line = self.bots[me]
            w.anchor[(me, 0)] = w.anchor[(me, 1)] = anc
            w.line[me] = line
        w.T = 10 ** 9
        own, ents, _, _ = build(w, np.array([L + d + 1 for _, _, d in rows]), np.array([s for _, s, _ in rows]),
                                np.array([d for _, _, d in rows]), self.net.extra)
        w.T = CAP
        with torch.no_grad():
            lg = policy_logits(self.net, torch.from_numpy(own), torch.from_numpy(ents), self.ens,
                               [me for me, _, _ in rows], [self.ticks] * len(rows))
            ids = [me for me, _, _ in rows]
            acts = sample(lg, self.temp, prev=np.array([self.prev.get(me, -1) for me in ids]), hyst=self.hyst,
                          **self._noise(ids, [s for _, s, _ in rows]))
        out = {}
        for (me, s, d), a in zip(rows, acts):
            a = int(a)
            self.prev[me] = a
            if s >= 11:
                a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
            out[me] = int(bits_of_action(np.array([a]))[0]) & (~16 if (self._no_kick(s) or self._kick_blocked(s, d))
                                                                 else ~0)
        return out

    def decide(self, me, delay):
        """Teclas del bot (bits de HaxBall), o None si no le toca decidir en este tick."""
        if self.ticks % self.every or self.n < MAX_BACK + 1:
            return None
        s = self.slot_of.get(int(me))
        if s is None or self.anchor is None:
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
            lg = policy_logits(self.net, torch.from_numpy(own), torch.from_numpy(ents), self.ens, [int(me)], [self.ticks])
            a = int(sample(lg, self.temp, prev=np.array([self.prev.get(int(me), -1)]), hyst=self.hyst,
                           **self._noise([int(me)], [s]))[0])
        self.prev[int(me)] = a
        if s >= 11:
            a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
        bits = int(bits_of_action(np.array([a]))[0])
        if self._no_kick(s) or self._kick_blocked(s, d):
            bits &= ~16                      # pelota parada que saca un humano, o regla de patada
        self.pending = [(tk, b) for tk, b in self.pending if tk > last_abs - 2] + [(last_abs + d + 1, bits)]
        return bits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos", default=None, help="puesto de un bot; sin esto, modo de varios bots (register)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--hyst", type=float, default=0.0,
                    help="mantiene la dirección anterior si su probabilidad es >= hyst × la máxima (menos cambios de tecla)")
    ap.add_argument("--ens", type=int, default=0, help="acciones en bloque: ensamble temporal de K decisiones")
    ap.add_argument("--ens-m", type=float, default=0.0)
    ap.add_argument("--renovar", type=float, default=1.0, help="ruido persistente de la dirección (1 = no)")
    ap.add_argument("--renovar-patada", type=float, default=1.0,
                    help="ruido persistente de la tecla de patear (1 = sorteo independiente)")
    ap.add_argument("--regla-saques", action="store_true",
                    help="en córner, tiro libre y saque de arco propios: patea solo desde atrás y con la barra alta")
    ap.add_argument("--regla-juego", action="store_true",
                    help="en juego abierto: no patea hacia afuera pegado a la banda ni atrás en campo propio sin un "
                         "compañero en esa dirección")
    ap.add_argument("--regla-lateral", action="store_true",
                    help="en el lateral propio: patea solo desde afuera, con ángulo hacia la cancha (con compañeros "
                         "humanos no conviene: ver LAT_MIN_DEG en learn/hx11_policy.py)")
    ap.add_argument("--sin-saques", default="",
                    help="pelotas paradas de su equipo que el bot no patea: corner,tiro_libre,saque_arco,lateral,penal")
    a = ap.parse_args()
    torch.set_num_threads(2)
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    if a.pos and lookup(vocab, a.pos) is None:
        sys.exit(f"puesto desconocido {a.pos}; opciones: {', '.join(k for k in vocab if k != 'roles')}")
    kinds = {"lateral": 1, "corner": 2, "saque_arco": 3, "tiro_libre": 4, "penal": 5}
    sin = [kinds[k.strip()] for k in a.sin_saques.split(",") if k.strip()]
    c = Cerebro(a.model, a.pos, vocab, a.temp, a.every, a.hyst, a.ens, a.ens_m, a.renovar, a.renovar_patada, sin,
                a.regla_saques, a.regla_lateral, a.regla_juego)
    print(json.dumps({"listo": True, "pos": a.pos, "linea": LINES[c.line]}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        fr = json.loads(line)
        if "set_pos" in fr:
            # cambio de puesto desde la consola del usuario (lo elige solo él: P2 y §4 del plan)
            v = lookup(vocab, fr["set_pos"])
            if v:
                c.anchor = np.array(v["ancla"], np.float32)
                c.line = int(v["linea_idx"])
                print(json.dumps({"pos": fr["set_pos"], "linea": LINES[c.line]}), flush=True)
            else:
                print(json.dumps({"error": f"puesto desconocido {fr['set_pos']}"}), flush=True)
            continue
        if "register" in fr:
            # varios bots: cada uno con su puesto o rol
            r = fr["register"]
            v = lookup(vocab, r["pos"])
            if v:
                c.bots[int(r["id"])] = (np.array(v["ancla"], np.float32), int(v["linea_idx"]))
            print(json.dumps({"registrado": r["id"], "pos": r["pos"], "ok": bool(v)}), flush=True)
            continue
        c.push(fr)
        if "bots" in fr:
            out = c.decide_many([(b["me"], b.get("delay", 20)) for b in fr["bots"]])
            if out:
                print(json.dumps({"t": fr.get("t"), "inputs": {str(k): v for k, v in out.items()}}), flush=True)
            continue
        bits = c.decide(fr.get("me"), fr.get("delay", 20))
        if bits is not None:
            print(json.dumps({"t": fr.get("t"), "input": bits}), flush=True)


if __name__ == "__main__":
    main()
