"""Fase de juego en vivo para el bot de sala de HaxEleven (causal: solo usa el pasado).

D0 etiqueta la fase de los datos de entrenamiento mirando la grabación entera (`tools/hx11_d0.build`): por ejemplo, el
equipo que saca una pelota parada sale de la patada siguiente. En la sala el bot tiene que decidir con lo que ya pasó.
Este detector usa las mismas señales que llegan a cualquier cliente:

* **estado del juego** 0 (saque inicial) → KICKOFF;
* **gol** (evento de la sala) → GOAL hasta el saque inicial;
* **colocación de la pelota por el script** (`setDiscProperties` del disco 0 con posición, el mismo evento que D0) →
  pelota parada del tipo del punto (`tools.hx11_script.piece_kind`):
  * tiro libre o penal: congelamiento (FREEZE) hasta la segunda colocación o 30 ticks (falta u offside), después PIECE;
  * PIECE hasta la primera patada (de cualquiera: durante la pelota parada solo patea el que saca);
* **equipo que saca** (D0 lo toma de la patada siguiente; acá, la regla del árbitro con el último toque):
  lateral → el rival del último que tocó; córner → el equipo que ataca ese arco; saque de arco → el que lo defiende;
  tiro libre/penal → si hubo patada en el tick de la colocación es offside (saca el rival del último que tocó),
  si no, falta (saca el equipo del último que tocó);
* la "salida pendiente" de D0 (marcada hacia atrás) no se detecta en vivo: queda juego abierto hasta la colocación.

`Fase.step(...)` se llama una vez por tick. `validar` compara con las etiquetas de D0 tick a tick en partidos de prueba.

  python -m learn.hx11_fase --partidos 10
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from env.hx11 import contract as HC
from tools.hx11_d0 import FREEZE, GOAL, KICKOFF, OPEN, OUT_PENDING, PIECE

KINDS = dict(lateral=HC.LATERAL, corner=HC.CORNER, saque_arco=HC.GOAL_KICK, tiro_libre=HC.FREE_KICK, penal=HC.PENALTY)
NAMES = {OPEN: "abierto", OUT_PENDING: "salida", PIECE: "parada", FREEZE: "congelado", GOAL: "gol", KICKOFF: "inicial"}
TOUCH = 0.6


def piece_kind(x, y):
    ax, ay = abs(x), abs(y)
    if abs(ay - 1230) < 0.5:
        return "lateral"
    if abs(ax - 1765) < 0.5 and abs(ay - 1165) < 0.5:
        return "corner"
    if abs(ax - 1572) < 0.5:
        return "saque_arco"
    if abs(ax - 1275) < 0.5 and ay < 0.5:
        return "penal"
    return "tiro_libre"


class Fase:
    """Estado de la fase; `step` devuelve (phase, piece_kind, piece_team, phase_ticks, spot_x, spot_y)."""

    def __init__(self):
        self.phase, self.kind, self.team, self.since, self.spot = OPEN, 0, -1, 0, (0.0, 0.0)
        self.last_touch_team = -1
        self.freeze_until = -1
        self.t = -1

    def _set(self, ph, kind=0, team=-1, spot=(0.0, 0.0)):
        if (ph, kind, team) != (self.phase, self.kind, self.team):
            self.since = self.t
        self.phase, self.kind, self.team, self.spot = ph, kind, team, spot

    def step(self, game_state, placement, kicks, goal, toucher_team):
        """game_state: estado de la sala (0 = saque inicial); placement: (x, y) si el script colocó la pelota en este
        tick, si no None; kicks: equipos (0/1) que patearon en este tick; goal: hubo gol; toucher_team: equipo del
        jugador que toca la pelota en este tick (o -1)."""
        self.t += 1
        kick_now = len(kicks) > 0
        if goal:
            self._set(GOAL)
        elif game_state == 0:
            self._set(KICKOFF)
        elif placement is not None:
            x, y = placement
            k = piece_kind(x, y)
            if self.phase == FREEZE and self.t <= self.freeze_until:
                # segunda colocación de una falta u offside: ya es la pelota parada
                self._set(PIECE, KINDS[k], self.team, (x, y))
            elif k in ("tiro_libre", "penal"):
                last = self.last_touch_team
                taker = (1 - last if kick_now else last) if last >= 0 else -1
                self.freeze_until = self.t + 30
                self._set(FREEZE, KINDS[k], taker, (x, y))
            else:
                last = self.last_touch_team
                if k == "lateral":
                    taker = 1 - last if last >= 0 else -1
                elif k == "corner":
                    taker = 0 if x > 0 else 1          # el rojo ataca hacia +x
                else:
                    taker = 1 if x > 0 else 0          # saque de arco: el que defiende ese arco
                self._set(PIECE, KINDS[k], taker, (x, y))
        elif self.phase == FREEZE and self.t > self.freeze_until:
            self._set(PIECE, self.kind, self.team, self.spot)
        elif self.phase == PIECE and kick_now:
            # durante la pelota parada solo patea el que saca (la barrera aleja al otro): cualquier patada la termina
            # (esperar la del equipo deducido dejaba la pelota parada colgada cuando el equipo estaba mal)
            self._set(OPEN)
        elif self.phase == GOAL and game_state != 0:
            pass
        elif self.phase == KICKOFF and game_state != 0:
            self._set(OPEN)
        if toucher_team >= 0:
            self.last_touch_team = toucher_team
        for k in kicks:
            self.last_touch_team = k
        # como D0: el contador de la fase solo corre en pelota parada, congelamiento y gol (en juego abierto y saque
        # inicial queda en 0)
        ticks = self.t - self.since if self.phase in (PIECE, FREEZE, GOAL) else 0
        return self.phase, self.kind, self.team, ticks, self.spot[0], self.spot[1]


def validar(names):
    """Acuerdo tick a tick con las etiquetas de D0, alimentando el detector con lo que vería un cliente."""
    from tools.hx11_d0 import load_d0
    from tools.hx11_script import Match
    ROOT = Path(__file__).resolve().parent.parent
    conf = collections.Counter()
    team_ok = team_n = kind_ok = kind_n = 0
    for n in names:
        o, meta = load_d0(ROOT / "data" / "hx11" / "d0" / n)
        T = len(o["phase"])
        pos = o["pos"].astype(np.float64) / 10
        ball = o["ball_pos"].astype(np.float64) / 10
        on = o["slot_pid"] >= 0
        gap = np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)) - HC.PLAYER_R - HC.BALL_R
        gap[~on] = np.inf
        near = gap.argmin(1)
        touch_team = np.where(gap[np.arange(T), near] <= TOUCH, (near >= 11).astype(int), -1)
        kick = o["kick"]
        score = o["score"].astype(np.int64)
        goals = np.r_[False, (np.diff(score, axis=0) > 0).any(1)]
        # colocaciones: las mismas que usa D0 (decisiones de la sala, tick de cada PLACE / FOUL / OFFSIDE)
        place = {}
        for d in meta["decisions"]:
            place[d[0]] = (d[4], d[5])
        st = np.where(o["phase"] == KICKOFF, 0, 1)
        f = Fase()
        for t in range(T):
            ks = sorted({int(s >= 11) for s in np.nonzero(kick[t])[0]})
            ph, kd, tm, _, _, _ = f.step(st[t], place.get(t), ks, bool(goals[t]), int(touch_team[t]))
            ref = int(o["phase"][t])
            conf[(NAMES[ref], NAMES[ph])] += 1
            if ref in (PIECE, FREEZE) and ph in (PIECE, FREEZE):
                kind_n += 1
                kind_ok += int(kd == o["piece_kind"][t])
                if o["piece_team"][t] >= 0:
                    team_n += 1
                    team_ok += int(tm == o["piece_team"][t])
    tot = sum(conf.values())
    by_ref = collections.Counter()
    for (r, p), c in conf.items():
        by_ref[r] += c
    acc = {r: round(conf[(r, r)] / by_ref[r], 4) for r in by_ref}
    return dict(ticks=tot, acuerdo_total=round(sum(conf[(r, r)] for r in by_ref) / tot, 4), acuerdo_por_fase=acc,
                tipo_de_pelota_parada=round(kind_ok / max(kind_n, 1), 4),
                equipo_que_saca=round(team_ok / max(team_n, 1), 4),
                confusion={f"{r}->{p}": c for (r, p), c in conf.most_common(20)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partidos", type=int, default=10)
    a = ap.parse_args()
    from learn.hx11_data import l0_split
    names = l0_split()["test"][:a.partidos]
    print(json.dumps(validar(names), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
