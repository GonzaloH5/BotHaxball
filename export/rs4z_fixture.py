"""Fixture de paridad de la observación v2 (Python → Node): una secuencia jugada en el simulador con
saques, saque inicial, latencia, variantes del mapa y planteles incompletos.

  python -m export.rs4z_fixture --out deploy/rs4z/fixture_obs_v2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bots.rspro.policy import RSPro
from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from env.rs4z.obs_v2 import OBS_DIM, observe


def state_of(env, n):
    fp = env.fp
    players = []
    for p in range(8):
        players.append(dict(slot=p, team=int(env.team[p]), active=bool(env.active[n, p]),
                            x=float(env.pos[n, fp + p, 0]), y=float(env.pos[n, fp + p, 1]),
                            vx=float(env.vel[n, fp + p, 0]), vy=float(env.vel[n, fp + p, 1]),
                            kickCancel=bool(env.kick_cancel[n, p]), applied=int(env._s_act[n, p])))
    return dict(ball=dict(x=float(env.pos[n, 0, 0]), y=float(env.pos[n, 0, 1]), vx=float(env.vel[n, 0, 0]),
                          vy=float(env.vel[n, 0, 1]), r=float(env.radius[n, 0])),
                players=players, actHist=[[int(env.act_hist[n, p, h]) for h in range(3)] for p in range(8)],
                delay=[int(env.delay[n, p]) for p in range(8)], kickStrength=float(env.rf[n, K.RF_KSTR]),
                restart=dict(team=int(env.ri[n, K.RI_TEAM]), kind=int(env.ri[n, K.RI_KIND]), ticks=int(env.ri[n, K.RI_TICKS])),
                kickoff=dict(active=bool(env.ri[n, K.RI_KO]), team=int(env.ri[n, K.RI_KO_TEAM]), ticks=int(env.ri[n, K.RI_KO_TICKS])),
                massPhase=int(env.ri[n, K.RI_MASS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy/rs4z/fixture_obs_v2.json")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    env = RS4ZEnv(4, seed=3, max_delay=12)
    active = np.ones((4, 8), dtype=bool)
    active[1, [3, 7]] = False          # 3v3
    active[2, [1, 2, 3, 6, 7]] = False  # 1v2
    env.start_match(np.arange(4), active=active, match_ticks=10 ** 9,
                    delay=np.array([[0] * 8, [7] * 8, [3, 9, 0, 0, 11, 6, 0, 0], [10] * 8]),
                    kick_strength=np.array([5.85, 5.75, 5.85, 5.75]), ball_radius=np.array([8.325, 8.0, 8.325, 8.0]))
    bot = RSPro(env, seed=3)
    bot.sync(env, np.arange(4))
    ctrl = env.active.copy()
    out = np.zeros((4, 8), dtype=np.int64)
    frames = []
    rng = np.random.default_rng(0)
    for t in range(args.steps):
        if t == 60:
            env.start_restart(0, C.CORNER, 1, (-1140.0, 660.0))
            bot.sync(env, [0])
        if t == 150:
            env.start_restart(3, C.GOAL_KICK, 0, (-1030.0, -180.0))
            bot.sync(env, [3])
        if t == 250:
            env.reset_kickoff([1], 1)
            bot.sync(env, [1])
        out[:] = 0
        bot.act(env, ctrl, out)
        noise = rng.random((4, 8)) < 0.2
        out[noise] = rng.integers(0, 18, int(noise.sum()))
        env.step(out)
        bot.push(env)
        if t % 4 == 0:
            obs = observe(env)
            for n in range(4):
                for p in range(8):
                    if env.active[n, p]:
                        frames.append(dict(state=state_of(env, n), slot=p, obs=[float(v) for v in obs[n, p]]))
    kinds = {}
    for f in frames:
        k = f["state"]["restart"]["kind"]
        kinds[k] = kinds.get(k, 0) + 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(obs_dim=OBS_DIM, frames=frames)), encoding="utf-8")
    print(len(frames), "casos; tipos de saque presentes:", kinds)


if __name__ == "__main__":
    main()
