"""Bot por reglas, vectorizado. Sirve de baseline y de primer rival para el RL.

Trabaja en el marco propio del jugador (ataca hacia +x) y devuelve acciones 0..17 en ese marco,
igual que el agente de RL, así que se puede mezclar con él en HaxballEnv.

Estrategia:
  - Si está "detrás" de la pelota respecto al arco rival y bien alineado: ir a la pelota y patear.
  - Si no: ir al punto detrás de la pelota (rodeándola para no hacer gol en contra).
  - Si el rival está más cerca de la pelota y ésta está en campo propio: volver a cubrir el arco.
"""
from __future__ import annotations

import numpy as np

from sim.physics import MOVE_UNIT


def _dir_to_action(d: np.ndarray, stop_eps: float = 2.0) -> np.ndarray:
    """Vector deseado (…,2) -> acción de movimiento 0..8 (la dirección más parecida)."""
    norm = np.linalg.norm(d, axis=-1)
    unit = d / np.maximum(norm, 1e-9)[..., None]
    scores = unit @ MOVE_UNIT[1:].T
    move = 1 + np.argmax(scores, axis=-1)
    return np.where(norm < stop_eps, 0, move)


def scripted_actions(env, players: np.ndarray | None = None, eps: float = 0.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Acciones (N, len(players)) en marco propio para los jugadores `players` de `env`."""
    rng = rng or np.random.default_rng()
    sim = env.sim
    P = sim.P
    players = np.arange(P) if players is None else np.asarray(players)
    sign = env.sign[players]
    st = sim.st
    r_p = st.player["radius"]
    r_b = st.ball["radius"]
    gx = env.goal_x

    ball = np.repeat(sim.ball_pos[:, None], len(players), 1).copy()
    ball[..., 0] *= sign
    bvel = np.repeat(sim.ball_vel[:, None], len(players), 1).copy()
    bvel[..., 0] *= sign
    me = sim.player_pos[:, players].copy()
    me[..., 0] *= sign

    # anticipar un poco la pelota
    ball_f = ball + bvel * 6.0
    goal = np.array([gx, 0.0])
    to_goal = goal - ball_f
    to_goal /= np.maximum(np.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-9)
    behind = ball_f - to_goal * (r_p + r_b + 6.0)

    to_ball = ball_f - me
    dist_ball = np.linalg.norm(to_ball, axis=-1)
    unit_tb = to_ball / np.maximum(dist_ball, 1e-9)[..., None]
    aligned = np.sum(unit_tb * to_goal, axis=-1) > 0.8

    # rodear la pelota: si estoy delante de ella, desviarme lateralmente
    ahead = me[..., 0] > ball_f[..., 0] - 5.0
    side = np.sign(me[..., 1] - ball_f[..., 1] + 1e-6)
    detour = behind.copy()
    detour[..., 1] = ball_f[..., 1] + side * (r_p + r_b + 25.0)
    detour[..., 0] = np.minimum(ball_f[..., 0] - 10.0, me[..., 0])
    target = np.where(aligned[..., None], ball_f, np.where(ahead[..., None], detour, behind))

    # defensa: el rival más cercano a la pelota llega antes y la pelota está en mi mitad
    team = sim.player_team[players]
    d_all = np.linalg.norm(sim.player_pos - sim.ball_pos[:, None], axis=-1)  # (N, P)
    opp_best = np.stack([np.min(d_all[:, sim.player_team != t], axis=1) for t in team], axis=1)
    defend = (opp_best + 20.0 < dist_ball) & (ball[..., 0] < 0)
    guard = np.stack([np.full(ball.shape[:-1], -gx + 30.0), ball[..., 1] * 0.4], axis=-1)
    target = np.where(defend[..., None], guard, target)

    # roles por equipo: el más cercano a la pelota va a buscarla; el 2º se abre para apoyar
    # (atrás y hacia el centro) y el resto cubre el arco. Sin esto los tres se amontonan.
    if env.T > 1:
        rank = np.zeros((sim.N, len(players)), dtype=np.int64)
        for j, p in enumerate(players):
            mates = np.where(sim.player_team == sim.player_team[p])[0]
            dp = d_all[:, p][:, None]
            dm = d_all[:, mates]
            rank[:, j] = ((dm < dp) | ((dm == dp) & (mates[None, :] < p))).sum(axis=1)
        lateral = np.where(ball[..., 1] > 0, -1.0, 1.0) * 0.28 * st.field_half_h
        support = np.stack([ball[..., 0] - 0.18 * gx, ball[..., 1] * 0.4 + lateral], axis=-1)
        cover = np.stack([np.minimum(ball[..., 0] - 0.25 * gx, -0.55 * gx), ball[..., 1] * 0.35], axis=-1)
        target = np.where((rank == 1)[..., None], support, np.where((rank >= 2)[..., None], cover, target))

    move = _dir_to_action(target - me)
    kick = aligned & (dist_ball < r_p + r_b + 8.0)
    act = move + 9 * kick

    if eps > 0:
        rnd = rng.random(act.shape) < eps
        act = np.where(rnd, rng.integers(0, 18, act.shape), act)
    return act
