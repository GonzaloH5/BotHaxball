"""Rewards RS4-Z: identidad telescópica de PBRS, suma cero entre equipos y pruebas con tramposos."""
import numpy as np
import pytest

from env.rs4z.core import RS4ZEnv
from eval.rs4z.cheaters import Cheater
from train.rs4z.rewards import XT_PATH, XT, Rewards

GAMMA = 0.997


def _env_rewards(n=16, threat=0.25, access=0.05, seed=0):
    env = RS4ZEnv(n, seed=seed)
    env.start_match(np.arange(n), match_ticks=10 ** 9)
    r = Rewards(env, GAMMA, XT() if XT_PATH.exists() else None)
    r.coefs.threat = threat if r.xt is not None else 0.0
    r.coefs.access = access
    return env, r


def test_pbrs_telescopes_and_is_zero_sum():
    env, r = _env_rewards()
    rng = np.random.default_rng(1)
    ball_task = np.zeros(env.N, dtype=bool)
    phi0 = r.potentials(ball_task)
    start = phi0.copy()
    total = np.zeros((env.N, 2))
    disc = 1.0
    T = 300
    for t in range(T):
        a = rng.integers(0, 18, (env.N, 8))
        env.step(a)
        phi1 = r.potentials(ball_task)
        terminal = np.zeros(env.N, dtype=bool) if t < T - 1 else np.ones(env.N, dtype=bool)
        f = r.shaping(phi0, phi1, terminal)
        assert np.allclose(f[:, 0], -f[:, 1], atol=1e-12), "el shaping debe ser de suma cero"
        total += disc * f
        disc *= GAMMA
        phi0 = phi1
    # Σ γ^t F_t = γ^T Φ_T − Φ_0 con Φ_T = 0 (terminal)
    assert np.allclose(total, -start, atol=1e-6)


@pytest.mark.parametrize("kind", ["still", "oscillate", "all_chase", "park", "goal_camp", "kick_out", "long_ball"])
def test_cheaters_earn_no_net_shaping(kind):
    """Una política degenerada no acumula shaping: con terminal al final, su retorno moldeado
    descontado es −Φ(s0), el mismo que cualquier otra política desde el mismo estado inicial."""
    env, r = _env_rewards(n=8, seed=3)
    cheat = Cheater(kind)
    ctrl = np.ones((env.N, 8), dtype=bool)
    out = np.zeros((env.N, 8), dtype=np.int64)
    ball_task = np.zeros(env.N, dtype=bool)
    phi0 = r.potentials(ball_task)
    start = phi0.copy()
    total = np.zeros((env.N, 2))
    disc = 1.0
    T = 200
    for t in range(T):
        out[:] = 0
        cheat.act(env, ctrl, out)
        env.step(out)
        cheat.push(env)
        phi1 = r.potentials(ball_task)
        f = r.shaping(phi0, phi1, np.full(env.N, t == T - 1))
        total += disc * f
        disc *= GAMMA
        phi0 = phi1
    assert np.allclose(total, -start, atol=1e-6)
    # y por paso no hay un premio sostenido: el promedio por paso tiende a 0
    assert np.abs(total + start).max() < 1e-6


def test_behind_own_goal_cost_counts_only_open_play_dead_zone():
    """Costo por estar detrás de la propia línea de gol: sólo en juego abierto y sólo del lado propio."""
    from env.rs4z import kernel as K
    from env.rs4z.core import RS4ZEnv
    from train.rs4z.rewards import behind_own_goal
    env = RS4ZEnv(2, seed=0)
    pos = np.zeros((8, 2))
    pos[:, 1] = np.linspace(-300, 300, 8)
    pos[0, 0] = -1200.0        # rojo detrás de su línea: cuenta
    pos[1, 0] = +1200.0        # rojo detrás de la línea RIVAL: no cuenta
    pos[4, 0] = +1250.0        # azul detrás de su línea: cuenta
    for n in range(2):
        env.place(n, ball_pos=(0.0, 0.0), ball_vel=(0.0, 0.0), player_pos=pos, player_vel=np.zeros((8, 2)))
    env.start_restart(1, 1, 0, (300.0, 688.0))      # fila 1: lateral en curso → no hay costo
    cost = behind_own_goal(env)
    assert cost[0].tolist() == [1.0, 1.0]
    assert cost[1].tolist() == [0.0, 0.0]
