"""Observación v2: rango acotado, sin información privada, invariancia por color y simetría."""
import numpy as np

from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from env.rs4z.obs_v2 import CRITIC_DIM, OBS_DIM, SELF_FEATURES, critic, observe
from env.rs4z.parity import chase_actions

SWAP = np.array([4, 5, 6, 7, 0, 1, 2, 3])


def _random_play(env, decisions, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(decisions):
        env.step(chase_actions(env, rng))


def mirrored_copy(env):
    """Mismo partido con los colores cambiados (rojo↔azul) y x espejado."""
    out = RS4ZEnv(env.N, contract=env.contract, seed=1, max_delay=env.max_delay)
    out.restore(env.snapshot())
    fp = env.fp
    for arr in (out.pos, out.vel):
        arr[:, :, 0] *= -1.0
        arr[:, fp:] = arr[:, fp:][:, SWAP]
    for name in ("kick_cancel", "active", "spawn_rank", "outside", "delay", "_s_act"):
        a = getattr(out, name)
        a[:] = getattr(env, name)[:, SWAP]
    from env.rs4z.core import MIRROR_ACTION
    out.act_hist[:] = MIRROR_ACTION[env.act_hist[:, SWAP]]
    out._s_act[:] = MIRROR_ACTION[env._s_act[:, SWAP]]
    from sim.stadium import BLUE, BLUEKO, RED, REDKO
    mask = env.mask[:, fp:][:, SWAP]
    red_ko, blue_ko = (mask & REDKO) != 0, (mask & BLUEKO) != 0
    mask = mask & ~(REDKO | BLUEKO) | np.where(red_ko, BLUEKO, 0) | np.where(blue_ko, REDKO, 0)
    out.mask[:, fp:] = mask
    extra = env.group[:, fp:][:, SWAP] & ~(RED | BLUE)   # bits c0/c1 que puso el script
    out.group[:, fp:] = np.where(env.active[:, SWAP], np.where(np.arange(8) < 4, RED, BLUE) | extra, 0)
    # discos del script: el rojo y el azul cambian de lado con los colores
    for a, b in ((C.SD_RED, C.SD_BLUE), (C.SD_BLUE, C.SD_RED)):
        out.pos[:, a] = env.pos[:, b] * np.array([-1.0, 1.0])
        out.radius[:, a] = env.radius[:, b]
    for a in (C.SD_RED, C.SD_BLUE):
        home = out.radius[:, a] == 0
        out.pos[home, a] = C.SD_HOME[a - 1]
    out.inv[:, fp:] = env.inv[:, fp:][:, SWAP]
    out.grav[:, 0] *= -1.0
    for col in (K.RI_TEAM, K.RI_LAST, K.RI_KO_TEAM):
        v = env.ri[:, col]
        out.ri[:, col] = np.where(v >= 0, 1 - v, v)
    out.ri[:, K.RI_SCORE0], out.ri[:, K.RI_SCORE1] = env.ri[:, K.RI_SCORE1], env.ri[:, K.RI_SCORE0]
    out.rf[:, K.RF_SPOT_X] *= -1.0
    return out


def test_obs_dims_and_no_private_fields():
    names = " ".join(SELF_FEATURES)
    for forbidden in ("clock", "score", "tfrac", "last_touch", "deadline"):
        assert forbidden not in names
    env = RS4ZEnv(8, seed=0)
    assert observe(env).shape == (8, 8, OBS_DIM) and critic(env).shape == (8, 8, CRITIC_DIM)


def test_obs_range_is_bounded_in_play_and_restarts():
    env = RS4ZEnv(64, seed=3)
    rng = np.random.default_rng(3)
    worst = 0.0
    for t in range(1500):
        env.step(chase_actions(env, rng))
        if t % 25 == 0:
            o = observe(env)
            worst = max(worst, float(np.abs(o).max()))
            assert np.isfinite(o).all()
    assert worst <= 5.0, worst


def test_color_swap_gives_identical_observations():
    env = RS4ZEnv(32, seed=5)
    _random_play(env, 400, seed=5)
    twin = mirrored_copy(env)
    a = observe(env)
    b = observe(twin)
    assert np.allclose(a, b[:, SWAP], atol=1e-6)


def test_color_swap_dynamics_match_away_from_map_asymmetries():
    # Asimetrías conocidas (F21 y orden de colisiones rojo→azul) hacen que la igualdad no sea exacta
    # en choques simultáneos; un paso desde estados de juego abierto debe coincidir casi siempre.
    env = RS4ZEnv(64, seed=7)
    _random_play(env, 300, seed=7)
    twin = mirrored_copy(env)
    rng = np.random.default_rng(8)
    act = chase_actions(env, rng)
    env.step(act)
    twin.step(act[:, SWAP])
    fp = env.fp
    mine = env.pos[:, fp:].copy()
    theirs = twin.pos[:, fp:][:, SWAP].copy()
    theirs[..., 0] *= -1.0
    err = np.abs(mine - theirs).max(axis=(1, 2))
    ball = np.abs(env.pos[:, 0] - twin.pos[:, 0] * np.array([-1.0, 1.0])).max(axis=1)
    assert (np.maximum(err, ball) < 1e-6).mean() >= 0.95, np.sort(np.maximum(err, ball))[-5:]
