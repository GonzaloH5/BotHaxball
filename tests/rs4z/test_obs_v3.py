"""Observación v3: compatibilidad con v2, geometría por mapa y paridad sim/grabación de las acciones pendientes."""
import numpy as np
import pytest

from env.rs4z import kernel as K
from env.rs4z import obs_v2, obs_v3
from env.rs4z.core import MIRROR_ACTION, RS4ZEnv

V2_HIST = 27 + 9 * obs_v2.N_HIST + obs_v2.N_HIST   # fin del historial en v2
V3_HIST = 27 + 9 * obs_v3.N_HIST + obs_v3.N_HIST


def _run(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(rng.integers(0, 18, size=(env.N, 8)))


def test_shared_features_match_v2_on_rs_one():
    env = RS4ZEnv(8, map="rs_one", seed=1, max_delay=15)
    env.delay[:] = 0
    _run(env, 40)
    a = obs_v2.observe(env)
    b = obs_v3.observe(env)
    assert b.shape == (8, 8, obs_v3.OBS_DIM)
    assert np.isfinite(b).all()
    np.testing.assert_allclose(b[..., :27], a[..., :27], atol=1e-6)
    # 3 decisiones más nuevas: mismas columnas de movimiento y patada
    np.testing.assert_allclose(b[..., 27:27 + 27], a[..., 27:27 + 27], atol=1e-6)
    np.testing.assert_allclose(b[..., V3_HIST - 5:V3_HIST - 2], a[..., V2_HIST - 3:V2_HIST], atol=1e-6)
    # n_mates.. restart/kickoff (sin delay ni masa, que cambian de escala o de definición)
    np.testing.assert_allclose(b[..., V3_HIST + 1:V3_HIST + 5], a[..., V2_HIST + 1:V2_HIST + 5], atol=1e-6)
    np.testing.assert_allclose(b[..., V3_HIST + 6:V3_HIST + 16], a[..., V2_HIST + 6:V2_HIST + 16], atol=1e-6)
    # entidades
    np.testing.assert_allclose(b[..., obs_v3.SELF_DIM:], a[..., obs_v2.SELF_DIM:], atol=1e-6)


@pytest.mark.parametrize("map_name", obs_v3.MAP_NAMES)
def test_map_one_hot_and_lateral(map_name):
    env = RS4ZEnv(2, map=map_name, seed=2, max_delay=15)
    o = obs_v3.observe(env)
    j = V3_HIST + 16
    hot = o[..., j:j + obs_v3.N_MAPS]
    assert (hot[..., obs_v3.MAP_NAMES.index(map_name)] == 1).all()
    assert hot.sum(-1).max() == 1
    line_h = obs_v3.map_geometry(map_name)[0]
    y = o[..., 1] * obs_v3.SY
    np.testing.assert_allclose(o[..., 17] * obs_v3.SY, line_h + y, atol=1e-3)


@pytest.mark.parametrize("delay", [0, 6, 9, 12, 15])
def test_recording_samples_reproduce_pending_actions(delay):
    """Con retardo d, las acciones propias que arma `build_samples` desde las entradas por tick son las mismas
    que el historial de decisiones del simulador (la decisión tomada en S_t produce S_{t+d+1}..S_{t+d+3})."""
    env = RS4ZEnv(1, map="sanguchito_rs_x4", seed=3, max_delay=15)
    env.delay[:] = delay
    rng = np.random.default_rng(4)
    steps = 40
    T = 3 * steps + 30
    inp = np.zeros((T, 8), np.uint8)
    pos = np.zeros((T, 8, 2), np.float32)
    vel = np.zeros((T, 8, 2), np.float32)
    ball = np.zeros((T, 4), np.float32)
    obs_env = {}
    st = np.ones(T, np.int8)
    rk = np.zeros(T, np.int8)
    rt = np.full(T, -1, np.int8)
    ra = np.zeros(T, np.int32)
    kt = np.full(T, -1, np.int8)
    ka = np.zeros(T, np.int32)
    move_bits = [0, 1, 1 | 8, 8, 2 | 8, 2, 2 | 4, 4, 1 | 4]   # arriba 1, abajo 2, izq 4, der 8
    for k in range(steps):
        t = 3 * k
        obs_env[t] = obs_v3.observe(env)[0].copy()
        pos[t] = env.player_pos[0]
        vel[t] = env.player_vel[0]
        ball[t, :2] = env.ball_pos[0]
        ball[t, 2:] = env.ball_vel[0]
        if env.kickoff[0]:
            st[t], kt[t], ka[t] = 0, env.ri[0, K.RI_KO_TEAM], env.ri[0, K.RI_KO_TICKS]
        if env.ri[0, K.RI_TEAM] >= 0:
            rk[t], rt[t], ra[t] = env.ri[0, K.RI_KIND], env.ri[0, K.RI_TEAM], env.ri[0, K.RI_TICKS]
        own = rng.integers(0, 18, size=(1, 8))
        world = own.copy()
        world[:, 4:] = MIRROR_ACTION[own[:, 4:]]
        for u in range(t + delay + 1, t + delay + 4):     # registros producidos con esta decisión
            inp[u] = [move_bits[a % 9] | (16 if a >= 9 else 0) for a in world[0]]
        env.step(own)
    ts = np.array([t for t in obs_env if t >= 15 + 3], dtype=np.int64)
    B = len(ts) * 8
    t_obs = np.repeat(ts, 8)
    slot = np.tile(np.arange(8), len(ts))
    t_lab = t_obs + delay + 1
    dl = np.full(B, delay, np.int64)
    out = np.zeros((B, obs_v3.OBS_DIM), np.float32)
    obs_v3.build_samples(t_obs, t_lab, slot, dl, ball.astype(np.float64), np.full(T, 8.325),
                         pos.astype(np.float64), vel.astype(np.float64), inp, np.zeros((T, 8), np.bool_),
                         st, rk, rt, ra, kt, ka, np.full(T, 0.5, np.float32),
                         np.full(T, 5.85, np.float32), np.ones(T, np.int8), obs_v3.line_h_table(), 1150.0, out)
    ref = np.stack([obs_env[t][p] for t, p in zip(t_obs, slot)])
    # posiciones, pelota, geometría y entidades (sin las columnas de patada, que en grabaciones salen de la
    # tecla y el estado del motor)
    cols = [c for c in range(obs_v3.OBS_DIM) if c not in (4, 5)
            and not (c >= obs_v3.SELF_DIM and (c - obs_v3.SELF_DIM) % obs_v3.ENT_DIM == 8)
            and c != V3_HIST + 5]   # fase de masa: igual en Sanguchito, pero el test no la carga
    np.testing.assert_allclose(out[:, cols], ref[:, cols], atol=2e-5)
