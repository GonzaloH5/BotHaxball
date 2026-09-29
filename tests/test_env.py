import numpy as np

from env.haxball_env import HaxballEnv

RIGHT, STAY = 3, 0


def _speed_after(delay, steps=1):
    env = HaxballEnv(1, 1, random_reset_prob=0.0, seed=0, action_delay_max=6)
    env.reset()
    env.delay[:] = delay
    fp = env.sim.first_player
    for _ in range(steps):
        env.step(np.array([[RIGHT, STAY]]))
    return env.sim.vel[0, fp, 0]


def test_no_delay_moves_every_tick():
    env = HaxballEnv(1, 1, random_reset_prob=0.0, seed=0)
    env.reset()
    env.step(np.array([[RIGHT, STAY]]))
    assert env.sim.vel[0, env.sim.first_player, 0] > 0.25


def test_delay_postpones_action_by_d_ticks():
    # frame_skip 3: con d=0 acelera 3 ticks, con d=2 sólo 1, con d=3 ninguno en la primera decisión
    v0, v2, v3 = _speed_after(0), _speed_after(2), _speed_after(3)
    assert v0 > v2 > 0 and v3 == 0
    # con d=4 recién se mueve en la segunda decisión (último tick = 6 - 4 = 2 ticks de empuje)
    assert _speed_after(4, 1) == 0 and _speed_after(4, 2) > 0


def test_delay_zero_is_identical_to_default():
    a = HaxballEnv(8, 1, seed=3)
    b = HaxballEnv(8, 1, seed=3, action_delay_max=0)
    a.reset(); b.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = rng.integers(0, 18, (8, 2))
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
    np.testing.assert_array_equal(oa, ob)


def test_kickoff_stall_penalizes_kicking_team():
    env = HaxballEnv(1, 1, random_reset_prob=0.0, seed=0, kickoff_timeout=30)
    env.reset()
    env.sim.reset_kickoff([0], kickoff_team=0)
    env.kickoff_ticks[:] = 0
    total = np.zeros(2)
    for _ in range(10):
        _, r, d, info = env.step(np.array([[7, 0]]))  # rojo se aleja de la pelota
        total += r[0]
        if d[0]:
            break
    assert info["stall"][0] and d[0] and not info["truncated"][0]
    assert r[0, 0] < -0.4 and r[0, 1] > -0.1


def test_kickoff_approach_rewards_going_to_ball_only_for_kicking_team():
    from env.rewards import RewardConfig
    rc = RewardConfig(shaping_coef=0.0, kickoff_approach=1.0)
    env = HaxballEnv(1, 1, random_reset_prob=0.0, seed=0, reward=rc, kickoff_timeout=0)
    env.reset()
    env.sim.reset_kickoff([0], kickoff_team=0)
    tot = np.zeros(2)
    while env.sim.kickoff[0]:
        _, r, _, _ = env.step(np.array([[3, 7]]))  # ambos hacia el centro
        tot += r[0]
    assert 0.8 < tot[0] < 1.05   # rojo recorrió ~toda la distancia de spawn
    assert tot[1] == 0           # azul no saca: sin premio
