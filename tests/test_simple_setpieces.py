"""Protección del perfil real: estado privado, rivales bloqueados hasta sacar."""
import numpy as np
import pytest

from env.tasks import load_catalog, make_env


def setup_piece(team=0, kind=3, frame_skip=3, optimized=True, count=1):
    env = make_env(load_catalog()["rs4_3v3"], count, 5, seed=0,
                   random_reset_prob=0.0, frame_skip=frame_skip, optimize_rollout=optimized)
    env.reset()
    sim = env.sim
    sim.kickoff[:] = False
    sim.mask[:] = sim.base_mask
    sim.vel[:] = 0.0
    for p in range(env.P):
        sim.player_pos[:, p] = (-100 + 200 * sim.player_team[p], 100 + p * 30)
    sx = -1 if team == 0 else 1
    if kind == 2:
        sx = -sx  # el córner se saca en el arco rival
    if kind == 1:
        sim.pos[0, 0] = (100.0, env.field_h + 20)
        env.last_touch[0] = 1 - team
    else:
        sim.pos[0, 0] = (sx * (env.field_w + 20), 200.0)
        env.last_touch[0] = 1 - team
    env._set_piece([0])
    env._phi = env._potentials()
    assert env.setpiece_team[0] == team
    assert env.setpiece_kind[0] == kind
    return env


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("kind", [1, 2, 3])
@pytest.mark.parametrize("frame_skip", [1, 3])
def test_rivals_cannot_steal_or_kick_before_restart(team, kind, frame_skip):
    env = setup_piece(team, kind, frame_skip)
    sim = env.sim
    rival = np.flatnonzero(sim.player_team != team)[0]
    origin = sim.ball_pos.copy()
    actions = np.full((1, env.P), 9, dtype=np.int64)
    for _ in range(5):
        # Incluso entrando desde el área o encima de la pelota, antes de la física.
        sim.player_pos[0, rival] = origin[0] + (20.0, 0.0)
        sim.player_vel[0, rival] = (-4.0, 0.0)
        _, _, done, info = env.step(actions)
        assert not done.any() and not info["goal"].any()
        assert not info["kicked"][0, rival]
        assert env.setpiece_team[0] == team
        np.testing.assert_array_equal(sim.ball_pos, origin)
        if kind == 3:
            sx = -1 if team == 0 else 1
            assert sx * sim.player_pos[0, rival, 0] <= env.field_w * 840 / 1150 - sim.st.player["radius"]
    # Ningún nuevo feature privado entra a la observación del modelo.
    assert not env.observe()[..., 56:71].any()


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("kind", [1, 2, 3])
@pytest.mark.parametrize("optimized", [False, True])
def test_own_kick_releases_protection_and_rival_can_then_kick(team, kind, optimized):
    env = setup_piece(team, kind, frame_skip=1, optimized=optimized)
    sim = env.sim
    own = np.flatnonzero(sim.player_team == team)[0]
    rival = np.flatnonzero(sim.player_team != team)[0]
    sim.player_pos[0, own] = sim.ball_pos[0] - (20.0, 0.0)
    actions = np.zeros((1, env.P), dtype=np.int64)
    actions[0, own] = 9
    _, _, _, info = env.step(actions)
    assert info["kicked"][0, own]
    assert env.setpiece_team[0] == -1 and env.setpiece_kind[0] == 0
    assert np.linalg.norm(sim.ball_vel) > 0
    sim.player_pos[0, rival] = sim.ball_pos[0] - (20.0, 0.0)
    actions[:] = 0
    actions[0, rival] = 9
    _, _, _, info = env.step(actions)
    assert info["kicked"][0, rival]


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_timeout_releases_without_permanent_wait_or_out_reset_loop(kind):
    env = setup_piece(kind=kind)
    actions = np.zeros((1, env.P), dtype=np.int64)
    limit = int(env.setpiece_limit[0])
    for _ in range((limit + env.frame_skip - 1) // env.frame_skip):
        env.step(actions)
    assert env.setpiece_team[0] == -1
    assert limit <= env.setpiece_ticks[0] < limit + env.frame_skip


@pytest.mark.parametrize("team", [0, 1])
def test_ball_out_starts_goal_kick_through_normal_step(team):
    env = setup_piece(team)
    env.setpiece_team[:] = -1
    env.setpiece_kind[:] = 0
    sx = -1 if team == 0 else 1
    env.sim.pos[0, 0] = (sx * (env.field_w + 20), 200.0)
    env.sim.vel[0, 0] = (sx, 0.0)
    env.last_touch[0] = 1 - team
    _, _, done, info = env.step(np.zeros((1, env.P), dtype=np.int64))
    assert info["out"][0] and not done[0]
    assert env.setpiece_team[0] == team and env.setpiece_kind[0] == 3


def test_reset_only_clears_selected_rows_and_reference_matches_optimized():
    env = setup_piece(count=2)
    env._reset_envs(np.array([1]))
    assert np.array_equal(env.setpiece_team, [0, -1])
    env._reset_envs(np.array([0]))
    assert not env.setpiece_kind.any() and not env.setpiece_ticks.any()
    reference, optimized = setup_piece(optimized=False), setup_piece(optimized=True)
    actions = np.full((1, env.P), 12, dtype=np.int64)
    for _ in range(10):
        a, b = reference.step(actions), optimized.step(actions)
        for index in range(3):
            np.testing.assert_array_equal(a[index], b[index])
        np.testing.assert_array_equal(reference.sim.pos, optimized.sim.pos)
