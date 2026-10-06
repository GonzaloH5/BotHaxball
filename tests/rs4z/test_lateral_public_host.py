"""Reglas del lateral del Host Publico, adaptadas al mapa RS4-Z.

La variante es opt-in: no cambia el contrato v2 de entrenamientos existentes.
"""
import numpy as np
import pytest

from env.rs4z import contract as C, kernel as K
from env.rs4z.core import RS4ZEnv


def setup_lateral(owner=0, side=1, contract="v2_lateral"):
    env = RS4ZEnv(1, contract=contract, frame_skip=1, max_delay=0,
                  deadline=0, kickoff_deadline=0, seed=0)
    env.place(0, ball_pos=(0, 0), ball_vel=(0, 0),
              player_pos=np.array([(-400, 300), (-300, -300), (-200, 200), (-100, -100),
                                   (400, 300), (300, -300), (200, 200), (100, -100)], float),
              player_vel=np.zeros((8, 2)))
    env.start_restart(0, C.LATERAL, owner, (100, side * 688))
    return env


def step(env):
    return env.step(np.zeros((1, 8), dtype=np.int64))


@pytest.mark.parametrize("owner", [0, 1])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("dx", [-270.01, 270.01])
def test_out_of_range_returns_original_spot_to_opponent(owner, side, dx):
    env = setup_lateral(owner, side)
    env.pos[0, 0, 0] = 100 + dx
    ev = step(env)
    assert ev["forfeit"][0] == owner
    assert ev["restart_exec"][0] == 0
    assert env.restart_team[0] == 1 - owner
    np.testing.assert_allclose(env.ball_pos[0], (100, side * 688))
    assert env.ri[0, K.RI_LAT_KICKED] == 0


@pytest.mark.parametrize("dx", [-270, 270])
def test_exact_range_boundary_is_allowed(dx):
    env = setup_lateral()
    env.pos[0, 0, 0] = 100 + dx
    assert step(env)["forfeit"][0] == -1
    assert env.restart_team[0] == 0


@pytest.mark.parametrize("side", [-1, 1])
def test_pushing_inside_without_kick_is_invalid(side):
    env = setup_lateral(side=side)
    env.pos[0, 0, 1] = side * 681
    ev = step(env)
    assert ev["forfeit"][0] == 0
    assert env.restart_team[0] == 1


def test_previous_tick_kick_allows_entry():
    env = setup_lateral()
    # Ejecutar una patada horizontal real que todavía deja la pelota afuera.
    env.pos[0, env.fp] = (75, 688)
    act = np.zeros((1, 8), dtype=np.int64)
    act[0, 0] = 9
    env.step(act)
    assert env.ri[0, K.RI_LAT_KICKED] == 1
    env.pos[0, 0] = (100, 681)
    env.vel[0, 0] = 0
    env.pos[0, env.fp] = (0, 0)
    ev = step(env)
    assert ev["restart_exec"][0] == C.LATERAL
    assert ev["forfeit"][0] == -1
    assert env.restart_team[0] == -1


def test_seven_second_timeout_gives_restart_to_opponent():
    env = setup_lateral()
    for _ in range(419):
        assert step(env)["forfeit"][0] == -1
    ev = step(env)
    assert ev["forfeit"][0] == 0
    assert env.restart_team[0] == 1
    assert ev["safety"][0] == 0


def test_existing_v2_contract_does_not_change():
    env = setup_lateral(contract="v2")
    env.pos[0, 0, 0] = 500
    assert step(env)["forfeit"][0] == -1
    assert env.restart_team[0] == 0

