import numpy as np
import pytest

from sim.physics import BatchSim
from sim.stadium import load_stadium, BLUEKO, REDKO

STAY, UP, RIGHT, LEFT, DOWN = 0, 1, 3, 7, 5
KICK = 9


def test_classic_geometry():
    st = load_stadium("classic")
    assert st.d_pos.shape == (4, 2) and np.all(st.d_invmass == 0)
    # red da la vuelta: primer arco de la red izquierda redondea la esquina (-400,-64)
    arcs = np.where(st.s_curved > 0)[0]
    centers = {tuple(np.round(st.s_center[i], 3)) for i in arcs}
    assert (-378.0, -42.0) in centers and (378.0, 42.0) in centers
    # arco blueKO (saque azul) = mitad izquierda; redKO = mitad derecha
    for i in arcs:
        if st.s_group[i] in (REDKO, BLUEKO):
            mid = np.array([75.0, 0.0]) if st.s_group[i] == REDKO else np.array([-75.0, 0.0])
            c = mid - st.s_center[i]
            assert c @ st.s_t0[i] > 0 and c @ st.s_t1[i] > 0


def run(sim, actions, ticks):
    a = np.array([actions], dtype=np.int64)
    goals = []
    for _ in range(ticks):
        goals.append(sim.step(a)[0])
    return goals


def test_player_top_speed_and_damping():
    sim = BatchSim(1, 1, 1)
    sim.reset_random([0])
    sim.pos[0, 0] = (0, 150)  # pelota lejos
    fp = sim.first_player
    sim.pos[0, fp:] = [(-380, -100), (200, 100)]
    sim.vel[0] = 0
    run(sim, [RIGHT, STAY], 200)
    # velocidad terminal: v = (v + a) * 0.96  => v* = 0.1*0.96/0.04 = 2.4
    assert sim.vel[0, fp, 0] == pytest.approx(2.4, rel=1e-3)


def test_ball_damping_and_wall_bounce():
    sim = BatchSim(1, 1, 1)
    sim.reset_random([0])
    fp = sim.first_player
    sim.pos[0, fp:] = [(-300, 190), (300, 190)]
    sim.vel[0, fp:] = 0
    sim.pos[0, 0] = (0, 100)
    sim.vel[0, 0] = (0, 5)
    sim.step(np.zeros((1, 2), dtype=np.int64))
    assert sim.vel[0, 0, 1] == pytest.approx(5 * 0.99)
    for _ in range(30):
        sim.step(np.zeros((1, 2), dtype=np.int64))
    # rebotó contra el plano y=170 (bCoef 0.5*1): ahora va hacia arriba y nunca pasó 160
    assert sim.vel[0, 0, 1] < 0
    assert sim.pos[0, 0, 1] <= 160 + 1e-9


def test_kick_adds_kick_strength_once():
    sim = BatchSim(1, 1, 1)
    sim.reset_random([0])
    fp = sim.first_player
    sim.pos[0, 0] = (0, 0)
    sim.vel[0, 0] = 0
    sim.pos[0, fp:] = [(-27, 0), (300, 150)]  # a 2 unidades de alcance
    sim.vel[0, fp:] = 0
    sim.step(np.array([[KICK, STAY]]))
    assert sim.kicked[0, 0]
    assert sim.vel[0, 0, 0] == pytest.approx(5 * 0.99)
    # mantener apretado no vuelve a patear
    sim.pos[0, fp] = sim.pos[0, 0] - (27, 0)
    sim.step(np.array([[KICK, STAY]]))
    assert not sim.kicked[0, 0]


def test_goal_detection_and_side():
    sim = BatchSim(2, 1, 1)
    sim.reset_random([0, 1])
    fp = sim.first_player
    sim.pos[:, fp:] = [(0, 190), (0, -190)]
    sim.vel[:, fp:] = 0
    sim.pos[0, 0] = (340, 0); sim.vel[0, 0] = (6, 0)    # al arco azul -> anota rojo
    sim.pos[1, 0] = (-340, 0); sim.vel[1, 0] = (-6, 0)  # al arco rojo -> anota azul
    goals = []
    for _ in range(20):
        goals.append(sim.step(np.zeros((2, 2), dtype=np.int64)).copy())
    goals = np.array(goals)
    assert (goals[:, 0] == 1).any() and (goals[:, 1] == -1).any()


def test_ball_stays_in_net():
    """La pelota que entra al arco queda contenida por la red (no se escapa por atrás)."""
    sim = BatchSim(1, 1, 1)
    sim.reset_random([0])
    fp = sim.first_player
    sim.pos[0, fp:] = [(0, 190), (0, -190)]
    sim.vel[0, fp:] = 0
    sim.pos[0, 0] = (300, 30); sim.vel[0, 0] = (12, 3)
    for _ in range(300):
        sim.step(np.zeros((1, 2), dtype=np.int64))
        assert sim.pos[0, 0, 0] <= 400 - 10 + 1e-6


def test_kickoff_barrier_blocks_defending_team():
    sim = BatchSim(1, 1, 1)
    sim.reset_kickoff([0], kickoff_team=0)  # saca rojo
    fp = sim.first_player
    run(sim, [STAY, LEFT], 300)  # azul corre hacia el centro
    # azul no entra al círculo (r=75) ni cruza la mitad
    assert np.linalg.norm(sim.pos[0, fp + 1]) >= 75 + 15 - 1e-6
    assert sim.kickoff[0]  # nadie tocó la pelota
    # rojo sí puede llegar a la pelota y el saque termina
    run(sim, [RIGHT, STAY], 200)
    assert not sim.kickoff[0]


def test_parallel_envs_are_deterministic():
    a = BatchSim(64, 2, 2, seed=1)
    b = BatchSim(64, 2, 2, seed=1)
    a.reset_random(np.arange(64)); b.reset_random(np.arange(64))
    rng = np.random.default_rng(0)
    for _ in range(200):
        act = rng.integers(0, 18, (64, 4))
        a.step(act); b.step(act)
    np.testing.assert_array_equal(a.pos, b.pos)


def test_real_classic_matches_room_export():
    st = load_stadium("classic")
    assert st.spawn_distance == 277.5
    assert st.ball["cGroup"] & 64 and st.ball["cGroup"] & 128  # kick, score
    assert len(st.d_pos) == 4  # la pelota no está entre los postes
    assert np.all(st.p_bcoef[:4] == 0)


def test_repo_classic_still_loads():
    st = load_stadium("classic_repo")
    assert st.spawn_distance == 170
