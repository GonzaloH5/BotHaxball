"""Controlador de pase X4: la predicción con las teclas en cola coincide con el simulador y el plan patea alineado."""
import numpy as np

from env.rs4z import contract as C
from env.rs4z.core import RS4ZEnv
from learn.x4_pass_macro import PassMacro, queued_ticks, roll_queue


def test_queued_ticks_matches_delay():
    assert [queued_ticks(d) for d in (0, 1, 8, 9, 10, 11)] == [0, 1, 8, 9, 10, 11]


def test_roll_queue_matches_kernel():
    rng = np.random.default_rng(1)
    for delay in (6, 9, 12):
        env = RS4ZEnv(1, map="sanguchito_rs_x4", frame_skip=3, max_delay=15, deadline=0,
                      kickoff_deadline=C.KICKOFF_SAFETY, seed=0)
        env.start_match(np.arange(1), delay=np.full((1, 8), delay), match_ticks=3 * 3600)
        preds = {}
        for s in range(24):
            acts = np.zeros((1, 8), int)
            acts[0, 0] = rng.integers(1, 9)
            p, _, T = roll_queue(env.player_pos[0, 0], env.player_vel[0, 0], env.act_hist[0, 0], delay)
            preds[s + T // 3] = p
            env.step(acts)
            if s >= 8 and s + 1 in preds:
                assert np.allclose(preds[s + 1], env.player_pos[0, 0], atol=1e-6)


def test_plan_kicks_when_aligned_and_in_reach():
    m = PassMacro(1)
    mates = np.array([[0.0, 0.0], [300.0, 0.0]] + [[0.0, 500.0]] * 6)
    # pelota dentro del alcance, en la línea del receptor: patea ya (la dirección es jugador → pelota)
    ball = np.array([15.0 + 8.325 + 2.0, 0.0])
    a = m.act(0, ball, np.zeros(2), np.zeros(2), np.zeros(2), mates, np.zeros((8, 2)), 0,
              hist=np.zeros(6, int), delay=0)
    assert a >= 9 and m.aim is not None
    # receptor detrás: no patea, se mueve
    m = PassMacro(1)
    mates[1] = [-300.0, 0.0]
    a = m.act(0, ball, np.zeros(2), np.zeros(2), np.zeros(2), mates, np.zeros((8, 2)), 0,
              hist=np.zeros(6, int), delay=0)
    assert a < 9
