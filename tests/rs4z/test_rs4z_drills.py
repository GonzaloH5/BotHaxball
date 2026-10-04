"""Motor de ejercicios: planteles, colocaciones a través del árbitro y resultados."""
import numpy as np
import pytest

from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from env.rs4z.drills import TASKS, Drills, team_slots


def _ev(n, goal=0, touched=None, restart_start=0, ticks=3):
    return dict(goal=np.array([goal] * n), touched=np.zeros((n, 8), bool) if touched is None else touched,
                restart_start=np.array([restart_start] * n), ticks=np.full(n, ticks))


@pytest.mark.parametrize("name", [t for t in TASKS if TASKS[t].start not in ("recorded_open", "recorded_attack")])
def test_every_task_places_valid_rosters(name):
    env = RS4ZEnv(4, seed=0)
    d = Drills(env, np.random.default_rng(0))
    task = TASKS[name]
    d.start(np.arange(4), name, 0.5)
    for n in range(4):
        lt = d.st.learner_team[n]
        assert env.active[n, team_slots(lt, 4)].sum() == task.n_own
        assert env.active[n, team_slots(1 - lt, 4)].sum() == task.n_opp
        assert np.isfinite(env.pos[n]).all()
    learner, scripted = d.controllers()
    assert (learner & scripted).sum() == 0
    assert ((learner | scripted) == env.active).all()


def test_goal_success_and_concede_outcomes():
    env = RS4ZEnv(1, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0], "empty_goal", 0.0, learner_team=0)
    done, out, tr = d.check(_ev(1, goal=1))
    assert done[0] and out[0] == 1.0
    d.start([0], "defend_1v1", 0.0, learner_team=0)
    done, out, tr = d.check(_ev(1, goal=-1))
    assert done[0] and out[0] == -1.0


def test_regain_needs_sustained_control_and_rival_control_ends_attack():
    env = RS4ZEnv(1, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0], "defend_1v1", 0.0, learner_team=0)
    own = np.zeros((1, 8), bool)
    own[0, 0] = True
    done, out, _ = d.check(_ev(1, touched=own))
    assert not done[0]
    for _ in range(25):
        done, out, _ = d.check(_ev(1))
        if done[0]:
            break
    assert done[0] and out[0] == 1.0
    d.start([0], "attack_1v1", 0.0, learner_team=0)
    rival = np.zeros((1, 8), bool)
    rival[0, 4] = True
    done, out, _ = d.check(_ev(1, touched=rival))
    assert not done[0], "un roce del rival no termina el ataque"
    for _ in range(12):
        done, out, _ = d.check(_ev(1))
        if done[0]:
            break
    assert done[0] and out[0] == 0.0


def test_timeout_outcomes():
    env = RS4ZEnv(1, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0], "situation_open", 0.0, learner_team=0)
    for _ in range(400):
        done, out, tr = d.check(_ev(1))
        if done[0]:
            break
    assert done[0] and tr[0], "situación abierta: el corte por tiempo es truncación"
    d.start([0], "defend_2v2", 0.0, learner_team=0)
    for _ in range(400):
        done, out, tr = d.check(_ev(1))
        if done[0]:
            break
    assert done[0] and out[0] == 1.0 and not tr[0], "defensa: aguantar sin gol es éxito"


def test_clean_rival_win_ends_the_attack():
    """En ataque, si el rival toca la pelota sin ningún aprendiz cerca, el ataque terminó (aunque después la
    recupere); una disputa cuerpo a cuerpo sigue la regla de control sostenido."""
    env = RS4ZEnv(2, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0, 1], "attack_1v1", 0.0, learner_team=0)
    pp = env.player_pos.copy()
    pp[0, 0] = env.ball_pos[0] + np.array([-200.0, 0.0])     # fila 0: aprendiz lejos de la pelota
    pp[1, 0] = env.ball_pos[1] + np.array([-30.0, 0.0])      # fila 1: aprendiz pegado a la pelota
    for n in (0, 1):
        env.place(n, ball_pos=env.ball_pos[n], ball_vel=(0.0, 0.0), player_pos=pp[n], player_vel=np.zeros((8, 2)))
    touched = np.zeros((2, 8), bool)
    touched[:, 4] = True
    done, out, tr = d.check(_ev(2, touched=touched))
    assert done[0] and out[0] == 0.0
    assert not done[1]


def test_pass_drills_need_a_pass_before_the_goal():
    """Pared y pase en profundidad: un gol sin pase entre aprendices no es éxito (la tarea es el pase)."""
    env = RS4ZEnv(2, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0, 1], "one_two", 0.0, learner_team=0)
    touched = np.zeros((2, 8), bool)
    touched[1, 0] = True
    d.check(_ev(2, touched=touched))
    touched[1, 0], touched[1, 1] = False, True
    d.check(_ev(2, touched=touched))            # fila 1: R0 → R1 (pase)
    done, out, tr = d.check(_ev(2, goal=1))
    assert done.all()
    assert out[0] == 0.0 and out[1] == 1.0
    assert d.st.passed[1] and not d.st.passed[0]


def test_restart_take_succeeds_when_the_own_restart_is_executed():
    """Saque propio: éxito al ejecutarlo (o al sacar el inicial); el saque arranca del equipo aprendiz."""
    env = RS4ZEnv(16, seed=0)
    d = Drills(env, np.random.default_rng(1))
    d.start(np.arange(16), "restart_take", 0.5, learner_team=0)
    own_restart = (env.ri[:, K.RI_TEAM] == 0) | ((env.ri[:, K.RI_KO] != 0) & (env.ri[:, K.RI_KO_TEAM] == 0))
    assert own_restart.all()
    ev = _ev(16)
    ev["restart_exec"] = np.zeros(16, dtype=np.int64)
    ev["kickoff_taken"] = np.zeros(16, dtype=bool)
    ev["restart_exec"][3] = 1
    ev["kickoff_taken"][5] = True
    done, out, tr = d.check(ev)
    assert done[3] and out[3] == 1.0 and done[5] and out[5] == 1.0
    assert done.sum() == 2
