"""El conversor de replays a datos de imitación (tools/build_bc_dataset.py) reproduce la obs del entorno."""
import numpy as np

from env.haxball_env import MIRROR_ACTION, HaxballEnv
from sim.physics import MOVE_DIRS
from tools.build_bc_dataset import Loader, input_to_action


def _to_input(a: int) -> int:
    dx, dy = MOVE_DIRS[a % 9]
    return (1 if dy < 0 else 0) | (2 if dy > 0 else 0) | (4 if dx < 0 else 0) | (8 if dx > 0 else 0) | (16 if a >= 9 else 0)


def test_input_mapping_roundtrip():
    for a in range(18):
        assert input_to_action(_to_input(a)) == a


def test_loader_reproduces_env_obs():
    T = 2
    env = HaxballEnv(1, T, "futsalx3", obs_layout="universal", max_entities=2 * T - 1, seed=3, random_reset_prob=1.0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step(rng.integers(0, 18, (1, env.P)))
    sim = env.sim
    fp = sim.first_player
    world = rng.integers(0, 18, env.P)  # teclas "del humano" en coordenadas del mundo
    # estado del env como un tick de replay: discos [pelota, estadio..., jugadores], ids rojo < azul
    discs = [{"x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy)}
             for (x, y), (vx, vy) in zip(sim.pos[0], sim.vel[0])]
    players = []
    for p in range(env.P):
        inp = _to_input(int(world[p])) | (16 if sim.kick_cancel[0, p] else 0)
        # isKicking de HaxBall: patada apretada y todavía no usada (el sim lo guarda como kick_cancel)
        players.append({"id": p, "team": 1 if sim.player_team[p] == 0 else 2, "disc": fp + p, "input": inp,
                        "kicking": bool(inp & 16) and not sim.kick_cancel[0, p]})
    tick = {"discs": discs, "players": players, "state": 1 if not sim.kickoff[0] else 0,
            "ko": 1 if sim.kickoff_team[0] == 0 else 2}
    ld = Loader("futsalx3", T, False, False, batch=4)
    obs, act = ld.obs([(tick, players, int(env.ticks[0]))])
    expected = env.observe()[0]
    np.testing.assert_allclose(obs, expected, atol=1e-6)
    blue = sim.player_team == 1
    exp_act = np.array([input_to_action(pl["input"]) for pl in players])
    exp_act[blue] = MIRROR_ACTION[exp_act[blue]]
    np.testing.assert_array_equal(act, exp_act)
