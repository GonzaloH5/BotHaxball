"""Regresiones del humo de infraestructura en el Pod (2026-10-04).

- Los pesos de etapa son fracciones del tiempo simulado: con la probabilidad por episodio, los partidos
  (10–30× más largos) dejaban a ejercicios y repaso en <1% del tiempo de S5–S7.
- Desde S5 los partidos duran entre 3 y 10 min de reloj (la sala juega ~10 min).
- Un partido con el reloj congelado (nadie saca el inicial) se corta por truncación.
- La compuerta de saques mide los saques PROPIOS del candidato (antes no podía fallar).
"""
import json

import numpy as np
import pytest

from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from env.rs4z.drills import MATCH_STALL_EXTRA, MATCH_STALL_FACTOR, Drills
from eval.rs4z.cheaters import Cheater
from eval.rs4z.gates import RULES, evaluate_gate
from eval.rs4z.runner import play
from bots.rspro.controller import RSProController
from train.rs4z.run import Config, Trainer
from train.rs4z.stages import STAGES


@pytest.fixture(scope="module")
def s5(tmp_path_factory):
    return Trainer(Config(run="pytest_rs4z_s5", envs=8, rollout=8, device="cpu", minibatch=512, samples=1e12,
                          start_stage="S5"))


def test_stage_weights_are_time_shares(s5):
    """Proceso de renovación con la probabilidad de inicio del entrenador: la fracción de tiempo ≈ peso."""
    names, w = s5._task_weights()
    _, p = s5._task_probs()
    lens = np.array([s5.ep_len[t] for t in names])
    assert lens[names.index("match_4v4")] > 10 * lens[names.index("situation_open")]
    rng = np.random.default_rng(0)
    slots, horizon = 256, 200_000
    time_in = np.zeros(len(names))
    for _ in range(slots):
        t = 0.0
        while t < horizon:
            i = rng.choice(len(names), p=p)
            d = rng.exponential(lens[i])
            time_in[i] += min(d, horizon - t)
            t += d
    share = time_in / time_in.sum()
    assert np.abs(share - w).max() < 0.03, dict(zip(names, np.round(share - w, 3)))
    drills = [i for i, t in enumerate(names) if "match" not in t]
    assert share[drills].sum() > 0.3


def test_full_matches_have_random_length_from_s5():
    tr = Trainer(Config(run="pytest_rs4z_s5_len", envs=8, rollout=8, device="cpu", minibatch=512, samples=1e12,
                        start_stage="S5", only_tasks="match_4v4"))
    lengths = tr.env.ri[:, K.RI_LEN]
    assert tr.is_match.all()
    assert (lengths >= 3 * 3600).all() and (lengths <= 10 * 3600).all()
    assert len(np.unique(lengths)) > 1


def test_early_stages_keep_task_lengths():
    stage_names = [s.name for s in STAGES]
    assert stage_names.index("S4") < stage_names.index("S5")
    tr = Trainer(Config(run="pytest_rs4z_s2", envs=4, rollout=8, device="cpu", minibatch=512, samples=1e12,
                        start_stage="S2", only_tasks="match_1v1"))
    assert (tr.env.ri[:, K.RI_LEN] == 2 * 3600).all()


def test_stalled_match_is_truncated():
    env = RS4ZEnv(2, seed=0)
    d = Drills(env, np.random.default_rng(0))
    d.start([0, 1], "match_4v4", 0.0, match_ticks=3600)
    ev = dict(goal=np.zeros(2, int), touched=np.zeros((2, 8), bool), restart_start=np.zeros(2, int),
              ticks=np.full(2, 3))
    d.st.ticks[0] = MATCH_STALL_FACTOR * 3600 + MATCH_STALL_EXTRA
    done, out, tr = d.check(ev)
    assert done[0] and tr[0] and out[0] == 0.0
    assert not done[1]


def test_restart_log_records_both_teams():
    bot = RSProController(levels=np.full((6, 2), 5), seed=0)
    r = play(bot, bot, n_envs=6, match_ticks=3600, seed=3)
    for t in (0, 1):
        d = r["restarts"].team(t)
        n = sum(len(v) for v in d.values())
        assert n > 5
        allv = np.concatenate([np.asarray(v, float) for v in d.values() if v])
        assert (allv > 0).all() and np.median(allv) < 600


def test_restart_gate_fails_for_a_candidate_that_never_takes_restarts():
    res = evaluate_gate("restart_battery", Cheater("still"), calib={}, episodes=16, seed=0)
    assert not res["passed"]
    assert res["late_rate"] > RULES["restart_battery"]["max_late_rate"]


def test_restart_gate_passes_for_rspro():
    res = evaluate_gate("restart_battery", RSProController(levels=np.full((4, 2), 5), seed=1), calib={},
                        episodes=32, seed=0)
    assert res["late_rate"] <= RULES["restart_battery"]["max_late_rate"], res["late_rate"]
    assert not any(res["slow"].values()), res["medians"]


def test_exploiter_publication(tmp_path, monkeypatch):
    import torch

    import eval.rs4z.gates as gates
    from train.rs4z.league import League
    from train.rs4z.model import ActorCritic

    main_dir = tmp_path / "main"
    main_dir.mkdir()
    m = ActorCritic()
    torch.save(dict(model=m.state_dict(), model_config=m.config, samples=10), main_dir / "latest.pt")
    ex = Trainer(Config(run="pytest_rs4z_exploiter", envs=4, rollout=8, device="cpu", minibatch=512, samples=1e12,
                        exploiter_of=str(main_dir / "latest.pt")))
    assert ex._task_weights()[0] == ["league_4v4"]
    for pts, accepted in ((0.4, False), (0.7, True)):
        monkeypatch.setattr(gates, "head_to_head", lambda a, b, games, seed, _p=pts: (_p, (_p - 0.05, _p + 0.05)))
        ex._publish_exploiter()
        data = json.loads((main_dir / "exploiters.json").read_text(encoding="utf-8"))
        assert data["history"][-1]["accepted"] is accepted
    assert len(data["members"]) == 1 and len(data["history"]) == 2
    main_league = League(main_dir / "league")
    assert main_league.merge_exploiters(main_dir / "exploiters.json") == 1


def test_matches_where_nobody_kicks_off_end_at_the_stall_cap():
    """Dos redes que nunca sacan el inicial congelan el reloj: la evaluación entre redes las cierra (no falla)."""
    still = Cheater("still")
    with pytest.raises(RuntimeError):
        play(still, still, n_envs=2, match_ticks=60, seed=0)
    r = play(still, still, n_envs=2, match_ticks=600, seed=0, on_stall="end")
    assert r["events"]["stalled"].all()
    assert (r["score"] == 0).all()
