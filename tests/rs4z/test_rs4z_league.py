"""Liga RS4-Z: PFSP, registro de resultados, inferencia de miembros congelados y exploiters."""
import json

import numpy as np

from env.rs4z.core import RS4ZEnv
from env.rs4z.obs_v2 import observe
from train.rs4z.league import League
from train.rs4z.model import ActorCritic


def test_pfsp_prefers_members_that_still_win(tmp_path):
    lg = League(tmp_path / "league")
    m = ActorCritic()
    for s in (1, 2, 3):
        lg.add(m, s)
    # el aprendiz le gana siempre al 0, nunca al 2
    for _ in range(50):
        lg.record(0, 1.0)
        lg.record(2, 0.0)
    rng = np.random.default_rng(0)
    draws = np.bincount([lg.sample(rng) for _ in range(3000)], minlength=3)
    assert draws[2] > draws[1] > draws[0]


def test_frozen_member_inference_fills_only_masked_rows(tmp_path):
    lg = League(tmp_path / "league")
    lg.add(ActorCritic(), 10)
    env = RS4ZEnv(4, seed=0)
    obs = observe(env)
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 4:] = True
    member = np.array([0, 0, -1, 0])
    out = np.full((4, 8), -1, dtype=np.int64)
    lg.act(env, obs, mask, member, out)
    assert (out[[0, 1, 3], 4:] >= 0).all() and (out[[0, 1, 3], 4:] < 18).all()
    assert (out[2] == -1).all() and (out[:, :4] == -1).all()


def test_exploiters_merge_once(tmp_path):
    main = League(tmp_path / "main")
    main.add(ActorCritic(), 5)
    ext = League(tmp_path / "main")
    ext.load_state(main.state())
    ext.add(ActorCritic(), 7, exploiter=True)
    (tmp_path / "exploiters.json").write_text(json.dumps(ext.state()), encoding="utf-8")
    assert main.merge_exploiters(tmp_path / "exploiters.json") == 1
    assert main.merge_exploiters(tmp_path / "exploiters.json") == 0
    rng = np.random.default_rng(0)
    assert main.members[main.sample(rng, exploiter=True)]["exploiter"]


def test_member_indices_are_stable_when_retiring(tmp_path):
    lg = League(tmp_path / "league", max_members=3)
    m = ActorCritic()
    for s in range(5):
        lg.add(m, s)
        lg.record(len(lg) - 1, 1.0 if s % 2 else 0.0)
    assert len(lg) == 5                      # nunca se borra: los partidos en curso guardan el índice
    live = [i for i, x in enumerate(lg.members) if not x.get("retired")]
    assert len(live) == 3
    rng = np.random.default_rng(0)
    assert all(not lg.members[lg.sample(rng)].get("retired") for _ in range(200))
    assert [x["samples"] for x in lg.members] == list(range(5))


def test_active_pool_bounds_distinct_opponents_and_keeps_pfsp(tmp_path):
    """Con muchos miembros, pocos rivales distintos en juego a la vez (costo por paso) y PFSP en el largo plazo."""
    lg = League(tmp_path / "league", active_size=8, refresh_every=100)
    lg.members = [dict(path=f"m{i}.pt", samples=i, wins=1.0, games=2.0, exploiter=False) for i in range(30)]
    for i in range(15):                      # el aprendiz le gana siempre a los 15 primeros
        lg.members[i]["wins"], lg.members[i]["games"] = 99.0, 100.0
    rng = np.random.default_rng(0)
    draws = [lg.sample(rng) for _ in range(5000)]
    for k in range(0, 5000, 100):
        assert len(set(draws[k:k + 100])) <= 8
    easy = np.mean([d < 15 for d in draws])
    assert easy < 0.05
    assert len(set(draws)) > 8              # el grupo activo se renueva


def test_frozen_sampling_matches_the_policy_distribution(tmp_path):
    """El muestreo Gumbel-max reproduce softmax(logits) del miembro."""
    import torch
    lg = League(tmp_path / "league")
    model = ActorCritic()
    lg.add(model, 1)
    env = RS4ZEnv(1, seed=0)
    obs = np.repeat(observe(env)[:, :1], 8, axis=1)          # misma obs en los 8 lugares
    obs = np.repeat(obs, 512, axis=0)
    mask = np.ones((512, 8), dtype=bool)
    member = np.zeros(512, dtype=np.int64)
    out = np.zeros((512, 8), dtype=np.int64)
    lg.act(None, obs, mask, member, out)
    freq = np.bincount(out.ravel(), minlength=18) / out.size
    with torch.no_grad():
        p = torch.softmax(lg._model(0).logits(torch.from_numpy(obs[0, :1])), -1).numpy()[0]
    assert np.abs(freq - p).max() < 0.02


def test_stacked_inference_equals_each_member(tmp_path):
    """La llamada única (vmap sobre parámetros apilados) da los mismos logits que cada miembro por separado."""
    import torch
    lg = League(tmp_path / "league")
    for s in range(3):
        torch.manual_seed(s)
        lg.add(ActorCritic(), s)
    env = RS4ZEnv(6, seed=1)
    obs = observe(env)
    x = torch.from_numpy(obs[:, 0])
    params = lg._stacked(np.array([2, 0]))
    from torch.func import functional_call, vmap
    X = torch.stack([x, x])
    got = vmap(lambda p, xb: functional_call(lg._base, p, (xb,)))(params, X)
    with torch.no_grad():
        for k, i in enumerate((2, 0)):
            assert torch.allclose(got[k], lg._model(i).logits(x), atol=1e-5)
