import numpy as np
import torch

from env.haxball_env import ENT_DIM, HaxballEnv
from train.model import EntityActorCritic, build_model


def _obs(T, n=3):
    env = HaxballEnv(n, T, "x6_half" if T < 6 else "x6", powershot=True, out_of_bounds=True,
                     obs_layout="entities", seed=0, random_reset_prob=1.0)
    return env, torch.from_numpy(env.reset().reshape(-1, env.obs_dim))


def test_same_weights_run_every_team_size():
    env, _ = _obs(1)
    m = EntityActorCritic(env.self_dim)
    for T in (1, 2, 3, 6):
        env, o = _obs(T)
        assert env.obs_dim == env.self_dim + ENT_DIM * (2 * T - 1)
        logits, v = m(o)
        assert logits.shape == (o.shape[0], 18) and torch.isfinite(logits).all()


def test_entity_order_does_not_matter_within_group():
    env, o = _obs(3, n=1)
    m = EntityActorCritic(env.self_dim)
    m.update_norm(o)
    S = env.self_dim
    x = o[:1].clone()
    ents = x[:, S:].reshape(1, -1, ENT_DIM)  # 2 compañeros + 3 rivales
    perm = torch.cat([ents[:, [1, 0]], ents[:, [4, 2, 3]]], dim=1)
    y = torch.cat([x[:, :S], perm.reshape(1, -1)], dim=1)
    np.testing.assert_allclose(m.logits(x).detach(), m.logits(y).detach(), atol=1e-5)


def test_build_model_roundtrip():
    m = EntityActorCritic(28, hidden=64)
    m2 = build_model(m.config())
    m2.load_state_dict(m.state_dict())
    _, o = _obs(2)
    torch.testing.assert_close(m(o)[0], m2(o)[0])
