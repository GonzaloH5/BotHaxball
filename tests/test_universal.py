"""Obs universal + SetActorCritic: un solo modelo para cualquier mapa y formato."""
import numpy as np
import pytest
import torch

from env.geometry import StadiumRays
from env.haxball_env import U_ENT_DIM, U_SELF_DIM, HaxballEnv
from sim.stadium import load_stadium
from train.model import SetActorCritic, build_model

E_MAX = 7


def _env(stadium, T, n=4, **kw):
    return HaxballEnv(n, T, stadium, obs_layout="universal", max_entities=E_MAX, seed=0,
                      random_reset_prob=1.0, **kw)


def test_same_width_for_every_map_and_format():
    dims = set()
    for stadium, T, kw in [("classic", 1, {}), ("big", 2, {}), ("futsal_x4", 4, {}),
                           ("x6_half", 3, {"powershot": True, "out_of_bounds": True})]:
        env = _env(stadium, T, **kw)
        o = env.reset()
        assert o.shape == (4, 2 * T, U_SELF_DIM + E_MAX * U_ENT_DIM)
        assert np.isfinite(o).all()
        # entidades presentes = 2T-1, el resto relleno
        ents = o[..., U_SELF_DIM:].reshape(4, 2 * T, E_MAX, U_ENT_DIM)
        assert (ents[..., 0].sum(-1) == 2 * T - 1).all()
        dims.add(o.shape[-1])
    assert len(dims) == 1


def test_mirror_symmetry():
    """Estado espejado en x: el rojo del original ve lo mismo que el azul del espejo."""
    env = _env("classic", 1, n=1)
    env.reset()
    o1 = env.observe()
    sim = env.sim
    fp = sim.first_player
    sim.pos[:, :, 0] *= -1
    sim.vel[:, :, 0] *= -1
    sim.pos[:, [fp, fp + 1]] = sim.pos[:, [fp + 1, fp]].copy()
    sim.vel[:, [fp, fp + 1]] = sim.vel[:, [fp + 1, fp]].copy()
    sim.kick_cancel[:] = sim.kick_cancel[:, ::-1].copy()
    sim.touch[:] = sim.touch[:, ::-1].copy()
    o2 = env.observe()
    np.testing.assert_allclose(o1[0, 0], o2[0, 1], atol=1e-5)
    np.testing.assert_allclose(o1[0, 1], o2[0, 0], atol=1e-5)


def test_rays_classic_known_geometry():
    g = StadiumRays(load_stadium("classic"))
    r = g.cast(np.array([[0.0, 0.0], [0.0, 100.0]]), "ball")
    assert r[0, 0] == pytest.approx(390.0)        # centro -> red del arco (x=400) - radio
    assert r[0, 2] == pytest.approx(160.0)        # hacia abajo -> línea y=170 - radio
    assert r[1, 0] == pytest.approx(360.0)        # fuera del arco -> línea de fondo x=370
    rp = g.cast(np.array([[0.0, 0.0]]), "red")
    assert rp[0, 2] == pytest.approx(185.0)       # jugador: plano y=200 - 15


@pytest.mark.parametrize("pooling", ["meanmax", "attention"])
def test_padding_does_not_change_output(pooling):
    """La misma situación con 7 o con 11 lugares de entidades da la misma salida."""
    env_a = _env("classic", 2, n=2)
    env_b = HaxballEnv(2, 2, "classic", obs_layout="universal", max_entities=11, seed=0, random_reset_prob=1.0)
    oa, ob = env_a.reset(), env_b.reset()
    np.testing.assert_allclose(oa[..., :U_SELF_DIM], ob[..., :U_SELF_DIM])
    m = SetActorCritic(U_SELF_DIM, U_ENT_DIM, pooling=pooling).eval()
    ta = torch.from_numpy(oa.reshape(-1, oa.shape[-1]))
    tb = torch.from_numpy(ob.reshape(-1, ob.shape[-1]))
    m.update_norm(ta)
    torch.testing.assert_close(m.logits(ta), m.logits(tb), atol=1e-5, rtol=1e-5)


def test_mixed_formats_in_one_batch():
    m = SetActorCritic(U_SELF_DIM, U_ENT_DIM)
    obs = [torch.from_numpy(_env(s, T).reset().reshape(-1, U_SELF_DIM + E_MAX * U_ENT_DIM))
           for s, T in [("classic", 1), ("big", 3), ("futsal_x4", 4)]]
    x = torch.cat(obs)
    m.update_norm(x)
    logits, v = m(x)
    assert torch.isfinite(logits).all() and torch.isfinite(v).all()
    m2 = build_model(m.config())
    m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(x)[0], m2(x)[0])
