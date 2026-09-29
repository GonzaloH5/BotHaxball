"""tools/grow_obs.py: la cirugía conserva la salida del modelo y deja seguir entrenando."""
import torch

from tools.grow_obs import grow_checkpoint, insert
from train.model import ActorCritic, SetActorCritic, build_model


def _ck(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # un paso de entrenamiento para que Adam tenga momentos y la normalización estadísticas
    S = model.self_dim if hasattr(model, "self_dim") else model.obs_dim
    obs = torch.randn(64, S + (3 * model.ent_dim if hasattr(model, "ent_dim") else 0))
    model.update_norm(obs)
    lo, v = model(obs)
    (lo.sum() + v.sum()).backward()
    opt.step()
    return {"model": model.state_dict(), "model_config": model.config(), "opt": opt.state_dict(),
            "league": [("snap", model.state_dict(), 1200.0, 3, 10)]}


def test_insert_keeps_old_values_in_order():
    t = torch.arange(5.0)
    assert insert(t, 0, [(0, 1), (3, 2), (5, 1)], fill=-1).tolist() == [-1, 0, 1, 2, -1, -1, 3, 4, -1]


def test_set_model_self_and_entity_growth_is_exact():
    for pooling in ("meanmax", "attention"):
        m = SetActorCritic(71, 8, hidden=32, layers=2, ent_hidden=16, pooling=pooling, ent_layers=1)
        out, err = grow_checkpoint(_ck(m), ["10:3", "end:4"], ["end:2", "4:1"])
        assert err < 1e-5, err
        cfg = out["model_config"]
        assert cfg["self_dim"] == 78 and cfg["ent_dim"] == 11
        new = build_model(cfg)
        new.load_state_dict(out["model"])                          # formas correctas
        build_model(cfg).load_state_dict(out["league"][0][1])       # la liga también
        opt = torch.optim.Adam(new.parameters(), lr=1e-3)
        opt.load_state_dict(out["opt"])                             # Adam retoma sin error
        obs = torch.randn(8, 78 + 3 * 11)
        lo, v = new(obs)
        (lo.sum() + v.sum()).backward()
        opt.step()
        # las entradas nuevas reciben gradiente: se pueden aprender
        assert new.pi_body[0].weight[:, 10:13].abs().sum() > 0


def test_flat_model_growth_is_exact():
    m = ActorCritic(40, hidden=32, layers=2)
    out, err = grow_checkpoint(_ck(m), ["end:5"], [])
    assert err < 1e-5 and out["model_config"]["obs_dim"] == 45
