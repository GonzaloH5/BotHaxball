"""Exporta un checkpoint a ONNX + metadatos + fixture de prueba para el bot de Node.

python -m export.to_onnx runs/classic_1v1/latest.pt --out deploy/model

Genera:
  deploy/model.onnx          obs crudas (batch, OBS_DIM) -> logits (batch, 18)
  deploy/model.json          metadatos (tamaño de equipo, frame_skip, escalas...)
  deploy/fixture.json        estados de ejemplo + obs y logits esperados (lo usa test_obs.js)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bots.scripted import scripted_actions
from env.haxball_env import POS_SCALE, VEL_SCALE, HaxballEnv
from train.model import PolicyOnly, load_model


def make_fixture(model, n_per_team: int, n: int = 40):
    """Estados variados (con saque y sin saque) con su obs y logits esperados."""
    env = HaxballEnv(1, n_per_team, random_reset_prob=0.5, seed=123, max_ticks=7200)
    obs = env.reset()
    rng = np.random.default_rng(0)
    out = []
    for i in range(n * 7):
        a = scripted_actions(env, None, eps=0.3, rng=rng)
        obs, _, _, _ = env.step(a)
        if i % 7 == 0:
            s = env.sim
            fp = s.first_player
            with torch.no_grad():
                logits = model.logits(torch.from_numpy(obs[0])).numpy()
            out.append({
                "ball": {"pos": s.pos[0, 0].tolist(), "vel": s.vel[0, 0].tolist()},
                "players": [{"team": int(s.player_team[p]), "pos": s.pos[0, fp + p].tolist(),
                             "vel": s.vel[0, fp + p].tolist(), "canKick": bool(not s.kick_cancel[0, p]),
                             "touching": bool(s.touch[0, p])} for p in range(s.P)],
                "kickoff": bool(s.kickoff[0]), "kickoffTeam": int(s.kickoff_team[0]),
                "tfrac": float(env.ticks[0] / env.max_ticks),
                "obs": obs[0].tolist(), "logits": logits.tolist(),
            })
    return out


FIXTURE_TASKS = ["classic_1v1", "big_3v3", "rs_2v2", "rf_7v7"]


def make_fixture_universal(model, tasks=FIXTURE_TASKS, per_task: int = 6):
    """Estados de varias tareas (mapas y formatos distintos) con su obs y logits esperados, más la
    geometría de cada estadio (la misma que usa el bot vía export/stadium_geom.py)."""
    from env.tasks import load_catalog, make_env
    from export.stadium_geom import geometry

    catalog = load_catalog()
    rng = np.random.default_rng(0)
    out, geoms = [], {}
    recurrent = getattr(model, "is_recurrent", False)
    for name in tasks:
        t = catalog[name]
        geoms.setdefault(t.stadium, geometry(t.stadium))
        env = make_env(t, 1, t.n_entities, seed=7, random_reset_prob=0.5)
        obs = env.reset()
        memory = model.initial_state(env.P) if recurrent else None
        stride = 1 if recurrent else 9
        for i in range(per_task * stride):
            actions = scripted_actions(env, None, eps=0.3, rng=rng)
            obs, _, done, _ = env.step(actions)
            if i % stride:
                continue
            s = env.sim
            fp = s.first_player
            with torch.no_grad():
                extra = {}
                if recurrent:
                    previous = torch.from_numpy(actions[0].astype(np.int64))
                    if done[0]:
                        memory.zero_()
                        previous.fill_(model.n_actions)
                    extra = {"memory": memory.tolist(), "previous_action": previous.tolist(), "reset": bool(done[0])}
                    logits, _, memory = model.step(torch.from_numpy(obs[0]), memory, previous)
                    extra["memory_out"] = memory.tolist()
                    logits = logits.numpy()
                else:
                    logits = model.logits(torch.from_numpy(obs[0])).numpy()
            ps = None
            if s.ps_on:
                cfg = s.ps_cfg
                ps = {"comba": float(s.ps_comba[0]),
                      "prog": float(1.0 - s.ps_charge[0] / cfg["charge"]) if s.ps_charge[0] > 0 else 0.0,
                      "invb": float((s.inv_env[0, 0] - cfg["base_inv"]) / (cfg["power_inv"] - cfg["base_inv"])),
                      "grav": (s.ball_grav[0] / cfg["grav"]).tolist(), "holder": int(s.ps_held[0])}
            out.append({
                **extra,
                "task": name, "stadium": t.stadium,
                "opts": {"maxEntities": t.n_entities, "psOn": t.powershot,
                         "outOfBounds": t.out_of_bounds or t.rules is not None},
                "ball": {"pos": s.pos[0, 0].tolist(), "vel": s.vel[0, 0].tolist()},
                "players": [{"team": int(s.player_team[p]), "pos": s.pos[0, fp + p].tolist(),
                             "vel": s.vel[0, fp + p].tolist(), "canKick": bool(not s.kick_cancel[0, p]),
                             "touching": bool(s.touch[0, p]),
                             "expelled": bool(env.rules is not None and env.rules.expelled[0, p])}
                            for p in range(s.P)],
                "rules": env.rules.features()[0].tolist() if env.rules is not None else None,
                "kickoff": bool(s.kickoff[0]), "kickoffTeam": int(s.kickoff_team[0]),
                "tfrac": float(env.ticks[0] / env.max_ticks), "ps": ps,
                "obs": obs[0].tolist(), "logits": logits.tolist(),
            })
    return {"geoms": geoms, "states": out}


def export_universal(args, ck, model, out: Path):
    from env.haxball_env import U_SELF_DIM
    E = 3
    dummy = torch.zeros(1, U_SELF_DIM + model.ent_dim * E)
    recurrent = getattr(model, "is_recurrent", False)
    if recurrent:
        from train.recurrent_model import RecurrentPolicyOnly
        torch.onnx.export(RecurrentPolicyOnly(model),
                          (dummy, model.initial_state(1), torch.full((1,), model.n_actions, dtype=torch.long)),
                          str(out.with_suffix(".onnx")), input_names=["obs", "memory", "previous_action"],
                          output_names=["logits", "memory_out"],
                          dynamic_axes={"obs": {0: "batch", 1: "width"}, "memory": {0: "batch"},
                                        "previous_action": {0: "batch"}, "logits": {0: "batch"},
                                        "memory_out": {0: "batch"}}, opset_version=17, dynamo=False)
    else:
        torch.onnx.export(PolicyOnly(model), dummy, str(out.with_suffix(".onnx")),
                          input_names=["obs"], output_names=["logits"],
                          dynamic_axes={"obs": {0: "batch", 1: "width"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
    envcfg = ck.get("env", {})
    meta = {"layout": "universal", "self_dim": U_SELF_DIM, "ent_dim": model.ent_dim, "n_actions": model.n_actions,
            "rule_observation": model.rule_observation,
            "recurrent": recurrent, "memory_size": model.memory_size if recurrent else 0,
            "frame_skip": envcfg.get("frame_skip", 3), "max_ticks": 7200, "tasks": envcfg.get("tasks"),
            # powershot del script (sim/physics.py): para estimar sus features en salas reales
            "ps_cfg": {"charge": 96, "power_inv": 2.3, "grav": 0.1},
            "checkpoint": str(args.ckpt), "steps": ck.get("steps")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    fx = make_fixture_universal(model)
    (out.parent / "fixture.json").write_text(json.dumps(fx), encoding="utf-8")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out.with_suffix(".onnx")))
    err = 0.0
    for f in fx["states"]:  # cada tarea tiene otro ancho de obs: el ONNX acepta cualquiera
        feeds = {"obs": np.array(f["obs"], dtype=np.float32)}
        if recurrent:
            feeds.update(memory=np.array(f["memory"], dtype=np.float32),
                         previous_action=np.array(f["previous_action"], dtype=np.int64))
        result = sess.run(None, feeds)
        got = result[0]
        if recurrent:
            err = max(err, float(np.abs(result[1] - np.array(f["memory_out"], dtype=np.float32)).max()))
        err = max(err, float(np.abs(got - np.array(f["logits"], dtype=np.float32)).max()))
    print(f"exportado {out.with_suffix('.onnx')} (obs universal, {len(fx['states'])} estados de "
          f"{len(FIXTURE_TASKS)} tareas); error máx ONNX vs torch: {err:.2e}")
    assert err < 1e-3, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="deploy/model")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = load_model(args.ckpt)
    if ck.get("model_config", {}).get("type") in ("set", "recurrent_set"):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        return export_universal(args, ck, model, out)
    envcfg = ck.get("env", {"n_per_team": 1, "stadium": "classic", "frame_skip": 3})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, model.obs_dim)
    torch.onnx.export(PolicyOnly(model), dummy, str(out.with_suffix(".onnx")),
                      input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    st_env = HaxballEnv(1, envcfg["n_per_team"])
    meta = {
        "obs_dim": model.obs_dim, "n_actions": model.n_actions, "n_per_team": envcfg["n_per_team"],
        "stadium": envcfg["stadium"], "frame_skip": envcfg["frame_skip"],
        "pos_scale": POS_SCALE, "vel_scale": VEL_SCALE, "goal_x": st_env.goal_x,
        "max_ticks": 7200, "checkpoint": str(args.ckpt), "steps": ck.get("steps"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    fx = make_fixture(model, envcfg["n_per_team"])
    (out.parent / "fixture.json").write_text(json.dumps(fx), encoding="utf-8")

    # verificar que ONNX == PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out.with_suffix(".onnx")))
    obs = np.array([p for f in fx for p in f["obs"]], dtype=np.float32)
    got = sess.run(None, {"obs": obs})[0]
    exp = np.array([p for f in fx for p in f["logits"]], dtype=np.float32)
    err = float(np.abs(got - exp).max())
    print(f"exportado {out.with_suffix('.onnx')} (obs_dim {model.obs_dim}); error máx ONNX vs torch: {err:.2e}")
    assert err < 1e-3, err


if __name__ == "__main__":
    main()
