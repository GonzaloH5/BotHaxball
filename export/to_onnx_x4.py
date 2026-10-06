"""Exportar una política X4 (`learn.x4_bc` / `learn.x4_ppo`) a ONNX para `deploy/bot.js`.

Escribe <out>.onnx (entrada `obs` [N, OBS_DIM] float32, salida `logits` [N, 18]) y <out>.json con lo que el
bot necesita (versión de observación, frame_skip, retardo máximo, rangos de pelota y patada) y verifica con
onnxruntime que los logits coinciden con torch.

  python -m export.to_onnx_x4 --ckpt runs/x4_bc/final/best.pt --out deploy/rs4z/x4_bc.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.rs4z import obs_v3
from learn.x4_policy import SetPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    model = SetPolicy(hidden=ck.get("hidden", 256))
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(4, obs_v3.OBS_DIM)
    torch.onnx.export(model, (x,), str(out), input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    # mapas en los que se entrenó: la imitación los guarda en "maps"; el RL (learn/x4_ppo.py), en sus argumentos
    # ("sanguchito_rs_x4:1.0,..."). El bot avisa si la sala es de otro mapa.
    maps = ck.get("maps")
    if not maps and isinstance(ck.get("args"), dict) and ck["args"].get("maps"):
        maps = [m.split(":")[0] for m in str(ck["args"]["maps"]).split(",") if m]
    meta = dict(obs_version=obs_v3.OBS_VERSION, obs_dim=obs_v3.OBS_DIM, n_actions=18, frame_skip=3,
                max_delay=int(obs_v3.MAX_DELAY), default_delay=10, ball_radii=[8.0, 9.0], kick_strengths=[5.65, 5.85],
                # la política se evalúa y se certifica muestreando con temperatura 1 (deploy/bot.js la usa por defecto)
                temperature=1.0,
                maps=list(maps or obs_v3.MAP_NAMES), stage=a.stage or Path(a.ckpt).parent.name,
                source=str(a.ckpt), step=ck.get("step") or ck.get("update"), delay=ck.get("delay"))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, obs_v3.OBS_DIM)).astype(np.float32)
    obs[:, obs_v3.SELF_DIM::obs_v3.ENT_DIM] = (rng.random((64, obs_v3.N_ENT)) < 0.9).astype(np.float32)
    ref = model(torch.from_numpy(obs)).detach().numpy()
    got = sess.run(["logits"], {"obs": obs})[0]
    err = float(np.abs(ref - got).max())
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB), diferencia máxima torch/onnx {err:.2e}")
    if err > 1e-4:
        raise SystemExit("FALLA: ONNX no coincide con torch")


if __name__ == "__main__":
    main()
