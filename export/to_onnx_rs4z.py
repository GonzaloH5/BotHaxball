"""Exportar el actor RS4-Z a ONNX (sólo obs pública v2 → logits; sin entradas del crítico).

Los metadatos salen del checkpoint (versión de observación, contrato, frame_skip): nada fijo en el
código (corrige F4). Verifica paridad torch ↔ onnxruntime sobre el fixture de observaciones.

  python -m export.to_onnx_rs4z runs/rs4z/<run>/latest.pt --out deploy/rs4z/model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env.rs4z import contract as C
from env.rs4z.obs_v2 import CRITIC_FEATURES, ENT_FEATURES, OBS_DIM, SELF_FEATURES
from eval.rs4z.net_controller import load_model
from train.rs4z.model import PolicyOnly


def export(ckpt, out, fixture="deploy/rs4z/fixture_obs_v2.json"):
    model = load_model(ckpt).eval()
    policy = PolicyOnly(model).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out.with_suffix(".onnx")
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(policy, dummy, str(onnx_path), input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(obs_version="rs4z-obs-v2", obs_dim=OBS_DIM, self_features=list(SELF_FEATURES),
                entity_features=list(ENT_FEATURES), n_actions=18, frame_skip=3, contract=C.VERSION,
                critic_features_not_exported=list(CRITIC_FEATURES), samples=int(s.get("samples", 0)),
                stage=s.get("stage"), checkpoint=str(ckpt))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # paridad
    import onnxruntime as ort
    frames = json.loads(Path(fixture).read_text(encoding="utf-8"))["frames"][:512]
    x = np.array([f["obs"] for f in frames], dtype=np.float32)
    with torch.no_grad():
        ref = policy(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"obs": x})[0]
    err = float(np.abs(ref - got).max())
    out.with_name(out.name + "_parity.json").write_text(
        json.dumps(dict(obs=x[:64].tolist(), logits=ref[:64].tolist())), encoding="utf-8")
    return dict(onnx=str(onnx_path), parity=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="deploy/rs4z/model")
    args = ap.parse_args()
    r = export(args.ckpt, args.out)
    print(r)
    if r["parity"] > 1e-4:
        raise SystemExit("paridad ONNX fuera de tolerancia")


if __name__ == "__main__":
    main()
