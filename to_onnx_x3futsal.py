"""Exportar una política de Futsal x3 (`learn.x3futsal_bc` / `learn.x3futsal_rl`) a ONNX para despliegue aislado (F5).

Escribe `<out>.onnx` (entradas `own` [N, 28], `mates` [N, 2, 10], `rivals` [N, 3, 10], salida `logits` [N, 18])
y `<out>.json` con el contrato completo (`x3-futsal-observation-v3`), parámetros de decodificación desacoplada,
guardia física de `kick_cancel` y verificación de paridad numérica con `onnxruntime`.

  python -m export.to_onnx_x3futsal --ckpt runs/x3futsal_rl/ppo_v1/best.pt --out deploy/x3futsal/x3futsal_rl.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from learn.x3futsal_data import (
    EFFECTIVE_KICK_DIST,
    TACTICAL_DATA_VERSION,
    TACTICAL_ENTITY_DIM,
    TACTICAL_FEATURE_CONTRACT,
    TACTICAL_SELF_DIM,
)
from learn.x3futsal_eval import FRAME_SKIP, STADIUM
from learn.x3futsal_policy import TACTICAL_POLICY_VERSION, X3TacticalPolicy
from sim.stadium import load_stadium
from tools.x3futsal_audit import stadium_identity


def export_x3futsal_onnx(
    ckpt_path: Path,
    out_path: Path,
    *,
    temperature: float = 0.85,
    kick_threshold: float = 0.28,
    stay_escape_threshold: float = 0.36,
) -> dict:
    if torch is None:
        raise RuntimeError("torch es necesario para exportar modelos X3 Futsal a ONNX")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy_version = ck.get("policyVersion")
    if policy_version != TACTICAL_POLICY_VERSION:
        raise ValueError(f"se requiere policyVersion={TACTICAL_POLICY_VERSION!r}, recibido: {policy_version!r}")

    hidden = int(ck.get("hidden", 192))
    entity_hidden = int(ck.get("entityHidden", 96))
    use_history = bool(ck.get("useHistory", False))
    if use_history:
        raise ValueError("la exportación ONNX F5 requiere política táctica sin dependencia de acción previa (useHistory=False)")

    model = X3TacticalPolicy(hidden=hidden, entity_hidden=entity_hidden, use_history=False)
    model.load_state_dict(ck["model"])
    model.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    own = torch.zeros(6, TACTICAL_SELF_DIM, dtype=torch.float32)
    mates = torch.zeros(6, 2, TACTICAL_ENTITY_DIM, dtype=torch.float32)
    rivals = torch.zeros(6, 3, TACTICAL_ENTITY_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        (own, mates, rivals),
        str(out_path),
        input_names=["own", "mates", "rivals"],
        output_names=["logits"],
        dynamic_axes={
            "own": {0: "n"},
            "mates": {0: "n"},
            "rivals": {0: "n"},
            "logits": {0: "n"},
        },
        opset_version=17,
        dynamo=False,
    )

    stadium = load_stadium(STADIUM)
    target_id = ck.get("datasetSignature", {}).get("targetStadiumIdentity") or stadium_identity(STADIUM)
    meta = {
        "modality": "x3futsal",
        "obs_version": TACTICAL_DATA_VERSION,
        "policy_version": TACTICAL_POLICY_VERSION,
        "feature_contract": TACTICAL_FEATURE_CONTRACT,
        "self_dim": TACTICAL_SELF_DIM,
        "n_mates": 2,
        "n_rivals": 3,
        "entity_dim": TACTICAL_ENTITY_DIM,
        "n_actions": 18,
        "frame_skip": FRAME_SKIP,
        "stadium_name": stadium.name,
        "stadium_identity": target_id,
        "field_half_w": float(stadium.field_half_w),
        "field_half_h": float(stadium.field_half_h),
        "goal_x": float(stadium.goal_x),
        "goal_half_height": float(stadium.goal_half_height),
        "decoder": {
            "temperature": float(temperature),
            "kick_threshold": float(kick_threshold),
            "stay_escape_threshold": float(stay_escape_threshold),
            "max_kick_dist_px": float(EFFECTIVE_KICK_DIST + 3.5),
            "rearm_kick_cancel_pulse": True,
        },
        "source": str(ckpt_path),
        "step": ck.get("step") or ck.get("update"),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    o_np = rng.normal(size=(64, TACTICAL_SELF_DIM)).astype(np.float32)
    m_np = rng.normal(size=(64, 2, TACTICAL_ENTITY_DIM)).astype(np.float32)
    r_np = rng.normal(size=(64, 3, TACTICAL_ENTITY_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(o_np), torch.from_numpy(m_np), torch.from_numpy(r_np)).numpy()
    got = sess.run(["logits"], {"own": o_np, "mates": m_np, "rivals": r_np})[0]
    soft = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    err_p = float(np.abs(soft(ref) - soft(got)).max())
    if err_p > 1e-5:
        raise RuntimeError(f"FALLA: ONNX no coincide con torch (err_p={err_p:.2e})")
    meta["maxProbError"] = err_p
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--kick-threshold", type=float, default=0.28)
    args = ap.parse_args()
    meta = export_x3futsal_onnx(
        args.ckpt, args.out, temperature=args.temp, kick_threshold=args.kick_threshold
    )
    print(f"{args.out} exportado correctamente (maxProbError={meta['maxProbError']:.2e})", flush=True)


if __name__ == "__main__":
    main()