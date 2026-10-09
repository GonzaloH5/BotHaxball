"""Exportar una política de HaxEleven (`learn.hx11_bc`) a ONNX para el bot de sala (`deploy/hx11/`).

Escribe <out>.onnx (entradas `own` [N, SELF_DIM(+EXTRA)] y `ents` [N, 21, 9], salida `logits` [N, 18]) y <out>.json con
lo que necesita el bot: versión de observación, decisión cada 3 ticks (P11), retardo de entrenamiento, temperatura
sugerida, vocabulario de `--pos` (`reports/hx11/pos_vocab.json`) y los flags del modelo (sin historia, rasgos extra).
Verifica con onnxruntime que las probabilidades coinciden con torch.

  python -m export.to_onnx_hx11 --ckpt runs/hx11_l0/l0_sinhist_full/model.pt --out deploy/hx11/l0.onnx --temp 0.7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learn.hx11_data import D_MAX, D_MIN, ENT_DIM, EXTRA_DIM, LINES, N_ENT, SELF_DIM
from learn.hx11_policy import SetPolicy

ROOT = Path(__file__).resolve().parent.parent
OBS_VERSION = "hx11-obs-v1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=1.0, help="temperatura de la dirección sugerida para la sala")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = SetPolicy(**ck["config"])
    model.load_state_dict(ck["state"])
    model.eval()
    sd = SELF_DIM + (EXTRA_DIM if model.extra else 0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    own, ents = torch.zeros(4, sd), torch.zeros(4, N_ENT, ENT_DIM)
    torch.onnx.export(model, (own, ents), str(out), input_names=["own", "ents"], output_names=["logits"],
                      dynamic_axes={"own": {0: "n"}, "ents": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    vocab = json.loads((ROOT / "reports" / "hx11" / "pos_vocab.json").read_text(encoding="utf-8"))
    meta = dict(obs_version=OBS_VERSION, self_dim=sd, n_ent=N_ENT, ent_dim=ENT_DIM, n_actions=18, frame_skip=3,
                delay_min=D_MIN, delay_max=D_MAX, temperature_move=a.temp, lines=list(LINES),
                sin_historia=bool(ck["config"].get("sin_historia")), extra=bool(ck["config"].get("extra")),
                pos_vocab=vocab, source=str(a.ckpt), step=ck.get("paso"),
                train_matches=len(ck.get("train") or []))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    o = rng.normal(size=(64, sd)).astype(np.float32)
    e = rng.normal(size=(64, N_ENT, ENT_DIM)).astype(np.float32)
    e[..., 0] = (rng.random((64, N_ENT)) < 0.9).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(o), torch.from_numpy(e)).numpy()
    got = sess.run(["logits"], {"own": o, "ents": e})[0]
    soft = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    err_p = float(np.abs(soft(ref) - soft(got)).max())
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB), diferencia máxima de probabilidades torch/onnx {err_p:.2e}")
    if err_p > 1e-5:
        raise SystemExit("FALLA: ONNX no coincide con torch")


if __name__ == "__main__":
    main()
