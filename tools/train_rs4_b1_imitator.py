"""Imitador recurrente del candidato C (PLAN_RS4.md 3): arquitectura exacta de B, demostraciones corregidas.

Datos: data/rs4_b1_demos (tools/rs4_b1_demos.py). Pérdida: entropía cruzada ponderada sobre la tecla
del mismo tick (desfase 0), sin la espera previa al saque inicial (fase 1: enseñaría a quedarse quieto),
con las patadas x`--kick-weight` (2 % de las teclas) y la acción previa ocultada al azar
(`--previous-dropout`) para que el modelo no se limite a repetir la tecla anterior.
Ventanas de 64 decisiones con 16 de precalentamiento de memoria (sin pérdida) salvo al inicio de un
tramo, donde la memoria cero es exacta. Las proyecciones pública y privilegiada quedan en cero (como
nacen en B) y la cabeza de valor no se entrena: la recalibra el calentamiento del crítico de PPO.

El acierto global no aprueba nada (PLAN_RS4.md 2.3): se informa por tipo de decisión y la compuerta
real es actuar en el simulador (tools/rs4_b1_c_gate.py).

  python -m tools.train_rs4_b1_imitator --updates 3000 --threads 12
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
PHASE_WEIGHT = np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)  # ver PHASES en tools/rs4_b1_demos.py
NO_ACTION = 18


def model_config(config_path=ROOT / "train" / "config_rs4_b1_b.yaml"):
    """La misma configuración que MultiTrainer construye para B (initialize_from carga estricto)."""
    from env.haxball_env import U_SELF_DIM
    from train.multitask import ENT_DIM
    from train.runtime import load_config
    mc = load_config(config_path)["model"]
    return dict(type="recurrent_set", self_dim=U_SELF_DIM, ent_dim=ENT_DIM, hidden=mc["hidden"], layers=mc["layers"],
                ent_hidden=mc.get("ent_hidden", 64), pooling=mc.get("pooling", "meanmax"),
                ent_layers=mc.get("ent_layers", 2), rule_observation=mc.get("rule_observation", "masked"),
                public_signals_version=mc["public_signals_version"], critic_features=int(mc["critic_features"]),
                memory_size=mc.get("memory_size", 64))


def load_group(paths):
    """Concatena grabaciones y arma índices de ventanas (inicio, jugador, ¿inicio de tramo?)."""
    parts = [dict(np.load(p)) for p in paths]
    data = {key: np.concatenate([p[key] for p in parts]) for key in ("obs", "act", "seg", "phase", "ball_dist")}
    offset, segs = 0, []
    for p in parts:
        segs.append(p["seg"].astype(np.int64) + offset)
        offset = int(segs[-1].max()) + 1 if len(segs[-1]) else offset
    data["seg"] = np.concatenate(segs)
    # acción previa dentro del mismo tramo (NO_ACTION al comenzar)
    previous = np.full_like(data["act"], NO_ACTION)
    same = data["seg"][1:] == data["seg"][:-1]
    previous[1:][same] = data["act"][:-1][same]
    data["previous"] = previous
    return data


def windows(data, length, burn_in):
    starts = []
    seg = data["seg"]
    edges = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for begin, end in zip(edges[:-1], edges[1:]):
        if end - begin < 8:
            continue
        for s in range(begin, end - 1, length - burn_in):
            starts.append((s, min(s + length, end), s == begin))
    return starts


def batch_tensors(data, chosen, length, burn_in, rng, dropout):
    B = len(chosen) * 8
    obs = np.zeros((length, B, data["obs"].shape[-1]), dtype=np.float32)
    act = np.zeros((length, B), dtype=np.int64)
    prev = np.full((length, B), NO_ACTION, dtype=np.int64)
    start = np.zeros((length, B), dtype=bool)
    weight = np.zeros((length, B), dtype=np.float32)
    meta = {key: np.zeros((length, B), dtype=np.float32) for key in ("phase", "ball_dist")}
    for i, (s, e, first) in enumerate(chosen):
        n, cols = e - s, slice(8 * i, 8 * i + 8)
        obs[:n, cols] = data["obs"][s:e]
        act[:n, cols] = data["act"][s:e]
        prev[:n, cols] = data["previous"][s:e]
        start[0, cols] = first
        phase = data["phase"][s:e]
        w = PHASE_WEIGHT[phase]
        if not first:
            w[:burn_in] = 0.0
        weight[:n, cols] = w
        meta["phase"][:n, cols] = phase
        meta["ball_dist"][:n, cols] = data["ball_dist"][s:e]
    if dropout > 0:
        prev[rng.random(prev.shape) < dropout] = NO_ACTION
    prev[start] = NO_ACTION
    return obs, act, prev, start, weight, meta


def forward(model, obs, prev, start):
    memory = model.initial_state(obs.shape[1])
    logits, _, _ = model.sequence(torch.from_numpy(obs), memory, torch.from_numpy(prev), torch.from_numpy(start))
    return logits


@torch.no_grad()
def evaluate(model, data, chosen, length, burn_in):
    """Métricas por tipo de decisión (greedy) sobre ventanas fijas de desarrollo."""
    rng = np.random.default_rng(0)
    totals = {}
    for i in range(0, len(chosen), 64):
        obs, act, prev_true, start, weight, meta = batch_tensors(data, chosen[i:i + 64], length, burn_in, rng, 0.0)
        logits = forward(model, obs, prev_true, start)
        logp = logits.log_softmax(-1)
        nll = -logp.gather(-1, torch.from_numpy(act)[..., None]).squeeze(-1).numpy()
        guess = logits.argmax(-1).numpy()
        top3 = (logits.topk(3, dim=-1).indices.numpy() == act[..., None]).any(-1)
        used = weight > 0
        kinds = {"todas": used, "patadas": used & (act >= 9), "cambios_de_tecla": used & (act != prev_true) & ~start,
                 "recepcion": used & (meta["ball_dist"] < 30) & (act < 9),
                 "saque_propio": used & (meta["phase"] == 3), "ultimo_segundo_saque_inicial": used & (meta["phase"] == 2)}
        for name, mask in kinds.items():
            row = totals.setdefault(name, np.zeros(5))
            row += [mask.sum(), nll[mask].sum(), (guess == act)[mask].sum(), top3[mask].sum(),
                    ((guess >= 9) & (act >= 9))[mask].sum()]
    out = {}
    for name, (n, nll, hit, top3, kicks) in totals.items():
        out[name] = dict(n=int(n), nll=float(nll / max(n, 1)), accuracy=float(hit / max(n, 1)), top3=float(top3 / max(n, 1)))
        if name == "patadas":
            out[name]["kick_recall"] = float(kicks / max(n, 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default=str(ROOT / "data" / "rs4_b1_demos"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "rs4_b1" / "c_bc"))
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--windows", type=int, default=32, help="ventanas por lote (x8 jugadores)")
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--burn-in", type=int, default=16)
    ap.add_argument("--group", type=int, default=8, help="grabaciones en memoria a la vez")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kick-weight", type=float, default=2.0)
    ap.add_argument("--previous-dropout", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from train.model import build_model
    config = model_config()
    model = build_model(config)
    for name, parameter in model.named_parameters():
        if name.startswith(("public_proj.", "critic_proj.")):
            parameter.requires_grad_(False)  # nacen en cero en B; aquí no hay contrato público ni crítico
    train_files = sorted(glob.glob(str(Path(a.demos) / "entrenamiento" / "*.npz")))
    dev = load_group(sorted(glob.glob(str(Path(a.demos) / "desarrollo" / "*.npz"))))
    dev_windows = windows(dev, a.length, a.burn_in)
    dev_windows = [dev_windows[i] for i in np.sort(rng.choice(len(dev_windows), min(1500, len(dev_windows)), replace=False))]
    sample = load_group(list(rng.choice(train_files, min(6, len(train_files)), replace=False)))
    flat = sample["obs"].reshape(-1, sample["obs"].shape[-1])
    model.update_norm(torch.from_numpy(flat[rng.choice(len(flat), min(400_000, len(flat)), replace=False)].astype(np.float32)))
    del sample, flat
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.updates, eta_min=a.lr / 10)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best, update, started = [], None, 0, time.time()
    while update < a.updates:
        order = rng.permutation(len(train_files))
        for begin in range(0, len(order), a.group):  # una pasada usa cada ventana una vez
            data = load_group([train_files[i] for i in order[begin:begin + a.group]])
            pool = windows(data, a.length, a.burn_in)
            shuffled = rng.permutation(len(pool))
            for first in range(0, len(shuffled) - a.windows + 1, a.windows):
                chosen = [pool[i] for i in shuffled[first:first + a.windows]]
                obs, act, prev, start, weight, _ = batch_tensors(data, chosen, a.length, a.burn_in, rng, a.previous_dropout)
                weight = torch.from_numpy(weight * np.where(act >= 9, a.kick_weight, 1.0).astype(np.float32))
                logits = forward(model, obs, prev, start)
                loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.from_numpy(act).reshape(-1),
                                        reduction="none") * weight.reshape(-1)).sum() / weight.sum().clamp(min=1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                update += 1
                if update % 50 == 0:
                    print(f"actualización {update}/{a.updates} | pérdida {loss.item():.3f} | {time.time() - started:.0f} s", flush=True)
                if update % a.eval_every == 0 or update == a.updates:
                    model.eval()
                    metrics = evaluate(model, dev, dev_windows, a.length, a.burn_in)
                    model.train()
                    history.append(dict(update=update, seconds=round(time.time() - started), dev=metrics))
                    print("  desarrollo | " + " | ".join(f"{k} acierto {v['accuracy']:.3f} nll {v['nll']:.3f}"
                                                        for k, v in metrics.items()), flush=True)
                    if best is None or metrics["todas"]["nll"] < best["dev"]["todas"]["nll"]:
                        best = history[-1]
                        torch.save({"model": model.state_dict(), "model_config": model.config(), "steps": 0,
                                    "bc": dict(version="RS4-b1-c-1", label_lag=0, update=update, args=vars(a),
                                               dev=metrics)}, out / "bc.pt")
                if update >= a.updates:
                    break
            if update >= a.updates:
                break
    report = dict(version="RS4-b1-c-imitation-1", config=config, args=vars(a), best=best, history=history,
                  train_recordings=len(train_files), seconds=round(time.time() - started))
    (ROOT / "reports" / "rs4_b1" / "c_imitation.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8")
    print(f"mejor actualización {best['update']} | desarrollo nll {best['dev']['todas']['nll']:.3f} "
          f"acierto {best['dev']['todas']['accuracy']:.3f} | {out / 'bc.pt'}")


if __name__ == "__main__":
    main()
