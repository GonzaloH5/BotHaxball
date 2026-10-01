"""Preentrenamiento por imitación (behavior cloning) con partidos reales de liga.

python -m train.bc [--run bc] [--epochs 4] [--config train/config_multi.yaml]

Entrena el MISMO modelo universal que usa el RL (SetActorCritic, arquitectura de `model` en el config) a predecir
la tecla que apretó cada jugador real, a partir de data/bc/**/*.npz (tools/build_bc_dataset.py). Se separan
replays enteros para validación (no ticks sueltos: si no, la validación ve el mismo partido).

Resultado: runs/<run>/bc.pt, que sirve para
  * arrancar el RL desde un modelo que ya se mueve como humano:  train.multitask --init-from runs/<run>/bc.pt
  * regularizar el RL para que no se aleje del estilo humano (ppo.bc_kl_coef en config_multi.yaml)
  * evaluarlo / exportarlo como cualquier checkpoint (eval.matrix, eval.render, export.to_onnx)
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from env.haxball_env import U_ENT_DIM, U_SELF_DIM

from .model import SetActorCritic
from .runtime import load_config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bc"


def load_dataset(val_frac: float, label: str = "act", folders=None, stadium=None, team_size=None):
    """Carga todos los shards, rellena entidades hasta el máximo y separa validación por replay."""
    shards = sorted(DATA.glob("*/*.npz"))
    if not shards:
        raise SystemExit(f"no hay datos en {DATA}: correr primero `python -m tools.build_bc_dataset`")
    parts = []
    e_max = 0
    for f in shards:
        if folders is not None and f.parent.name not in folders:
            continue
        z = np.load(f)
        if stadium is not None and ("stadium" not in z.files or str(z["stadium"]) != stadium):
            z.close()
            continue
        if stadium == "rs_one" and "powershot" in z.files and bool(z["powershot"]):
            print(f"salteo {f.name}: shard RS4 antiguo con powershot; reconstruirlo antes de usar")
            z.close()
            continue
        # un replay puede tener tramos de distinto formato (p. ej. 2v2 y 3v3): cada tramo tiene su ancho de obs
        obs_parts = [z[k] for k in sorted(z.files, key=lambda k: int(k[4:]) if k.startswith("obs_") else -1)
                     if k.startswith("obs_")]
        # validación: ~val_frac de los replays, elegidos por hash del nombre (estable entre corridas)
        h = int(hashlib.md5(f.name.encode()).hexdigest(), 16) % 1000
        if label not in z.files:
            raise SystemExit(f"{f} no tiene '{label}': reconstruir con `python -m tools.build_bc_dataset --overwrite`")
        y = z[label]
        prev = z["act_lagm3"] if "act_lagm3" in z.files else z["act"]
        if team_size is not None:
            if "T" not in z.files or len(z["T"]) != len(y):
                raise SystemExit(f"{f}: falta metadata T válida para filtrar el formato")
            selected = z["T"] == team_size
            start = 0
            kept = []
            for o in obs_parts:
                kept.append(o[selected[start:start + len(o)]])
                start += len(o)
            if start != len(y):
                raise SystemExit(f"{f}: observaciones y acciones desalineadas")
            obs_parts = [o for o in kept if len(o)]
            y, prev = y[selected], prev[selected]
            if not len(y):
                z.close()
                continue
        for o in obs_parts:
            e_max = max(e_max, (o.shape[1] - U_SELF_DIM) // U_ENT_DIM)
        parts.append({"obs": obs_parts, "act": y, "change": y != prev, "w": float(z["weight"]),
                      "stadium": str(z["stadium"]), "folder": f.parent.name, "val": h < val_frac * 1000})
        z.close()
    D = U_SELF_DIM + U_ENT_DIM * e_max
    out = {}
    for split in (False, True):
        ps = [p for p in parts if p["val"] == split]
        n = sum(len(p["act"]) for p in ps)
        obs = np.zeros((n, D), dtype=np.float16)
        act = np.zeros(n, dtype=np.int64)
        change = np.zeros(n, dtype=bool)
        w = np.zeros(n, dtype=np.float32)
        stadium = np.empty(n, dtype=object)
        k = 0
        for p in ps:
            m = len(p["act"])
            j = k
            for o in p["obs"]:  # entidades de relleno quedan en 0 (presente = 0)
                obs[j:j + len(o), : o.shape[1]] = o
                j += len(o)
            assert j == k + m, (j, k, m)
            act[k:k + m] = p["act"]
            change[k:k + m] = p["change"]
            w[k:k + m] = p["w"]
            stadium[k:k + m] = p["stadium"]
            k += m
        ok = np.isfinite(obs).all(axis=1)
        if not ok.all():  # un solo NaN arruina todo el entrenamiento (loss nan): se descartan esas filas
            print(f"aviso: {int((~ok).sum())} muestras con valores no finitos descartadas", flush=True)
            obs, act, w, stadium, change = obs[ok], act[ok], w[ok], stadium[ok], change[ok]
        out["val" if split else "train"] = {"obs": obs, "act": act, "w": w, "stadium": stadium,
                                            "change": change, "replays": len(ps)}
    return out, e_max


@torch.no_grad()
def evaluate(model, d, batch=16384):
    model.eval()
    n = len(d["act"])
    correct = top3 = move_ok = kick_ok = 0
    ch_ok = ch_n = 0
    per = defaultdict(lambda: [0, 0])
    loss = 0.0
    for s in range(0, n, batch):
        x = torch.from_numpy(d["obs"][s:s + batch].astype(np.float32))
        y = torch.from_numpy(d["act"][s:s + batch])
        lg = model.logits(x)
        loss += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item()
        pred = lg.argmax(-1)
        ok = (pred == y).numpy()
        correct += ok.sum()
        ch = d["change"][s:s + batch]
        ch_ok += int(ok[ch].sum())
        ch_n += int(ch.sum())
        top3 += (lg.topk(3, -1).indices == y[:, None]).any(-1).sum().item()
        move_ok += ((pred % 9) == (y % 9)).sum().item()
        kick_ok += ((pred >= 9) == (y >= 9)).sum().item()
        for st, o in zip(d["stadium"][s:s + batch], ok):
            per[st][0] += int(o)
            per[st][1] += 1
    model.train()
    return {"loss": loss / n, "acc": correct / n, "top3": top3 / n, "move_acc": move_ok / n, "kick_acc": kick_ok / n,
            "change_acc": ch_ok / max(ch_n, 1),  # acierto cuando el jugador CAMBIA de tecla: ahí copiar no sirve
            "per_stadium": {k: v[0] / v[1] for k, v in sorted(per.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bc")
    ap.add_argument("--config", default=str(ROOT / "train" / "config_multi.yaml"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--folders", default=None, help="carpetas separadas por coma; RS4: rsx4")
    ap.add_argument("--data-dir", default=None, help="dataset separado; RS4: data/bc_rs4_v2")
    ap.add_argument("--stadium", default=None, help="clave exacta de estadio; RS4: rs_one")
    ap.add_argument("--team-size", type=int, default=None, help="filtrar filas NvN, no sólo el nombre del replay")
    ap.add_argument("--init-from", default=None, help="afinar un imitador existente; no carga ni modifica PPO")
    ap.add_argument("--label", default="act_lag6",
                    help="tecla a imitar: act (la del instante), act_lag6 / act_lag12 (100/200 ms después: tiempo de "
                         "reacción, evita que el modelo copie su propio movimiento)")
    ap.add_argument("--keyframe-weight", type=float, default=4.0,
                    help="peso extra de las muestras donde el jugador cambia de tecla (las decisiones)")
    a = ap.parse_args()
    global DATA
    if a.data_dir:
        DATA = Path(a.data_dir).resolve()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cfg = load_config(a.config)
    mc = cfg["model"]
    if a.team_size is not None and a.team_size < 1:
        ap.error("--team-size debe ser >=1")
    if a.init_from and (ROOT / "runs" / a.run / "bc.pt").exists():
        ap.error("Para fine-tuning elegir un run nuevo; no se sobrescribe un imitador existente")

    t0 = time.time()
    data, e_max = load_dataset(a.val_frac, a.label,
                              a.folders.split(",") if a.folders else None, a.stadium, a.team_size)
    tr, va = data["train"], data["val"]
    if not len(tr["act"]) or not len(va["act"]):
        raise SystemExit("El filtro requiere muestras de entrenamiento y validación en replays distintos")
    print(f"datos: {len(tr['act']):,} muestras de entrenamiento ({tr['replays']} replays), {len(va['act']):,} de "
          f"validación ({va['replays']} replays), hasta {e_max} entidades | {time.time() - t0:.0f}s", flush=True)
    base = np.bincount(va["act"], minlength=18).max() / len(va["act"])
    print(f"referencia: predecir siempre la tecla más común acierta {base:.1%}", flush=True)

    model = SetActorCritic(U_SELF_DIM, U_ENT_DIM, hidden=mc["hidden"], layers=mc["layers"],
                           ent_hidden=mc.get("ent_hidden", 64), pooling=mc.get("pooling", "meanmax"),
                           ent_layers=mc.get("ent_layers", 2))
    if a.init_from:
        source = torch.load(a.init_from, map_location="cpu", weights_only=False)
        if "bc" not in source:
            raise SystemExit("--init-from requiere un checkpoint BC; no afinar el PPO por imitación aquí")
        model.load_state_dict(source["model"], strict=True)
        model.rule_observation = source["model_config"].get("rule_observation", "full")
    else:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(tr["act"]), size=min(400_000, len(tr["act"])), replace=False)
        model.update_norm(torch.from_numpy(tr["obs"][np.sort(sample)].astype(np.float32)))
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = len(tr["act"])
    steps_total = a.epochs * (n // a.batch)
    step = 0
    run_dir = ROOT / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(a.epochs):
        perm = rng.permutation(n)
        t1 = time.time()
        run_loss = 0.0
        for s in range(0, n - a.batch + 1, a.batch):
            lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(steps_total, 1))))
            for g in opt.param_groups:
                g["lr"] = lr
            i = np.sort(perm[s:s + a.batch])
            x = torch.from_numpy(tr["obs"][i].astype(np.float32))
            y = torch.from_numpy(tr["act"][i])
            w = torch.from_numpy(tr["w"][i] * np.where(tr["change"][i], a.keyframe_weight, 1.0).astype(np.float32))
            ce = torch.nn.functional.cross_entropy(model.logits(x), y, reduction="none")
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            step += 1
        ev = evaluate(model, va)
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(n // a.batch, 1), **ev})
        print(f"época {ep + 1}/{a.epochs} ({time.time() - t1:.0f}s) | train {run_loss / max(n // a.batch, 1):.3f} | "
              f"val loss {ev['loss']:.3f} acierto {ev['acc']:.1%} top3 {ev['top3']:.1%} | "
              f"dirección {ev['move_acc']:.1%} patada {ev['kick_acc']:.1%} | en cambios de decisión {ev['change_acc']:.1%}",
              flush=True)
        print("      por mapa: " + " | ".join(f"{k} {v:.1%}" for k, v in ev["per_stadium"].items()), flush=True)

    tasks = []
    for st in cfg["stages"]:
        tasks += [t for t in st["tasks"] if t not in tasks]
    torch.save({"model": model.state_dict(), "model_config": model.config(), "steps": 0, "iteration": 0,
                "env": {"obs_layout": "universal", "tasks": tasks, "max_entities": e_max, "frame_skip": 3},
                "bc": {"history": history, "train_samples": n, "val_samples": len(va["act"]), "label": a.label,
                       "keyframe_weight": a.keyframe_weight, "folders": a.folders,
                       "stadium": a.stadium, "team_size": a.team_size, "init_from": a.init_from,
                       "data_dir": str(DATA)}},
               run_dir / "bc.pt")
    print(f"guardado en {run_dir / 'bc.pt'}")


if __name__ == "__main__":
    main()
