"""Imitación humana X4 (E1): política de conjuntos sobre obs v3 con retardo de observación por muestra.

  python -m learn.x4_bc --out runs/x4_bc/d0 --delay 0 --steps 20000
  python -m learn.x4_bc --out runs/x4_bc/rand --delay 0:15 --steps 60000

Métricas en desarrollo (conjunto fijo): NLL, precisión top-1 de la acción conjunta, de la dirección y de la
patada, por contexto (juego abierto, saques, saque inicial). Con `--delay-scan` evalúa el mismo modelo con
varios retardos (el modelo recibe el retardo como feature).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learn import x4_data as XD
from learn.x4_policy import SetPolicy, param_count

ROOT = Path(__file__).resolve().parent.parent


def parse_delay(text):
    if ":" in text:
        lo, hi = text.split(":")
        return (int(lo), int(hi))
    return int(text)


@torch.no_grad()
def evaluate(model, data, t, p, delay, batch=8192):
    model.eval()
    nll, acc, acc_move, acc_kick, kick_tp, kick_fp, kick_fn = [], [], [], [], 0, 0, 0
    changed = data.label[t, p] != data.label[t - 3, p]     # el humano cambió de tecla respecto de hace 3 ticks
    ctx_all = data.ctx[t, p]
    per_ctx = {}
    for i in range(0, len(t), batch):
        tt, pp = t[i:i + batch], p[i:i + batch]
        d = np.full(len(tt), delay, np.int64) if np.isscalar(delay) else delay[i:i + batch]
        obs = torch.from_numpy(XD.featurize(data, tt, pp, d))
        y = torch.from_numpy(data.label[tt, pp].astype(np.int64))
        logits = model(obs)
        l = F.cross_entropy(logits, y, reduction="none")
        pred = logits.argmax(-1)
        nll.append(l.numpy())
        acc.append((pred == y).numpy())
        acc_move.append((pred % 9 == y % 9).numpy())
        pk, yk = pred >= 9, y >= 9
        acc_kick.append((pk == yk).numpy())
        kick_tp += int((pk & yk).sum())
        kick_fp += int((pk & ~yk).sum())
        kick_fn += int((~pk & yk).sum())
    nll, acc, acc_move, acc_kick = (np.concatenate(x) for x in (nll, acc, acc_move, acc_kick))
    for c, name in enumerate(XD.STATE_NAMES):
        m = ctx_all == c
        if m.any():
            per_ctx[name] = dict(n=int(m.sum()), nll=float(nll[m].mean()), acc=float(acc[m].mean()),
                                 acc_move=float(acc_move[m].mean()),
                                 acc_change=float(acc[m & changed].mean()) if (m & changed).any() else None)
    model.train()
    return dict(nll=float(nll.mean()), acc=float(acc.mean()), acc_move=float(acc_move.mean()),
                nll_change=float(nll[changed].mean()), acc_change=float(acc[changed].mean()),
                frac_change=float(changed.mean()),
                acc_kick=float(acc_kick.mean()), kick_precision=kick_tp / max(1, kick_tp + kick_fp),
                kick_recall=kick_tp / max(1, kick_tp + kick_fn), per_context=per_ctx)


def baseline(data, t, p, train_labels):
    """Líneas base: acción más frecuente y "repetir la tecla de hace 3 ticks"."""
    y = data.label[t, p].astype(np.int64)
    freq = np.bincount(train_labels, minlength=18) + 1.0
    freq /= freq.sum()
    prev = data.label[t - 3, p].astype(np.int64)
    return dict(majority_acc=float((y == np.argmax(freq)).mean()), prior_nll=float(-np.log(freq[y]).mean()),
                repeat_acc=float((y == prev).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--maps", default="sanguchito_rs_x4", help="mapas separados por coma")
    ap.add_argument("--families", default="", help="familias de sala (vacío = todas)")
    ap.add_argument("--delay", default="0", help="retardo de observación en ticks: n o lo:hi")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=100000)
    ap.add_argument("--train-limit", type=int, default=0, help="máximo de grabaciones de entrenamiento")
    ap.add_argument("--delay-scan", default="", help="retardos a evaluar al final, p. ej. 0,3,6,9,12,15")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--init", default="", help="checkpoint para continuar")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maps = tuple(a.maps.split(","))
    fams = tuple(f for f in a.families.split(",") if f) or None
    train_names = XD.split_names(a.splits, "train", fams, maps)
    if a.train_limit:
        train_names = train_names[:a.train_limit]
    dev_names = XD.split_names(a.splits, "dev", fams, maps)
    t0 = time.time()
    train = XD.load(train_names, maps=maps)
    dev = XD.load(dev_names, maps=maps)
    print(f"datos: train {len(train.names)} grabaciones, {len(train.valid)} ticks válidos; "
          f"dev {len(dev.names)} / {len(dev.valid)} ({time.time() - t0:.0f} s)", flush=True)
    delay = parse_delay(a.delay)
    sampler = XD.Sampler(train, delay=delay, seed=a.seed)
    dev_t, dev_p = XD.Sampler(dev).fixed(min(a.eval_n, 8 * len(dev.valid)))
    eval_delay = delay if isinstance(delay, int) else int(round((delay[0] + delay[1]) / 2))
    model = SetPolicy(hidden=a.hidden)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    base = baseline(dev, dev_t, dev_p, train.label[train.valid[::7]].ravel().astype(np.int64))
    print(f"modelo {param_count(model) / 1e3:.0f}k parámetros; líneas base dev {json.dumps(base)}", flush=True)
    log = dict(args=vars(a), baseline=base, train_recordings=train.names, dev_recordings=dev.names, evals=[])
    best = None
    t0 = time.time()
    seen = 0
    for step in range(1, a.steps + 1):
        obs, y, ctx, _ = sampler.batch(a.batch)
        logits = model(torch.from_numpy(obs))
        loss = F.cross_entropy(logits, torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += len(y)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, dev, dev_t, dev_p, eval_delay)
            ev.update(step=step, samples=seen, train_loss=float(loss), seconds=round(time.time() - t0),
                      samples_per_s=round(seen / (time.time() - t0)))
            log["evals"].append(ev)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()
                              if k != "per_context"}), flush=True)
            ckpt = dict(model=model.state_dict(), hidden=a.hidden, obs_version="x4-obs-v3", step=step,
                        delay=a.delay, maps=maps)
            torch.save(ckpt, out / "last.pt")
            if best is None or ev["nll"] < best:
                best = ev["nll"]
                torch.save(ckpt, out / "best.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.delay_scan:
        scan = {}
        for d in (int(x) for x in a.delay_scan.split(",")):
            scan[d] = evaluate(model, dev, dev_t, dev_p, d)
            print(f"retardo {d:2d}: nll {scan[d]['nll']:.4f} acc {scan[d]['acc']:.4f}", flush=True)
        log["delay_scan"] = scan
        (out / "log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
