"""Imitación de HaxEleven (L0): entrenamiento por máxima verosimilitud de las teclas humanas.

Prototipo en CPU (L0-mini, revisión de la auditoría §6): un subconjunto de partidos de entrenamiento y de desarrollo,
para probar la tubería y alimentar el banco de lazo cerrado. El lazo abierto (estas métricas) es **diagnóstico**, no
puerta (P3):
* NLL total y NLL en los ticks donde la tecla cambia (el copycat se ve ahí: repetir la tecla anterior acierta casi
  siempre en el total);
* NLL de patear (P(patada) = suma de las 9 acciones con patada) y acierto;
* por línea del puesto.

Aumento por espejo (y → −y), que además intercambia izquierda y derecha en el ancla.

Acciones en bloque (`--chunk K`, ACT de Zhao et al. 2023): la red predice también las acciones de las K − 1
decisiones siguientes (cada 3 ticks) y la pérdida promedia las K cabezas. `--init` arranca de otro modelo (la cabeza de
la acción actual se copia en las K) para un ajuste corto en vez de entrenar de cero.

  python -m learn.hx11_bc --train 60 --dev 15 --steps 15000 --out runs/hx11_l0/mini
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from learn.hx11_data import LINES, Sampler, l0_split
from learn.hx11_policy import HIST_HI, HIST_LO, SetPolicy

Y_FLIP = np.array([0, 5, 4, 3, 2, 1, 8, 7, 6])
OWN_Y = [1, 3, 5, 7, 9, 13, 16, 19, 22, 25, 28, 50, 56, 58, 65]
ENT_Y = [2, 4, 6]


def mirror(own, ents, y):
    own, ents = own.copy(), ents.copy()
    own[:, OWN_Y] *= -1
    ents[..., ENT_Y] *= -1
    return own, ents, flip_y(y)


def flip_y(y):
    """Acción espejada; respeta las etiquetas ignoradas (−100) de las acciones en bloque."""
    return np.where(y >= 0, Y_FLIP[np.maximum(y, 0) % 9] + 9 * (np.maximum(y, 0) // 9), y)


def evaluate(net, data):
    own, ents, y, ch, line = data[:5]
    net.eval()
    with torch.no_grad():
        lp = torch.log_softmax(net(torch.from_numpy(own), torch.from_numpy(ents)), -1).numpy()
    nll = -lp[np.arange(len(y)), y]
    pk = np.exp(lp[:, 9:]).sum(1).clip(1e-6, 1 - 1e-6)
    kick = y >= 9
    out = dict(nll=float(nll.mean()), nll_cambio=float(nll[ch].mean()), acierto=float((lp.argmax(1) == y).mean()),
               acierto_cambio=float((lp.argmax(1)[ch] == y[ch]).mean()),
               nll_patada=float(-(kick * np.log(pk) + (~kick) * np.log(1 - pk)).mean()),
               p_patada_media=float(pk.mean()), tasa_patada=float(kick.mean()))
    out["por_linea"] = {LINES[k]: round(float(nll[line == k].mean()), 4) for k in range(len(LINES)) if (line == k).any()}
    if getattr(net, "hold", False) and len(data) > 5:
        with torch.no_grad():
            _, hl = net.forward_hold(torch.from_numpy(own), torch.from_numpy(ents))
            hp = torch.log_softmax(hl, -1).numpy()
        out["nll_duracion"] = float(-hp[np.arange(len(y)), data[5]].mean())
    net.train()
    return out


def batch_with_line(sampler, B):
    bt = sampler.batch(B)
    own, ents, y, ch = bt[:4]
    return (own, ents, y, ch, own[:, 59:64].argmax(1)) + tuple(bt[4:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--dev", type=int, default=15)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sin-historia", action="store_true", help="brazo: sin las entradas propias anteriores")
    ap.add_argument("--extra", action="store_true", help="observación v2: rasgos de 'voy yo'")
    ap.add_argument("--cerca-frac", type=float, default=0.0, help="fracción del lote con el jugador cerca de la pelota")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="corta (evaluando y guardando) al pasar este tiempo")
    ap.add_argument("--hold", action="store_true", help="cabeza de duración: cuánto sostiene el humano la tecla")
    ap.add_argument("--hidden", type=int, default=256, help="ancho del tronco")
    ap.add_argument("--ent-hidden", type=int, default=128, help="ancho del codificador de entidades")
    ap.add_argument("--split-file", default=None, help="JSON con train/dev (para máquinas sin el manifiesto)")
    ap.add_argument("--drop-hist", type=float, default=0.0,
                    help="brazo: borra la historia propia en esta fracción de las muestras (en la inferencia va entera)")
    ap.add_argument("--chunk", type=int, default=0, help="acciones en bloque: cuántas decisiones predice (ACT)")
    ap.add_argument("--init", default=None, help="arranca con los pesos de este modelo (ajuste corto)")
    ap.add_argument("--extra-data", default=None,
                    help="npz con own/ents/y (p. ej. learn.hx11_corner humanos); varios separados por coma")
    ap.add_argument("--extra-frac", default="0", help="fracción de cada lote que sale de cada --extra-data (por coma)")
    ap.add_argument("--out", default="runs/hx11_l0/mini")
    ap.add_argument("--reanudar", action="store_true",
                    help="si hay out/ultimo.pt con el optimizador, sigue desde ese paso (para PCs que se apagan)")
    ap.add_argument("--guardar-min", type=float, default=5.0, help="checkpoint completo al menos cada tantos minutos")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    modos = json.loads(Path("data/hx11/modos.json").read_text(encoding="utf-8"))
    sp = json.loads(Path(a.split_file).read_text(encoding="utf-8")) if a.split_file else l0_split()
    rng = np.random.default_rng(a.seed)
    train = sorted(rng.permutation(sp["train"])[:a.train].tolist())
    dev = sorted(rng.permutation(sp["dev"])[:a.dev].tolist())
    t0 = time.time()
    tr = Sampler(train, stride=a.stride, seed=a.seed, modos=modos, extra=a.extra, cerca_frac=a.cerca_frac, hold=a.hold,
                 chunk=a.chunk)
    dv = Sampler(dev, stride=a.stride, seed=a.seed + 1, modos=modos, extra=a.extra, hold=a.hold)
    dev_data = batch_with_line(dv, 40000)
    print(f"datos: {len(train)} partidos de entrenamiento, {len(dev)} de desarrollo ({time.time() - t0:.0f} s)",
          flush=True)
    net = SetPolicy(hidden=a.hidden, ent_hidden=a.ent_hidden, sin_historia=a.sin_historia, extra=a.extra, hold=a.hold,
                    chunk=a.chunk)
    if a.init:
        src = torch.load(a.init, map_location="cpu", weights_only=False)["state"]
        if net.chunk > 1 and src["logits.weight"].shape[0] == net.n_actions:
            # la cabeza de la acción actual del modelo de partida, copiada en las K del bloque
            src = dict(src, **{"logits.weight": src["logits.weight"].repeat(net.chunk, 1),
                               "logits.bias": src["logits.bias"].repeat(net.chunk)})
        net.load_state_dict(src)
        print(f"pesos iniciales de {a.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    extras = []
    if a.extra_data:
        fracs = [float(x) for x in str(a.extra_frac).split(",")]
        for path, fr in zip(a.extra_data.split(","), fracs):
            if fr <= 0:
                continue
            assert net.chunk == 1 and not a.hold, "--extra-data solo con la cabeza de una acción"
            z = np.load(path)
            extras.append((z["own"], z["ents"], z["y"].astype(np.int64), fr))
            print(f"datos extra: {len(extras[-1][2])} muestras de {path}, {fr:.0%} de cada lote", flush=True)
    ci = 4 + int(a.hold)             # índice de las acciones en bloque en el lote
    hist = []
    best = 1e9
    first = 1
    ck_path = out / "ultimo.pt"
    if a.reanudar and ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if "opt" in ck:
            net.load_state_dict(ck["state"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            hist, best, first = ck.get("hist", []), ck.get("best", 1e9), ck["paso"] + 1
            rng = np.random.default_rng(a.seed + first)
            tr.rng = np.random.default_rng(a.seed + 7919 * first)
            print(f"reanudado desde el paso {ck['paso']}", flush=True)
    last_save = time.time()

    def save(step, final=False):
        """Checkpoint completo y atómico (si la PC se apaga a mitad, queda el anterior entero)."""
        tmp = out / "ultimo.tmp"
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), config=net.config,
                        train=train, dev=dev, paso=step, best=best, hist=hist, args=vars(a), terminado=final), tmp)
        os.replace(tmp, ck_path)

    for step in range(first, a.steps + 1):
        bt = tr.batch(a.batch)
        own, ents, y = bt[0], bt[1], bt[2]
        if extras:
            B0 = len(y)
            parts_o, parts_e, parts_y, used = [], [], [], 0
            for eo, ee, ey, fr in extras:
                ne = int(round(B0 * fr))
                j = rng.integers(0, len(ey), ne)
                parts_o.append(eo[j]); parts_e.append(ee[j]); parts_y.append(ey[j])
                used += ne
            own = np.concatenate([own[used:]] + parts_o)
            ents = np.concatenate([ents[used:]] + parts_e)
            y = np.concatenate([y[used:]] + parts_y)
        yc = bt[ci] if net.chunk > 1 else None
        if rng.random() < 0.5:
            own, ents, y = mirror(own, ents, y)
            if yc is not None:
                yc = flip_y(yc)
        if a.drop_hist > 0:
            own = own.copy()
            own[rng.random(len(own)) < a.drop_hist, HIST_LO:HIST_HI] = 0.0
        if a.hold:
            lg, hl = net.forward_hold(torch.from_numpy(own), torch.from_numpy(ents))
            loss = lossf(lg, torch.from_numpy(y)) + lossf(hl, torch.from_numpy(bt[4]))
        elif yc is not None:
            lg = net.forward_chunk(torch.from_numpy(own), torch.from_numpy(ents))
            loss = lossf(lg.reshape(-1, net.n_actions), torch.from_numpy(yc.reshape(-1)))
        else:
            loss = lossf(net(torch.from_numpy(own), torch.from_numpy(ents)), torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        late = a.max_minutes > 0 and time.time() - t0 > a.max_minutes * 60
        if step % a.eval_every == 0 or step == a.steps or late:
            ev = evaluate(net, dev_data)
            ev.update(paso=step, perdida_train=float(loss.item()), segundos=round(time.time() - t0))
            hist.append(ev)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}), flush=True)
            if ev["nll"] < best:
                best = ev["nll"]
                tmp = out / "model.tmp"
                torch.save(dict(state=net.state_dict(), config=net.config, train=train, dev=dev, paso=step,
                                args=vars(a)), tmp)
                os.replace(tmp, out / "model.pt")
            save(step, final=step == a.steps or late)
            last_save = time.time()
            (out / "historia.tmp").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            os.replace(out / "historia.tmp", out / "historia.json")
        elif time.time() - last_save > a.guardar_min * 60:
            save(step)
            last_save = time.time()
        if late:
            break


if __name__ == "__main__":
    main()
