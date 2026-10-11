"""Ajuste en lazo cerrado con discriminador (GAIL, Ho y Ermon 2016) para la población de L1: el remedio que prevé el
plan si la puerta de realismo P5 no pasa (docs/PLAN_HAXELEVEN.md). Aprobado por el usuario el 2026-10-10.

Con e1 y las reglas de patada, la población queda en 18% de métricas dentro del techo humano y el C2ST la separa con
AUC 0,99. La delatan el temblor (4,9 cambios de tecla por segundo contra 3,5), los jugadores quietos, el doble de
tiempo de pelota parada, pocos pases y el equipo estirado (`reports/hx11/realismo_e1.md`). Los ajustes baratos no
alcanzan, y premios hechos a mano solo cambiaban una cosa por otra (piloto A3).

* **Juego:** 22 bots con la misma política (arranca de `--init`, e1), en partidos del simulador desde estados reales
  del banco de entrenamiento, con las reglas de la población (córner, tiro libre y saque de arco, lateral, juego
  abierto), temperatura 0,7 y ruido persistente, como en la puerta de realismo.
* **Discriminador** D(observación, acción): ve lo mismo que la política más las teclas propias anteriores (la política
  no las ve: `sin_historia`; sin ellas no podría notar el temblor) y la acción ejecutada. Separa decisiones humanas del
  banco (`hx11_data.Sampler`, mismas observaciones con retardo) de las de los bots. Premio de cada decisión de cada
  jugador: el logit de D (log D − log(1 − D)), recortado.
* **Política:** PPO sobre las decisiones de los 22, con penalización KL(π‖π_ref) exacta hacia e1 congelada
  (coeficiente adaptativo hacia `--kl-objetivo`), para no perder lo que ya hace bien (P3, coordinación). Las primeras
  `--calentar` iteraciones solo se entrenan D y el crítico.
* **Registro por iteración:** precisión de D, premio, KL y, de los partidos de esa iteración, cambios de tecla por
  segundo, velocidad, jugadores quietos, juego abierto y pases (las cosas que delata el C2ST).

Guarda `model.pt` (formato de L0: sirve para `learn.hx11_realismo sinteticos`, la puerta P3 y el cerebro),
`ultimo.pt` (todo, para `--reanudar`) e `historia.json`.

  python -m learn.hx11_gail --init runs/hx11_l0/l0_quienva_e1/model.pt --out runs/hx11_l0/gail_p1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from env.hx11 import kernel as K
from learn.hx11_closed_loop import FRAME_SKIP, bits_of_action
from learn.hx11_data import MAX_BACK, ROT_MOVE, Sampler, build, load_match
from learn.hx11_policy import SetPolicy, near_rows, open_kick_blocked, piece_rows, sample
from learn.hx11_rl import Games, gae, game_starts, load_net, make_critic, tempered_logp
from learn.hx11_selfplay import kick_blocked, store_all
from tools.hx11_d0 import OPEN

ROOT = Path(__file__).resolve().parent.parent


class Disc(nn.Module):
    """D(observación con historia propia, acción) → logit de "humano"."""

    def __init__(self, hidden=256, ent_hidden=128, extra=True):
        super().__init__()
        self.body = SetPolicy(hidden=hidden, ent_hidden=ent_hidden, n_actions=1, sin_historia=False, extra=extra)
        self.act = nn.Embedding(18, 64)
        self.out = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, own, ents, a):
        return self.out(torch.cat([self.body.trunk(own, ents), self.act(a)], -1))[:, 0]


def rollout_all(g: Games, pol, critic, T, temp, every=FRAME_SKIP):
    """T decisiones de los 22 de cada partido. Devuelve el lote (T, N) con N = partidos × 22 y estadísticas."""
    n = g.n
    N = n * 22
    buf = {k: [] for k in ("own", "ents", "act", "exe", "logp", "val", "done", "on")}
    k_start = g.k
    for _ in range(T):
        kk = MAX_BACK + g.k
        own = np.zeros((N, pol.self_net[0].in_features), np.float32)
        ents = None
        on = np.zeros(N, bool)
        rows = []
        for i, w in enumerate(g.W):
            slots = np.nonzero(w.pid[kk] >= 0)[0]
            o, e, _, _ = build(w, np.full(len(slots), kk), slots, g.D[i][slots], pol.extra)
            if ents is None:
                ents = np.zeros((N,) + e.shape[1:], np.float32)
            j = i * 22 + slots
            own[j], ents[j], on[j] = o, e, True
            rows += [(i, int(s)) for s in slots]
        idx = np.array([i * 22 + s for i, s in rows])
        keys = [(g.episode, i, s) for i, s in rows]
        fresh = near_rows([g.env.pos[i, g.env.fp + s] for i, s in rows], [g.env.pos[i, 0] for i, s in rows])
        ot, et = torch.from_numpy(own[idx]), torch.from_numpy(ents[idx])
        with torch.no_grad():
            lg = pol(ot, et)
            v = critic(ot, et)[:, 0].numpy()
        act = sample(lg, temp, gumbel=g.noise.get(keys, fresh),
                     kick_u=g.noise.get_kick(keys, piece_rows([g.env.ri[i, K.RI_MODE] for i, s in rows])))
        lp = tempered_logp(lg, temp).gather(1, torch.as_tensor(act).unsqueeze(1))[:, 0].numpy()
        A = np.zeros(N, np.int64)
        E = np.zeros(N, np.int64)
        LP = np.zeros(N, np.float32)
        V = np.zeros(N, np.float32)
        for j, ((i, s), a) in enumerate(zip(rows, act)):
            a = int(a)
            A[i * 22 + s] = a
            L = kk - int(g.D[i][s]) - 1
            if a >= 9 and (kick_blocked(g.W[i], L, s, int(g.D[i][s]), True, True)
                           or open_kick_blocked(g.W[i], L, s, int(g.D[i][s]))):
                a %= 9
            E[i * 22 + s] = a                                        # acción ejecutada (coordenadas del equipo)
            LP[i * 22 + s], V[i * 22 + s] = lp[j], v[j]
            if s >= 11:
                a = int(ROT_MOVE[a % 9] + 9 * (a // 9))
            g.acts[i, s] = a
            g.W[i].inp[kk, s] = bits_of_action(np.array([a]))[0]
        for f in range(every):
            kk = MAX_BACK + g.k
            if f > 0:
                for w in g.W:
                    w.inp[kk] = w.inp[kk - 1]
            g.env.step(g.acts, frames=1)
            for i, w in enumerate(g.W):
                store_all(g.env, i, w, kk + 1)
                w.score[kk + 1] = g.env.ri[i, [K.RI_SCORE0, K.RI_SCORE1]]
            g.k += 1
        done = g.k + every > g.H
        for k_, v_ in (("own", own), ("ents", ents), ("act", A), ("exe", E), ("logp", LP), ("val", V),
                       ("done", np.full(N, done)), ("on", on)):
            buf[k_].append(v_)
        if done:
            st = behaviour(g, k_start)
            g.reset()
            k_start = 0
    st = behaviour(g, k_start) if g.k > k_start else {}
    with torch.no_grad():
        kk = MAX_BACK + g.k
        last_v = np.zeros(N, np.float32)
        for i, w in enumerate(g.W):
            slots = np.nonzero(w.pid[kk] >= 0)[0]
            o, e, _, _ = build(w, np.full(len(slots), kk), slots, g.D[i][slots], pol.extra)
            last_v[i * 22 + slots] = critic(torch.from_numpy(o), torch.from_numpy(e))[:, 0].numpy()
    out = {k: np.stack(v) for k, v in buf.items()}
    out["last_v"] = last_v
    return out, st


def behaviour(g: Games, k0):
    """Lo que delata el C2ST, en los ticks [k0, k) de los partidos actuales."""
    a, b = MAX_BACK + k0, MAX_BACK + g.k
    if b - a < 30:
        return {}
    sp, idle, chg, opn, near = [], [], [], [], []
    for w in g.W:
        pos = w.pos[a:b + 1].astype(np.float32) / 10
        on = w.pid[a:b + 1] >= 0
        v = np.hypot(*np.diff(pos, axis=0).transpose(2, 0, 1))
        ok = on[1:] & on[:-1] & (v < 10)
        sp.append(v[ok].mean())
        idle.append((v[ok] < 0.3).mean())
        inp = w.inp[a:b + 1].astype(np.int64)
        chg.append(((inp[1:] != inp[:-1]) & ok).sum() / max(ok.sum(), 1) * 60)
        ph = w.phase[a:b + 1]
        opn.append((ph == OPEN).mean())
        ball = w.ball[a:b + 1].astype(np.float32) / 10
        d = np.where(on, np.hypot(*(pos - ball[:, None]).transpose(2, 0, 1)), np.inf).min(1)
        near.append(np.median(d[ph == OPEN]) if (ph == OPEN).any() else np.nan)
    return dict(vel=float(np.mean(sp)), quietos=float(np.mean(idle)), cambios_tecla_s=float(np.mean(chg)),
                juego_abierto=float(np.mean(opn)), mas_cercano=float(np.nanmean(near)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="política de partida y referencia de la KL (e1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file", default="runs/hx11_l0/split.json")
    ap.add_argument("--partidos", type=int, default=60)
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--segundos", type=int, default=60)
    ap.add_argument("--T", type=int, default=64, help="decisiones por partido y por iteración")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--calentar", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-critico", type=float, default=3e-4)
    ap.add_argument("--lr-disc", type=float, default=1e-4)
    ap.add_argument("--disc-pasos", type=int, default=20, help="pasos de D por iteración")
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lote", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--premio-max", type=float, default=5.0)
    ap.add_argument("--kl-objetivo", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--guardar", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reanudar", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sp = json.loads((ROOT / a.split_file).read_text(encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    names = sorted(rng.permutation(sp["train"])[:a.partidos].tolist())
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    pol = load_net(a.init)
    ref = load_net(a.init).eval()
    hum = Sampler(names, stride=3, seed=a.seed, modos=modos, extra=pol.extra)
    H = a.segundos * 60
    starts = game_starts(hum.matches, H, rng)
    print(f"datos: {len(names)} partidos, {len(starts)} comienzos ({time.time() - t0:.0f} s)", flush=True)
    critic = make_critic(pol)
    disc = Disc(extra=pol.extra)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=a.lr_critico)
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr_disc, weight_decay=1e-4)
    beta, it0, hist = a.beta, 0, []
    ult = out / "ultimo.pt"
    if a.reanudar and ult.exists():
        ck = torch.load(ult, map_location="cpu", weights_only=False)
        pol.load_state_dict(ck["state"])
        critic.load_state_dict(ck["critico"])
        disc.load_state_dict(ck["disc"])
        opt.load_state_dict(ck["opt"])
        opt_c.load_state_dict(ck["opt_c"])
        opt_d.load_state_dict(ck["opt_d"])
        beta, it0, hist = ck["beta"], ck["paso"], ck["hist"]
        print(f"reanudado en la iteración {it0}", flush=True)
    games = Games(a.n_env, hum.matches, starts, H, a.seed + 1000 * it0)
    for it in range(it0, a.iters):
        ti = time.time()
        pol.eval()
        critic.eval()
        b, st = rollout_all(games, pol, critic, a.T, a.temp)
        tr = time.time() - ti
        on = b["on"].reshape(-1)
        own = torch.from_numpy(b["own"].reshape(-1, b["own"].shape[-1])[on])
        ents = torch.from_numpy(b["ents"].reshape(-1, *b["ents"].shape[2:])[on])
        exe = torch.from_numpy(b["exe"].reshape(-1)[on])
        # 1) discriminador: humanos del banco contra las decisiones de esta iteración
        disc.train()
        accs = []
        for _ in range(a.disc_pasos):
            ho, he, hy, _ = hum.batch(512)
            okh = hy >= 0                                    # etiquetas válidas (−100: sin tecla en la grabación)
            ho, he, hy = ho[okh], he[okh], hy[okh]
            j = torch.randint(0, len(own), (512,))
            lh = disc(torch.from_numpy(ho), torch.from_numpy(he), torch.from_numpy(hy.astype(np.int64)))
            lb = disc(own[j], ents[j], exe[j])
            loss_d = (nn.functional.binary_cross_entropy_with_logits(lh, torch.full_like(lh, 0.9))
                      + nn.functional.binary_cross_entropy_with_logits(lb, torch.zeros_like(lb)))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            accs.append(float(((lh > 0).float().mean() + (lb < 0).float().mean()) / 2))
        # 2) premio de cada decisión con el D actualizado
        disc.eval()
        with torch.no_grad():
            r_on = torch.cat([disc(own[j:j + 4096], ents[j:j + 4096], exe[j:j + 4096])
                              for j in range(0, len(own), 4096)]).clamp(-a.premio_max, a.premio_max).numpy()
        rew = np.zeros(on.shape, np.float32)
        rew[on] = r_on
        rew = rew.reshape(b["on"].shape)
        val = np.where(b["on"], b["val"], 0.0)
        adv, ret = gae(rew, val, b["done"] | ~b["on"], b["last_v"], a.gamma, a.lam)
        adv_t = torch.from_numpy(adv.reshape(-1)[on])
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(ret.reshape(-1)[on])
        act = torch.from_numpy(b["act"].reshape(-1)[on])
        logp_old = torch.from_numpy(b["logp"].reshape(-1)[on])
        Nn = len(own)
        with torch.no_grad():
            lp_ref = torch.cat([tempered_logp(ref(own[j:j + 4096], ents[j:j + 4096]), a.temp)
                                for j in range(0, Nn, 4096)])
        # 3) PPO con KL a la referencia
        pol.train()
        critic.train()
        train_pol = it >= a.calentar
        kls, clips, vls = [], [], []
        for _ in range(a.epocas):
            perm = torch.randperm(Nn)
            for j in range(0, Nn, a.lote):
                mb = perm[j:j + a.lote]
                vl = ((critic(own[mb], ents[mb])[:, 0] - ret_t[mb]) ** 2).mean()
                opt_c.zero_grad()
                vl.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()
                vls.append(float(vl.detach()))
                if not train_pol:
                    continue
                lp = tempered_logp(pol(own[mb], ents[mb]), a.temp)
                ratio = torch.exp(lp.gather(1, act[mb].unsqueeze(1))[:, 0] - logp_old[mb])
                pg = -torch.min(ratio * adv_t[mb], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[mb]).mean()
                kl = (lp.exp() * (lp - lp_ref[mb])).sum(-1).mean()
                opt.zero_grad()
                (pg + beta * kl).backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()
                kls.append(float(kl.detach()))
                clips.append(float(((ratio.detach() - 1).abs() > a.clip).float().mean()))
        kl_m = float(np.mean(kls)) if kls else 0.0
        if train_pol:
            beta = float(np.clip(beta * (1.5 if kl_m > 1.5 * a.kl_objetivo else 1 / 1.5 if kl_m < a.kl_objetivo / 1.5
                                         else 1.0), 1e-3, 100.0))
        row = dict(iter=it + 1, decisiones=int(Nn), disc_acierto=round(float(np.mean(accs)), 3),
                   premio=round(float(r_on.mean()), 3), kl_ref=round(kl_m, 4), beta=round(beta, 4),
                   clip=round(float(np.mean(clips)), 3) if clips else None, v_loss=round(float(np.mean(vls)), 4),
                   **{k: round(v, 3) for k, v in st.items()}, seg_rollout=round(tr), seg=round(time.time() - ti),
                   total=round(time.time() - t0))
        hist.append(row)
        print(json.dumps(row), flush=True)
        last = it + 1 == a.iters
        if (it + 1) % a.guardar == 0 or last:
            model = dict(state=pol.state_dict(), config=pol.config, paso=it + 1, args=vars(a))
            torch.save(model, out / "model.pt")
            torch.save(dict(model, critico=critic.state_dict(), disc=disc.state_dict(), opt=opt.state_dict(),
                            opt_c=opt_c.state_dict(), opt_d=opt_d.state_dict(), beta=beta, hist=hist,
                            terminado=last), ult)
            (out / "historia.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            if (it + 1) % (a.guardar * 4) == 0:
                torch.save(model, out / f"model_{it + 1}.pt")


if __name__ == "__main__":
    main()
