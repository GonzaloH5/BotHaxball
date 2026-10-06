"""Valor esperado de la posesión (EPV) aprendido de las grabaciones humanas de Sanguchito.

φ(s) = P(el rojo marca en los próximos `HORIZON_S` segundos | s) − P(el azul marca | s), estimado con un MLP
f(s) ≈ P(rojo marca) y antisimétrico por construcción: φ(s) = f(s) − f(espejo(s)), donde el espejo rota la cancha
180° e intercambia los equipos. Así φ(espejo(s)) = −φ(s) exactamente y la misma red da el valor para los dos lados.

Usos:
* medir cuánto valor genera cada pase, recepción y posesión (`tools.x4_pass_chain`; VAEP, Decroos et al. 2019,
  arXiv 1802.07127; EPV, Fernández, Bornn y Cervone 2021, arXiv 2011.09426);
* shaping basado en potencial en el RL (Ng, Harada y Russell 1999): F = γ·φ(s') − φ(s) para el rojo y −F para el
  azul. No cambia la política óptima, así que un φ imperfecto no se puede explotar a la larga; densifica la señal de
  ganar o perder la pelota, progresar y ubicarse como lo hacen los humanos que terminan marcando.

Entrada: pelota (x, y, vx, vy), los 4 jugadores de cada equipo ordenados por distancia a la pelota (posición,
velocidad, posición relativa a la pelota y distancia) y el saque en curso (tipo y equipo, saque inicial). Las mismas
features salen de un `Episode` (grabaciones y simulación) o del estado del entorno (`features_env`).

  python -m learn.x4_epv --out runs/x4_epv/epv.pt --report reports/x4/epv.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
HORIZON_S = 10.0
SX, SY = 1150.0, 670.0
N_FEAT = 4 + 8 * 7 + 4


def features(ball, pos, vel, rkind, rteam, kickoff):
    """Features desde el punto de vista del rojo (ataca +x). ball (N, 4), pos/vel (N, 8, 2), rkind/rteam/kickoff (N,)."""
    ball = np.asarray(ball, np.float32)
    pos = np.asarray(pos, np.float32)
    vel = np.asarray(vel, np.float32)
    N = len(ball)
    out = np.zeros((N, N_FEAT), np.float32)
    bx, by = ball[:, 0], ball[:, 1]
    out[:, 0] = bx / SX
    out[:, 1] = by / SY
    out[:, 2] = ball[:, 2] / 10.0
    out[:, 3] = ball[:, 3] / 10.0
    dx = pos[..., 0] - bx[:, None]
    dy = pos[..., 1] - by[:, None]
    d = np.hypot(dx, dy)
    col = 4
    for sl in (slice(0, 4), slice(4, 8)):
        order = np.argsort(d[:, sl], axis=1)
        take = lambda a: np.take_along_axis(a[:, sl], order, axis=1)
        block = np.stack([take(pos[..., 0]) / SX, take(pos[..., 1]) / SY, take(vel[..., 0]) / 3.0,
                          take(vel[..., 1]) / 3.0, take(dx) / SX, take(dy) / SY, take(d) / 1000.0], -1)
        out[:, col:col + 28] = block.reshape(N, 28)
        col += 28
    rk = np.asarray(rkind)
    sign = np.where(np.asarray(rteam) == 0, 1.0, np.where(np.asarray(rteam) == 1, -1.0, 0.0))
    for k in (1, 2, 3):
        out[:, col] = np.where(rk == k, sign, 0.0)
        col += 1
    out[:, col] = np.asarray(kickoff, np.float32)
    return out


def mirror_state(ball, pos, vel, rteam):
    """Cancha rotada 180° y equipos intercambiados (el azul pasa a atacar +x en los lugares 0..3)."""
    ball = np.asarray(ball, np.float32).copy()
    ball[:, :4] *= -1.0
    swap = np.r_[4:8, 0:4]
    pos = -np.asarray(pos, np.float32)[:, swap]
    vel = -np.asarray(vel, np.float32)[:, swap]
    rteam = np.where(np.asarray(rteam) >= 0, 1 - np.asarray(rteam), -1)
    return ball, pos, vel, rteam


class EPVNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):          # logit de P(rojo marca en el horizonte)
        return self.net(x).squeeze(-1)


class EPV:
    """φ(s) en [−1, 1] para lotes de estados (numpy) en el dispositivo de la red."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location="cpu")
        self.net = EPVNet(ck.get("hidden", 256))
        self.net.load_state_dict(ck["model"])
        self.net.to(device).eval()
        self.device = device
        self.horizon_s = float(ck.get("horizon_s", HORIZON_S))

    @torch.no_grad()
    def phi(self, ball, pos, vel, rkind, rteam, kickoff):
        f = features(ball, pos, vel, rkind, rteam, kickoff)
        mb, mp, mv, mt = mirror_state(ball, pos, vel, rteam)
        g = features(mb, mp, mv, rkind, mt, kickoff)
        x = torch.from_numpy(np.concatenate([f, g])).to(self.device)
        p = torch.sigmoid(self.net(x)).float().cpu().numpy()
        n = len(f)
        return p[:n] - p[n:]

    def phi_episode(self, ep):
        rt = ep.restart_team if ep.restart_team is not None else np.full(len(ep.ball), -1)
        return self.phi(ep.ball, ep.pos, ep.vel, ep.restart_kind, rt, ep.kickoff)

    def phi_env(self, env):
        """φ de cada partido de un `RS4ZEnv` (estado actual)."""
        from env.rs4z import kernel as K
        ball = np.concatenate([env.pos[:, 0], env.vel[:, 0]], -1)
        team = env.ri[:, K.RI_TEAM]
        rk = np.where(team >= 0, env.ri[:, K.RI_KIND], 0)
        return self.phi(ball, env.player_pos, env.player_vel, rk, np.where(rk > 0, team, -1), env.ri[:, K.RI_KO] != 0)


# ------------------------------------------------------------------------------------- datos
def episode_labels(ep, horizon_s=HORIZON_S):
    """(y_rojo, y_azul, válido) por muestreo: ¿marca cada equipo dentro del horizonte? Se descartan los muestreos
    cuyo horizonte pasa el final del tramo sin gol (censurados)."""
    T = len(ep.ball)
    H = int(round(horizon_s / (ep.stride / 60.0)))
    g = ep.goal_ev if ep.goal_ev is not None else np.zeros(T, np.int8)
    yr = np.zeros(T, np.float32)
    yb = np.zeros(T, np.float32)
    valid = np.zeros(T, bool)
    nxt = T          # índice del próximo gol (T si no hay)
    nxt_team = 0
    for t in range(T - 1, -1, -1):
        if g[t] != 0:
            nxt, nxt_team = t, int(g[t])
        if nxt < T and nxt - t <= H:
            valid[t] = True
            yr[t] = float(nxt_team > 0)
            yb[t] = float(nxt_team < 0)
        elif t + H < T:
            valid[t] = True           # el horizonte entero entra en el tramo y no hubo gol
    return yr, yb, valid


def build(split, map_name="sanguchito_rs_x4", every=2, limit=None):
    """Features y etiquetas (con el espejo) de las grabaciones de una partición."""
    from tools import x4_metrics as XM
    from tools.x4_ticks import MAP_IDS
    rows = json.loads((ROOT / "reports" / "x4" / "splits.json").read_text(encoding="utf-8"))["recordings"]
    X, Y, G = [], [], []
    names = [n for n, r in sorted(rows.items()) if r["split"] == split and map_name in (r.get("maps") or {})]
    for k, name in enumerate(names[:limit]):
        path = ROOT / "data" / "x4_ticks" / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        for ep in XM.episodes_from_ticks(str(path), map_id=MAP_IDS[map_name]):
            yr, yb, ok = episode_labels(ep)
            idx = np.flatnonzero(ok)[::every]
            if not len(idx):
                continue
            rt = ep.restart_team[idx]
            f = features(ep.ball[idx], ep.pos[idx], ep.vel[idx], ep.restart_kind[idx], rt, ep.kickoff[idx])
            mb, mp, mv, mt = mirror_state(ep.ball[idx], ep.pos[idx], ep.vel[idx], rt)
            fm = features(mb, mp, mv, ep.restart_kind[idx], mt, ep.kickoff[idx])
            X += [f, fm]
            Y += [yr[idx], yb[idx]]
            G += [np.full(len(idx), k), np.full(len(idx), k)]
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def evaluate(net, X, Y, device="cpu"):
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(net(torch.from_numpy(X[i:i + 65536]).to(device))).cpu()
                       for i in range(0, len(X), 65536)]).numpy()
    base = float(Y.mean())
    brier = float(np.mean((p - Y) ** 2))
    brier0 = base * (1 - base)
    eps = 1e-6
    ll = float(-np.mean(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)))
    ll0 = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    bins = np.quantile(p, np.linspace(0, 1, 11))
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p <= hi)
        if m.any():
            cal.append(dict(p_media=round(float(p[m].mean()), 4), frecuencia=round(float(Y[m].mean()), 4), n=int(m.sum())))
    return dict(n=int(len(Y)), tasa_base=round(base, 4), brier=round(brier, 5), brier_skill=round(1 - brier / brier0, 4),
                logloss=round(ll, 5), logloss_skill=round(1 - ll / ll0, 4), auc=round(auc(Y, p), 4), calibracion=cal)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "x4_epv" / "epv.pt"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "x4" / "epv.json"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--every", type=int, default=2, help="un muestreo cada N (N·3 ticks)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    Xtr, Ytr, Gtr = build("train", every=a.every)
    Xdv, Ydv, _ = build("dev", every=a.every)
    Xte, Yte, _ = build("test", every=a.every)
    print(f"datos: train {len(Xtr)}, dev {len(Xdv)}, test {len(Xte)} ({time.time() - t0:.0f} s)", flush=True)
    net = EPVNet(a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
    rng = np.random.default_rng(a.seed)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, hist = 1e9, None, []
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(a.epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            loss = lossf(net(Xt[b]), Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        dv = evaluate(net, Xdv, Ydv)
        hist.append(dict(epoch=ep + 1, train_logloss=round(tot / len(perm), 5), dev=dict(
            logloss=dv["logloss"], brier_skill=dv["brier_skill"], auc=dv["auc"])))
        print(json.dumps(hist[-1]), flush=True)
        if dv["logloss"] < best:
            best = dv["logloss"]
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    report = dict(version="x4-epv-1", horizon_s=HORIZON_S, every=a.every, epochs=hist,
                  dev=evaluate(net, Xdv, Ydv), test=evaluate(net, Xte, Yte))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=net.state_dict(), hidden=a.hidden, horizon_s=HORIZON_S, n_feat=N_FEAT, version="x4-epv-1"),
               a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(test={k: v for k, v in report["test"].items() if k != "calibracion"}), ensure_ascii=False))


if __name__ == "__main__":
    main()
