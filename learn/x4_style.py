"""Recompensa de estilo humano para el RL X4 (AMP: Peng et al. 2021, arXiv 2104.02180; GAIL: Ho y Ermon 2016,
arXiv 1606.03476).

Un discriminador mira el último segundo de juego visto desde un equipo: la pelota, sus 4 jugadores y los 4 rivales
(posición y velocidad) hace 1 s, 0,5 s, 0,25 s y ahora, en el marco del equipo (ataca hacia +x) y con los jugadores
ordenados por distancia a la pelota. Aprende a separar los tramos de las grabaciones humanas de Sanguchito (+1) de
los del agente (−1) por mínimos cuadrados, con penalización de gradiente sobre los humanos (AMP). Cada decisión cuyo
último segundo fue entero de juego abierto le da al equipo r = max(0, 1 − ¼(D − 1)²), centrada con su media móvil: los
saques y el saque inicial valen 0, así que el estilo no empuja ni a buscar ni a evitar pelotas muertas. No es de suma
cero: cada equipo cobra por su propio parecido.

Por qué (diagnóstico del 2026-10-07, `docs/PRELANZAMIENTO.md` §5): el ancla KL copia las decisiones de la imitación una
por una, con sus defectos, y se apaga con λ → 0,05; las métricas humanas sólo controlan. Ninguna premia que los
partidos del propio agente se parezcan a los humanos. En duelos 1v1 el agente pasa el 13% de las veces y pierde la
pelota el 61% (humanos: 22% y 49%; la imitación, 15% y 53%): un tramo "me encaran → la pierdo" no se ve humano.

  python -m learn.x4_style --bank --out data/x4_style_bank.npz            # ventanas humanas (una vez)
  python -m learn.x4_style --probe --agent runs/x4_cola/rl_checkpoint/evals/ep_01950.npz,...   # ¿las separa?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from env.rs4z import kernel as K

ROOT = Path(__file__).resolve().parent.parent
OFFSETS = (20, 10, 5, 0)              # muestreos hacia atrás (cada uno = 3 ticks = 1/20 s): 1 s, 0,5 s, 0,25 s y ahora
DEPTH = max(OFFSETS) + 1
BODIES = 9                            # pelota + 8 lugares
FEAT = len(OFFSETS) * BODIES * 4
SCALE = np.array([1150.0, 670.0, 6.0, 6.0], np.float32)     # x, y, vx, vy
# columnas de y y vy (para el espejo vertical con que se aumentan los datos del discriminador)
Y_COLS = np.zeros(FEAT, bool)
Y_COLS[1::4] = True
Y_COLS[3::4] = True


# ------------------------------------------------------------------------------------- ventanas
def features(win, team):
    """win: (M, K, 9, 4) estados del mundo (pelota y los 8 lugares; x, y, vx, vy) en los momentos de OFFSETS, del más
    viejo al actual; team: (M,) 0 rojo, 1 azul. Devuelve (M, FEAT) en el marco del equipo: pelota, sus 4 jugadores y
    los 4 rivales, ordenados por distancia a la pelota en el momento actual (el mismo orden en los K momentos)."""
    team = np.asarray(team, np.int64)
    w = np.array(win, np.float32, copy=True)
    M, Kn = w.shape[:2]
    s = np.where(team == 1, -1.0, 1.0).astype(np.float32)[:, None, None]
    w[..., 0] *= s
    w[..., 2] *= s
    now = w[:, -1]
    rows = np.arange(M)[:, None]

    def by_distance(idx):
        d = np.hypot(now[rows, idx, 0] - now[:, None, 0, 0], now[rows, idx, 1] - now[:, None, 0, 1])
        return np.take_along_axis(idx, np.argsort(d, 1, kind="stable"), 1)

    red, blue = np.arange(1, 5)[None].repeat(M, 0), np.arange(5, 9)[None].repeat(M, 0)
    mates = np.where(team[:, None] == 0, red, blue)
    rivals = np.where(team[:, None] == 0, blue, red)
    sel = np.concatenate([np.zeros((M, 1), np.int64), by_distance(mates), by_distance(rivals)], 1)
    out = w[np.arange(M)[:, None, None], np.arange(Kn)[None, :, None], sel[:, None, :]] / SCALE
    return out.reshape(M, -1)


def open_play(env):
    """Juego abierto en cada partido de un `RS4ZEnv`: sin saque inicial, sin saque y sin cobro pendiente."""
    return (env.ri[:, K.RI_KO] == 0) & (env.ri[:, K.RI_TEAM] < 0) & (env.ri[:, K.RI_PEND] == 0)


class History:
    """Últimos DEPTH estados de cada partido de un `RS4ZEnv` (uno por decisión) y si eran juego abierto."""

    def __init__(self, n):
        self.buf = np.zeros((n, DEPTH, BODIES, 4), np.float32)
        self.open = np.zeros((n, DEPTH), bool)
        self.i = 0

    def push(self, env):
        self.i = (self.i + 1) % DEPTH
        b = self.buf[:, self.i]
        b[:, 0, :2] = env.pos[:, 0]
        b[:, 0, 2:] = env.vel[:, 0]
        b[:, 1:, :2] = env.player_pos
        b[:, 1:, 2:] = env.player_vel
        self.open[:, self.i] = open_play(env)

    def reset(self, rows):
        """Partidos que empiezan de nuevo: su historia deja de valer."""
        self.open[np.atleast_1d(rows)] = False

    def windows(self):
        """(N, K, 9, 4) ventanas que terminan en el último estado y (N,) si el último segundo entero fue juego abierto."""
        idx = [(self.i - o) % DEPTH for o in OFFSETS]
        return self.buf[:, idx], self.open.all(1)


def episode_windows(ep, every=1):
    """(M, FEAT) ventanas de los dos equipos de un `tools.x4_metrics.Episode` (grabación humana o partido grabado del
    agente; muestreo cada 3 ticks), en los momentos con el último segundo entero de juego abierto."""
    T = len(ep.ball)
    if T <= max(OFFSETS):
        return np.zeros((0, FEAT), np.float32)
    st = np.zeros((T, BODIES, 4), np.float32)
    st[:, 0] = ep.ball[:, :4]
    st[:, 1:, :2] = ep.pos
    st[:, 1:, 2:] = ep.vel
    bad = np.concatenate([[0], np.cumsum(~np.asarray(ep.open_play, bool))])
    t = np.arange(max(OFFSETS), T)
    t = t[bad[t + 1] - bad[t - max(OFFSETS)] == 0][::every]
    if not len(t):
        return np.zeros((0, FEAT), np.float32)
    win = st[t[:, None] - np.asarray(OFFSETS)[None, :]]
    return np.concatenate([features(win, np.zeros(len(t), np.int64)), features(win, np.ones(len(t), np.int64))])


# ------------------------------------------------------------------------------------- discriminador
class Discriminator(nn.Module):
    def __init__(self, mean, std, hidden=256):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def style_reward(d):
    """AMP: r = max(0, 1 − ¼(D − 1)²) ∈ [0, 1]; 1 cuando D da el valor humano (+1)."""
    return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)


class Style:
    """Discriminador, su entrenamiento en línea y la recompensa centrada. `bank`: archivo de `--bank` (ventanas humanas
    del split de entrenamiento y una muestra del de desarrollo)."""

    def __init__(self, bank, device, lr=1e-4, gp=5.0, steps=8, batch=4096, keep=65536, seed=0):
        d = np.load(bank)
        self.human = torch.from_numpy(np.ascontiguousarray(d["x"]))            # float16 en CPU
        self.human_dev = torch.from_numpy(np.ascontiguousarray(d["x_dev"])) if "x_dev" in d.files else None
        self.dev = torch.device(device)
        self.D = Discriminator(d["mean"], d["std"]).to(self.dev)
        self.opt = torch.optim.Adam(self.D.parameters(), lr=lr)
        self.gp, self.steps, self.batch, self.keep = gp, steps, batch, keep
        self.gen = torch.Generator().manual_seed(seed)
        self.r_mean, self.r_w = 0.0, 0.0                                         # media móvil de r (EMA con corrección)
        self.agent = []
        self.flip = torch.from_numpy(np.where(Y_COLS, -1.0, 1.0).astype(np.float32)).to(self.dev)

    @property
    def baseline(self):
        return self.r_mean / self.r_w if self.r_w > 1e-6 else 0.0

    @torch.no_grad()
    def reward(self, feats):
        """r ∈ [0, 1] de cada ventana (sin centrar)."""
        if not len(feats):
            return np.zeros(0, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).to(self.dev)
        return style_reward(self.D(x)).float().cpu().numpy()

    def add(self, feats, r):
        """Ventanas del agente de esta actualización (para entrenar el discriminador) y su r (para la media móvil)."""
        if len(feats):
            self.agent.append(np.asarray(feats, np.float16))
            b = 0.99
            self.r_mean = b * self.r_mean + (1 - b) * float(np.mean(r))
            self.r_w = b * self.r_w + (1 - b)

    def _augment(self, x):
        flip = torch.rand(len(x), 1, generator=self.gen).to(self.dev) < 0.5
        return torch.where(flip, x * self.flip, x)

    def fit(self):
        """Pasos del discriminador con las ventanas del agente juntadas desde el último `fit`. Devuelve estadísticas."""
        if not self.agent:
            return {}
        agent = np.concatenate(self.agent)
        self.agent = []
        if len(agent) > self.keep:
            agent = agent[torch.randperm(len(agent), generator=self.gen)[:self.keep].numpy()]
        agent = torch.from_numpy(agent)
        sums = np.zeros(4)
        for _ in range(self.steps):
            ia = torch.randint(len(agent), (self.batch,), generator=self.gen)
            ih = torch.randint(len(self.human), (self.batch,), generator=self.gen)
            xa = self._augment(agent[ia].to(self.dev).float())
            xh = self._augment(self.human[ih].to(self.dev).float()).requires_grad_(True)
            dh, da = self.D(xh), self.D(xa)
            loss_ls = ((dh - 1) ** 2).mean() + ((da + 1) ** 2).mean()
            g = torch.autograd.grad(dh.sum(), xh, create_graph=True)[0]
            gp = (g ** 2).sum(-1).mean()
            loss = loss_ls + self.gp * gp
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            sums += [float(loss_ls.detach()), float(gp.detach()), float(dh.detach().mean()), float(da.detach().mean())]
        s = sums / self.steps
        return dict(style_d_loss=round(s[0], 4), style_gp=round(s[1], 4), style_d_human=round(s[2], 3),
                    style_d_agent=round(s[3], 3), style_baseline=round(self.baseline, 4))

    @torch.no_grad()
    def score(self, feats, n=20000):
        """r medio de un conjunto de ventanas (muestra de hasta n)."""
        if feats is None or not len(feats):
            return None
        if len(feats) > n:
            feats = feats[torch.randperm(len(feats), generator=self.gen)[:n].numpy()]
        x = torch.as_tensor(np.asarray(feats, np.float32)).to(self.dev)
        return round(float(style_reward(self.D(x)).mean()), 4)

    def state_dict(self):
        return dict(D=self.D.state_dict(), opt=self.opt.state_dict(), r_mean=float(self.r_mean), r_w=float(self.r_w))

    def load_state_dict(self, sd):
        self.D.load_state_dict(sd["D"])
        self.opt.load_state_dict(sd["opt"])
        self.r_mean, self.r_w = float(sd["r_mean"]), float(sd["r_w"])


# ------------------------------------------------------------------------------------- banco humano
def human_windows(names, ticks_dir, every, map_name="sanguchito_rs_x4"):
    from tools import x4_metrics as XM
    from tools.x4_ticks import MAP_IDS
    out = []
    for name in names:
        path = Path(ticks_dir) / f"{Path(name).stem}.npz"
        if not path.exists():
            continue
        for ep in XM.episodes_from_ticks(path, map_id=MAP_IDS[map_name]):
            w = episode_windows(ep, every)
            if len(w):
                out.append(w.astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, FEAT), np.float16)


def build_bank(out, splits, ticks_dir, every=2, dev_max=50000, seed=0):
    from learn import x4_data as XD
    train = XD.split_names(splits, "train", None, ("sanguchito_rs_x4",))
    dev = XD.split_names(splits, "dev", None, ("sanguchito_rs_x4",))
    x = human_windows(train, ticks_dir, every)
    x_dev = human_windows(dev, ticks_dir, every)
    rng = np.random.default_rng(seed)
    if len(x_dev) > dev_max:
        x_dev = x_dev[rng.choice(len(x_dev), dev_max, replace=False)]
    xf = x.astype(np.float32)
    mean, std = xf.mean(0), np.maximum(xf.std(0), 1e-3)
    np.savez(out, x=x, x_dev=x_dev, mean=mean.astype(np.float32), std=std.astype(np.float32),
             meta=json.dumps(dict(offsets=OFFSETS, every=every, train=len(train), dev=len(dev))))
    return dict(ventanas=len(x), ventanas_dev=len(x_dev), grabaciones_train=len(train), grabaciones_dev=len(dev))


# ------------------------------------------------------------------------------------- prueba offline
def probe(bank, agent_files, device="cpu", steps=3000, batch=2048, seed=0):
    """Entrena un discriminador sólo con ventanas humanas y del agente (partidos guardados por las evaluaciones) y mide
    en datos que no vio: AUC humanos de desarrollo contra partidos del agente apartados, y r medio de cada lado."""
    from tools import x4_metrics as XM
    files = list(agent_files)
    held = files[len(files) // 2:] if len(files) > 1 else files
    fit_files = files[:len(files) // 2] if len(files) > 1 else files
    load = lambda fs: np.concatenate([episode_windows(e) for f in fs for e in XM.episodes_from_npz(f)]).astype(np.float16)
    st = Style(bank, device, steps=steps, batch=batch, keep=10 ** 9, seed=seed)
    xa, xa_held = load(fit_files), load(held)
    st.agent = [xa]
    stats = st.fit()
    with torch.no_grad():
        dh = st.D(torch.as_tensor(st.human_dev.float()).to(st.dev)).cpu().numpy()
        da = st.D(torch.as_tensor(xa_held.astype(np.float32)).to(st.dev)).cpu().numpy()
    lab = np.r_[np.ones(len(dh)), np.zeros(len(da))]
    sc = np.r_[dh, da]
    order = np.argsort(sc)
    ranks = np.empty(len(sc))
    ranks[order] = np.arange(1, len(sc) + 1)
    auc = (ranks[lab == 1].sum() - len(dh) * (len(dh) + 1) / 2) / (len(dh) * len(da))
    rh = float(np.clip(1 - 0.25 * (dh - 1) ** 2, 0, None).mean())
    ra = float(np.clip(1 - 0.25 * (da - 1) ** 2, 0, None).mean())
    return dict(auc=round(float(auc), 4), r_humanos_dev=round(rh, 4), r_agente_apartado=round(ra, 4),
                ventanas_agente=len(xa), ventanas_agente_apartadas=len(xa_held), **stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", action="store_true", help="armar el banco de ventanas humanas")
    ap.add_argument("--probe", action="store_true", help="¿el discriminador separa al agente de los humanos?")
    ap.add_argument("--out", default=str(ROOT / "data" / "x4_style_bank.npz"))
    ap.add_argument("--bank-path", default=str(ROOT / "data" / "x4_style_bank.npz"))
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--ticks-dir", default=str(ROOT / "data" / "x4_ticks"))
    ap.add_argument("--every", type=int, default=2, help="una ventana cada N muestreos (las vecinas son casi iguales)")
    ap.add_argument("--agent", default="", help="partidos guardados del agente (evals/ep_*.npz), separados por coma")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.bank:
        print(json.dumps(build_bank(a.out, a.splits, a.ticks_dir, a.every)), flush=True)
    if a.probe:
        print(json.dumps(probe(a.bank_path, [f for f in a.agent.split(",") if f], a.device, a.steps)), flush=True)


if __name__ == "__main__":
    main()
