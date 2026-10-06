"""RL X4 (E2/E3 del plan): MAPPO con parámetros compartidos y ancla KL a la imitación humana.

Receta y fuentes (docs/PLAN.md §2, revisión §8):
* MAPPO (arXiv 2103.01955): una política para los 8 jugadores, valor normalizado, 5 épocas, clip 0,2,
  lote grande. Crítico asimétrico (AlphaStar; MAPPO "agent-specific global state"): observación del actor
  + reloj y marcador + la acción que cada uno de los 8 jugadores va a aplicar (información privilegiada que
  el actor no tiene por la latencia).
* Pérdida (1−λ)·L_PPO + λ·KL(BC‖π) (HR-PPO arXiv 2403.19648, λ=0,06; VPT arXiv 2206.11795; AlphaStar).
  Con `--lambda-dist` λ se sortea por partido entre valores (DiL-piKL arXiv 2210.05492). El actor no ve λ:
  con varios valores, la política única aprende el promedio (DiL-piKL usa un tipo por λ); el crítico sí lo ve.
* Recompensa: gol ±1 de suma cero + shaping de progreso con tope tipo CHECKPOINT de GRF (arXiv 1907.11180):
  10 franjas del campo rival, +0,1 la primera vez por punto que el equipo (último toque) lleva la pelota a
  cada franja, el resto al marcar; suma cero, escalado por `--shaping` y retirable (MARLadona arXiv
  2409.20326). El pase NO se premia (Liu 2022, arXiv 2105.12196).
* Precalentar el crítico con el actor congelado (`--critic-warmup`; plan E2, motivado por Wołczyk 2024).
* [inferencia] Penalización de suma cero `--forfeit-penalty` (0,1) al equipo que deja vencer un saque o el saque
  inicial (plazo de entrenamiento). Sin ella, en self-play quedarse quieto en el saque inicial vale exactamente 0
  (reloj congelado y el saque pasa al rival, que es la misma política): una corrida en CPU desde la BC dejó de
  sacar a las 50 actualizaciones. En la sala no hay plazo y quedarse quieto no es una opción.
* Arranques desde estados humanos grabados (`--human-starts`; Salimans & Chen arXiv 1812.03381, Backplay
  arXiv 1807.06919): juego abierto y saques del split de entrenamiento.
* Rivales (TiZero arXiv 2302.07515, OpenAI Five arXiv 1912.06680): `--pool-frac` de los partidos contra el
  pool congelado (BC incluida) con PFSP (1−x)², el resto self-play.
* Latencia por jugador y partido sorteada de la distribución de sala (`--delays`), mapas mezclados
  (`--maps`).

  python -m learn.x4_ppo --bc runs/x4_bc/best.pt --out runs/x4_ppo/a --envs 1024 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from env.rs4z import kernel as K
from env.rs4z import obs_v2, obs_v3
from env.rs4z.core import MIRROR_ACTION, RS4ZEnv
from learn.x4_policy import SetPolicy

ROOT = Path(__file__).resolve().parent.parent
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# orden de los 8 jugadores visto desde cada lugar: él, compañeros, rivales (por lugar)
ORDER = np.array([[p] + [q for q in range(8) if q != p and TEAM[q] == TEAM[p]] + [q for q in range(8) if TEAM[q] != TEAM[p]]
                  for p in range(8)])
CRITIC_IN = obs_v3.OBS_DIM + obs_v2.CRITIC_DIM + 8 * 18 + 1
N_REGIONS = 10


# ------------------------------------------------------------------------------------- crítico
class Critic(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CRITIC_IN, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RunningNorm:
    """Media y desvío móviles (EMA con corrección de sesgo) de los retornos (normalización del valor, MAPPO)."""

    def __init__(self, beta=0.99):
        self.beta, self.mean, self.sq, self.w = beta, 0.0, 0.0, 0.0

    def update(self, x):
        x = np.asarray(x, np.float64)
        b = self.beta
        self.mean = b * self.mean + (1 - b) * x.mean()
        self.sq = b * self.sq + (1 - b) * (x ** 2).mean()
        self.w = b * self.w + (1 - b)

    @property
    def mu(self):
        return self.mean / self.w if self.w > 1e-6 else 0.0

    @property
    def std(self):
        if self.w <= 1e-6:
            return 1.0
        return math.sqrt(max(self.sq / self.w - self.mu ** 2, 2.5e-3))


def upcoming_actions(env):
    """(N, 8, 8*18) acción que cada jugador aplicará en el próximo tick, en el orden y marco de cada lugar."""
    N = env.N
    lag = env.delay
    h = np.where(lag > 0, (lag + env.frame_skip - 1) // env.frame_skip, 0)
    h = np.minimum(h, env.H - 1)
    world = np.take_along_axis(env.act_hist, h[..., None], axis=2)[..., 0]     # (N, 8)
    world = np.where(env.active, world, -1)
    out = np.zeros((N, 8, 8, 18), np.float32)
    for p in range(8):
        acts = world[:, ORDER[p]]                                             # (N, 8)
        if TEAM[p] == 1:
            acts = np.where(acts >= 0, MIRROR_ACTION[np.maximum(acts, 0)], -1)
        n_idx, k_idx = np.nonzero(acts >= 0)
        out[n_idx, p, k_idx, acts[n_idx, k_idx]] = 1.0
    return out.reshape(N, 8, 8 * 18)


# ------------------------------------------------------------------------------------- estados humanos
class StateBank:
    """Estados de juego abierto y de inicio de saque de las grabaciones de entrenamiento."""

    def __init__(self, data, seed=0):
        from learn.x4_data import STATE_OPEN
        self.d = data
        self.rng = np.random.default_rng(seed)
        v = data.valid
        open_ = (data.state[v] == 1) & (data.rkind[v] == 0)
        self.open = v[open_]
        rs = (data.state[v] == 1) & (data.rkind[v] > 0) & (data.rage[v] == 0) & (data.rteam[v] >= 0)
        self.restart = v[rs]

    def apply(self, env, row, restart_frac=0.3):
        d = self.d
        use_restart = len(self.restart) and self.rng.random() < restart_frac
        t = int(self.rng.choice(self.restart if use_restart else self.open))
        held = ((d.inp[t] & 16) != 0) & ~d.kicking[t]
        env.radius[row, 0] = d.ball_r[t]
        env.rf[row, K.RF_KSTR] = d.kstr[t]
        env.place(row, ball_pos=d.ball[t, :2], ball_vel=d.ball[t, 2:4], player_pos=d.pos[t], player_vel=d.vel[t],
                  kick_held=held, last_touch=-1, mass_phase=0 if abs(d.mass[t] - 0.5) < 1e-6 else 1)
        if use_restart:
            env.start_restart(row, int(d.rkind[t]), int(d.rteam[t]), (float(d.ball[t, 0]), float(d.ball[t, 1])))
        return t


# ------------------------------------------------------------------------------------- arena
class Arena:
    """Un `RS4ZEnv` de un mapa con su bookkeeping: quién controla cada lugar, λ, shaping, latencias."""

    # [inferencia] La imitación no tiene datos humanos de 2K23 (one-hot nunca visto): el ancla KL y los rivales del
    # pool ven ese mapa como RS ONE, del que 2K23 es la versión 70 px más angosta (la geometría ya va en la obs).
    BC_MAP_ALIAS = {"haxarg_2k23": "rs_one"}

    def __init__(self, map_name, n, args, rng, bank=None, pool_size=0):
        self.map, self.N, self.a, self.rng, self.bank = map_name, n, args, rng, bank
        j = obs_v3.SELF_FEATURES.index("map_rs_one")
        self.map_cols = (j + obs_v3.MAP_NAMES.index(map_name), j + obs_v3.MAP_NAMES.index(self.BC_MAP_ALIAS[map_name])) \
            if map_name in self.BC_MAP_ALIAS else None
        self.env = RS4ZEnv(n, map=map_name, frame_skip=3, max_delay=15, seed=int(rng.integers(1 << 30)))
        self.learner = np.ones((n, 8), bool)      # lugares que controla el aprendiz
        self.opp = np.full(n, -1)                 # índice del rival del pool (-1 self-play)
        self.lam = np.zeros(n)
        self.regions = np.zeros((n, 2, N_REGIONS), bool)
        self.pool_size = pool_size
        self.ep_goals = np.zeros((n, 2))
        self.restart_all()

    def _delays(self, k):
        vals, w = self.a.delay_values, self.a.delay_weights
        return self.rng.choice(vals, size=(k, 8), p=w)

    def new_match(self, rows, pfsp_probs=None):
        rows = np.atleast_1d(rows)
        if len(rows) == 0:
            return
        env = self.env
        lo, hi = self.a.match_minutes
        ticks = (self.rng.uniform(lo, hi, len(rows)) * 3600).astype(np.int64)
        env.start_match(rows, delay=self._delays(len(rows)), match_ticks=ticks)
        for r in rows:
            if self.bank is not None and self.rng.random() < self.a.human_starts:
                self.bank.apply(env, r, self.a.human_restart_frac)
            if self.pool_size and self.rng.random() < self.a.pool_frac:
                p = pfsp_probs if pfsp_probs is not None else np.full(self.pool_size, 1.0 / self.pool_size)
                self.opp[r] = int(self.rng.choice(self.pool_size, p=p))
                side = int(self.rng.integers(2))
                self.learner[r] = TEAM == side
            else:
                self.opp[r] = -1
                self.learner[r] = True
            self.lam[r] = self.rng.choice(self.a.lambda_values)
            self.regions[r] = False
            self.ep_goals[r] = 0

    def restart_all(self):
        self.new_match(np.arange(self.N))

    def shaping(self, goal):
        """Recompensa de progreso por equipo (N, 2) tras un paso, suma cero (CHECKPOINT de GRF)."""
        env = self.env
        out = np.zeros((self.N, 2))
        if self.a.shaping <= 0:
            return out
        bx = env.ball_pos[:, 0]
        last = env.ri[:, K.RI_LAST]
        # sólo juego vivo: sin saque inicial, sin saque del script y sin cobro pendiente (la pelota colocada por el
        # script no es progreso de nadie)
        live = (env.ri[:, K.RI_KO] == 0) & (env.ri[:, K.RI_TEAM] < 0) & (env.ri[:, K.RI_PEND] == 0)
        for t, s in ((0, 1.0), (1, -1.0)):
            xa = bx * s
            k = np.clip((xa / C_GOAL_X * N_REGIONS).astype(np.int64), 0, N_REGIONS - 1)
            hit = live & (last == t) & (xa > 0)
            rows = np.flatnonzero(hit & ~self.regions[np.arange(self.N), t, k])
            # se cobran todas las franjas hasta k (la pelota pudo saltear franjas en una decisión)
            for r in rows:
                new = ~self.regions[r, t, :k[r] + 1]
                out[r, t] += 0.1 * new.sum()
                self.regions[r, t, :k[r] + 1] = True
        scorer = np.where(goal > 0, 0, np.where(goal < 0, 1, -1))
        for r in np.flatnonzero(scorer >= 0):
            t = scorer[r]
            out[r, t] += 0.1 * (~self.regions[r, t]).sum()
            self.regions[r] = False                       # nuevo punto tras el gol
        out *= self.a.shaping
        return out - out[:, ::-1]


C_GOAL_X = 1150.0


def forfeit_reward(forfeit, penalty):
    """(N, 2) suma cero: −penalty al equipo que perdió un saque por plazo (forfeit = equipo, −1 ninguno)."""
    out = np.zeros((len(forfeit), 2))
    rows = np.flatnonzero(forfeit >= 0)
    out[rows, forfeit[rows]] -= penalty
    out[rows, 1 - forfeit[rows]] += penalty
    return out


# ------------------------------------------------------------------------------------- trainer
def load_policy(path, device):
    ck = torch.load(path, map_location="cpu")
    m = SetPolicy(hidden=ck.get("hidden", 256))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc", required=True, help="checkpoint de la imitación (ancla, inicialización y pool)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--maps", default="sanguchito_rs_x4:0.7,rs_one:0.15,haxarg_2k23:0.15")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64, help="decisiones por partido y por actualización")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--lambda-dist", default="0.06", help="valores de λ (KL a la BC) sorteados por partido")
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--forfeit-penalty", type=float, default=0.1,
                    help="penalización (suma cero) por dejar vencer un saque o el saque inicial")
    ap.add_argument("--shaping-anneal", type=float, default=0.0, help="decisiones hasta llevar el shaping a 0 (0 = fijo)")
    ap.add_argument("--critic-warmup", type=int, default=20, help="actualizaciones sólo del crítico al empezar")
    ap.add_argument("--human-starts", type=float, default=0.4)
    ap.add_argument("--human-restart-frac", type=float, default=0.3)
    ap.add_argument("--pool-frac", type=float, default=0.2)
    ap.add_argument("--pool", default="", help="checkpoints extra del pool, separados por coma (la BC siempre está)")
    ap.add_argument("--snapshot-every", type=int, default=100, help="actualizaciones entre snapshots al pool")
    ap.add_argument("--delays", default="8:0.2,9:0.25,10:0.25,11:0.2,14:0.1",
                    help="retardo D en ticks:peso. En sala se midió un lag (frame aplicado − observado) de 9–12 ticks; "
                         "en la convención del kernel D = lag − 1 (decisión en S_t aplicada desde S_{t+D+1})")
    ap.add_argument("--match-minutes", default="1:3")
    ap.add_argument("--splits", default=str(ROOT / "reports" / "x4" / "splits.json"))
    ap.add_argument("--bank-recordings", type=int, default=200, help="grabaciones para estados humanos (0 = ninguna)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-matches", type=int, default=32, help="partidos por lado contra la BC en cada evaluación")
    ap.add_argument("--eval-minutes", type=float, default=2.0)
    ap.add_argument("--shaping-retire", type=float, default=0.75,
                    help="victorias contra la BC a partir de las cuales el shaping se retira para siempre (0 = nunca)")
    ap.add_argument("--human-gate", type=float, default=1.0,
                    help="W1 normalizada media máxima (métricas clave) para que un checkpoint cuente como mejor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    a.delay_values = np.array([int(x.split(":")[0]) for x in a.delays.split(",")])
    w = np.array([float(x.split(":")[1]) for x in a.delays.split(",")])
    a.delay_weights = w / w.sum()
    a.lambda_values = [float(x) for x in a.lambda_dist.split(",")]
    lo, hi = (float(x) for x in a.match_minutes.split(":"))
    a.match_minutes = (lo, hi)
    a.map_mix = [(m.split(":")[0], float(m.split(":")[1])) for m in a.maps.split(",")]
    return a


class Trainer:
    def __init__(self, a):
        self.a = a
        torch.set_num_threads(a.threads)
        torch.manual_seed(a.seed)
        self.rng = np.random.default_rng(a.seed)
        self.dev = torch.device(a.device)
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        ck = torch.load(a.bc, map_location="cpu")
        self.hidden = ck.get("hidden", 256)
        self.policy = SetPolicy(hidden=self.hidden).to(self.dev)
        self.policy.load_state_dict(ck["model"])
        self.bc = load_policy(a.bc, self.dev)
        for p in self.bc.parameters():
            p.requires_grad_(False)
        self.critic = Critic().to(self.dev)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=a.lr, eps=1e-5)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=a.critic_lr, eps=1e-5)
        self.vnorm = RunningNorm()
        self.pool = [("bc", self.bc)] + [(Path(p).stem, load_policy(p, self.dev)) for p in a.pool.split(",") if p]
        self.pool_wins = np.full(len(self.pool), 0.5)   # tasa de victoria del aprendiz contra cada miembro
        bank = None
        if a.bank_recordings and a.human_starts > 0:
            from learn import x4_data as XD
            names = XD.split_names(a.splits, "train", None, ("sanguchito_rs_x4",))[:a.bank_recordings]
            if names:
                bank = StateBank(XD.load(names, maps=("sanguchito_rs_x4",)), seed=a.seed)
        self.bank = bank
        total = sum(w for _, w in a.map_mix)
        self.arenas = []
        for m, w in a.map_mix:
            n = max(1, int(round(a.envs * w / total)))
            self.arenas.append(Arena(m, n, a, self.rng, bank if m == "sanguchito_rs_x4" else None, len(self.pool)))
        self.decisions = 0
        self.update = 0
        self.best_score = -1.0
        self.log = open(self.out / "log.jsonl", "a", encoding="utf-8")

    # -------------------------------------------------------------- inferencia
    def pfsp(self):
        x = self.pool_wins
        w = (1.0 - x) ** 2 + 1e-3
        return w / w.sum()

    @staticmethod
    def bc_view(arena, obs):
        """Observación que ven la BC y el pool: con el mapa reemplazado si no hay datos humanos de él."""
        if arena.map_cols is None:
            return obs
        o = obs.copy()
        real, alias = arena.map_cols
        o[..., real] = 0.0
        o[..., alias] = np.where(obs[..., real] > 0, 1.0, o[..., alias])
        return o

    @torch.no_grad()
    def act(self, arena, obs, crit):
        """Acciones (N, 8) y datos del aprendiz para un paso de una arena."""
        N = arena.N
        o = torch.from_numpy(obs.reshape(N * 8, -1)).to(self.dev)
        logits = self.policy(o)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        obs_bc = self.bc_view(arena, obs)
        o_bc = o if obs_bc is obs else torch.from_numpy(obs_bc.reshape(N * 8, -1)).to(self.dev)
        bc_logp = F.log_softmax(self.bc(o_bc), -1)
        v = self.critic(torch.from_numpy(crit.reshape(N * 8, -1)).to(self.dev))
        acts = a.view(N, 8).cpu().numpy()
        for k in np.unique(arena.opp[arena.opp >= 0]):
            rows = np.flatnonzero(arena.opp == k)
            sel = ~arena.learner[rows]
            ob = torch.from_numpy(obs_bc[rows][sel]).to(self.dev)
            oa = torch.distributions.Categorical(logits=self.pool[k][1](ob)).sample().cpu().numpy()
            sub = acts[rows]
            sub[sel] = oa
            acts[rows] = sub
        return acts, logp.view(N, 8).cpu().numpy(), bc_logp.view(N, 8, 18).cpu().numpy(), v.view(N, 8).cpu().numpy()

    def critic_input(self, arena, obs):
        env = arena.env
        c = obs_v2.critic(env)
        up = upcoming_actions(env)
        lam = np.broadcast_to(arena.lam[:, None, None], (arena.N, 8, 1)).astype(np.float32)
        return np.concatenate([obs, c, up, lam], -1).astype(np.float32)

    # -------------------------------------------------------------- rollout
    def rollout(self):
        a = self.a
        T = a.rollout
        buf = []
        stats = dict(goals=0, matches=0, shaping=0.0, kickoff_wait=[], decisions=0, forfeits=0)
        shaping_scale = 1.0
        if a.shaping_anneal > 0:
            shaping_scale = max(0.0, 1.0 - self.decisions / a.shaping_anneal)
        for arena in self.arenas:
            env = arena.env
            N = arena.N
            ob = np.zeros((T, N, 8, obs_v3.OBS_DIM), np.float32)
            cr = np.zeros((T, N, 8, CRITIC_IN), np.float32)
            ac = np.zeros((T, N, 8), np.int64)
            lp = np.zeros((T, N, 8), np.float32)
            bl = np.zeros((T, N, 8, 18), np.float32)
            vv = np.zeros((T, N, 8), np.float32)
            rw = np.zeros((T, N, 8), np.float32)
            dn = np.zeros((T, N), bool)
            lm = np.zeros((T, N, 8), bool)
            la = np.zeros((T, N), np.float32)
            pfsp = self.pfsp()
            for t in range(T):
                obs = obs_v3.observe(env)
                crit = self.critic_input(arena, obs)
                acts, logp, bc_logp, v = self.act(arena, obs, crit)
                ob[t], cr[t], ac[t], lp[t], bl[t], vv[t] = obs, crit, acts, logp, bc_logp, v
                lm[t] = arena.learner & env.active
                la[t] = arena.lam
                ev = env.step(acts)
                goal = ev["goal"]
                team_r = np.zeros((N, 2))
                team_r[:, 0] += np.sign(goal)
                team_r[:, 1] -= np.sign(goal)
                sh = arena.shaping(goal) * shaping_scale
                team_r += sh
                team_r += forfeit_reward(ev["forfeit"], a.forfeit_penalty)
                stats["forfeits"] += int((ev["forfeit"] >= 0).sum())
                rw[t] = team_r[:, TEAM]
                arena.ep_goals[:, 0] += goal > 0
                arena.ep_goals[:, 1] += goal < 0
                stats["goals"] += int((goal != 0).sum())
                stats["shaping"] += float(np.abs(sh).sum())
                done = ev["match_end"]
                dn[t] = done
                ends = np.flatnonzero(done)
                for r in ends:
                    k = arena.opp[r]
                    if k >= 0:
                        side = int(TEAM[np.flatnonzero(arena.learner[r])[0]])
                        diff = arena.ep_goals[r, side] - arena.ep_goals[r, 1 - side]
                        res = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
                        self.pool_wins[k] = 0.97 * self.pool_wins[k] + 0.03 * res
                stats["matches"] += len(ends)
                arena.new_match(ends, pfsp)
            obs = obs_v3.observe(env)
            crit = self.critic_input(arena, obs)
            with torch.no_grad():
                last_v = self.critic(torch.from_numpy(crit.reshape(N * 8, -1)).to(self.dev)).view(N, 8).cpu().numpy()
            buf.append((ob, cr, ac, lp, bl, vv, rw, dn, lm, la, last_v))
            stats["decisions"] += T * N
        self.decisions += stats["decisions"]
        return buf, stats

    def gae(self, vv, rw, dn, last_v):
        a = self.a
        T = rw.shape[0]
        mu, sd = self.vnorm.mu, self.vnorm.std
        v = vv * sd + mu
        lv = last_v * sd + mu
        adv = np.zeros_like(rw)
        g = np.zeros(rw.shape[1:], np.float32)
        for t in reversed(range(T)):
            nv = lv if t == T - 1 else v[t + 1]
            nd = 1.0 - dn[t][:, None].astype(np.float32)
            delta = rw[t] + a.gamma * nv * nd - v[t]
            g = delta + a.gamma * a.gae * nd * g
            adv[t] = g
        return adv, adv + v

    # -------------------------------------------------------------- actualización
    def learn(self, buf, critic_only=False):
        a = self.a
        cols = [[] for _ in range(8)]
        for ob, cr, ac, lp, bl, vv, rw, dn, lm, la, last_v in buf:
            adv, ret = self.gae(vv, rw, dn, last_v)
            m = lm
            for i, x in enumerate((ob, cr, ac, lp, bl, adv, ret, np.broadcast_to(la[..., None], lm.shape))):
                cols[i].append(x[m])
        ob, cr, ac, lp, bl, adv, ret, lam = (np.concatenate(c) for c in cols)
        self.vnorm.update(ret)
        ret_n = (ret - self.vnorm.mu) / self.vnorm.std
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(ac)
        idx = np.arange(n)
        info = dict(samples=n, pi_loss=0.0, v_loss=0.0, kl_bc=0.0, entropy=0.0, clipfrac=0.0, approx_kl=0.0)
        steps = 0
        dev = self.dev
        T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
        for _ in range(a.epochs):
            self.rng.shuffle(idx)
            for mb in np.array_split(idx, a.minibatches):
                v = self.critic(T(cr[mb]))
                v_loss = F.mse_loss(v, T(ret_n[mb]))
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
                info["v_loss"] += float(v_loss.detach())
                if not critic_only:
                    logits = self.policy(T(ob[mb]))
                    logp_all = F.log_softmax(logits, -1)
                    act = T(ac[mb], torch.int64)
                    logp = logp_all.gather(1, act[:, None])[:, 0]
                    ratio = torch.exp(logp - T(lp[mb]))
                    A = T(adv[mb])
                    pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A)
                    bc = T(bl[mb])
                    kl = (bc.exp() * (bc - logp_all)).sum(-1)              # KL(BC‖π) por muestra
                    ent = -(logp_all.exp() * logp_all).sum(-1)
                    L = T(lam[mb])
                    loss = ((1 - L) * (pg - a.ent * ent) + L * kl).mean()
                    self.opt_pi.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.opt_pi.step()
                    with torch.no_grad():
                        info["pi_loss"] += float(pg.mean())
                        info["kl_bc"] += float(kl.mean())
                        info["entropy"] += float(ent.mean())
                        info["clipfrac"] += float(((ratio - 1).abs() > a.clip).float().mean())
                        info["approx_kl"] += float((T(lp[mb]) - logp).mean())
                steps += 1
        for k in ("pi_loss", "v_loss", "kl_bc", "entropy", "clipfrac", "approx_kl"):
            info[k] /= max(1, steps)
        info["ret_mean"] = float(ret.mean())
        return info

    @torch.no_grad()
    def evaluate(self):
        """Fuerza contra la BC (latencia de sala, ambos lados, muestreado) y parecido humano en self-play."""
        from learn.x4_eval import ModelPolicy, human_compare, play
        a = self.a
        delays = tuple(int(d) for d in a.delay_values)
        me = ModelPolicy(self.policy, self.dev, "aprendiz")
        bc = ModelPolicy(self.bc, self.dev, "bc")
        self.policy.eval()
        w = dr = l = gf = ga = 0
        for side, (red, blue) in enumerate(((me, bc), (bc, me))):
            r = play(red, blue, map_name="sanguchito_rs_x4", matches=a.eval_matches, minutes=a.eval_minutes,
                     delays=delays, seed=1000 + self.update * 2 + side, record=0)
            g = r["goals"] if side == 0 else r["goals"][:, ::-1]
            d = g[:, 0] - g[:, 1]
            w += int((d > 0).sum()); dr += int((d == 0).sum()); l += int((d < 0).sum())
            gf += int(g[:, 0].sum()); ga += int(g[:, 1].sum())
        selfplay = play(me, me, map_name="sanguchito_rs_x4", matches=4, minutes=a.eval_minutes, delays=delays,
                        seed=77 + self.update, record=4)
        hum = human_compare(selfplay["episodes"]) or {}
        self.policy.train()
        n = max(1, w + dr + l)
        key = ("passes_per_min", "possession_s", "pass_length", "depth", "width", "dist_ball_2", "still_frac",
               "key_changes_per_s", "kickoff_wait_s", "restart_s_lateral")
        w1 = hum.get("w1_norm", {})
        vals = [w1[k] for k in key if w1.get(k) is not None]
        out = dict(vs_bc=dict(wins=w, draws=dr, losses=l, score=round((w + 0.5 * dr) / n, 3),
                              goal_diff=round((gf - ga) / n, 3)),
                   selfplay_safety=float(selfplay["safety"].mean()),
                   human_w1_mean=round(float(np.mean(vals)), 3) if vals else None,
                   human_w1={k: (round(w1[k], 3) if w1.get(k) is not None else None) for k in key},
                   # valores crudos de self-play (humanos p50 en Sanguchito: patadas 3,2/min, pases 8,8/min,
                   # goles 0,25/min por tramo, espera del saque inicial 3,9 s)
                   selfplay={k: (round(hum["summary"][k]["mean"], 3) if hum.get("summary", {}).get(k, {}).get("n") else None)
                             for k in ("kicks_per_min", "passes_per_min", "goals_per_min", "kickoff_wait_s",
                                       "dist_ball_1", "still_frac")})
        return out

    def save(self, name="last.pt"):
        # sólo tipos planos: los checkpoints se cargan con torch.load(weights_only=True)
        def plain(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v
        torch.save(dict(model=self.policy.state_dict(), critic=self.critic.state_dict(), hidden=self.hidden,
                        obs_version=obs_v3.OBS_VERSION, update=int(self.update), decisions=int(self.decisions),
                        vnorm=[float(self.vnorm.mean), float(self.vnorm.sq), float(self.vnorm.w)],
                        pool_wins=[float(x) for x in self.pool_wins],
                        args={k: plain(v) for k, v in vars(self.a).items()}),
                   self.out / name)

    def snapshot(self):
        path = self.out / f"snap_{self.update:05d}.pt"
        self.save(path.name)
        m = SetPolicy(hidden=self.hidden).to(self.dev)
        m.load_state_dict(self.policy.state_dict())
        m.eval()
        self.pool.append((path.stem, m))
        self.pool_wins = np.append(self.pool_wins, 0.5)
        for ar in self.arenas:
            ar.pool_size = len(self.pool)

    def train(self):
        a = self.a
        t0 = time.time()
        for u in range(a.updates):
            self.update = u + 1
            tr = time.time()
            buf, stats = self.rollout()
            tl = time.time()
            info = self.learn(buf, critic_only=u < a.critic_warmup)
            row = dict(update=self.update, decisions=self.decisions, critic_only=u < a.critic_warmup,
                       rollout_s=round(tl - tr, 2), learn_s=round(time.time() - tl, 2),
                       dps=round(stats["decisions"] / max(1e-6, time.time() - tr)), goals=stats["goals"],
                       matches=stats["matches"], shaping_abs=round(stats["shaping"], 3), forfeits=stats["forfeits"],
                       pool_wins={n: round(float(w), 3) for (n, _), w in zip(self.pool, self.pool_wins)},
                       **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()})
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
            print(json.dumps(row), flush=True)
            if self.update % 10 == 0 or self.update == a.updates:
                self.save()
            if a.snapshot_every and self.update % a.snapshot_every == 0:
                self.snapshot()
            if a.eval_every and self.update % a.eval_every == 0 and u >= a.critic_warmup:
                ev = self.evaluate()
                ev.update(update=self.update, decisions=self.decisions, shaping=a.shaping)
                if a.shaping > 0 and a.shaping_retire > 0 and ev["vs_bc"]["score"] >= a.shaping_retire:
                    a.shaping = 0.0                       # MARLadona: se retira para siempre
                    ev["shaping_retired"] = True
                hw = ev["human_w1_mean"]
                if hw is not None and hw <= a.human_gate and ev["vs_bc"]["score"] > self.best_score:
                    self.best_score = ev["vs_bc"]["score"]
                    self.save("best.pt")
                    ev["new_best"] = True
                with open(self.out / "eval.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev) + "\n")
                print("EVAL " + json.dumps(ev), flush=True)
        self.save()
        return time.time() - t0


def main(argv=None):
    a = parse_args(argv)
    Trainer(a).train()


if __name__ == "__main__":
    main()
