"""PPO multi-agente con self-play, liga de snapshots (PFSP) y currículo.

python -m train.ppo_selfplay --config train/config.yaml --run classic_1v1
python -m train.ppo_selfplay --run classic_1v1 --resume      # continúa desde latest.pt

Currículo de tamaño de equipo (opcional, sección `teams` del config; requiere model.type: entity):
  1v1 -> 2v2 -> 3v3 -> 6v6. Se pasa a la etapa siguiente cuando el currículo de rivales llegó a la
  liga y se cumplieron `min_steps` en la etapa (o a los `max_steps`). El modelo se conserva; el entorno
  y los buffers se reconstruyen con el nuevo tamaño.

Tipos de partido (se reasignan en cada iteración):
  selfplay : los dos equipos los juega el agente actual (ambos generan datos)
  pool     : rojo = agente, azul = snapshot congelado de la liga
  scripted : rojo = agente, azul = bot por reglas
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from bots.scripted import scripted_actions
from env.haxball_env import HaxballEnv
from env.rewards import RewardConfig

from .league import League
from .model import ActorCritic, EntityActorCritic, build_model

ROOT = Path(__file__).resolve().parent.parent

SELF, POOL, SCRIPTED = 0, 1, 2


def lerp(a, b, t):
    return a + (b - a) * min(max(t, 0.0), 1.0)


def resolve_device(name: str | None) -> torch.device:
    """ppo.device: "cpu" (por defecto), "cuda" o "auto" (cuda si hay GPU NVIDIA)."""
    if name in (None, "cpu"):
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, cfg: dict, run: str, resume: bool):
        self.cfg = cfg
        self.run_dir = ROOT / "runs" / run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        e, p, r = cfg["env"], cfg["ppo"], cfg["reward"]
        torch.set_num_threads(p["torch_threads"])
        torch.manual_seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])

        self.rcfg = RewardConfig(goal=r["goal"], w_ball_progress=r["w_ball_progress"],
                                 w_near_ball=r["w_near_ball"], kick_to_goal=r["kick_to_goal"],
                                 kickoff_stall=r.get("kickoff_stall", 0.5),
                                 kickoff_approach=r.get("kickoff_approach", 0.0),
                                 out_penalty=r.get("out_penalty", 0.1),
                                 w_spread=r.get("w_spread", 0.0),
                                 gamma=p["gamma"])
        self.teams = cfg.get("teams")
        self.team_stage = 0
        self.team_steps = 0
        self._make_env()
        mc = cfg["model"]
        if mc.get("type", "mlp") == "entity":
            self.model = EntityActorCritic(self.env.self_dim, n_actions=self.env.n_actions, hidden=mc["hidden"],
                                           layers=mc["layers"], ent_hidden=mc.get("ent_hidden", 64))
        else:
            assert not self.teams, "el currículo de equipos necesita model.type: entity"
            self.model = ActorCritic(self.env.obs_dim, self.env.n_actions, mc["hidden"], mc["layers"])
        self.device = resolve_device(p.get("device"))
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p["lr"], eps=1e-5)
        lg = cfg["league"]
        self.league = League(max_size=lg["max_size"], pfsp_power=lg["pfsp_power"])
        self.steps = 0
        self.iteration = 0
        self.stage = 0
        self.wr_window: list[tuple[int, int]] = []
        if resume and (self.run_dir / "latest.pt").exists():
            self.load(self.run_dir / "latest.pt")
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    def env_settings(self) -> dict:
        """Config del entorno de la etapa actual (la de equipos pisa a la base)."""
        e = dict(self.cfg["env"])
        if self.teams:
            e.update(self.teams[self.team_stage])
        return e

    def _make_env(self) -> None:
        e = self.env_settings()
        self.env = HaxballEnv(e["n_envs"], e["n_per_team"], e["stadium"], e["frame_skip"],
                              e["max_ticks"], e["random_reset_prob"], self.rcfg, seed=self.cfg["seed"] + self.team_stage,
                              action_delay_max=e.get("action_delay_max", 0),
                              kickoff_timeout=e.get("kickoff_timeout", 0),
                              powershot=e.get("powershot", False),
                              out_of_bounds=e.get("out_of_bounds", False),
                              obs_layout=e.get("obs_layout", "flat"))
        self.N, self.P, self.T = self.env.N, self.env.P, self.env.T

    # ------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> None:
        torch.save({
            "model": self.model.state_dict(), "model_config": self.model.config(),
            "opt": self.opt.state_dict(), "steps": self.steps, "iteration": self.iteration,
            "stage": self.stage, "learner_elo": self.league.learner_elo,
            "scripted_elo": self.league.scripted_elo,
            "league": [(m.name, m.model.state_dict(), m.elo, m.wins, m.games) for m in self.league.members],
            "team_stage": self.team_stage, "team_steps": self.team_steps,
            "env": {"n_per_team": self.T, "stadium": self.env_settings()["stadium"],
                    "frame_skip": self.cfg["env"]["frame_skip"],
                    "powershot": self.cfg["env"].get("powershot", False),
                    "out_of_bounds": self.cfg["env"].get("out_of_bounds", False),
                    "obs_layout": self.cfg["env"].get("obs_layout", "flat")},
        }, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.steps, self.iteration, self.stage = ck["steps"], ck["iteration"], ck["stage"]
        self.league.learner_elo = ck["learner_elo"]
        self.league.scripted_elo = ck["scripted_elo"]
        self.team_stage, self.team_steps = ck.get("team_stage", 0), ck.get("team_steps", 0)
        if self.team_stage:
            self._make_env()
        for name, sd, elo, wins, games in ck["league"]:
            m = build_model(ck["model_config"])
            m.load_state_dict(sd)
            self.league.add_snapshot(m.to(self.device), name)
            mem = self.league.members[-1]
            mem.elo, mem.wins, mem.games = elo, wins, games
        print(f"reanudado desde {path} (paso {self.steps:,}, iter {self.iteration}, etapa {self.stage}, "
              f"{self.T}v{self.T})")

    # ------------------------------------------------------------ asignación de rivales
    def assign_modes(self):
        st = self.cfg["curriculum"][self.stage]
        fr = np.array([st["selfplay"], st["pool"] if self.league.members else 0.0, st["scripted"]])
        fr = fr / fr.sum()
        counts = np.floor(fr * self.N).astype(int)
        counts[0] += self.N - counts.sum()
        modes = np.concatenate([np.full(c, m) for m, c in zip((SELF, POOL, SCRIPTED), counts)])
        self.rng.shuffle(modes)
        opp_id = np.full(self.N, -1)
        pool_envs = np.where(modes == POOL)[0]
        if len(pool_envs):
            ids = self.league.sample(self.cfg["league"]["opponents_per_iter"], self.rng)
            opp_id[pool_envs] = np.array(ids)[np.arange(len(pool_envs)) % len(ids)]
        self.modes, self.opp_id = modes, opp_id
        # máscara de agentes que entrenan: en selfplay ambos equipos, si no sólo rojo (primeros T)
        learner = np.zeros((self.N, self.P), dtype=bool)
        learner[:, : self.T] = True
        learner[modes == SELF, self.T:] = True
        self.learner = learner
        self.scripted_eps = st["scripted_eps"]

    # ------------------------------------------------------------ actuar
    @torch.no_grad()
    def act(self, obs: np.ndarray):
        N, P, dev = self.N, self.P, self.device
        o = torch.from_numpy(obs.reshape(N * P, -1)).to(dev)
        logits, value = self.model(o)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        actions = a.cpu().numpy().reshape(N, P)
        blue = slice(self.T, self.P)
        # rivales congelados
        for oid in np.unique(self.opp_id[self.opp_id >= 0]):
            envs = np.where(self.opp_id == oid)[0]
            ob = torch.from_numpy(obs[envs, blue].reshape(len(envs) * self.T, -1)).to(dev)
            lg = self.league.members[oid].model.logits(ob)
            actions[envs, blue] = torch.distributions.Categorical(logits=lg).sample().cpu().numpy().reshape(len(envs), self.T)
        sc = np.where(self.modes == SCRIPTED)[0]
        if len(sc):
            actions[sc, blue] = scripted_actions(self.env, np.arange(self.T, self.P), self.scripted_eps, self.rng)[sc]
        return actions, logp.cpu().numpy().reshape(N, P), value.cpu().numpy().reshape(N, P)

    # ------------------------------------------------------------ loop
    def train(self, until_team: int | None = None):
        """until_team: frenar al llegar a esa etapa de equipos (p. ej. 1 = al terminar el 1v1)."""
        try:
            while self.steps < self.cfg["ppo"]["total_steps"]:
                if until_team is not None and self.team_stage >= until_team:
                    print(f"llegó a la etapa de equipos {until_team}: fin de la pasada")
                    break
                self._train_phase()
        except KeyboardInterrupt:
            print("interrumpido: guardando...")
        self.save(self.run_dir / "latest.pt")
        print(f"guardado en {self.run_dir / 'latest.pt'}")

    def _maybe_advance_team(self) -> bool:
        if not self.teams or self.team_stage >= len(self.teams) - 1:
            return False
        t = self.teams[self.team_stage]
        in_league = self.stage >= len(self.cfg["curriculum"]) - 1
        if (in_league and self.team_steps >= t.get("min_steps", 0)) or self.team_steps >= t.get("max_steps", float("inf")):
            self.save(self.run_dir / f"fin_{self.T}v{self.T}.pt")
            self.team_stage += 1
            self.team_steps = 0
            self.stage = 0
            self.wr_window = []
            # la liga arranca de nuevo con el modelo actual (los rivales de otro tamaño ya no sirven)
            self.league.members.clear()
            self._make_env()
            print(f"*** equipos: pasa a {self.T}v{self.T} en {self.env_settings()['stadium']} ***", flush=True)
            return True
        return False

    def _train_phase(self):
        cfg, p = self.cfg, self.cfg["ppo"]
        L, N, P = p["rollout_len"], self.N, self.P
        D = self.env.obs_dim
        obs = self.env.reset()
        buf_obs = np.zeros((L, N, P, D), np.float32)
        buf_act = np.zeros((L, N, P), np.int64)
        buf_logp = np.zeros((L, N, P), np.float32)
        buf_val = np.zeros((L, N, P), np.float32)
        buf_rew = np.zeros((L, N, P), np.float32)
        buf_done = np.zeros((L, N), np.float32)
        ep_goals = {SELF: [0, 0], POOL: [0, 0], SCRIPTED: [0, 0]}
        t_start = time.time()
        steps_start = self.steps
        while self.steps < p["total_steps"]:
            frac = self.steps / p["total_steps"]
            lr = lerp(p["lr"], p["lr_final"], frac)
            for g in self.opt.param_groups:
                g["lr"] = lr
            ent_coef = lerp(p["ent_coef"], p["ent_coef_final"], frac)
            self.rcfg.shaping_coef = max(0.0, 1.0 - self.steps / cfg["reward"]["shaping_decay_steps"])
            self.assign_modes()
            pool_goals = {}
            n_stall = n_done = 0
            t0 = time.time()

            for t in range(L):
                actions, logp, val = self.act(obs)
                buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = obs, actions, logp, val
                obs, rew, done, info = self.env.step(actions)
                if info["truncated"].any():
                    # bootstrap del valor en cortes por tiempo
                    tr = np.where(info["truncated"])[0]
                    with torch.no_grad():
                        fo = torch.from_numpy(info["final_obs"][tr].reshape(len(tr) * P, -1)).to(self.device)
                        v_final = self.model(fo)[1].cpu().numpy().reshape(len(tr), P)
                    rew[tr] += p["gamma"] * v_final
                buf_rew[t], buf_done[t] = rew, done
                n_stall += int(info["stall"].sum())
                n_done += int(done.sum())
                # estadísticas de goles desde el punto de vista del rojo (aprendiz)
                g = info["goal"]
                for m in (SELF, POOL, SCRIPTED):
                    sel = self.modes == m
                    ep_goals[m][0] += int((g[sel] == 1).sum())
                    ep_goals[m][1] += int((g[sel] == -1).sum())
                for e in np.where((g != 0) & (self.modes == POOL))[0]:
                    k = self.opp_id[e]
                    pool_goals.setdefault(k, [0, 0])[0 if g[e] == 1 else 1] += 1
                sc = (g != 0) & (self.modes == SCRIPTED)
                if sc.any():
                    self.league.record(None, int((g[sc] == 1).sum()), int((g[sc] == -1).sum()))
            for k, (a_, b_) in pool_goals.items():
                self.league.record(int(k), a_, b_)
            t_roll = time.time() - t0

            # GAE
            with torch.no_grad():
                last_v = self.model(torch.from_numpy(obs.reshape(N * P, -1)).to(self.device))[1].cpu().numpy().reshape(N, P)
            adv = np.zeros_like(buf_rew)
            gae = np.zeros((N, P), np.float32)
            for t in reversed(range(L)):
                nv = last_v if t == L - 1 else buf_val[t + 1]
                nd = 1.0 - buf_done[t][:, None]
                delta = buf_rew[t] + p["gamma"] * nv * nd - buf_val[t]
                gae = delta + p["gamma"] * p["gae_lambda"] * nd * gae
                adv[t] = gae
            ret = adv + buf_val

            # sólo muestras de agentes que aprenden
            m = np.broadcast_to(self.learner[None], (L, N, P)).reshape(-1)
            dev = self.device
            b_obs = torch.from_numpy(buf_obs.reshape(-1, D)[m]).to(dev)
            b_act = torch.from_numpy(buf_act.reshape(-1)[m]).to(dev)
            b_logp = torch.from_numpy(buf_logp.reshape(-1)[m]).to(dev)
            b_adv = torch.from_numpy(adv.reshape(-1)[m]).to(dev)
            b_ret = torch.from_numpy(ret.reshape(-1)[m]).to(dev)
            self.model.update_norm(b_obs)
            n_samples = len(b_act)
            self.steps += n_samples
            self.team_steps += n_samples

            t1 = time.time()
            stats = self.update(b_obs, b_act, b_logp, b_adv, b_ret, ent_coef)
            t_upd = time.time() - t1
            self.iteration += 1

            # currículo
            gs = ep_goals[SCRIPTED]
            self.wr_window.append((gs[0], gs[1]))
            ep_goals[SCRIPTED] = [0, 0]
            self.wr_window = self.wr_window[-20:]
            w = sum(x for x, _ in self.wr_window)
            l_ = sum(y for _, y in self.wr_window)
            wr = w / max(w + l_, 1)
            st = cfg["curriculum"][self.stage]
            if w + l_ >= 200 and wr >= st["advance_winrate"] and self.stage < len(cfg["curriculum"]) - 1:
                self.stage += 1
                self.wr_window = []
                print(f"*** etapa {self.stage}: {cfg['curriculum'][self.stage]['name']} ***")
                if not self.league.members:
                    self.league.add_snapshot(self.model, f"it{self.iteration}")

            lg = cfg["league"]
            if self.iteration % lg["snapshot_every"] == 0 and self.stage >= 1:
                self.league.add_snapshot(self.model, f"it{self.iteration}")
            from .checkpoints import maybe_save_checkpoints
            maybe_save_checkpoints(self)

            sps = (self.steps - steps_start) / (time.time() - t_start)
            wr_pool = self._rate(ep_goals[POOL])
            wr_self = self._rate(ep_goals[SELF])
            scalars = {**stats, "goal_share_vs_scripted": wr, "goal_share_vs_pool": wr_pool,
                       "red_goal_share_selfplay": wr_self, "training/goal_elo": self.league.learner_elo,
                       "training/scripted_goal_elo": self.league.scripted_elo, "stage": self.stage,
                       "goals_in_window": w + l_,
                       "shaping_coef": self.rcfg.shaping_coef, "lr": lr, "sps": sps,
                       "league_size": len(self.league.members),
                       "stall_frac": n_stall / max(n_done, 1), "team_size": self.T}
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, self.steps)
            if self.iteration % cfg["log"]["every"] == 0:
                print(f"it {self.iteration:5d} | {self.T}v{self.T} | pasos {self.steps/1e6:8.1f}M | {sps:7.0f}/s "
                      f"(rollout {t_roll:.1f}s upd {t_upd:.1f}s) | etapa {self.stage} | "
                      f"goles vs bot {wr:.2f} | goles vs liga {wr_pool:.2f} | elo por goles {self.league.learner_elo:6.0f} | "
                      f"ent {stats['entropy']:.2f} | shaping {self.rcfg.shaping_coef:.2f} | "
                      f"saques trabados {n_stall / max(n_done, 1):.2f}", flush=True)
                ep_goals[POOL] = [0, 0]
                ep_goals[SELF] = [0, 0]
            if self._maybe_advance_team():
                return

    @staticmethod
    def _rate(g):
        return g[0] / max(g[0] + g[1], 1)

    def update(self, obs, act, logp_old, adv, ret, ent_coef):
        p = self.cfg["ppo"]
        n = len(act)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = min(p["minibatch"], n)
        agg = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        cnt = 0
        for _ in range(p["epochs"]):
            perm = torch.randperm(n, device=obs.device)
            for s in range(0, n - mb + 1, mb):
                i = perm[s:s + mb]
                logits, v = self.model(obs[i])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act[i])
                ratio = torch.exp(lp - logp_old[i])
                a = adv[i]
                pg = -torch.min(ratio * a, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a).mean()
                vl = 0.5 * (v - ret[i]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + p["vf_coef"] * vl - ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    agg["pg_loss"] += pg.item()
                    agg["v_loss"] += vl.item()
                    agg["entropy"] += ent.item()
                    agg["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                    agg["clipfrac"] += ((ratio - 1).abs() > p["clip"]).float().mean().item()
                cnt += 1
        return {k: v / max(cnt, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "train" / "config.yaml"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--until-team", type=int, default=None,
                    help="frenar al llegar a esta etapa de equipos (1 = cuando termina el 1v1)")
    ap.add_argument("--override", nargs="*", default=[], help="clave.sub=valor, ej. env.n_envs=64")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for ov in args.override:
        k, v = ov.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for part in path:
            d = d[part]
        d[last] = yaml.safe_load(v)
    Trainer(cfg, args.run or cfg["run_name"], args.resume).train(args.until_team)


if __name__ == "__main__":
    main()
