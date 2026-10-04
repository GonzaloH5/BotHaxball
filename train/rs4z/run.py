"""Entrenamiento RS4-Z desde pesos aleatorios: curriculum con compuertas, PPO y crítico centralizado.

  python -m train.rs4z.run --run rs4z_main [--envs 576] [--device cuda] [--resume]
  python -m train.rs4z.run --run smoke --smoke --samples 20e6 --envs 64 --device cpu

Semántica (ver auditoría): el gol no termina el episodio de partido; el fin de partido y el final de un
ejercicio son terminales reales (Φ' = 0); el corte por tiempo de un ejercicio "truncate" y el borde del
rollout se bootstrapean con el valor del estado. Las filas que no controla el aprendiz (RS-Pro, redes
congeladas) nunca entran a la pérdida.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from bots.rspro.policy import RSPro, sample_style
from env.rs4z import contract as C
from env.rs4z import kernel as K
from env.rs4z.core import RS4ZEnv
from env.rs4z.drills import TASK_NAMES, TASKS, Drills, team_slots
from env.rs4z.obs_v2 import CRITIC_DIM, OBS_DIM, critic as build_critic, observe

from .league import League
from .model import ActorCritic
from .rewards import XT, XT_PATH, Rewards, per_player
from .stages import STAGE_INDEX, STAGES

ROOT = Path(__file__).resolve().parent.parent.parent
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
LEARNER, RSPRO, FROZEN = 0, 1, 2


@dataclass
class Config:
    run: str = "rs4z"
    envs: int = 576
    rollout: int = 128
    epochs: int = 3
    minibatch: int = 16384
    lr: float = 3e-4
    lr_min: float = 5e-5
    clip: float = 0.2
    ent_start: float = 0.01
    ent_end: float = 0.003
    vf_coef: float = 0.5
    max_grad: float = 0.5
    target_kl: float = 0.03
    gae_lambda: float = 0.95
    device: str = "cuda"
    seed: int = 0
    start_stage: str = "S1"
    samples: float = 0.0           # techo total (0 = presupuesto del curriculum)
    eval_every: float = 50e6
    eval_min_frac: float = 0.25    # no evaluar compuertas antes de este % del presupuesto de la etapa
    ckpt_every: float = 25e6
    snapshot_every: float = 250e6
    smoke: bool = False
    gate_episodes: int = 256
    target_success: float = 0.6    # dificultad adaptativa: éxito buscado por tarea


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = ROOT / "runs" / "rs4z" / cfg.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        N = cfg.envs
        self.env = RS4ZEnv(N, contract="v2", seed=cfg.seed, deadline=C.TRAINING_DEADLINE,
                           kickoff_deadline=C.TRAINING_DEADLINE, max_delay=12)
        bank = None
        bank_path = ROOT / "data" / "rs4_states" / "rs4z_entrenamiento.npz"
        if not bank_path.exists():
            bank_path = ROOT / "data" / "rs4_states" / "entrenamiento.npz"
        if bank_path.exists():
            from env.rs4_states import StateBank
            bank = StateBank(bank_path)
        self.drills = Drills(self.env, self.rng, state_bank=bank)
        self.bot = RSPro(self.env, seed=cfg.seed + 17)
        self.xt = XT() if XT_PATH.exists() else None
        self.stage_i = STAGE_INDEX[cfg.start_stage]
        self.rewards = Rewards(self.env, STAGES[self.stage_i].gamma, self.xt)
        self.model = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.league = League(self.dir / "league", device=self.device)
        self.samples = 0
        self.stage_samples = 0
        self.iter = 0
        self.difficulty = {t: 0.0 for t in TASK_NAMES}
        self.success = {t: 0.5 for t in TASK_NAMES}
        self.ret_mean, self.ret_var, self.ret_count = 0.0, 1.0, 1e-4
        self.ctrl = np.zeros((N, 8), dtype=np.int64)        # LEARNER / RSPRO / FROZEN
        self.frozen_id = np.full(N, -1, dtype=np.int64)     # miembro de la liga en filas con FROZEN
        self.is_match = np.zeros(N, dtype=bool)
        self.ball_task = np.zeros(N, dtype=bool)
        self.task_log = {t: [0, 0.0] for t in TASK_NAMES}
        self.history = []
        self._last_eval = 0
        self._last_ckpt = 0
        self._last_snapshot = 0
        self.stop_reason = None
        self._apply_stage()
        self.assign(np.arange(N))

    # ------------------------------------------------------------------ curriculum
    @property
    def stage(self):
        return STAGES[self.stage_i]

    def _apply_stage(self, progress=0.0):
        st = self.stage
        c = dict(st.coefs)
        for k, end in st.coefs_end.items():
            c[k] = c.get(k, 0.0) * (1.0 - progress) + end * progress
        for k in ("goal", "result", "drill", "threat", "access", "ball", "crowd"):
            setattr(self.rewards.coefs, k, float(c.get(k, 0.0)))
        self.rewards.gamma = st.gamma

    def _task_weights(self):
        st = self.stage
        names = list(st.tasks)
        w = np.array([st.tasks[t] for t in names], dtype=np.float64)
        w /= w.sum()
        prev = [t for s in STAGES[:self.stage_i] for t in s.tasks if t not in st.tasks]
        if prev and st.retention > 0:
            w = w * (1 - st.retention)
            names += prev
            w = np.concatenate([w, np.full(len(prev), st.retention / len(prev))])
        return names, w

    def assign(self, rows):
        """Episodio nuevo en `rows`: tarea, rivales, compañeros, latencia y variante del mapa."""
        env, st, rng = self.env, self.stage, self.rng
        names, w = self._task_weights()
        for n in np.atleast_1d(rows):
            task = TASKS[names[rng.choice(len(names), p=w)]]
            diff = self.difficulty[task.name]
            self.drills.start([n], task.name, diff)
            lt = int(self.drills.st.learner_team[n])
            ctrl = np.where(env.active[n], RSPRO, RSPRO)
            ctrl[team_slots(lt, task.n_own)] = LEARNER
            self.frozen_id[n] = -1
            match = task.timeout == 0
            self.is_match[n] = match
            self.ball_task[n] = task.n_opp == 0
            if match and task.n_opp > 0:
                mode = self._opponent_mode()
                opp = team_slots(1 - lt, task.n_opp)
                if mode == "mirror":
                    ctrl[opp] = LEARNER
                elif mode in ("pfsp", "exploiter") and len(self.league):
                    member = self.league.sample(rng, exploiter=mode == "exploiter")
                    if member is not None:
                        ctrl[opp] = FROZEN
                        self.frozen_id[n] = member
                # compañeros ad-hoc: uno del equipo aprendiz (sólo 4v4/4v3) controlado por RS-Pro
                if st.frozen_mates and task.n_own >= 3 and rng.random() < st.frozen_mates:
                    mate = team_slots(lt, task.n_own)[rng.integers(1, task.n_own)]
                    ctrl[mate] = RSPRO
            ctrl[~env.active[n]] = RSPRO
            self.ctrl[n] = ctrl
            # niveles y estilos de RS-Pro (nunca la región reservada para evaluar)
            lo, hi = task.opp_levels
            for t in (0, 1):
                level = int(np.clip(round(lo + diff * (hi - lo) + rng.normal(0, 0.7)), lo, hi))
                if t == lt:
                    level = int(rng.integers(3, 6))     # compañeros scripted: competentes
                self.bot.configure([n], t, level, sample_style(rng))
            # latencia por jugador y variante del mapa
            lat = st.latency
            host = rng.random(8) < lat["host"]
            client = rng.integers(lat["client"][0], lat["client"][1] + 1, 8)
            env.delay[n] = np.where(host, 0, client)
            if st.variants and rng.random() < st.variants:
                env.rf[n, K.RF_KSTR] = C.KICK_STRENGTHS[int(rng.integers(0, 2))]
                env.radius[n, 0] = C.BALL_RADII[int(rng.integers(0, 2))]
            self.bot.sync(env, [n])

    def _opponent_mode(self):
        opp = self.stage.opponents
        modes = list(opp)
        p = np.array([opp[m] for m in modes], dtype=np.float64)
        return modes[self.rng.choice(len(modes), p=p / p.sum())]

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def _policy(self, obs, crit):
        x = torch.from_numpy(obs).to(self.device)
        c = torch.from_numpy(crit).to(self.device)
        logits, value = self.model(x, c)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), value.cpu().numpy()

    def rollout(self):
        cfg, env = self.cfg, self.env
        T, N = cfg.rollout, env.N
        buf = dict(obs=np.zeros((T, N, 8, OBS_DIM), np.float32), crit=np.zeros((T, N, 8, CRITIC_DIM), np.float32),
                   act=np.zeros((T, N, 8), np.int64), logp=np.zeros((T, N, 8), np.float32),
                   val=np.zeros((T, N, 8), np.float32), rew=np.zeros((T, N, 8), np.float32),
                   learn=np.zeros((T, N, 8), bool), term=np.zeros((T, N), bool), trunc=np.zeros((T, N), bool),
                   final_val=np.zeros((T, N, 8), np.float32))
        stats = dict(goals=0, matches=0, drill_done=0, forfeits=0, reward_terms={})
        obs = np.zeros((N, 8, OBS_DIM), np.float32)
        crit = np.zeros((N, 8, CRITIC_DIM), np.float32)
        out = np.zeros((N, 8), dtype=np.int64)
        for t in range(T):
            observe(env, obs)
            build_critic(env, crit)
            learn = (self.ctrl == LEARNER) & env.active
            idx = np.nonzero(learn)
            out[:] = 0
            a = np.zeros((N, 8), np.int64)
            logp = np.zeros((N, 8), np.float32)
            v = np.zeros((N, 8), np.float32)
            if len(idx[0]):
                a_l, logp_l, v_l = self._policy(obs[idx], crit[idx])
                a[idx], logp[idx], v[idx] = a_l, logp_l, v_l
                out[idx] = a_l
            scripted = (self.ctrl == RSPRO) & env.active
            if scripted.any():
                self.bot.act(env, scripted, out)
            frozen = (self.ctrl == FROZEN) & env.active
            if frozen.any():
                self.league.act(env, obs, frozen, self.frozen_id, out)
            phi0 = self.rewards.potentials(self.ball_task)
            ev = env.step(out)
            self.bot.push(env)
            done, outcome, trunc = self.drills.check(ev)
            match_end = ev["match_end"] & self.is_match
            terminal = (done & ~trunc) | match_end
            phi1 = self.rewards.potentials(self.ball_task)
            c = self.rewards.coefs
            team_r = self.rewards.shaping(phi0, phi1, terminal)
            stats["reward_terms"]["shaping"] = stats["reward_terms"].get("shaping", 0.0) + float(np.abs(team_r).sum())
            g = ev["goal"]
            goal_r = np.stack([np.where(g == 1, 1.0, np.where(g == -1, -1.0, 0.0)),
                               np.where(g == -1, 1.0, np.where(g == 1, -1.0, 0.0))], axis=1)
            team_r += c.goal * goal_r * self.is_match[:, None]
            if c.result:
                sc = env.score
                res = np.sign(sc[:, 0] - sc[:, 1]).astype(np.float64)
                team_r += c.result * np.stack([res, -res], axis=1) * match_end[:, None]
            lt = self.drills.st.learner_team
            drill_r = np.zeros((N, 2))
            drill_r[np.arange(N), lt] = outcome * (done & ~self.is_match)
            team_r += c.drill * drill_r
            buf["obs"][t], buf["crit"][t], buf["act"][t] = obs, crit, a
            buf["logp"][t], buf["val"][t], buf["learn"][t] = logp, v, learn
            buf["rew"][t] = per_player(team_r)
            buf["term"][t], buf["trunc"][t] = terminal, trunc & done
            stats["goals"] += int((g != 0).sum())
            stats["forfeits"] += int((ev["forfeit"] >= 0).sum())
            ended = np.flatnonzero(done | match_end)
            if len(trunc.nonzero()[0]):
                tr = np.flatnonzero(trunc & done)
                if len(tr):
                    fo = observe(env)
                    fc = build_critic(env)
                    sel = np.zeros((N, 8), bool)
                    sel[tr] = learn[tr]
                    fi = np.nonzero(sel)
                    if len(fi[0]):
                        with torch.no_grad():
                            fv = self.model.value(torch.from_numpy(fo[fi]).to(self.device),
                                                  torch.from_numpy(fc[fi]).to(self.device)).cpu().numpy()
                        buf["final_val"][t][fi] = fv
            for n in ended:
                name = TASK_NAMES[self.drills.st.task[n]]
                ok = outcome[n] > 0.5 if not self.is_match[n] else (env.score[n, lt[n]] > env.score[n, 1 - lt[n]])
                if self.is_match[n] and self.frozen_id[n] >= 0:
                    mine, theirs = env.score[n, lt[n]], env.score[n, 1 - lt[n]]
                    self.league.record(int(self.frozen_id[n]), 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0)
                self.success[name] = 0.98 * self.success[name] + 0.02 * float(ok)
                self.task_log[name][0] += 1
                self.task_log[name][1] += float(ok)
                d = self.difficulty[name]
                self.difficulty[name] = float(np.clip(d + (0.01 if self.success[name] > cfg.target_success + 0.1 else
                                                           -0.01 if self.success[name] < cfg.target_success - 0.1 else 0.0), 0, 1))
                stats["matches" if self.is_match[n] else "drill_done"] += 1
            goal_rows = np.flatnonzero((g != 0) & ~done)
            if len(goal_rows):
                self.bot.sync(env, goal_rows)
            if len(ended):
                self.assign(ended)
        observe(env, obs)
        build_critic(env, crit)
        last_v = np.zeros((N, 8), np.float32)
        li = np.nonzero((self.ctrl == LEARNER) & env.active)
        if len(li[0]):
            with torch.no_grad():
                last_v[li] = self.model.value(torch.from_numpy(obs[li]).to(self.device),
                                              torch.from_numpy(crit[li]).to(self.device)).cpu().numpy()
        return buf, last_v, stats

    # ------------------------------------------------------------------ PPO
    def update(self, buf, last_v):
        cfg = self.cfg
        gamma, lam = self.rewards.gamma, cfg.gae_lambda
        std = math.sqrt(self.ret_var) + 1e-8
        val = buf["val"] * std + self.ret_mean       # el crítico predice valores normalizados
        fval = buf["final_val"] * std + self.ret_mean
        lv = last_v * std + self.ret_mean
        T = val.shape[0]
        adv = np.zeros_like(val)
        acc = np.zeros(val.shape[1:], np.float32)
        for t in reversed(range(T)):
            nxt = lv if t == T - 1 else val[t + 1]
            term = buf["term"][t][:, None]
            trunc = buf["trunc"][t][:, None]
            nxt = np.where(trunc, fval[t], nxt)
            nxt = np.where(term, 0.0, nxt)
            delta = buf["rew"][t] + gamma * nxt - val[t]
            cont = ~(term | trunc)
            acc = delta + gamma * lam * cont * acc
            adv[t] = acc
        ret = adv + val
        mask = buf["learn"]
        r = ret[mask]
        # normalización de retornos (sólo crítico)
        b_mean, b_var, b_n = float(r.mean()), float(r.var()), r.size
        tot = self.ret_count + b_n
        delta = b_mean - self.ret_mean
        self.ret_mean += delta * b_n / tot
        self.ret_var = (self.ret_var * self.ret_count + b_var * b_n + delta ** 2 * self.ret_count * b_n / tot) / tot
        self.ret_count = tot
        std = math.sqrt(self.ret_var) + 1e-8
        D = lambda k: torch.from_numpy(buf[k][mask]).to(self.device)
        obs, crit, act, old_logp = D("obs"), D("crit"), D("act"), D("logp")
        adv_t = torch.from_numpy(adv[mask]).to(self.device)
        ret_t = torch.from_numpy(((ret[mask] - self.ret_mean) / std).astype(np.float32)).to(self.device)
        n = obs.shape[0]
        frac = min(1.0, self.samples / max(self._budget(), 1.0))
        ent_coef = cfg.ent_start + (cfg.ent_end - cfg.ent_start) * frac
        lr = cfg.lr + (cfg.lr_min - cfg.lr) * frac
        for g in self.opt.param_groups:
            g["lr"] = lr
        logs = dict(kl=0.0, clip=0.0, entropy=0.0, vloss=0.0, ploss=0.0, epochs=0)
        mb = min(cfg.minibatch, n)
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            kls = []
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logits, value = self.model(obs[idx], crit[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - old_logp[idx])
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = 0.5 * ((value - ret_t[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + cfg.vf_coef * vl - ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    kls.append(kl)
                    logs["clip"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                logs["entropy"] += ent.item()
                logs["vloss"] += vl.item()
                logs["ploss"] += pl.item()
            logs["epochs"] += 1
            logs["kl"] = float(np.mean(kls))
            if logs["kl"] > cfg.target_kl:
                break
        steps = max(1, logs["epochs"] * math.ceil(n / mb))
        for k in ("clip", "entropy", "vloss", "ploss"):
            logs[k] /= steps
        logs.update(lr=lr, ent_coef=ent_coef, rows=int(n))
        return logs, int(n)

    def _budget(self):
        return self.cfg.samples if self.cfg.samples else sum(s.budget for s in STAGES)

    # ------------------------------------------------------------------ compuertas
    def check_gates(self):
        from eval.rs4z.gates import evaluate_stage
        st = self.stage
        if not st.gates:
            return None
        cpu_model = copy.deepcopy(self.model).to("cpu")
        result = evaluate_stage(st.name, cpu_model, episodes=self.cfg.gate_episodes, seed=1000 + self.iter)
        result["samples"] = self.samples
        with (self.dir / "gates.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        return result

    # ------------------------------------------------------------------ persistencia
    def save(self, name="latest.pt"):
        state = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), cfg=asdict(self.cfg),
                     model_config=self.model.config, samples=self.samples, stage=self.stage.name,
                     stage_samples=self.stage_samples, iter=self.iter, difficulty=self.difficulty,
                     success=self.success, ret=(self.ret_mean, self.ret_var, self.ret_count),
                     rng=self.rng.bit_generator.state, league=self.league.state())
        tmp = self.dir / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(self.dir / name)

    def load(self, path):
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.samples, self.stage_samples, self.iter = s["samples"], s["stage_samples"], s["iter"]
        self.stage_i = STAGE_INDEX[s["stage"]]
        self.difficulty, self.success = s["difficulty"], s["success"]
        self.ret_mean, self.ret_var, self.ret_count = s["ret"]
        self.rng.bit_generator.state = s["rng"]
        self.league.load_state(s["league"])
        self._apply_stage(min(1.0, self.stage_samples / self.stage.budget))
        self.assign(np.arange(self.env.N))

    # ------------------------------------------------------------------ bucle
    def train(self):
        cfg = self.cfg
        budget = self._budget()
        log_path = self.dir / "log.jsonl"
        while self.samples < budget and self.stop_reason is None:
            t0 = time.time()
            buf, last_v, stats = self.rollout()
            t1 = time.time()
            logs, rows = self.update(buf, last_v)
            t2 = time.time()
            self.samples += rows
            self.stage_samples += rows
            self.iter += 1
            progress = min(1.0, self.stage_samples / self.stage.budget)
            self._apply_stage(progress)
            row = dict(iter=self.iter, samples=self.samples, stage=self.stage.name, stage_progress=progress,
                       sps=rows / (t2 - t0), rollout_s=t1 - t0, update_s=t2 - t1, goals=stats["goals"],
                       matches=stats["matches"], drills=stats["drill_done"], forfeits=stats["forfeits"],
                       coefs=asdict(self.rewards.coefs),
                       success={k: round(v, 3) for k, v in self.success.items() if k in self.stage.tasks},
                       difficulty={k: round(v, 2) for k, v in self.difficulty.items() if k in self.stage.tasks},
                       **logs)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if self.iter % 5 == 0 or cfg.smoke:
                print(f"[{self.stage.name} {progress:4.0%}] it {self.iter} muestras {self.samples / 1e6:7.1f}M "
                      f"{row['sps'] / 1e3:5.1f}k/s kl {logs['kl']:.4f} ent {logs['entropy']:.3f} "
                      f"éxito {row['success']}", flush=True)
            if self.samples - self._last_ckpt >= cfg.ckpt_every:
                self.save()
                self._last_ckpt = self.samples
            if self.stage.name in ("S5", "S6", "S7") and self.samples - self._last_snapshot >= cfg.snapshot_every:
                self.league.add(self.model, self.samples)
                self._last_snapshot = self.samples
            if (progress >= cfg.eval_min_frac and self.samples - self._last_eval >= cfg.eval_every) or progress >= 1.0:
                self._last_eval = self.samples
                result = self.check_gates()
                if result is not None and result["passed"]:
                    print(f"compuertas de {self.stage.name} aprobadas: {result['summary']}", flush=True)
                    self.save(f"stage_{self.stage.name}.pt")
                    if self.stage_i + 1 < len(STAGES):
                        self.stage_i += 1
                        self.stage_samples = 0
                        self._apply_stage()
                    else:
                        self.stop_reason = "curriculum completo"
                elif progress >= 1.0:
                    self.stop_reason = (f"presupuesto de {self.stage.name} agotado sin aprobar compuertas: "
                                        f"{None if result is None else result['summary']}")
        self.save()
        print("fin:", self.stop_reason or "presupuesto total", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for f in Config.__dataclass_fields__.values():
        if f.type in ("bool", bool):
            ap.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
    trainer = Trainer(cfg)
    if args.resume and (trainer.dir / "latest.pt").exists():
        trainer.load(trainer.dir / "latest.pt")
    trainer.train()


if __name__ == "__main__":
    main()
