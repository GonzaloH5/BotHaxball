"""Competitive RL (F4) for Futsal x3: Shared-parameter PPO with KL anchor to BC/Tactical prior.

  python -m learn.x3futsal_rl --bc runs/x3futsal_bc/tactical_v3/best_competitive.pt --out runs/x3futsal_rl/ppo_v1
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learn.x3futsal_data import (
    TACTICAL_DATA_VERSION,
    TACTICAL_ENTITY_DIM,
    TACTICAL_FEATURE_CONTRACT,
    TACTICAL_SELF_DIM,
    featurize_tactical,
)
from learn.x3futsal_eval import (
    FRAME_SKIP,
    KICKOFF_SAFETY_TICKS,
    STADIUM,
    ScriptedPolicy,
    TacticalPolicy,
    _aggregate,
    _player_states,
    action_to_sim,
    play_batch,
)
from learn.x3futsal_policy import TACTICAL_POLICY_VERSION, X3TacticalPolicy
from sim.physics import BatchSim
from sim.stadium import load_stadium
from tools.x3futsal_audit import stadium_identity


class X3CentralCritic(nn.Module):
    """Centralized value function V(s) over the oriented 3v3 state for MAPPO."""

    def __init__(self, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TACTICAL_SELF_DIM + 5 * TACTICAL_ENTITY_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, own: torch.Tensor, mates: torch.Tensor, rivals: torch.Tensor) -> torch.Tensor:
        flat = torch.cat((own, mates.reshape(len(own), -1), rivals.reshape(len(own), -1)), dim=-1)
        return self.net(flat).squeeze(-1)


def _potential(sim: BatchSim, goal_x: float) -> np.ndarray:
    """Zero-sum potential function phi(s) in [-1, 1] measuring ball progression toward blue goal (+x)."""
    bx = sim.ball_pos[:, 0] / max(goal_x, 1.0)
    bvx = sim.ball_vel[:, 0] / 12.0
    return np.clip(0.85 * bx + 0.15 * bvx, -1.0, 1.0)


def _observe_team(sim: BatchSim, team: int, half_w: float, half_h: float, goal_x: float):
    slots = np.arange(3 * team, 3 * team + 3, dtype=np.int64)
    states = _player_states(sim)
    count = sim.N * 3
    repeated_states = np.repeat(states[:, None, :, :], 3, axis=1).reshape(count, 7, 4)
    actor_slots = np.tile(slots, sim.N)
    kick_cancel = sim.kick_cancel[:, slots].reshape(-1)
    return featurize_tactical(
        repeated_states, actor_slots, half_w, half_h, goal_x, kick_cancel=kick_cancel
    )


def train_ppo(
    *,
    out: Path,
    bc_checkpoint: Path | None = None,
    updates: int = 200,
    envs: int = 16,
    horizon: int = 128,
    epochs: int = 4,
    hidden: int = 192,
    entity_hidden: int = 96,
    lr: float = 3e-4,
    kl_coef: float = 0.15,
    shaping_coef: float = 0.08,
    forfeit_penalty: float = 0.15,
    deep_defender_penalty: float = 0.025,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    eval_every: int = 20,
    eval_matches: int = 4,
    seed: int = 20261028,
    threads: int = 2,
    device_name: str = "cpu",
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    stadium = load_stadium(STADIUM)
    half_w, half_h, goal_x = stadium.field_half_w, stadium.field_half_h, stadium.goal_x
    target_id = stadium_identity(STADIUM)

    actor = X3TacticalPolicy(hidden=hidden, entity_hidden=entity_hidden, use_history=False).to(device)
    if bc_checkpoint is not None and bc_checkpoint.is_file():
        ckpt = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("policyVersion") == TACTICAL_POLICY_VERSION:
            actor.load_state_dict(ckpt["model"])
    anchor = copy.deepcopy(actor).to(device).eval()
    for param in anchor.parameters():
        param.requires_grad_(False)

    critic = X3CentralCritic(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=lr, weight_decay=1e-4
    )

    sim = BatchSim(envs, 3, 3, stadium=stadium, seed=seed)
    tactical_opp = TacticalPolicy(stadium)
    scripted_opp = ScriptedPolicy(stadium)
    kickoff_ticks = np.zeros(envs, dtype=np.int64)
    dummy_scores = np.zeros(envs, dtype=np.int64)
    dummy_prev = np.zeros((envs, 6), dtype=np.int64)
    dummy_has_prev = np.zeros((envs, 6), dtype=bool)

    out.mkdir(parents=True, exist_ok=True)
    log = {"updates": [], "bestCompetitiveIndex": None}
    best_index = None
    t0 = time.time()

    for update in range(1, updates + 1):
        # Alternate opponent mode across environments: half tactical, half scripted/self-play
        use_tactical_opp = (update % 2 == 1)
        obs_own, obs_mates, obs_rivals = [], [], []
        act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], []

        for _ in range(horizon):
            own, mates, rivals = _observe_team(sim, 0, half_w, half_h, goal_x)
            own_t = torch.from_numpy(own).to(device)
            mates_t = torch.from_numpy(mates).to(device)
            rivals_t = torch.from_numpy(rivals).to(device)
            with torch.no_grad():
                logits = actor(own_t, mates_t, rivals_t)
                dist = torch.distributions.Categorical(logits=logits)
                actions_red = dist.sample()
                logp = dist.log_prob(actions_red)
                values = critic(own_t, mates_t, rivals_t).reshape(envs, 3).mean(dim=1)

            red_np = actions_red.cpu().numpy().reshape(envs, 3)
            if use_tactical_opp:
                blue_np = tactical_opp.actions(sim, 1, dummy_scores, 0.0, dummy_prev, dummy_has_prev)
            else:
                blue_np = scripted_opp.actions(sim, 1, dummy_scores, 0.0, dummy_prev, dummy_has_prev)

            native = np.zeros((envs, 6), dtype=np.int64)
            for match in range(envs):
                native[match, :3] = [action_to_sim(a, 0) for a in red_np[match]]
                native[match, 3:] = [action_to_sim(a, 1) for a in blue_np[match]]

            phi_before = _potential(sim, goal_x)
            step_reward = np.zeros(envs, dtype=np.float32)
            step_done = np.zeros(envs, dtype=bool)
            ball_advanced = sim.ball_pos[:, 0] > (-half_w + 190.0)
            deep_players = sim.player_pos[:, :, 0] < (-half_w + 115.0)
            step_reward -= deep_defender_penalty * (
                deep_players & ball_advanced[:, None]
            ).sum(axis=1).astype(np.float32)

            rearm = (native >= 9) & sim.kick_cancel
            cur_actions = native.copy()
            for sub in range(FRAME_SKIP):
                if sub == 0 and np.any(rearm):
                    cur_actions = np.where(rearm, native % 9, native)
                elif sub == 1 and np.any(rearm):
                    cur_actions = native.copy()
                sim.step(cur_actions)
                if np.any(sim.kicked):
                    cur_actions = np.where(sim.kicked, cur_actions % 9, cur_actions)
                for match in range(envs):
                    g = int(sim.goal[match])
                    if g != 0:
                        step_reward[match] += 1.0 if g > 0 else -1.0
                        step_done[match] = True
                        conceding = 1 if g > 0 else 0
                        sim.reset_kickoff([match], kickoff_team=conceding)
                        kickoff_ticks[match] = 0
                    elif sim.kickoff[match]:
                        kickoff_ticks[match] += 1
                        if kickoff_ticks[match] >= KICKOFF_SAFETY_TICKS:
                            ko_team = int(sim.kickoff_team[match])
                            step_reward[match] += -forfeit_penalty if ko_team == 0 else forfeit_penalty
                            sim.kickoff[match] = False
                            sim.mask[match, sim.first_player:] = sim.base_mask[sim.first_player:]
                            kickoff_ticks[match] = 0
                    else:
                        kickoff_ticks[match] = 0

            phi_after = np.where(step_done, 0.0, _potential(sim, goal_x))
            step_reward += shaping_coef * (gamma * phi_after - phi_before).astype(np.float32)

            obs_own.append(own)
            obs_mates.append(mates)
            obs_rivals.append(rivals)
            act_buf.append(actions_red.cpu().numpy().reshape(envs, 3))
            logp_buf.append(logp.cpu().numpy().reshape(envs, 3))
            val_buf.append(values.cpu().numpy())
            rew_buf.append(step_reward)
            done_buf.append(step_done)

        with torch.no_grad():
            own, mates, rivals = _observe_team(sim, 0, half_w, half_h, goal_x)
            next_val = critic(
                torch.from_numpy(own).to(device),
                torch.from_numpy(mates).to(device),
                torch.from_numpy(rivals).to(device),
            ).reshape(envs, 3).mean(dim=1).cpu().numpy()

        # GAE calculation
        adv = np.zeros((horizon, envs), dtype=np.float32)
        gae = np.zeros(envs, dtype=np.float32)
        for t in range(horizon - 1, -1, -1):
            nonterminal = (~done_buf[t]).astype(np.float32)
            v_next = next_val if t == horizon - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * v_next * nonterminal - val_buf[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + np.asarray(val_buf, dtype=np.float32)

        flat_own = torch.from_numpy(np.concatenate(obs_own, axis=0)).to(device)
        flat_mates = torch.from_numpy(np.concatenate(obs_mates, axis=0)).to(device)
        flat_rivals = torch.from_numpy(np.concatenate(obs_rivals, axis=0)).to(device)
        flat_act = torch.from_numpy(np.asarray(act_buf).reshape(-1)).to(device)
        flat_old_logp = torch.from_numpy(np.asarray(logp_buf).reshape(-1)).to(device)
        flat_adv = torch.from_numpy(np.repeat(adv, 3, axis=1).reshape(-1)).to(device)
        flat_ret = torch.from_numpy(np.repeat(returns, 3, axis=1).reshape(-1)).to(device)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-6)

        for _ in range(epochs):
            logits = actor(flat_own, flat_mates, flat_rivals)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(flat_act)
            ratio = torch.exp(new_logp - flat_old_logp)
            surr1 = ratio * flat_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * flat_adv
            ppo_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

            with torch.no_grad():
                ref_logits = anchor(flat_own, flat_mates, flat_rivals)
                ref_probs = F.softmax(ref_logits, dim=-1)
            kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), ref_probs, reduction="batchmean")

            pred_val = critic(flat_own, flat_mates, flat_rivals)
            val_loss = F.mse_loss(pred_val, flat_ret)
            total_loss = (1.0 - kl_coef) * ppo_loss + kl_coef * kl_loss + 0.5 * val_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0
            )
            optimizer.step()

        if update % eval_every == 0 or update == updates:
            ckpt = {
                "model": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                "hidden": hidden,
                "entityHidden": entity_hidden,
                "useHistory": False,
                "obs_version": TACTICAL_DATA_VERSION,
                "policyVersion": TACTICAL_POLICY_VERSION,
                "featureContract": TACTICAL_FEATURE_CONTRACT,
                "datasetSignature": {"targetStadiumIdentity": target_id},
                "update": update,
            }
            torch.save(ckpt, out / "last.pt")
            from learn.x3futsal_eval import ModelPolicy
            pol = ModelPolicy(out / "last.pt", device)
            games = []
            for red, blue, role in ((pol, scripted_opp, 0), (scripted_opp, pol, 1)):
                side_games = play_batch(red, blue, matches=eval_matches, minutes=0.5, seed=seed + update)
                games.extend({**g, "firstPolicyTeam": role} for g in side_games)
            agg = _aggregate(games, 0)
            comp_idx = agg["goalDifferencePerMatch"] + 0.05 * agg["shotsPerMatch"]
            agg.update(update=update, competitiveIndex=float(comp_idx), seconds=round(time.time() - t0, 1))
            log["updates"].append(agg)
            if best_index is None or comp_idx > best_index:
                best_index = float(comp_idx)
                log["bestCompetitiveIndex"] = best_index
                torch.save(ckpt, out / "best.pt")
            (out / "metrics.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(agg), flush=True)

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bc", type=Path, default=None, help="checkpoint BC táctico inicial (ancla KL)")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--entity-hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-coef", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-matches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_ppo(
        out=args.out,
        bc_checkpoint=args.bc,
        updates=args.updates,
        envs=args.envs,
        horizon=args.horizon,
        hidden=args.hidden,
        entity_hidden=args.entity_hidden,
        lr=args.lr,
        kl_coef=args.kl_coef,
        eval_every=args.eval_every,
        eval_matches=args.eval_matches,
        seed=args.seed,
        threads=args.threads,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()