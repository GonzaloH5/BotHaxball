"""Competitive RL (F4) for Futsal x3: High-throughput shared-parameter MAPPO with KL anchor and 3v3 rotational shaping.

  python -m learn.x3futsal_rl --bc runs/x3futsal_bc/tactical_v3/best_competitive.pt --out runs/x3futsal_rl/ppo_v1
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None

from learn.x3futsal_data import (
    MIRROR_X_ACTION,
    TACTICAL_DATA_VERSION,
    TACTICAL_ENTITY_DIM,
    TACTICAL_FEATURE_CONTRACT,
    TACTICAL_SELF_DIM,
    featurize_tactical,
    featurize_tactical_team,
)
from learn.x3futsal_eval import (
    FRAME_SKIP,
    KICKOFF_SAFETY_TICKS,
    STADIUM,
    ScriptedPolicy,
    TacticalPolicy,
    _aggregate,
    _player_states,
    _scripted_actions,
    _tactical_actions,
    action_to_sim,
    play_batch,
)
from learn.x3futsal_policy import TACTICAL_POLICY_VERSION, X3TacticalPolicy, decode_actions_np
from sim.physics import BatchSim, set_physics_threads
from sim.stadium import load_stadium
from tools.x3futsal_audit import stadium_identity


if nn is not None:
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
else:  # pragma: no cover
    X3CentralCritic = None


def _potential(sim: BatchSim, goal_x: float, half_w: float = 550.0) -> np.ndarray:
    """Zero-sum potential function phi(s) in [-1, 1] combining ball progression and 3v3 high-line support."""
    bx = sim.ball_pos[:, 0] / max(goal_x, 1.0)
    bvx = sim.ball_vel[:, 0] / 12.0
    # Premiar en potencial que el último hombre (cierre) acompañe la jugada fuera del área propia
    p_pos = sim.player_pos
    red_last_x = p_pos[:, :3, 0].min(axis=1) / max(half_w, 1.0)
    blue_last_x = (-p_pos[:, 3:, 0]).min(axis=1) / max(half_w, 1.0)
    high_line_diff = 0.5 * (red_last_x - blue_last_x)
    return np.clip(0.75 * bx + 0.15 * bvx + 0.10 * high_line_diff, -1.0, 1.0)


def _observe_team(sim: BatchSim, team: int, half_w: float, half_h: float, goal_x: float,
                  out_own: np.ndarray | None = None,
                  out_mates: np.ndarray | None = None,
                  out_rivals: np.ndarray | None = None):
    return featurize_tactical_team(
        sim, team, half_w, half_h, goal_x,
        out_own=out_own, out_mates=out_mates, out_rivals=out_rivals,
    )


def train_ppo(
    *,
    out: Path,
    bc_checkpoint: Path | None = None,
    updates: int = 200,
    envs: int = 64,
    horizon: int = 128,
    epochs: int = 4,
    minibatch_size: int = 4096,
    hidden: int = 192,
    entity_hidden: int = 96,
    lr: float = 3e-4,
    kl_coef: float = 0.15,
    shaping_coef: float = 0.08,
    forfeit_penalty: float = 0.15,
    deep_defender_penalty: float = 0.035,
    pass_reward: float = 0.06,
    press_recovery_reward: float = 0.025,
    rotation_reward: float = 0.015,
    selfplay_frac: float = 0.25,
    open_play_reset_frac: float = 0.25,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    eval_every: int = 20,
    eval_matches: int = 4,
    seed: int = 20261028,
    threads: int = 4,
    device_name: str = "cpu",
) -> dict:
    if torch is None:
        raise RuntimeError("torch es necesario para ejecutar learn.x3futsal_rl")
    set_physics_threads(threads)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device_name)
    stadium = load_stadium(STADIUM)
    half_w, half_h, goal_x = stadium.field_half_w, stadium.field_half_h, stadium.goal_x
    goal_hh = stadium.goal_half_height
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
    open_init = rng.random(envs) < open_play_reset_frac
    if np.any(open_init):
        sim.reset_open_play_tactical(np.flatnonzero(open_init))

    tactical_opp = TacticalPolicy(stadium)
    scripted_opp = ScriptedPolicy(stadium)
    kickoff_ticks = np.zeros(envs, dtype=np.int64)
    box_stay_ticks = np.zeros((envs, 3), dtype=np.float32)
    owner = np.full(envs, -1, dtype=np.int64)
    pending_passer = np.full(envs, -1, dtype=np.int64)
    prev_last_man = np.argmin(sim.player_pos[:, :3, 0], axis=1)

    # Pre-reservar buffers de rollout para eliminar asignaciones dinámicas y concatenaciones por paso
    n_Transition_actors = envs * 3
    obs_own = np.empty((horizon, n_Transition_actors, TACTICAL_SELF_DIM), dtype=np.float32)
    obs_mates = np.empty((horizon, n_Transition_actors, 2, TACTICAL_ENTITY_DIM), dtype=np.float32)
    obs_rivals = np.empty((horizon, n_Transition_actors, 3, TACTICAL_ENTITY_DIM), dtype=np.float32)
    blue_own_buf = np.empty((n_Transition_actors, TACTICAL_SELF_DIM), dtype=np.float32)
    blue_mates_buf = np.empty((n_Transition_actors, 2, TACTICAL_ENTITY_DIM), dtype=np.float32)
    blue_rivals_buf = np.empty((n_Transition_actors, 3, TACTICAL_ENTITY_DIM), dtype=np.float32)

    act_buf = np.empty((horizon, envs, 3), dtype=np.int64)
    logp_buf = np.empty((horizon, envs, 3), dtype=np.float32)
    val_buf = np.empty((horizon, envs), dtype=np.float32)
    rew_buf = np.empty((horizon, envs), dtype=np.float32)
    done_buf = np.empty((horizon, envs), dtype=bool)
    native = np.zeros((envs, 6), dtype=np.int64)

    # División concurrente de rivales por entorno: Táctico rotativo + Self-Play + Scripted con jitter
    n_sp = int(round(envs * float(np.clip(selfplay_frac, 0.0, 0.5))))
    n_scr = max(1, int(round(envs * 0.20))) if envs >= 4 else 0
    n_tac = max(0, envs - n_sp - n_scr)
    env_opp_mode = np.zeros(envs, dtype=np.int64)  # 0: tactical, 1: scripted, 2: self-play
    if n_scr > 0:
        env_opp_mode[n_tac:n_tac + n_scr] = 1
    if n_sp > 0:
        env_opp_mode[n_tac + n_scr:] = 2

    out.mkdir(parents=True, exist_ok=True)
    log = {"updates": [], "bestCompetitiveIndex": None}
    best_index = None
    t0 = time.time()
    infer_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    for update in range(1, updates + 1):
        # Rotar asignaciones de rival entre entornos en cada update
        env_opp_mode = np.roll(env_opp_mode, shift=max(1, envs // 4))
        sp_mask = env_opp_mode == 2
        scr_mask = env_opp_mode == 1

        for step_idx in range(horizon):
            own, mates, rivals = featurize_tactical_team(
                sim, 0, half_w, half_h, goal_x,
                out_own=obs_own[step_idx],
                out_mates=obs_mates[step_idx],
                out_rivals=obs_rivals[step_idx],
            )
            own_t = torch.from_numpy(own).to(device)
            mates_t = torch.from_numpy(mates).to(device)
            rivals_t = torch.from_numpy(rivals).to(device)
            with infer_ctx():
                logits = actor(own_t, mates_t, rivals_t)
                dist = torch.distributions.Categorical(logits=logits)
                actions_red = dist.sample()
                logp = dist.log_prob(actions_red)
                # Evaluar el crítico centralizado una sola vez por equipo/entorno (3x más rápido)
                values = critic(own_t[0::3], mates_t[0::3], rivals_t[0::3])

            red_np = actions_red.cpu().numpy().reshape(envs, 3)

            # Acciones del equipo azul: mezcla de Táctico rotativo, Scripted con jitter y Self-Play
            blue_np = _tactical_actions(sim, 1, half_w, half_h, goal_x, goal_hh, rng=rng)
            if np.any(scr_mask):
                scr_acts = _scripted_actions(sim, 1, half_w, half_h, rng=rng)
                blue_np[scr_mask] = scr_acts[scr_mask]
            if np.any(sp_mask):
                b_own, b_mates, b_rivals = featurize_tactical_team(
                    sim, 1, half_w, half_h, goal_x,
                    out_own=blue_own_buf, out_mates=blue_mates_buf, out_rivals=blue_rivals_buf,
                )
                with infer_ctx():
                    b_logits = actor(
                        torch.from_numpy(b_own).to(device),
                        torch.from_numpy(b_mates).to(device),
                        torch.from_numpy(b_rivals).to(device),
                    ).cpu().numpy()
                b_decoded = decode_actions_np(
                    b_logits,
                    kick_cancel=sim.kick_cancel[:, 3:].reshape(-1),
                    greedy=False,
                    temperature=0.9,
                    rng=rng,
                ).reshape(envs, 3)
                blue_np[sp_mask] = b_decoded[sp_mask]

            native[:, :3] = red_np
            native[:, 3:] = MIRROR_X_ACTION[blue_np]

            phi_before = _potential(sim, goal_x, half_w)
            step_reward = np.zeros(envs, dtype=np.float32)
            step_done = np.zeros(envs, dtype=bool)

            # 1. Penalización escalonada contra el anclaje en el área propia (solo jugadores rojos :3)
            red_px = sim.player_pos[:, :3, 0]
            ball_advanced = sim.ball_pos[:, 0] > (-half_w + 210.0)
            in_own_box = (red_px < (-half_w + 125.0)) & ball_advanced[:, None] & (~sim.kickoff[:, None])
            box_stay_ticks = np.where(in_own_box, np.minimum(box_stay_ticks + 1.0, 12.0), 0.0)
            if deep_defender_penalty > 0.0 and np.any(in_own_box):
                escalation = 1.0 + 0.15 * box_stay_ticks
                step_reward -= deep_defender_penalty * (in_own_box * escalation).sum(axis=1).astype(np.float32)

            # 2. Recompensa por rotación dinámica del último hombre en juego abierto
            cur_last_man = np.argmin(red_px, axis=1)
            rotated = (~sim.kickoff) & (cur_last_man != prev_last_man) & (sim.ball_vel[:, 0] > -0.5)
            if rotation_reward > 0.0 and np.any(rotated):
                step_reward[rotated] += rotation_reward
            prev_last_man = cur_last_man

            # 3. Ventana de 3 sub-frames con rearme de kick_cancel y eventos tácticos vectorizados
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
                    kicked_envs = np.flatnonzero(sim.kicked.any(axis=1))
                    for m in kicked_envs:
                        ks = np.flatnonzero(sim.kicked[m])
                        pending_passer[m] = int(ks[-1])

                if np.any(sim.touch):
                    touched_envs = np.flatnonzero(sim.touch.any(axis=1))
                    for m in touched_envs:
                        ts = np.flatnonzero(sim.touch[m])
                        p = int(ts[-1])
                        t_idx = p // 3
                        if owner[m] >= 0 and (owner[m] // 3) != t_idx:
                            # Recuperación de pelota (presión efectiva en mitad/campo rival)
                            if t_idx == 0 and sim.ball_pos[m, 0] > -120.0:
                                step_reward[m] += press_recovery_reward
                            pending_passer[m] = -1
                        elif (pending_passer[m] >= 0 and (pending_passer[m] // 3) == t_idx
                              and pending_passer[m] != p):
                            # Pase completado entre compañeros
                            if t_idx == 0:
                                step_reward[m] += pass_reward
                            else:
                                step_reward[m] -= 0.5 * pass_reward
                            pending_passer[m] = -1
                        owner[m] = p

                scored_envs = np.flatnonzero(sim.goal != 0)
                if len(scored_envs):
                    for m in scored_envs:
                        g = int(sim.goal[m])
                        step_reward[m] += 1.0 if g > 0 else -1.0
                        step_done[m] = True
                        conceding = 1 if g > 0 else 0
                        if rng.random() < open_play_reset_frac:
                            sim.reset_open_play_tactical([m])
                        else:
                            sim.reset_kickoff([m], kickoff_team=conceding)
                        kickoff_ticks[m] = 0
                        box_stay_ticks[m] = 0.0
                        owner[m] = -1
                        pending_passer[m] = -1

                no_goal = sim.goal == 0
                ko_mask = no_goal & sim.kickoff
                kickoff_ticks[ko_mask] += 1
                kickoff_ticks[no_goal & (~sim.kickoff)] = 0
                expired = np.flatnonzero(ko_mask & (kickoff_ticks >= KICKOFF_SAFETY_TICKS))
                if len(expired):
                    for m in expired:
                        ko_team = int(sim.kickoff_team[m])
                        step_reward[m] += -forfeit_penalty if ko_team == 0 else forfeit_penalty
                        sim.kickoff[m] = False
                        sim.mask[m, sim.first_player:] = sim.base_mask[sim.first_player:]
                        kickoff_ticks[m] = 0

            phi_after = np.where(step_done, 0.0, _potential(sim, goal_x, half_w))
            step_reward += shaping_coef * (gamma * phi_after - phi_before).astype(np.float32)

            act_buf[step_idx] = red_np
            logp_buf[step_idx] = logp.cpu().numpy().reshape(envs, 3)
            val_buf[step_idx] = values.cpu().numpy()
            rew_buf[step_idx] = step_reward
            done_buf[step_idx] = step_done

        with infer_ctx():
            own, mates, rivals = featurize_tactical_team(
                sim, 0, half_w, half_h, goal_x,
                out_own=blue_own_buf, out_mates=blue_mates_buf, out_rivals=blue_rivals_buf,
            )
            next_val = critic(
                torch.from_numpy(own[0::3]).to(device),
                torch.from_numpy(mates[0::3]).to(device),
                torch.from_numpy(rivals[0::3]).to(device),
            ).cpu().numpy()

        # Cálculo vectorizado de GAE
        adv = np.zeros((horizon, envs), dtype=np.float32)
        gae = np.zeros(envs, dtype=np.float32)
        for t in range(horizon - 1, -1, -1):
            nonterminal = (~done_buf[t]).astype(np.float32)
            v_next = next_val if t == horizon - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * v_next * nonterminal - val_buf[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + val_buf

        total_actor_samples = horizon * n_Transition_actors
        flat_own = torch.from_numpy(obs_own.reshape(total_actor_samples, TACTICAL_SELF_DIM)).to(device)
        flat_mates = torch.from_numpy(obs_mates.reshape(total_actor_samples, 2, TACTICAL_ENTITY_DIM)).to(device)
        flat_rivals = torch.from_numpy(obs_rivals.reshape(total_actor_samples, 3, TACTICAL_ENTITY_DIM)).to(device)
        flat_act = torch.from_numpy(act_buf.reshape(-1)).to(device)
        flat_old_logp = torch.from_numpy(logp_buf.reshape(-1)).to(device)
        flat_adv = torch.from_numpy(np.repeat(adv, 3, axis=1).reshape(-1)).to(device)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-6)

        # Crítico centralizado evaluado sobre estados de equipo (1 por env-step = 3x menos cómputo)
        total_team_samples = horizon * envs
        team_own = flat_own[0::3]
        team_mates = flat_mates[0::3]
        team_rivals = flat_rivals[0::3]
        flat_ret_team = torch.from_numpy(returns.reshape(-1)).to(device)

        # Pre-calcular probabilidades del ancla congelada UNA sola vez antes de las épocas PPO (4x menos cómputo)
        if kl_coef > 0.0:
            with infer_ctx():
                ref_probs_parts = []
                mb = max(1024, int(minibatch_size))
                for start in range(0, total_actor_samples, mb):
                    stop = start + mb
                    ref_logits = anchor(flat_own[start:stop], flat_mates[start:stop], flat_rivals[start:stop])
                    ref_probs_parts.append(F.softmax(ref_logits, dim=-1))
                flat_ref_probs = torch.cat(ref_probs_parts, dim=0)
        else:
            flat_ref_probs = None

        mb_actor = min(total_actor_samples, max(1024, (int(minibatch_size) // 3) * 3))
        mb_team = max(1, mb_actor // 3)

        for _ in range(epochs):
            team_perm = torch.from_numpy(rng.permutation(total_team_samples)).to(device)
            actor_perm = (team_perm.unsqueeze(1) * 3 + torch.arange(3, device=device)).reshape(-1)
            for t_start in range(0, total_team_samples, mb_team):
                t_stop = min(total_team_samples, t_start + mb_team)
                a_start, a_stop = t_start * 3, t_stop * 3
                a_idx = actor_perm[a_start:a_stop]
                t_idx = team_perm[t_start:t_stop]

                b_own = flat_own[a_idx]
                b_mates = flat_mates[a_idx]
                b_rivals = flat_rivals[a_idx]
                logits = actor(b_own, b_mates, b_rivals)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(flat_act[a_idx])
                ratio = torch.exp(new_logp - flat_old_logp[a_idx])
                b_adv = flat_adv[a_idx]
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

                if flat_ref_probs is not None:
                    kl_loss = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        flat_ref_probs[a_idx],
                        reduction="batchmean",
                    )
                else:
                    kl_loss = torch.tensor(0.0, device=device)

                pred_val = critic(team_own[t_idx], team_mates[t_idx], team_rivals[t_idx])
                val_loss = F.mse_loss(pred_val, flat_ret_team[t_idx])
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
            pol = ModelPolicy(out / "last.pt", device, assist_mode="kickoff")
            games_scr, games_tac = [], []
            for red, blue, role in ((pol, scripted_opp, 0), (scripted_opp, pol, 1)):
                side_games = play_batch(red, blue, matches=eval_matches, minutes=0.5, seed=seed + update)
                games_scr.extend({**g, "firstPolicyTeam": role} for g in side_games)
            for red, blue, role in ((pol, tactical_opp, 0), (tactical_opp, pol, 1)):
                side_games = play_batch(red, blue, matches=max(2, eval_matches // 2), minutes=0.5, seed=seed + update + 777)
                games_tac.extend({**g, "firstPolicyTeam": role} for g in side_games)
            agg = _aggregate(games_scr, 0)
            agg_tac = _aggregate(games_tac, 0)
            box_camp = agg.get("ownBoxCampingFracPerMatch", 0.0)
            rotations = agg.get("roleRotationsPerMatch", 0.0)
            comp_idx = (
                0.55 * agg["goalDifferencePerMatch"]
                + 0.45 * agg_tac["goalDifferencePerMatch"]
                + 0.05 * agg["shotsPerMatch"]
                + 0.04 * agg["completedPassesPerMatch"]
                + 0.01 * min(rotations, 25.0)
                - 0.40 * box_camp
                - 0.50 * agg["safetyKickoffsPerMatch"]
            )
            elapsed_s = max(1e-3, time.time() - t0)
            samples_processed = update * envs * horizon
            agg.update(
                update=update,
                vsTacticalGoalDiff=agg_tac["goalDifferencePerMatch"],
                vsTacticalWinRate=agg_tac["winRate"],
                competitiveIndex=float(comp_idx),
                samplesProcessed=int(samples_processed),
                samplesPerSecond=round(samples_processed / elapsed_s, 1),
                seconds=round(elapsed_s, 1),
            )
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
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--entity-hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-coef", type=float, default=0.15)
    parser.add_argument("--deep-defender-penalty", type=float, default=0.035)
    parser.add_argument("--pass-reward", type=float, default=0.06)
    parser.add_argument("--selfplay-frac", type=float, default=0.25)
    parser.add_argument("--open-play-reset-frac", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-matches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_ppo(
        out=args.out,
        bc_checkpoint=args.bc,
        updates=args.updates,
        envs=args.envs,
        horizon=args.horizon,
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        hidden=args.hidden,
        entity_hidden=args.entity_hidden,
        lr=args.lr,
        kl_coef=args.kl_coef,
        deep_defender_penalty=args.deep_defender_penalty,
        pass_reward=args.pass_reward,
        selfplay_frac=args.selfplay_frac,
        open_play_reset_frac=args.open_play_reset_frac,
        eval_every=args.eval_every,
        eval_matches=args.eval_matches,
        seed=args.seed,
        threads=args.threads,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()