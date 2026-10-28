"""Preflight del banco cooperativo: validación de estados y baseline de recepción.

Lee SÓLO grabaciones train. No construye optimizadores ni ejecuta aprendizaje.
STILL y la política original congelada reciben los mismos resets de etapa 0,
incluyendo velocidad, vuelo y comandos humanos que ya estaban en cola. Las
entradas futuras inmediatas inicializan esa cola retardada; no son decisiones
nuevas del modelo. Un éxito STILL mide cuánto resuelve la herencia del estado.
Un segundo baseline STILL en etapa 1 exige cero pases propios y cero éxitos:
verifica que la cola heredada no entregue gratuitamente el premio del pase.

Los lotes tienen 16 filas y reserva cooperativa 0.5. Cada brazo termina exactamente
los mismos ejercicios antes de generar el siguiente lote, evitando que su distinta
duración cambie los ejemplos. La fracción de decisiones vivas describe estos lotes,
no estima la mezcla del entrenamiento continuo. Valida también resets de etapas
0..4 y finitud de la entrada real del crítico. No prueba fuerza ni rendimiento test.

  python -m learn.x4_coop_preflight --bc bc.pt --init original.pt \
      --recordings 8 --episodes 64 --device cpu --threads 2 --out preflight.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from env.rs4z import contract as C, obs_v3
from learn import x4_data as XD
from learn.x4_coop_curriculum import CoopBank, CoopCurriculum, STAGES, TEAM
from learn.x4_eval import ModelPolicy, Policy
from learn.x4_pass_macro import pass_table
from learn.x4_ppo import Arena, CRITIC_IN, Trainer, parse_args as ppo_args

ROOT = Path(__file__).resolve().parent.parent
MAP = "sanguchito_rs_x4"
ENVS = 16
FRACTION = 0.5


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def config_for(a):
    # Reuse trainer parsing, with the cooperative pilot's D=8..11. Parsing has
    # no filesystem side effects and does not construct a Trainer.
    cfg = ppo_args(["--bc", a.bc, "--out", a.out, "--envs", str(ENVS),
                    "--bank-recordings", "0", "--human-starts", "0", "--pool-frac", "0",
                    "--eval-every", "0", "--shaping-kind", "none", "--auto-pass-arm-update", "0",
                    "--delays", "8:1,9:1,10:1,11:1"])
    cfg.match_minutes = (3.0, 3.0)
    return cfg


def make_arena(bank, cfg, stage, seed):
    rng = np.random.default_rng(seed)
    arena = Arena(MAP, ENVS, cfg, rng, bank=None, pool_size=0)
    arena.coop = CoopCurriculum(bank, ENVS, rng, frac=FRACTION)
    arena.coop.stage = stage
    arena.coop._reserve_rows()
    arena.new_match(np.arange(ENVS))
    return arena


def critic_features(arena, obs):
    # Use the real concatenation/padding interface, without creating a model,
    # optimizer, training log, checkpoint or evaluation machinery.
    from types import SimpleNamespace
    stub = SimpleNamespace(lam_eff=lambda ar: ar.lam.astype(np.float32), bc_slot=-1, epv=None)
    return Trainer.critic_input(stub, arena, obs, np.zeros(arena.N))


def inspect_reset(arena):
    env, cur = arena.env, arena.coop
    rows = np.flatnonzero(cur.active)
    minimum_player_distance = minimum_player_gap = minimum_ball_gap = None
    player_overlaps = ball_overlaps = 0
    for row in rows:
        ids = np.flatnonzero(env.active[row])
        pos = env.player_pos[row, ids]
        radii = env.radius[row, env.fp + ids]
        if len(ids) >= 2:
            distance = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            take = np.triu_indices(len(ids), 1)
            d, gap = distance[take], (distance - radii[:, None] - radii[None])[take]
            value, separation = float(d.min()), float(gap.min())
            minimum_player_distance = value if minimum_player_distance is None else min(minimum_player_distance, value)
            minimum_player_gap = separation if minimum_player_gap is None else min(minimum_player_gap, separation)
            player_overlaps += int((gap < -0.1).sum())
        gap = np.linalg.norm(pos - env.ball_pos[row], axis=1) - radii - env.radius[row, 0]
        value = float(gap.min())
        minimum_ball_gap = value if minimum_ball_gap is None else min(minimum_ball_gap, value)
        ball_overlaps += int((gap < -0.1).sum())
    obs = obs_v3.observe(env, delay=arena.delay_obs)
    critic = critic_features(arena, obs)
    mask = arena.learner & env.active & cur.active[:, None]
    return dict(active_exercises=len(rows), reserved_exercises=int(cur.exercise_rows.sum()),
                active_fraction=len(rows) / arena.N, min_player_distance_px=minimum_player_distance,
                min_player_gap_px=minimum_player_gap, min_ball_gap_px=minimum_ball_gap,
                overlapping_player_pairs=player_overlaps, overlapping_ball_pairs=ball_overlaps,
                geometry_tolerance_px=0.1, actor_control_slots=int(mask.sum()),
                finite_observation=bool(np.isfinite(obs).all()),
                finite_critic_input=bool(np.isfinite(critic).all()),
                critic_shape=list(critic.shape), critic_expected_dimension=CRITIC_IN,
                active_delays=env.delay[rows].tolist())


def policy_for(pol, seed):
    if pol.model is None:
        return pol
    generator = torch.Generator(device=pol.device).manual_seed(seed)
    return ModelPolicy(pol.model, device=pol.device, name=pol.name, greedy=False, generator=generator)


@torch.no_grad()
def run_arm(arena, policy, selected, seed):
    env, cur = arena.env, arena.coop
    actor = policy_for(policy, seed)
    rng = np.random.default_rng(seed)
    selected = np.asarray(selected, int)
    chosen = np.zeros(ENVS, bool)
    chosen[selected] = True
    cur.active &= chosen
    done = ~chosen
    records = {}
    stats = dict(actor_control_slot_samples=0, actor_forward_samples=0, live_exercise_decisions=0,
                 all_row_decisions=0, controlled_ball_decisions=0, macro_starts=0, macro_busy_samples=0,
                 macro_kicks=0, human_receptions=0, receptions_before_new_commands=0,
                 policy_passes=0,
                 finite_observation=True, finite_critic_input=True, curriculum_rewards_zero_sum=True)
    limit = int(np.ceil(cur.timeout_s * 60 / env.frame_skip)) + 2
    for step in range(limit):
        live = chosen & ~done
        if not live.any():
            break
        obs = obs_v3.observe(env, delay=arena.delay_obs)
        critic = critic_features(arena, obs)
        stats["finite_observation"] &= bool(np.isfinite(obs).all())
        stats["finite_critic_input"] &= bool(np.isfinite(critic).all())
        mask = arena.learner & env.active & live[:, None]
        count = int(mask.sum())
        stats["actor_control_slot_samples"] += count
        stats["actor_forward_samples"] += count if policy.model is not None else 0
        stats["live_exercise_decisions"] += int(live.sum())
        stats["all_row_decisions"] += ENVS
        distance = np.linalg.norm(env.player_pos - env.ball_pos[:, None], axis=-1)
        close = distance <= C.PLAYER_RADIUS + env.radius[:, 0, None] + 15.0
        stats["controlled_ball_decisions"] += int((close & mask).any(1).sum())
        actions = np.zeros((ENVS, 8), np.int64)
        if getattr(actor, "n_pass", 0):
            ready = bool(getattr(actor.model, "pass_ready", False))
            table = pass_table(env, ready_only=ready)
            actions[mask] = actor(obs[mask], rng, pass_mask=table[mask] >= 0)
            sampled = actions
            actions, busy = arena.macro.apply(sampled, table)
            stats["macro_starts"] += int((~busy & (sampled >= 18) & mask).sum())
            stats["macro_busy_samples"] += int((busy & mask).sum())
        else:
            actions[mask] = actor(obs[mask], rng)
        previous_received = cur.received.copy()
        cur.before_step(env)
        ev = env.step(actions)
        if getattr(actor, "n_pass", 0):
            stats["macro_kicks"] += int((arena.macro.running() & ev["kicked"] & mask).sum())
            arena.macro.after(ev["kicked"], np.flatnonzero(ev["goal"] != 0))
        reward, terminal, event_stats = cur.step(env, ev)
        stats["curriculum_rewards_zero_sum"] &= bool(np.allclose(reward.sum(1), 0))
        stats["human_receptions"] += int(event_stats.get("human_receptions", 0))
        stats["policy_passes"] += int(event_stats.get("policy_passes", 0))
        just_received = live & ~previous_received & cur.received
        for row in np.flatnonzero(just_received):
            # Conservative window: a receipt within D+one decision can mostly
            # reflect the inherited queue. It is a diagnostic, not attribution.
            receiver = int(cur.receiver[row])
            stats["receptions_before_new_commands"] += int(cur.age[row] <= env.delay[row, receiver] + env.frame_skip)
        for row in np.flatnonzero(live & terminal):
            team = int(cur.attack_team[row])
            records[int(row)] = dict(success=bool(reward[row, team] > 0),
                                     received=bool(cur.received[row]), duration_ticks=int(cur.age[row]),
                                     retained_ticks=int(cur.retained[row]),
                                     goal=int(ev["goal"][row]), out=bool(ev["out"][row]),
                                     timeout=bool(cur.age[row] >= int(cur.timeout_s * 60)),
                                     attacking_team=team, row=int(row))
        done |= terminal
    for row in selected:
        if int(row) not in records:
            records[int(row)] = dict(success=False, received=bool(cur.received[row]),
                                     duration_ticks=int(cur.age[row]), retained_ticks=int(cur.retained[row]),
                                     goal=0, out=False, timeout=False, attacking_team=int(cur.attack_team[row]),
                                     row=int(row), unfinished=True)
    return [records[int(row)] for row in selected], stats


def summarize_arm(records, stats):
    n = len(records)
    successes = sum(row["success"] for row in records)
    received = sum(row["received"] for row in records)
    live = stats["live_exercise_decisions"]
    return dict(episodes=n, successes=successes, success_rate=successes / n if n else None,
                received=received, reception_rate=received / n if n else None,
                control_survival_after_reception=successes / received if received else None,
                duration_seconds_mean=float(np.mean([r["duration_ticks"] / 60 for r in records])) if n else None,
                unfinished=sum(bool(r.get("unfinished")) for r in records),
                exercise_fraction_of_live_decisions=live / max(1, stats["all_row_decisions"]),
                controlled_ball_fraction=stats["controlled_ball_decisions"] / max(1, live),
                **stats, state_results=records)


def paired_difference(still, original, seed):
    values = np.array([int(b["success"]) - int(a["success"]) for a, b in zip(still, original)], float)
    if not len(values):
        return dict(original_minus_still=None, ic90=[None, None])
    rng = np.random.default_rng(seed)
    draw = values[rng.integers(len(values), size=(1000, len(values)))].mean(1)
    return dict(original_minus_still=float(values.mean()), ic90=np.percentile(draw, (5, 95)).tolist(),
                bootstrap_unit="paired exercise; repeated source examples are not independent humans")


def still_stage1(bank, cfg, episodes, seed):
    """No newly issued keys: inherited human queue must not buy the pass reward."""
    records, counters, batches = [], {}, []
    for batch in range(max(16, episodes * 2)):
        if len(records) >= episodes:
            break
        scenario_seed = seed + 100_000 + batch
        arena = make_arena(bank, cfg, 1, scenario_seed)
        clone = make_arena(bank, cfg, 1, scenario_seed)
        snap_a, snap_b = arena.env.snapshot(), clone.env.snapshot()
        identical = all(np.array_equal(snap_a[k], snap_b[k]) for k in snap_a)
        identical &= np.array_equal(arena.coop.receiver, clone.coop.receiver)
        if not identical:
            raise RuntimeError("stage-1 cloned preflight resets differ")
        selected = np.flatnonzero(arena.coop.active)[:episodes - len(records)]
        batches.append(dict(seed=scenario_seed, selected_rows=selected.tolist(), identical_states=bool(identical),
                            reset=inspect_reset(arena)))
        if not len(selected):
            continue
        rr, ss = run_arm(arena, Policy("still"), selected, scenario_seed)
        for row in rr:
            row["scenario_seed"] = scenario_seed
        records.extend(rr)
        for key, value in ss.items():
            if isinstance(value, bool):
                counters[key] = counters.get(key, True) and value
            else:
                counters[key] = counters.get(key, 0) + value
    if not records:
        raise RuntimeError("no usable stage-1 exercises in the selected training bank")
    summary = summarize_arm(records, counters)
    summary["no_free_reward_verified"] = bool(summary["successes"] == 0 and summary["policy_passes"] == 0
                                             and len(records) == episodes and summary["unfinished"] == 0)
    return summary, batches


def evaluate(bank, cfg, original, episodes=64, seed=7):
    stages = {}
    for stage, name in enumerate(STAGES):
        checks = [inspect_reset(make_arena(bank, cfg, stage, seed + 50_000 + j)) for j in range(4)]
        stages[name] = dict(stage=stage, resets=checks,
                            finite=all(r["finite_observation"] and r["finite_critic_input"] for r in checks),
                            active_exercises=sum(r["active_exercises"] for r in checks),
                            overlapping_player_pairs=sum(r["overlapping_player_pairs"] for r in checks),
                            overlapping_ball_pairs=sum(r["overlapping_ball_pairs"] for r in checks))
    records = {"still": [], "original": []}
    counters = {"still": {}, "original": {}}
    batches = []
    for batch in range(max(16, episodes * 2)):
        if len(records["still"]) >= episodes:
            break
        scenario_seed = seed + 1000 + batch
        a = make_arena(bank, cfg, 0, scenario_seed)
        b = make_arena(bank, cfg, 0, scenario_seed)
        snap_a, snap_b = a.env.snapshot(), b.env.snapshot()
        identical = all(np.array_equal(snap_a[k], snap_b[k]) for k in snap_a)
        identical &= np.array_equal(a.coop.receiver, b.coop.receiver)
        if not identical:
            raise RuntimeError("paired preflight resets differ")
        selected = np.flatnonzero(a.coop.active)[:episodes - len(records["still"])]
        check = inspect_reset(a)
        batches.append(dict(seed=scenario_seed, selected_rows=selected.tolist(), identical_states=bool(identical),
                            reset=check))
        if not len(selected):
            continue
        for name, arena, pol in (("still", a, Policy("still")), ("original", b, original)):
            rr, ss = run_arm(arena, pol, selected, scenario_seed)
            for row in rr:
                row["scenario_seed"] = scenario_seed
            records[name].extend(rr)
            for key, value in ss.items():
                if isinstance(value, bool):
                    counters[name][key] = counters[name].get(key, True) and value
                else:
                    counters[name][key] = counters[name].get(key, 0) + value
    if not records["still"]:
        raise RuntimeError("no usable stage-0 exercises in the selected training bank")
    baseline = {name: summarize_arm(records[name], counters[name]) for name in records}
    baseline1, batches1 = still_stage1(bank, cfg, episodes, seed)
    valid = (len(records["still"]) == episodes
             and all(not row.get("unfinished") for rr in records.values() for row in rr)
             and all(s["finite_observation"] and s["finite_critic_input"]
                     and s["curriculum_rewards_zero_sum"] for s in baseline.values())
             and baseline1["no_free_reward_verified"] and baseline1["finite_observation"]
             and baseline1["finite_critic_input"] and baseline1["curriculum_rewards_zero_sum"]
             and all(s["finite"] and s["active_exercises"] > 0
                     and s["overlapping_player_pairs"] == 0 and s["overlapping_ball_pairs"] == 0 for s in stages.values()))
    return dict(technical_preflight_passed=bool(valid), stages=stages, baseline_stage0=baseline,
                baseline_stage1_still=baseline1, stage1_batches=batches1,
                paired_success_difference=paired_difference(records["still"], records["original"], seed),
                batches=batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc", required=True)
    ap.add_argument("--init", required=True, help="política original congelada, incluyendo sus macros")
    ap.add_argument("--recordings", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if min(a.recordings, a.episodes, a.threads) < 1:
        ap.error("recordings/episodes/threads deben ser positivos")
    started = time.perf_counter()
    torch.set_num_threads(a.threads)
    # Numba and Torch are separate CPU pools. Avoid the host's unbounded default.
    import numba
    numba.set_num_threads(min(a.threads, numba.config.NUMBA_NUM_THREADS))
    cfg = config_for(a)
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", "train", None, (MAP,))[:a.recordings]
    data = XD.load(names, maps=(MAP,))
    bank = CoopBank(data, max_records=12000, max_candidates=20000)
    original = Policy(a.init, device=a.device)
    for parameter in original.model.parameters():
        parameter.requires_grad_(False)
    report = dict(version="x4-coop-preflight-1", learning=False, split="train", map=MAP,
                  seed=a.seed, envs=ENVS, reserved_exercise_fraction=FRACTION,
                  requested_episodes_per_baseline=a.episodes,
                  requested_recordings=a.recordings, train_recordings=data.names,
                  valid_source_ticks=int(len(data.valid)), source_ticks=int(data.ticks), records_mined=len(bank),
                  latency_distribution=dict(values=cfg.delay_values.tolist(), weights=cfg.delay_weights.tolist()),
                  observation_delay_mode=cfg.delay_feature, observation_fixed_delay=cfg.delay_feature_fixed,
                  bc=fingerprint(a.bc), original=fingerprint(a.init), original_n_pass=int(original.n_pass),
                  original_parameters_frozen=all(not p.requires_grad for p in original.model.parameters()),
                  runtime=dict(torch=torch.__version__, numpy=np.__version__, device=a.device, threads=a.threads),
                  limitations=["Etapa0 hereda vuelo, velocidad y cola humana; éxito STILL cuantifica esa herencia.",
                               "La cola se reconstruye con entradas futuras inmediatas del registro humano.",
                               "Etapa0 no tiene defensores: el baseline no mide fuerza ni cooperación en 4v4.",
                               "Los lotes pareados no estiman la mezcla temporal de entrenamiento continuo.",
                               "La validación geométrica tolera 0.1 px de solape numérico; no certifica toda la física.",
                               "macro_kicks cuenta sólo macros todavía activos; no atribuye comandos tardíos tras cancelar.",
                               "Bootstrap por ejercicio; puede repetir fuente humana. No es intervalo poblacional humano."])
    report.update(evaluate(bank, cfg, original, a.episodes, a.seed))
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(records_mined=len(bank), technical_preflight_passed=report["technical_preflight_passed"],
                          still=report["baseline_stage0"]["still"]["success_rate"],
                          original=report["baseline_stage0"]["original"]["success_rate"],
                          stage1_still_successes=report["baseline_stage1_still"]["successes"],
                          stage1_still_policy_passes=report["baseline_stage1_still"]["policy_passes"],
                          stage1_no_free_reward=report["baseline_stage1_still"]["no_free_reward_verified"],
                          elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
