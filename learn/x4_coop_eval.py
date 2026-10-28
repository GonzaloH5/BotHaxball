"""Evaluación reproducible de juego colectivo, separada del entrenamiento.

--policy es el candidato; --reference se puede repetir para control/original: también
se evalúan y entran al pool rival junto a BC y scripted. Cada combinación juega
--matches partidos POR LADO con D=8..11. No hay recompensa ni requisito de pase.
Las métricas de partido usan x4_pass_chain con el equipo correcto y bootstrap por
partido. Las sondas 2v1/3v2 usan únicamente estados de DEV/TEST de Sanguchito,
compartidos entre checkpoints y rivales, con acciones pendientes humanas reconstruidas.

Las sondas alteran el plantel y no prueban nivel humano ni fuerza 4v4. Su referencia
humana conserva los 8 jugadores y sólo describe lo ocurrido desde el mismo estado.
La retención de una recepción se observa durante 2 s; horizontes incompletos se
censuran. Un gol propio antes de 2 s cuenta como éxito. "Pase útil" es diagnóstico:
retención + avance >=60 px, o salida de presión; un gol sin pase también es éxito.
El bootstrap de las sondas agrupa por grabación, no trata ticks cercanos como humanos
independientes. Usar --split dev para decidir el piloto; reservar --split test
(por defecto) para evaluación final. Train no está permitido. Semillas, estados,
latencias y versiones se guardan en el JSON.

  python -m learn.x4_coop_eval --policy nuevo.pt --reference control.pt \
      --reference original.pt --bc bc.pt --matches 8 --probe-states 64 \
      --recordings 40 --minutes 2 --device cuda --threads 8 --out coop.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from env.rs4z import obs_v3
from env.rs4z.core import RS4ZEnv
from learn import x4_data as XD
from learn.x4_eval import ModelPolicy, Policy, play
from learn.x4_pass_macro import MacroState, pass_table
from learn.x4_probes import pick_states
from tools import x4_metrics as XM
from tools import x4_pass_chain as PC

ROOT = Path(__file__).resolve().parent.parent
MAP = "sanguchito_rs_x4"
DELAYS = (8, 9, 10, 11)
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])
RET_STEPS = 40                 # 2 s con frame_skip=3
NATURAL_METRICS = ("retencion_tras_recibir", "retencion_bajo_presion",
                   "perdida_sin_pase_bajo_presion", "progreso_tras_pase_avance",
                   "salidas_presion_por_min", "goles_por_min", "tiros_por_min", "pases_por_posesion",
                   "precision_pase", "desmarques_utiles_frac", "reofrece_tras_pasar_frac")


def seeded_policy(policy, seed):
    """Cada modelo tiene su propio RNG: evaluar otro modelo no altera su muestreo."""
    if getattr(policy, "model", None) is None:
        return policy
    generator = torch.Generator(device=policy.device).manual_seed(int(seed))
    return ModelPolicy(policy.model, device=policy.device, name=policy.name,
                       greedy=policy.greedy, generator=generator)


def mean_ci(values, seed=0, n=300):
    x = np.asarray(values, float)
    if len(x) == 0:
        return dict(valor=None, ic90=[None, None], n=0)
    rng = np.random.default_rng(seed)
    draw = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return dict(valor=round(float(x.mean()), 4),
                ic90=[round(float(v), 4) for v in np.percentile(draw, (5, 95))], n=len(x))


def match_summary(goals, units, finished, safety, seed):
    g = np.asarray(goals, int).reshape(-1, 2)
    diff = g[:, 0] - g[:, 1]
    result = dict(matches=len(g), finished=int(finished), wins=int((diff > 0).sum()),
                  draws=int((diff == 0).sum()), losses=int((diff < 0).sum()),
                  goals_for=int(g[:, 0].sum()), goals_against=int(g[:, 1].sum()),
                  point_fraction=mean_ci((diff > 0) + 0.5 * (diff == 0), seed),
                  goal_difference=mean_ci(diff, seed), safety_kickoff=int(safety))
    result["valid_for_strength"] = finished == len(g)
    if finished != len(g):
        # play informa cuántos terminaron, pero no entrega la máscara por fila.
        # No adjudicar marcadores parciales como victorias/empates certificados.
        result["scores_at_cap"] = g.tolist()
        result["wins"] = result["draws"] = result["losses"] = None
        result["point_fraction"] = dict(valor=None, ic90=[None, None], n=0)
        result["goal_difference"] = dict(valor=None, ic90=[None, None], n=0)
    if units:
        metrics = PC.bootstrap(units, n=300, seed=seed)
        result["natural_metrics"] = {k: metrics[k] for k in NATURAL_METRICS}
        result["support"] = PC.support(units)
    return result


def evaluate_matches(candidate, opponent, *, matches, minutes, seed):
    goals, units, finished, safety = [], [], 0, 0
    for side in (0, 1):
        own = seeded_policy(candidate, seed + 2 * side)
        rival = seeded_policy(opponent, seed + 2 * side + 1)
        red, blue = (own, rival) if side == 0 else (rival, own)
        r = play(red, blue, map_name=MAP, matches=matches, minutes=minutes,
                 delays=DELAYS, seed=seed + side, record=matches)
        goals.extend(r["goals"] if side == 0 else r["goals"][:, ::-1])
        units.extend(PC.unit_counts([ep], team=side) for ep in r["episodes"])
        finished += r["finished"]
        safety += int(r["safety_kickoff"].sum())
    return match_summary(goals, units, finished, safety, seed)


def select_states(data, n_att, n_def, n, seed=0, min_spacing=300):
    """Muestra equilibrada por lado; evita ventanas solapadas de la misma grabación.

    pick_states comparte las reglas de la sonda existente. Se exige además contacto
    inicial limpio: inferir posesión sólo por cercanía de 30 px puede inventarla.
    Si faltan estados/lados no se duplican: se declara el tamaño efectivo.
    """
    if n <= 0:
        return []
    candidates = pick_states(data, n_att, n_def, max(16 * n, n), np.random.default_rng(seed))
    quota = [(n + 1) // 2, n // 2]
    count, selected, ticks = [0, 0], [], {}
    for t, team, carrier, active in candidates:
        if count[team] >= quota[team]:
            continue
        dist = np.linalg.norm(data.pos[t] - data.ball[t, :2], axis=1)
        reach = XM.PLAYER_R + data.ball_r[t] + XM.CONTACT_EPS
        if dist[carrier] > reach or np.any(dist[TEAM != team] <= reach):
            continue
        rec = int(data.rec_of_tick[t])
        if any(abs(t - other) < min_spacing for other in ticks.get(rec, ())):
            continue
        ticks.setdefault(rec, []).append(t)
        selected.append((t, team, carrier, active))
        count[team] += 1
        if len(selected) == n:
            break
    return selected


def restore_states(env, data, states, seed):
    """Restaura posiciones, física y cola; el historial son acciones del MUNDO.

    El actor decide en t; su etiqueta retardada sería t+D+1. Como en
    obs_v3.build_samples, hist[h] toma la entrada t+D+1-3*(h+1).
    Los futuros inmediatos sólo inicializan la cola observada, no entrenan al actor.
    """
    rng = np.random.default_rng(seed)
    delays = rng.choice(DELAYS, size=(len(states), 8))
    active = np.stack([s[3] for s in states])
    env.start_match(np.arange(len(states)), active=active, delay=delays, match_ticks=10 ** 9,
                    ball_radius=np.array([data.ball_r[s[0]] for s in states]),
                    kick_strength=np.array([data.kstr[s[0]] for s in states]))
    for i, (t, team, _, _) in enumerate(states):
        held = ((data.inp[t] & 16) != 0) & ~data.kicking[t]
        env.place(i, ball_pos=data.ball[t, :2], ball_vel=data.ball[t, 2:4],
                  player_pos=data.pos[t], player_vel=data.vel[t], kick_held=held,
                  last_touch=team, mass_phase=0 if abs(data.mass[t] - 0.5) < 1e-6 else 1)
        env._s_act[i] = XD.move_of_input(data.inp[t]) + 9 * ((data.inp[t] & 16) != 0)
        for p in range(8):
            for h in range(env.H):
                u = t + int(delays[i, p]) + 1 - 3 * (h + 1)
                # Una decisión pasada en cola afecta un bloque futuro. No tomar
                # entradas de otro plantel/grabación si el extremo está cortado.
                if (0 <= u < data.ticks and data.rec_of_tick[u] == data.rec_of_tick[t]
                        and np.all(data.pid[u] == data.pid[t])):
                    mv = int(XD.move_of_input(np.array([data.inp[u, p]]))[0])
                    env.act_hist[i, p, h] = mv + 9 * bool(data.inp[u, p] & 16)
    return delays


def choose_actions(env, candidate, opponent, teams, rng, macro):
    """Interfaz Policy/ModelPolicy, incluyendo pase listo y compromiso MacroState."""
    obs = obs_v3.observe(env)
    actions = np.zeros((env.N, 8), np.int64)
    table = np.full((env.N, 8, 3), -1, np.int64) if macro is not None else None
    for pol, own in ((candidate, True), (opponent, False)):
        mask = TEAM[None] == teams[:, None]
        if not own:
            mask = ~mask
        mask &= env.active
        if not mask.any():
            continue
        kw = {}
        if getattr(pol, "n_pass", 0):
            ready = bool(getattr(getattr(pol, "model", None), "pass_ready", False))
            tab = pass_table(env, ready_only=ready)
            table[mask] = tab[mask]
            kw["pass_mask"] = tab[mask] >= 0
        actions[mask] = pol(obs[mask], rng, **kw)
    if macro is not None:
        actions, _ = macro.apply(actions, table)
    return actions


def probe_row(ep, team, carrier, terminal="horizon"):
    """Resultado por estado con censura explícita de recepción, sin gate de pase."""
    owner, _ = XM._touch_owner(ep)
    T = len(ep.ball)
    sign = 1 if team == 0 else -1
    clean_loss = np.flatnonzero((owner >= 0) & (TEAM[np.maximum(owner, 0)] != team))
    loss = int(clean_loss[0]) if len(clean_loss) else T
    if terminal in ("out", "goal_against", "loss"):
        loss = min(loss, T - 1)
    own_goal = terminal == "goal"
    last, release = int(carrier), 0
    passes = []
    for t in range(min(T, loss)):
        q = int(owner[t])
        if q < 0:
            continue
        if q != last and TEAM[q] == team:
            if np.linalg.norm(ep.ball[t, :2] - ep.ball[release, :2]) >= XM.MIN_PASS:
                passes.append((release, t, last, q))
        last, release = q, t
    retained = eligible = useful = censored = 0
    for r, c, passer, receiver in passes:
        end = c + RET_STEPS
        failed = loss < T and loss <= end
        observed = failed or end < T or own_goal
        eligible += observed
        censored += not observed
        ok = observed and not failed
        retained += ok
        opponents = np.flatnonzero(TEAM != team)
        p0 = np.linalg.norm(ep.pos[r, opponents] - ep.pos[r, passer], axis=1).min()
        p1 = np.linalg.norm(ep.pos[c, opponents] - ep.pos[c, receiver], axis=1).min()
        w = min(end + 1, loss, T)
        progress = max(0.0, float(((ep.ball[c:w, 0] - ep.ball[c, 0]) * sign).max())) if w > c else 0.0
        useful += ok and (progress >= PC.PROG_MIN or (p0 <= PC.PRESS_R and p1 >= p0 + 20.0))
    rivals = np.flatnonzero(TEAM != team)
    pressured = np.linalg.norm(ep.pos[0, rivals] - ep.pos[0, carrier], axis=1).min() <= PC.PRESS_R
    w = max(1, min(loss, T))
    progress = max(0.0, float(((ep.ball[:w, 0] - ep.ball[0, 0]) * sign).max()))
    lost = terminal in ("out", "goal_against", "loss") or loss < T
    return dict(n=1, goal=int(own_goal), loss=int(lost), out=int(terminal == "out"),
                pass_any=int(bool(passes)), passes=len(passes), reception_observed=int(eligible),
                reception_retained=int(retained), reception_censored=int(censored), useful_passes=int(useful),
                pressure=int(pressured), pressure_loss_no_pass=int(pressured and lost and not passes),
                max_progress=progress, kept_to_horizon=int(terminal == "horizon" and not lost),
                recording_cut=int(terminal == "cut"))


def probe_rates(rows):
    S = lambda key: sum(row[key] for row in rows)
    divide = lambda a, b: S(a) / S(b) if S(b) else None
    return dict(goal_rate=divide("goal", "n"), loss_rate=divide("loss", "n"), out_rate=divide("out", "n"),
                pass_state_rate=divide("pass_any", "n"), useful_passes_per_state=divide("useful_passes", "n"),
                retention_after_reception=divide("reception_retained", "reception_observed"),
                pressure_loss_without_pass=divide("pressure_loss_no_pass", "pressure"),
                max_progress_px=divide("max_progress", "n"), kept_to_horizon_rate=divide("kept_to_horizon", "n"))


def summarize_probes(rows, groups, seed=0, n_boot=300):
    if not rows:
        return dict(n=0, metrics={}, support={})
    group_ids = np.unique(groups)
    grouped = [[row for row, g in zip(rows, groups) if g == key] for key in group_ids]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in probe_rates(rows)}
    for _ in range(n_boot):
        selected = [row for i in rng.integers(0, len(grouped), len(grouped)) for row in grouped[i]]
        for key, value in probe_rates(selected).items():
            if value is not None:
                draws[key].append(value)
    metrics = {}
    for key, value in probe_rates(rows).items():
        metrics[key] = dict(valor=round(float(value), 4) if value is not None else None,
                            ic90=[round(float(v), 4) for v in np.percentile(draws[key], (5, 95))]
                            if draws[key] else [None, None])
    return dict(n=len(rows), recording_units=len(grouped), metrics=metrics,
                support={key: int(sum(r[key] for r in rows)) for key in
                         ("passes", "reception_observed", "reception_retained", "reception_censored",
                          "pressure", "pressure_loss_no_pass", "recording_cut")})


@torch.no_grad()
def run_probes(candidate, opponent, data, states, *, horizon=200, seed=0):
    if not states:
        return [], []
    env = RS4ZEnv(len(states), map=MAP, frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=0, seed=seed)
    delays = restore_states(env, data, states, seed)
    teams = np.array([s[1] for s in states])
    own, rival = seeded_policy(candidate, seed + 100), seeded_policy(opponent, seed + 101)
    macro = MacroState(env.N) if getattr(own, "n_pass", 0) or getattr(rival, "n_pass", 0) else None
    rng = np.random.default_rng(seed)
    recorders = [XM.EpisodeRecorder(env, row=i) for i in range(env.N)]
    done = np.zeros(env.N, bool)
    terminal = np.full(env.N, "horizon", object)
    for _ in range(horizon):
        actions = choose_actions(env, own, rival, teams, rng, macro)
        for i in np.flatnonzero(~done):
            recorders[i].pre(env)
        ev = env.step(actions)
        if macro is not None:
            macro.after(ev["kicked"], np.flatnonzero(ev["goal"] != 0))
        for i in np.flatnonzero(~done):
            recorders[i].record(env, ev, actions)
            if ev["goal"][i]:
                terminal[i] = "goal" if (ev["goal"][i] > 0) == (teams[i] == 0) else "goal_against"
            elif ev["out"][i] or ev["restart_start"][i]:
                terminal[i] = "out"
            else:
                dist = np.linalg.norm(env.player_pos[i] - env.ball_pos[i], axis=1)
                touch = ((dist <= XM.PLAYER_R + env.radius[i, 0] + XM.CONTACT_EPS) | ev["kicked"][i]) & env.active[i]
                if touch[TEAM != teams[i]].any() and not touch[TEAM == teams[i]].any():
                    terminal[i] = "loss"
            done[i] = terminal[i] != "horizon"
        if done.all():
            break
    rows = [probe_row(rec.episode(), s[1], s[2], terminal[i]) for i, (rec, s) in enumerate(zip(recorders, states))]
    return rows, delays.tolist()


def human_probes(data, states, horizon=200):
    """Mismos estados y reglas de observación, con ocho humanos y cola original."""
    rows = []
    for t, team, carrier, _ in states:
        idx, terminal = [t], "horizon"
        for j in range(1, horizon):
            u = t + 3 * j
            if (u >= data.ticks or data.rec_of_tick[u] != data.rec_of_tick[t]
                    or np.any(data.pid[u] != data.pid[t])):
                terminal = "cut"
                break
            idx.append(u)
            if data.state[u] == 2:
                terminal = "goal" if (data.ball[u, 0] > 0) == (team == 0) else "goal_against"
                break
            if data.state[u] != 1 or data.rkind[u] > 0:
                terminal = "out"
                break
            dist = np.linalg.norm(data.pos[u] - data.ball[u, :2], axis=1)
            touched = (dist <= XM.PLAYER_R + data.ball_r[u] + XM.CONTACT_EPS) | data.kick_ev[u - 2:u + 1].any(0)
            if touched[TEAM != team].any() and not touched[TEAM == team].any():
                terminal = "loss"
                break
        idx = np.asarray(idx)
        kicked = np.array([data.kick_ev[u:min(u + 3, data.ticks)].any(0) for u in idx])
        ep = XM.Episode(stride=3, ball=data.ball[idx].copy(), ball_r=float(data.ball_r[t]),
                        pos=data.pos[idx].copy(), vel=data.vel[idx].copy(),
                        move=XD.move_of_input(data.inp[idx]), kick_key=(data.inp[idx] & 16) != 0,
                        kicked=kicked, open_play=(data.state[idx] == 1) & (data.rkind[idx] == 0),
                        restart_kind=data.rkind[idx], restart_age=data.rage[idx], kickoff=data.state[idx] == 0)
        rows.append(probe_row(ep, team, carrier, terminal))
    return rows


def state_manifest(data, states):
    ids, offsets = np.unique(data.rec_of_tick, return_index=True)
    start = dict(zip(ids.tolist(), offsets.tolist()))
    return [dict(recording=data.names[int(data.rec_of_tick[t])],
                 tick_in_loaded_recording=int(t - start[int(data.rec_of_tick[t])]),
                 concatenated_tick=int(t), attacking_team=team, carrier=carrier,
                 active_slots=np.flatnonzero(active).tolist()) for t, team, carrier, active in states]


def checkpoint_fingerprint(spec):
    path = Path(spec)
    if not path.is_file():
        return dict(spec=spec)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(spec=str(path), sha256=digest.hexdigest())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", required=True, help="candidato; se puede repetir")
    ap.add_argument("--reference", action="append", default=[], help="control/original; evalúa y suma al pool")
    ap.add_argument("--bc", required=True, help="checkpoint BC del pool fijo")
    ap.add_argument("--matches", type=int, default=8, help="partidos por candidato/rival POR LADO")
    ap.add_argument("--probe-states", type=int, default=64, help="estados held-out por sonda; 0 omite sondas")
    ap.add_argument("--split", choices=("dev", "test"), default="test", help="dev para piloto; test reservado al final")
    ap.add_argument("--recordings", type=int, default=40, help="grabaciones del split, seleccionadas con seed")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--horizon", type=int, default=200, help="decisiones por sonda (200 = 10 s)")
    ap.add_argument("--greedy", action="store_true")
    a = ap.parse_args(argv)
    if min(a.matches, a.recordings, a.threads, a.horizon) < 1 or a.probe_states < 0 or a.minutes <= 0:
        ap.error("matches/recordings/threads/horizon/minutes deben ser positivos; probe-states >= 0")
    return a


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    candidates = list(dict.fromkeys(a.policy + a.reference))
    rivals = list(dict.fromkeys([a.bc, "scripted"] + a.reference))
    policies = {s: Policy(s, device=a.device, greedy=a.greedy) for s in dict.fromkeys(candidates + rivals)}
    report = dict(version="x4-coop-eval-1", map=MAP, seed=a.seed, delays=list(DELAYS),
                  greedy=a.greedy, matches_per_side=a.matches, minutes=a.minutes, probe_horizon=a.horizon,
                  checkpoints={s: checkpoint_fingerprint(s) for s in policies},
                  runtime=dict(torch=torch.__version__, numpy=np.__version__, device=a.device, threads=a.threads),
                  training_access=False, probe_split=a.split, matches={}, probes={},
                  limitations=["Sondas 2v1/3v2 alteran el plantel; humanos conservan 4v4.",
                               "Resultados de simulación no demuestran fuerza contra humanos en sala.",
                               "Pase útil es diagnóstico; gol y progreso sin pase también cuentan.",
                               "Salida termina la sonda aun si el equipo obtendría el saque.",
                               "Bootstrap 90% por partido/grabación; con pocas unidades es incierto.",
                               "Cadena de partido usa definiciones existentes; sondas censuran recepciones tardías."])
    for candidate in candidates:
        report["matches"][candidate] = {}
        for j, rival in enumerate(rivals):
            if candidate == rival:
                continue
            row = evaluate_matches(policies[candidate], policies[rival], matches=a.matches,
                                   minutes=a.minutes, seed=a.seed + 1000 * j)
            report["matches"][candidate][rival] = row
            print(candidate, "vs", rival, json.dumps({k: row[k] for k in ("wins", "draws", "losses", "point_fraction")}), flush=True)
    if a.probe_states:
        names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", a.split, None, (MAP,))
        np.random.default_rng(a.seed).shuffle(names)
        data = XD.load(names[:a.recordings], maps=(MAP,))
        report["probe_recordings"] = data.names
        for kind, (na, nd) in (("2v1", (2, 1)), ("3v2", (3, 2))):
            states = select_states(data, na, nd, a.probe_states, a.seed, min_spacing=3 * a.horizon)
            groups = [int(data.rec_of_tick[s[0]]) for s in states]
            entry = dict(requested_states=a.probe_states, selected_states=len(states),
                         states=state_manifest(data, states),
                         human_4v4_approximate=summarize_probes(human_probes(data, states, a.horizon), groups, a.seed),
                         policies={})
            for candidate in candidates:
                entry["policies"][candidate] = {}
                for j, rival in enumerate(rivals):
                    if candidate == rival:
                        continue
                    rows, delays = run_probes(policies[candidate], policies[rival], data, states,
                                             horizon=a.horizon, seed=a.seed + 1000 * j + na)
                    entry["policies"][candidate][rival] = dict(
                        **summarize_probes(rows, groups, a.seed), delays=delays, state_results=rows)
                    print(kind, candidate, "vs", rival, "states", len(rows), flush=True)
            report["probes"][kind] = entry
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
