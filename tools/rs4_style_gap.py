"""Comparar el estilo de juego RS4 4v4 del bot contra partidos humanos.

Mide lo mismo sobre las grabaciones humanas convertidas (npz de BC) y sobre
partidos simulados entre dos controladores:

* estructura: fracción del tiempo con 3+ jugadores detrás del balón, dispersión
  respecto al centro del equipo, profundidad (x máx − x mín) y anchura;
* territorio: x medio del balón en coordenadas del equipo (+x ataca);
* dinámica (sólo simulado): goles por partido, partidos 0-0, pases y pérdidas
  por minuto y tiempo hasta presionar tras perder el balón.

Ejemplos:
  python -m tools.rs4_style_gap --human data/bc_rs4_v2/rsx4
  python -m tools.rs4_style_gap --human data/bc_rs4_v2/rsx4 --red runs/bc_rs4_v5_attn32/bc.pt \\
      --blue runs/bc_rs4_v5_attn32/bc.pt --games 32 --out reports/rs4_style_bc.json

Es un tablero de diagnóstico, no una prueba de calidad: parecerse a humanos en
estas cifras no garantiza buen fútbol, pero alejarse mucho (todos detrás del
balón, sin profundidad ni goles) delata el bloque defensivo de v3.
"""
from __future__ import annotations

import argparse
import glob
import sys
import json
from pathlib import Path

import numpy as np

W, H = 1150.0, 600.0  # rs_one; la obs universal normaliza posiciones por W/H
PRESS_DISTANCE = 60.0
PRESS_CENSOR_TICKS = 300


def _structure(x, y, ball_x):
    """x, y: (S, 4) posiciones del equipo en su marco (+x ataca); ball_x: (S,)."""
    cx, cy = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    return dict(behind3=float(((x < ball_x[:, None]).sum(axis=1) >= 3).mean()),
                spread=float(np.hypot(x - cx, y - cy).mean()),
                depth=float((x.max(axis=1) - x.min(axis=1)).mean()),
                width=float((y.max(axis=1) - y.min(axis=1)).mean()),
                ball_x=float(ball_x.mean()))


def human_style(directory, stride=5):
    """Estructura humana por grabación (media y percentiles 10/90 entre grabaciones)."""
    rows = []
    for path in sorted(glob.glob(str(Path(directory) / "*.npz"))):
        with np.load(path) as z:
            obs = np.concatenate([z[k] for k in z.files if k.startswith("obs_")]).astype(np.float32)
        # 4v4 (obs[55] = T/10) y fuera del saque inicial.
        obs = obs[(np.abs(obs[:, 55] - .4) < 1e-3) & (obs[:, 18] == 0)][::stride]
        if len(obs) < 50:
            continue
        px, py = obs[:, 0] * W, obs[:, 1] * H
        xs, ys = [px], [py]
        for e in range(7):
            j = 71 + 8 * e
            mate = (obs[:, j] > .5) & (obs[:, j + 1] < .5)
            xs.append(np.where(mate, px + obs[:, j + 2] * 400, np.nan))
            ys.append(np.where(mate, py + obs[:, j + 3] * 400, np.nan))
        x, y = np.stack(xs, 1), np.stack(ys, 1)
        complete = (~np.isnan(x)).sum(axis=1) == 4
        if complete.sum() < 50:
            continue
        # Exactamente 4 columnas presentes por fila: compactar conservando pares (x, y).
        keep = ~np.isnan(x[complete])
        row = _structure(x[complete][keep].reshape(-1, 4), y[complete][keep].reshape(-1, 4),
                         obs[complete, 4] * W)
        row["samples"] = int(complete.sum())
        rows.append(row)
    if not rows:
        raise SystemExit(f"sin muestras 4v4 en {directory}")
    keys = ("behind3", "spread", "depth", "width", "ball_x")
    return dict(recordings=len(rows), samples=int(sum(r["samples"] for r in rows)),
                **{k: float(np.mean([r[k] for r in rows])) for k in keys},
                p10={k: float(np.percentile([r[k] for r in rows], 10)) for k in keys},
                p90={k: float(np.percentile([r[k] for r in rows], 90)) for k in keys})


def simulated_style(red, blue, games=32, minutes=2.0, seed=51, greedy=True, device="cpu"):
    from eval.agents import make_agent
    from eval.rs4_v3 import environment, record_executed, reset_agents
    agents = (make_agent(red, greedy, device=device), make_agent(blue, greedy, device=device))
    ticks = int(minutes * 3600)
    env = environment(games, seed, max_ticks=ticks)
    env.reset()
    env._reset_envs(np.arange(games), kickoff_team=0)
    env.sim.player_pos[:] += env.rng.uniform(-8, 8, env.sim.player_pos.shape)
    env._phi = env._potentials()
    obs = env.observe()
    reset_agents(agents, env)
    players = (np.arange(4), np.arange(4, 8))
    samples = {0: [], 1: []}
    passes, turnovers = np.zeros(2), np.zeros(2)
    pressing = np.full((games, 2), -1, dtype=np.int64)  # ticks desde la pérdida, -1 inactivo
    press_ticks, press_count, press_censored = np.zeros(2), np.zeros(2), np.zeros(2)
    for step in range((ticks + 2) // 3):
        actions = np.empty((games, 8), dtype=np.int64)
        for team in (0, 1):
            actions[:, players[team]] = agents[team](env, obs, players[team])
        obs, _, done, info = env.step(actions)
        record_executed(agents, env, info["executed_actions"])
        passes += info["events"]["passes"].sum(axis=0)
        lost = info["events"]["turnovers"]
        turnovers += lost.sum(axis=0)
        live = ~env.sim.kickoff & ~done
        ball = env.sim.ball_pos
        for team in (0, 1):
            sign = 1.0 if team == 0 else -1.0
            own = env.sim.player_pos[:, env.sim.player_team == team]
            if step % 2 == 0 and live.any():
                samples[team].append((sign * own[live, :, 0], own[live, :, 1], sign * ball[live, 0]))
            pressing[lost[:, team] > 0, team] = 0
            active = pressing[:, team] >= 0
            pressing[active, team] += env.frame_skip
            near = np.linalg.norm(own - ball[:, None], axis=-1).min(axis=1) < PRESS_DISTANCE
            solved = active & near & ~done
            censored = active & ~solved & (done | (pressing[:, team] >= PRESS_CENSOR_TICKS))
            press_ticks[team] += pressing[solved, team].sum()
            press_count[team] += solved.sum()
            press_censored[team] += censored.sum()
            pressing[solved | censored, team] = -1
        if done.any():
            reset_agents(agents, env, done)
    score = env.score.astype(np.float64)
    total_minutes = games * minutes
    result = dict(red_spec=red, blue_spec=blue, games=games, minutes=minutes, seed=seed, greedy=greedy,
                  goals_per_game=float(score.sum() / games), scoreless=float((score.sum(axis=1) == 0).mean()),
                  red_points=float(((score[:, 0] > score[:, 1]) + .5 * (score[:, 0] == score[:, 1])).mean()))
    live_fraction = len(samples[0]) / max(1, ((ticks + 2) // 3 + 1) // 2)
    result["live_play_fraction"] = float(live_fraction)  # decisiones fuera del saque inicial
    for team, label in ((0, "red"), (1, "blue")):
        if samples[team]:
            x = np.concatenate([s[0] for s in samples[team]])
            y = np.concatenate([s[1] for s in samples[team]])
            b = np.concatenate([s[2] for s in samples[team]])
            row = _structure(x, y, b)
        else:  # p. ej. un imitador greedy que nunca ejecuta el saque inicial
            row = {key: float("nan") for key in ("behind3", "spread", "depth", "width", "ball_x")}
        row.update(passes_per_min=float(passes[team] / total_minutes),
                   turnovers_per_min=float(turnovers[team] / total_minutes),
                   pass_turnover_ratio=float(passes[team] / max(turnovers[team], 1)),
                   press_seconds=float(press_ticks[team] / max(press_count[team], 1) / 60),
                   press_censored=float(press_censored[team] / max(press_count[team] + press_censored[team], 1)))
        result[label] = row
    return result


def _table(human, sims):
    keys = (("behind3", "3+ detrás del balón", "{:.0%}"), ("spread", "dispersión (px)", "{:.0f}"),
            ("depth", "profundidad (px)", "{:.0f}"), ("width", "anchura (px)", "{:.0f}"),
            ("ball_x", "x del balón (px)", "{:+.0f}"))
    header = ["métrica"] + (["humanos"] if human else []) + [f"{Path(s['red_spec']).parent.name or s['red_spec']} vs {Path(s['blue_spec']).parent.name or s['blue_spec']}" for s in sims]
    lines = [" | ".join(header)]
    for key, label, fmt in keys:
        cells = [label] + ([fmt.format(human[key])] if human else []) + [fmt.format(s["red"][key]) for s in sims]
        lines.append(" | ".join(cells))
    for key, label, fmt in (("goals_per_game", "goles por partido", "{:.2f}"), ("scoreless", "partidos 0-0", "{:.0%}"),
                            ("live_play_fraction", "tiempo con juego en curso", "{:.0%}")):
        lines.append(" | ".join([label] + (["—"] if human else []) + [fmt.format(s[key]) for s in sims]))
    for key, label, fmt in (("passes_per_min", "pases/min", "{:.1f}"), ("turnovers_per_min", "pérdidas/min", "{:.1f}"),
                            ("pass_turnover_ratio", "pases/pérdidas", "{:.2f}"), ("press_seconds", "s hasta presionar", "{:.1f}")):
        lines.append(" | ".join([label] + (["—"] if human else []) + [fmt.format(s["red"][key]) for s in sims]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", help="carpeta de npz humanos RS4 (data/bc_rs4_v2/rsx4)")
    ap.add_argument("--red", action="append", default=[], help="controlador rojo (checkpoint o scripted:r3)")
    ap.add_argument("--blue", action="append", default=[], help="rival azul; uno por --red o uno para todos")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--sample", action="store_true",
                    help="muestrear acciones; por defecto greedy, como deploy/bot.js (--temp 0)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    if len(a.blue) not in (0, 1, len(a.red)):
        ap.error("usar un --blue por cada --red, o uno compartido")
    human = human_style(a.human) if a.human else None
    blues = a.blue * len(a.red) if len(a.blue) == 1 else (a.blue or a.red)
    sims = [simulated_style(r, b, a.games, a.minutes, a.seed, not a.sample, a.device) for r, b in zip(a.red, blues)]
    sys.stdout.reconfigure(encoding="utf-8")
    print(_table(human, sims))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(human=human, matches=sims), indent=2), encoding="utf-8")
        print(f"guardado en {a.out}")


if __name__ == "__main__":
    main()
