"""¿Por qué la imitación pasa menos que los humanos? Descompone el déficit de patadas y pases.

1. Lazo abierto (estados humanos de prueba, retardo de sala): en los estados de juego abierto con el jugador al
   alcance de la pelota, ¿la BC predice la tecla de patada con la misma frecuencia que la apretaron los humanos?
   (calibración: media de P(patada) predicha contra frecuencia real, por distancia a la pelota).
2. Lazo cerrado (self-play de la BC contra la misma métrica en las grabaciones): fracción del tiempo con algún jugador
   al alcance de la pelota, patadas por minuto y P(patada efectiva | al alcance). Separa "llega menos a la pelota"
   de "llega pero no patea" y de "patea pero no le llega al compañero".

  python -m learn.x4_bc_diagnose --policy runs/x4_bc/final_sangu_rsone/best.pt --out reports/x4/bc_diagnose.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools import x4_metrics as XM

ROOT = Path(__file__).resolve().parent.parent
REACH = XM.PLAYER_R + 8.325 + 4.0      # alcance de patada (radio jugador + pelota + 4 px del motor)


def open_loop(pol, data, n=200_000, delay=10, seed=0):
    from learn import x4_data as XD
    rng = np.random.default_rng(seed)
    v = data.valid
    t = v[rng.integers(0, len(v), n)]
    p = rng.integers(0, 8, n)
    d = np.hypot(data.pos[t, p, 0] - data.ball[t, 0], data.pos[t, p, 1] - data.ball[t, 1])
    open_ = data.ctx[t, p] == XD.STATE_OPEN
    keep = open_ & (d <= 120.0)
    t, p, d = t[keep], p[keep], d[keep]
    obs = XD.featurize(data, t, p, np.full(len(t), delay))
    with torch.no_grad():
        pk = []
        for i in range(0, len(obs), 8192):
            lg = pol.model(torch.from_numpy(obs[i:i + 8192]).to(pol.device))
            pk.append(torch.softmax(lg, -1)[:, 9:].sum(-1).cpu().numpy())
    pk = np.concatenate(pk)
    human = data.label[t, p] >= 9
    out = []
    for lo, hi in ((0, REACH), (REACH, 40), (40, 70), (70, 120)):
        m = (d >= lo) & (d < hi)
        out.append(dict(distancia=[round(lo, 1), hi], n=int(m.sum()), humanos_tecla=round(float(human[m].mean()), 4),
                        bc_p_patada=round(float(pk[m].mean()), 4)))
    return out


def closed_loop_stats(episodes):
    reach_samples = kicks = kicks_reach = samples = 0
    team_reach = 0
    minutes = 0.0
    for ep in episodes:
        idx = np.flatnonzero(ep.open_play)
        d = np.hypot(ep.pos[idx, :, 0] - ep.ball[idx, None, 0], ep.pos[idx, :, 1] - ep.ball[idx, None, 1])
        inr = d <= REACH
        samples += len(idx)
        reach_samples += int(inr.any(1).sum())
        team_reach += int(inr[:, :4].any(1).sum() + inr[:, 4:].any(1).sum())
        k = ep.kicked[idx]
        kicks += int(k.sum())
        kicks_reach += int((k & inr).sum())
        minutes += len(idx) * ep.stride / 3600.0
    return dict(minutos_juego_abierto=round(minutes, 1),
                frac_alguien_al_alcance=round(reach_samples / max(1, samples), 4),
                patadas_por_min_y_jugador=round(kicks / 8 / max(1e-9, minutes), 3),
                patadas_por_muestreo_al_alcance=round(kicks_reach / max(1, team_reach), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=str(ROOT / "runs" / "x4_bc" / "final_sangu_rsone" / "best.pt"))
    ap.add_argument("--recordings", type=int, default=40)
    ap.add_argument("--matches", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from learn import x4_data as XD
    from learn.x4_eval import Policy, play
    from tools.x4_ticks import MAP_IDS
    pol = Policy(a.policy)
    names = XD.split_names(ROOT / "reports" / "x4" / "splits.json", "test", None, ("sanguchito_rs_x4",))
    data = XD.load(names[:a.recordings], maps=("sanguchito_rs_x4",))
    report = dict(policy=a.policy, lazo_abierto=open_loop(pol, data))
    print(json.dumps(report["lazo_abierto"], indent=1), flush=True)
    hum = []
    for n in names:
        p = ROOT / "data" / "x4_ticks" / f"{Path(n).stem}.npz"
        if p.exists():
            hum += XM.episodes_from_ticks(str(p), map_id=MAP_IDS["sanguchito_rs_x4"])
    report["lazo_cerrado_humanos"] = closed_loop_stats(hum)
    r = play(pol, pol, matches=a.matches, minutes=a.minutes, seed=5, record=a.matches)
    report["lazo_cerrado_bc"] = closed_loop_stats(r["episodes"])
    print(json.dumps({k: v for k, v in report.items() if k.startswith("lazo_cerrado")}, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
