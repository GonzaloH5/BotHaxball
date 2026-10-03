"""Grabaciones del bloque RS4-b1 para explicar resultados (PLAN_RS4.md 5): casos elegidos y al azar.

Usa el entorno del protocolo (árbitro real) y el visor HTML de eval/render.py. El jugador marcado
con un anillo amarillo es el candidato. Las grabaciones explican resultados; nunca los sustituyen.

  python -m tools.rs4_b1_replays --out reports/rs4_b1/replays
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RING = ("disc(f[0],f[1],M.r_ball,ballColor,'#000',2);",
        "disc(f[0],f[1],M.r_ball,ballColor,'#000',2);"
        "if(M.mark!=null){x.beginPath();x.arc(tx(f[2+2*M.mark]),ty(f[3+2*M.mark]),(M.r_player+7)*S,0,7);"
        "x.lineWidth=3*S;x.strokeStyle='#ffd43b';x.stroke();}")


def record(controllers, bank, state, mirror, seconds, seed, mark=None, names=("rojo", "azul")):
    """controllers: lista de (Policy, máscara (8,) bool). Devuelve el HTML autocontenido."""
    from env.rs4_states import apply_states
    from eval.render import HTML, stadium_lines
    from eval.rs4_b1 import environment
    env = environment(1, seed, minutes=10)
    env.reset()
    apply_states(env, np.array([0]), bank, np.array([state]), mirror=np.array([mirror]))
    env.match_ticks[:] = 0
    env.match_score[:] = 0
    env.score[:] = 0
    obs = env.observe()
    rng = np.random.default_rng(seed)
    for policy, _ in controllers:
        policy.reset(1, 8)
    s = env.sim
    frames = []
    released = None  # tick en que se liberó el saque inicial de la grabación (si arrancó en uno)
    for step in range(int(seconds * 20)):
        actions = np.zeros((1, 8), dtype=np.int64)
        for policy, mask in controllers:
            chosen = policy.act(env, obs, mask[None], rng, True)
            actions[0, mask] = chosen[0, mask]
        active = env.setpiece_team[0] >= 0
        obs, _, done, _ = env.step(actions)
        if active and env.setpiece_team[0] < 0 and released is None:
            released = 3 * (step + 1)
        frame = ([round(float(v), 1) for v in s.pos[0, [0, *range(s.first_player, s.K)]].ravel()]
                 + [int(k) for k in (actions[0] >= 9)] + [0, int(env.score[0, 0]), int(env.score[0, 1])])
        frames += [frame] * 3  # el visor avanza 60 cuadros por segundo; una decisión dura 3 ticks
        if done[0]:
            for policy, _ in controllers:
                policy.reset(1, 8, np.array([0]))
    st = s.st
    W = max(st.width, st.field_half_w + 30)
    # La banda real es la del árbitro (670); st.field_half_h (600) es sólo la escala de la observación.
    line_h = env._rs1.c["line_half_h"] if getattr(env, "_rs1", None) is not None else st.field_half_h
    meta = {"lines": stadium_lines(st), "posts": st.d_pos.tolist(), "post_r": st.d_radius.tolist(), "P": env.P,
            "T": env.T, "r_player": st.player["radius"], "r_ball": st.ball["radius"], "W": W, "H": st.height,
            "fw": st.field_half_w, "fh": line_h, "ko": st.kickoff_radius, "S": min(1.6, 1100 / (2 * W + 20)),
            "red": names[0], "blue": names[1], "kickoff_stalls": 0, "public_cues": [], "mark": mark}
    page = HTML.replace(*RING).replace("__META__", json.dumps(meta))
    return page.replace("__FRAMES__", json.dumps(frames, separators=(",", ":"))), released


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", default=str(ROOT / "data" / "rs4_states" / "desarrollo.npz"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rs4_b1" / "replays"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from env.rs4_states import KIND_CORNER, KIND_OPEN, StateBank
    from eval.rs4_b1 import Policy
    from tools.evaluate_rs4_b1 import EVAL_RIVALS, EVAL_TEAMMATES, _resolve
    bank = StateBank(a.states)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    red = np.arange(8) < 4
    champion = _resolve(EVAL_RIVALS[0])
    reference = "runs/rs4_b1/inputs/v3_2568M.pt"
    # Elegido: el primer córner en que A1 deja llegar la liberación de seguridad (900 ticks).
    corner = None
    for candidate_corner in bank.select([KIND_CORNER])[:20]:
        flip = bool(bank.arrays["taker"][candidate_corner] == 1)
        _, released = record([(Policy("runs/rs4_b1_a_s1/latest.pt"), red), (Policy(champion), ~red)], bank,
                             int(candidate_corner), flip, 16, a.seed)
        if released is None or released >= 900:
            corner = int(candidate_corner)
            break
    corner = int(bank.select([KIND_CORNER])[0]) if corner is None else corner
    taker_blue = bool(bank.arrays["taker"][corner] == 1)
    cases = []
    # Elegidos: el mismo córner con A1 y con la referencia (A1 deja vencer el 64% de sus córners).
    for label, spec in (("A1", "runs/rs4_b1_a_s1/latest.pt"), ("referencia v3 2568M", reference)):
        cases.append((f"corner_{label.split()[0]}.html", [(Policy(spec), red), (Policy(champion), ~red)],
                      corner, taker_blue, 25, None, (f"{label} (los 4)", "campeón v3")))
    # Al azar: candidato en un puesto con tres compañeros reservados (R3:2) contra el campeón.
    open_ids = bank.select([KIND_OPEN])
    for label, spec in (("A1", "runs/rs4_b1_a_s1/latest.pt"), ("A2", "runs/rs4_b1_a_s2/latest.pt")):
        state, slot = int(rng.choice(open_ids)), int(rng.integers(0, 4))
        candidate = np.zeros(8, dtype=bool)
        candidate[slot] = True
        teammates = red & ~candidate
        cases.append((f"partido_{label}_azar.html", [(Policy(spec), candidate), (Policy(EVAL_TEAMMATES[0]), teammates),
                                                      (Policy(champion), ~red)],
                      state, False, 90, slot, (f"{label} (anillo) + R3:2", "campeón v3")))
    index = []
    for name, controllers, state, mirror, seconds, mark, names in cases:
        page, released = record(controllers, bank, state, mirror, seconds, a.seed, mark, names)
        (out / name).write_text(page, encoding="utf-8")
        index.append(dict(file=name, state=state, mirror=mirror, seconds=seconds, red=names[0], blue=names[1],
                          set_piece_released_tick=released))
        print(f"{out / name} | saque liberado en el tick {released}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
