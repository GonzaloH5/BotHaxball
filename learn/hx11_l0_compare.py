"""Compara brazos de L0 en los dos lazos cerrados, con las mismas ventanas y los mismos arranques.

* **No reactivo** (`learn/hx11_closed_loop`): 640 ventanas de 5 s desde el jugador de campo más cercano a la pelota en
  10 partidos de desarrollo; los otros 21 juegan sus teclas grabadas. Se compara contra el humano en las mismas ventanas.
* **Reactivo** (`learn/hx11_selfplay`): 8 juegos de 60 s con los 22 lugares jugando el modelo. La referencia humana son
  las mismas ventanas grabadas.

Escribe `reports/hx11/l0_brazos.{md,json}`. Es diagnóstico de la puerta P3, no la puerta: la puerta usa los datos de
prueba y más ventanas.

  python -m learn.hx11_l0_compare runs/hx11_l0/conhist runs/hx11_l0/sinhist ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learn import hx11_closed_loop as CL
from learn import hx11_selfplay as SP
from learn.hx11_data import l0_split, load_match
from learn.hx11_policy import SetPolicy

ROOT = Path(__file__).resolve().parent.parent
NR_KEYS = ("patadas_efectivas_min", "toques_min", "vel", "dist_pelota", "dist_ancla", "cambios_s")
SP_KEYS = ("patadas_jugador_min", "toques_jugador_min", "cambios_posesion_min", "vel_media", "mas_cercano_pelota_p50",
           "largo_equipo", "juego_abierto")


def load(path):
    ck = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
    net = SetPolicy(**ck["config"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--starts", type=int, default=640)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="reports/hx11/l0_brazos")
    ap.add_argument("--split", default="dev", help="dev para elegir; test para la puerta")
    ap.add_argument("--hold-off", action="store_true", help="ignorar la cabeza de duración (decidir cada 3 ticks)")
    a = ap.parse_args()
    torch.set_num_threads(6)
    modos = json.loads((ROOT / "data" / "hx11" / "modos.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    names = sorted(rng.permutation(l0_split()[a.split])[:10].tolist())
    matches = [load_match(n, modos) for n in names]
    owners = [np.load(ROOT / "data" / "hx11" / "d1" / f"{n}.npz")["owner"] for n in names]
    rng = np.random.default_rng(0)
    per = max(1, a.starts // len(matches))
    sl = [(mi, t0, s) for mi, m in enumerate(matches) for t0, s in CL.starts(m, owners[mi], per, rng, near=True)]
    sp_names = names[:4]
    sp_matches = [m for m in matches if m.name in sp_names]
    res = {}
    for run in a.runs:
        net, ck = load(run)
        if a.hold_off:
            net.hold = False
        rows = CL.rollout(net, matches, owners, sl, n_env=64, seed=0, temp_move=a.temp)
        nr = CL.summarize(rows)
        sp_rows, sp_ref, _ = SP.play(net, sp_matches, a.games, a.seconds * 60, seed=0, n_env=8, temp_move=a.temp)
        res[Path(run).name] = dict(config=ck["config"], args={k: ck["args"].get(k) for k in ("train", "steps", "drop_hist",
                                                                                               "sin_historia", "extra")},
                                   no_reactivo=nr, reactivo=SP.summarize(sp_rows),
                                   reactivo_humano=SP.summarize(sp_ref) if sp_ref else None)
        print(Path(run).name, json.dumps({k: nr.get(f"{k}_bot", {}).get("media") for k in NR_KEYS}),
              json.dumps({k: res[Path(run).name]["reactivo"][k]["media"] for k in SP_KEYS}), flush=True)
    hum_nr = {k: res[next(iter(res))]["no_reactivo"].get(f"{k}_humano", {}).get("media") for k in NR_KEYS}
    hum_sp = res[next(iter(res))]["reactivo_humano"]
    Path(a.out + ".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["# L0: brazos comparados en lazo cerrado", "",
          f"Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura "
          f"{a.temp}. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de "
          f"{'prueba' if a.split == 'test' else 'desarrollo'}. Las tasas son totales sobre el tiempo total; las patadas "
          f"son efectivas (la física las ejecuta).", "",
          f"## No reactivo: {len(sl)} ventanas de 5 s desde el jugador más cercano a la pelota", "",
          "| Brazo | Patadas efectivas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |",
          "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        md.append(f"| {k} | " + " | ".join(str(r["no_reactivo"].get(f"{x}_bot", {}).get("media")) for x in NR_KEYS) + " |")
    md.append("| **humano** | " + " | ".join(str(hum_nr[x]) for x in NR_KEYS) + " |")
    md += ["", f"## Reactivo: {a.games} juegos de {a.seconds} s con los 22 lugares jugando el modelo", "",
           "| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la "
           "pelota (p50) | Largo del equipo | Juego abierto |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        md.append(f"| {k} | " + " | ".join(str(r["reactivo"][x]["media"]) for x in SP_KEYS) + " |")
    if hum_sp:
        md.append("| **humano** | " + " | ".join(str(hum_sp[x]["media"]) for x in SP_KEYS) + " |")
    Path(a.out + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
