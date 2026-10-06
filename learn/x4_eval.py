"""Evaluación en simulación (E0): partidos entre políticas con latencia de sala + parecido humano.

Fuerza: diferencia de gol y victorias de A contra B (y de B contra A cambiando de lado), siempre con
latencia de sala; nunca contra un solo rival greedy (revisión §7). Con varias políticas, todos contra todos y
ranking Bradley–Terry sobre los resultados (Elo con tamaño de muestra declarado).

Parecido humano: los mismos partidos se graban con `tools.x4_metrics.EpisodeRecorder` y se comparan con la
referencia humana (`data/human_metrics_sanguchito.samples.npz` de `tools.x4_metrics`, split de entrenamiento) con W1 normalizada. El
techo es la distancia test-vs-train de los propios humanos (`reports/x4/human_metrics_sanguchito.json`).

Políticas: ruta a checkpoint (`learn.x4_bc` / `learn.x4_ppo`), `scripted` (`learn.x4_scripted`), `still` (no se
mueve) o `random`.
Modo: `--greedy` toma la acción más probable; por defecto se muestrea (temperatura 1).

  python -m learn.x4_eval --policies runs/x4_bc/best.pt,still,random --matches 32 --out reports/x4/eval_bc.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from env.rs4z import kernel as K
from env.rs4z import obs_v3
from env.rs4z.core import RS4ZEnv
from tools import x4_metrics as XM

ROOT = Path(__file__).resolve().parent.parent


class Policy:
    def __new__(cls, spec, *args, **kw):
        if spec == "scripted":
            from learn.x4_scripted import ScriptedPolicy
            return ScriptedPolicy()
        return super().__new__(cls)

    def __init__(self, spec, device="cpu", greedy=False, temperature=1.0):
        self.name = Path(spec).stem if spec not in ("still", "random") else spec
        self.spec, self.greedy, self.temperature = spec, greedy, temperature
        self.model = None
        if spec not in ("still", "random"):
            from learn.x4_policy import SetPolicy
            ck = torch.load(spec, map_location="cpu")
            self.model = SetPolicy(hidden=ck.get("hidden", 256))
            self.model.load_state_dict(ck["model"])
            self.model.to(device).eval()
            self.name = f"{Path(spec).parent.name}/{Path(spec).stem}"
        self.device = device

    @torch.no_grad()
    def __call__(self, obs, rng):
        n = len(obs)
        if self.spec == "still":
            return np.zeros(n, np.int64)
        if self.spec == "random":
            return rng.integers(0, 18, size=n)
        logits = self.model(torch.from_numpy(obs).to(self.device))
        if self.greedy:
            return logits.argmax(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits / self.temperature).sample().cpu().numpy()


class ModelPolicy:
    """Adaptador de un SetPolicy en memoria a la interfaz de `Policy` (muestreado o greedy)."""

    def __init__(self, model, device="cpu", name="modelo", greedy=False, generator=None):
        """`generator`: un `torch.Generator` del mismo dispositivo para que el muestreo sea reproducible
        (las evaluaciones de distintos checkpoints ven así el mismo azar)."""
        self.model, self.device, self.name, self.greedy, self.spec = model, device, name, greedy, name
        self.generator = generator

    @torch.no_grad()
    def __call__(self, obs, rng):
        logits = self.model(torch.from_numpy(obs).to(self.device))
        if self.greedy:
            return logits.argmax(-1).cpu().numpy()
        if self.generator is not None:
            p = torch.softmax(logits.float(), -1)
            return torch.multinomial(p, 1, generator=self.generator).squeeze(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()


def play(red, blue, *, map_name="sanguchito_rs_x4", matches=16, minutes=3.0, delays=(8, 9, 10, 11), seed=0,
         record=4, max_decisions=None):
    """Partidos completos red vs blue. Devuelve goles por partido, eventos de seguridad y episodios grabados."""
    rng = np.random.default_rng(seed)
    # sin plazos de saque (como la sala); el saque inicial que nadie ejecuta se libera por seguridad a los
    # KICKOFF_SAFETY ticks y se cuenta en `safety` (si no, el reloj congelado detendría el partido para siempre)
    from env.rs4z import contract as C
    env = RS4ZEnv(matches, map=map_name, frame_skip=3, max_delay=15, deadline=0, kickoff_deadline=C.KICKOFF_SAFETY,
                  seed=seed)
    ticks = int(minutes * 3600)
    env.start_match(np.arange(matches), delay=rng.choice(delays, size=(matches, 8)), match_ticks=ticks)
    recs = [XM.EpisodeRecorder(env, row=r) for r in range(min(record, matches))]
    goals = np.zeros((matches, 2), np.int64)
    safety = np.zeros(matches, np.int64)
    done = np.zeros(matches, bool)
    limit = max_decisions or int(ticks / 3 * 4)
    steps = 0
    while not done.all() and steps < limit:
        obs = obs_v3.observe(env)
        acts = np.zeros((matches, 8), np.int64)
        acts[:, :4] = red(obs[:, :4].reshape(-1, obs.shape[-1]), rng).reshape(matches, 4)
        acts[:, 4:] = blue(obs[:, 4:].reshape(-1, obs.shape[-1]), rng).reshape(matches, 4)
        ev = env.step(acts)
        live = ~done
        goals[live, 0] += ev["goal"][live] > 0
        goals[live, 1] += ev["goal"][live] < 0
        safety[live] += ev["safety"][live] != 0
        for rec in recs:
            if not done[rec.row]:
                rec.record(env, ev, acts)
        done |= ev["match_end"]
        steps += 1
    return dict(goals=goals, safety=safety, finished=int(done.sum()), episodes=[r.episode() for r in recs])


def bradley_terry(names, results, iters=200):
    """Fuerza (escala Elo) desde victorias/empates por par; empate = media victoria."""
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n))
    for (a, b), r in results.items():
        W[idx[a], idx[b]] += r["wins"] + 0.5 * r["draws"]
        W[idx[b], idx[a]] += r["losses"] + 0.5 * r["draws"]
    p = np.ones(n)
    for _ in range(iters):
        for i in range(n):
            num = W[i].sum() + 0.5
            den = sum((W[i, j] + W[j, i] + 1) / (p[i] + p[j]) for j in range(n) if j != i)
            p[i] = num / max(den, 1e-12)
        p /= np.exp(np.mean(np.log(p)))
    return {nm: round(float(400 * np.log10(p[idx[nm]])), 1) for nm in names}


HUMAN_SAMPLES = ROOT / "data" / "human_metrics_sanguchito.samples.npz"   # de `tools.x4_metrics --out .../human_metrics_sanguchito.json`


def human_compare(episodes, samples_path=HUMAN_SAMPLES, split="train"):
    if not Path(samples_path).exists() or not episodes:
        return None
    ref = np.load(samples_path)
    reference = {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(split + "/")}
    ss = [XM.episode_samples(ep)[0] for ep in episodes]
    cand = XM.merge(ss)
    cmp_ = XM.compare(cand, reference)
    return dict(summary=XM.summarize(cand), w1_norm={k: v["w1_norm"] for k, v in cmp_.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--map", default="sanguchito_rs_x4")
    ap.add_argument("--matches", type=int, default=16, help="partidos por par y por lado")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--delays", default="8,9,10,11", help="retardo D del kernel (lag medido en sala − 1)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--record", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    delays = tuple(int(x) for x in a.delays.split(","))
    pols = [Policy(s, greedy=a.greedy) for s in a.policies.split(",")]
    results, human = {}, {}
    pairs = list(itertools.combinations(range(len(pols)), 2))
    for i, j in pairs:
        A, B = pols[i], pols[j]
        agg = dict(wins=0, draws=0, losses=0, goals_for=0, goals_against=0, safety=0, matches=0)
        for side, (red, blue) in enumerate(((A, B), (B, A))):
            r = play(red, blue, map_name=a.map, matches=a.matches, minutes=a.minutes, delays=delays,
                     seed=1000 * i + 10 * j + side, record=0)
            g = r["goals"] if side == 0 else r["goals"][:, ::-1]
            diff = g[:, 0] - g[:, 1]
            agg["wins"] += int((diff > 0).sum())
            agg["draws"] += int((diff == 0).sum())
            agg["losses"] += int((diff < 0).sum())
            agg["goals_for"] += int(g[:, 0].sum())
            agg["goals_against"] += int(g[:, 1].sum())
            agg["safety"] += int(r["safety"].sum())
            agg["matches"] += len(diff)
        agg["goal_diff_per_match"] = round((agg["goals_for"] - agg["goals_against"]) / max(1, agg["matches"]), 3)
        results[(A.name, B.name)] = agg
        print(f"{A.name} vs {B.name}: {json.dumps(agg)}", flush=True)
    names = [p.name for p in pols]
    rating = bradley_terry(names, results) if len(pols) > 1 else {}
    # parecido humano: cada política contra sí misma (los dos equipos son la política)
    for k, p in enumerate(pols):
        if a.record > 0 and p.spec not in ("still", "random"):
            r = play(p, p, map_name=a.map, matches=a.record, minutes=a.minutes, delays=delays, seed=7 + k,
                     record=a.record)
            human[p.name] = dict(safety_per_match=float(r["safety"].mean()), goals_per_match=float(r["goals"].sum(1).mean()),
                                 **(human_compare(r["episodes"]) or {}))
    report = dict(version="x4-eval-1", map=a.map, delays=delays, greedy=a.greedy, minutes=a.minutes,
                  results={f"{x} vs {y}": v for (x, y), v in results.items()}, rating=rating, human=human)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(rating=rating), ensure_ascii=False))


if __name__ == "__main__":
    main()
