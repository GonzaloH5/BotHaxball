"""Liga barata del RL X4 (docs/PLAN.md §2 E3: "un explotador barato cada 200–500M decisiones"; pre-registro en
docs/LIGA.md). Corre sola y es idempotente como la cola (`<root>/queue_state.json`, mismas reglas de reintento).

Ciclo, para k = 0..S (S = --segments):
1. objetivo_k: copia congelada del agente principal (k = 0: el checkpoint de --init; después, su last.pt al final del
   tramo k).
2. explotador_k: arranca de objetivo_k y entrena sólo contra él (--no-pool-bc --pool-frac 1, sin snapshots propios)
   --exploiter-updates actualizaciones, con la misma recompensa y la misma ancla a la BC que el principal (busca
   debilidades que un humano podría aprovechar, no trucos del simulador).
3. h2h_k: el explotador contra su objetivo, --h2h-matches partidos por lado. Ese puntaje es la explotabilidad del
   principal en ese momento. Si llega a --accept, el explotador entra al pool fijo del principal.
4. principal (si k < S): un tramo de --segment-updates actualizaciones más (--resume), con los explotadores aceptados en
   el pool y --main-pool-frac de los partidos contra el pool (PFSP: más peso a los rivales que le ganan).
Al final: el principal contra la BC, el rival scripted, las referencias (--refs) y cada explotador; veredicto de la liga
(docs/LIGA.md §3); certificación y export como en la cola.

  python -m learn.x4_league --root runs/x4_liga --init runs/x4_estilo/rl_control/best.pt --device cuda \\
      --rl-extra "--shaping-kind none --auto-pass-arm-update 0 --forfeit-reasons 1,2,3,4 --lambda-dist 0.05 ..."
  python -m learn.x4_league --h2h A.pt B.pt --out r.json      # un enfrentamiento (lo usa la liga)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from learn.x4_queue import BC, ROOT, Queue

H2H_DELAYS = (8, 9, 10, 11)          # los de la certificación (lag de sala − 1)


# ------------------------------------------------------------------------------------- enfrentamientos
def h2h(a_spec, b_spec, matches=64, minutes=3.0, seed=0, device="cuda"):
    """A contra B, `matches` partidos por lado (A de rojo y de azul), muestreando como en la sala. Puntaje de A
    (victoria 1, empate ½)."""
    import torch
    from learn.x4_eval import Policy, play
    torch.manual_seed(seed)
    A, B = Policy(a_spec, device=device), Policy(b_spec, device=device)
    agg = dict(wins=0, draws=0, losses=0, goals_for=0, goals_against=0, matches=0)
    for side, (red, blue) in enumerate(((A, B), (B, A))):
        r = play(red, blue, matches=matches, minutes=minutes, delays=H2H_DELAYS, seed=seed * 10 + side, record=0)
        g = r["goals"] if side == 0 else r["goals"][:, ::-1]
        d = g[:, 0] - g[:, 1]
        agg["wins"] += int((d > 0).sum())
        agg["draws"] += int((d == 0).sum())
        agg["losses"] += int((d < 0).sum())
        agg["goals_for"] += int(g[:, 0].sum())
        agg["goals_against"] += int(g[:, 1].sum())
        agg["matches"] += len(d)
    agg["score"] = round((agg["wins"] + 0.5 * agg["draws"]) / max(1, agg["matches"]), 4)
    agg["a"], agg["b"] = str(a_spec), str(b_spec)
    return agg


class League(Queue):
    def pool_path(self, name):
        p = self.root / "pool"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{name}.pt"

    def freeze(self, name, src):
        """Copia congelada con nombre propio (el pool identifica a cada rival por el nombre del archivo)."""
        dst = self.pool_path(name)
        if not dst.exists():
            shutil.copyfile(src, dst)
        return dst

    def match(self, step, a_spec, b_spec, seed):
        """Enfrentamiento en un proceso aparte (idempotente como cualquier paso de la cola)."""
        a = self.a
        out = self.root / "h2h" / f"{step}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        code = self.run(step, ["learn.x4_league", "--h2h", a_spec, b_spec, "--out", out, "--matches", a.h2h_matches,
                               "--minutes", a.h2h_minutes, "--seed", seed, "--device", a.device], retries=2)
        return json.loads(out.read_text()) if code == 0 and out.exists() else None

    def exploiter(self, k, target):
        a = self.a
        name = f"explotador_{k}"
        cmd = self.rl_cmd(name, ["--init", target, "--pool", target, "--pool-frac", "1.0", "--snapshot-every", "0",
                                 "--eval-every", "0", "--envs", a.exploiter_envs], updates=a.exploiter_updates)
        cmd.append("--no-pool-bc")
        code = self.run(name, cmd)
        if not self.check_finished(name, name, a.exploiter_updates, code):
            return None
        return self.freeze(f"explotador_{k:02d}", self.root / name / "last.pt")

    def main(self):
        a = self.a
        st = self.state
        code = self.run("preflight", ["learn.x4_preflight", "--bc", a.bc, "--device", a.device,
                                      "--out", self.root / "preflight.json", "--init", a.init,
                                      *a.preflight_extra.split()], retries=1)
        if code != 0:
            self.note("preflight con FALLA: la liga se detiene (ver preflight.json)")
            return 1
        st.setdefault("explotadores", {})
        st.setdefault("pool", [])
        for k in range(a.segments + 1):
            # 1. objetivo congelado
            src = Path(a.init) if k == 0 else self.root / "principal" / "last.pt"
            target = self.freeze(f"objetivo_{k:02d}", src)
            # 2-3. explotador y su puntaje contra el objetivo
            key = str(k)
            if key not in st["explotadores"]:
                ex = self.exploiter(k, target)
                res = self.match(f"h2h_explotador_{k}", ex, target, seed=100 + k) if ex else None
                acc = bool(res and res["score"] >= a.accept)
                st["explotadores"][key] = dict(objetivo=str(target), explotador=str(ex) if ex else None,
                                               puntaje=(res or {}).get("score"), resultado=res, aceptado=acc)
                if acc and k < a.segments:
                    st["pool"].append(str(ex))
                self.save()
                self.note(f"explotador {k}: {res and res['score']} contra su objetivo"
                          + (" → entra al pool" if acc and k < a.segments else ""))
            # 4. tramo del principal
            if k < a.segments:
                step = f"principal_{k + 1}"
                target_upd = (k + 1) * a.segment_updates
                cmd = self.rl_cmd("principal", ["--init", a.init, "--pool", ",".join(st["pool"]),
                                                "--pool-frac", a.main_pool_frac], updates=target_upd)
                code = self.run(step, cmd)
                if not self.check_finished(step, "principal", target_upd, code):
                    self.note("el principal se cayó: la liga certifica lo que haya")
                    break
        self.final_evaluation()
        return self.certify(["principal"])

    def final_evaluation(self):
        """El principal al final contra el pool de referencia y los explotadores; veredicto pre-registrado."""
        a, st = self.a, self.state
        final = self.root / "principal" / "last.pt"
        if not final.exists() or "evaluacion_final" in st:
            return
        rivals = [("bc", a.bc), ("scripted", "scripted"), ("inicio", a.init)]
        rivals += [(Path(r).parent.name + "_" + Path(r).stem, r) for r in a.refs.split(",") if r]
        rivals += [(f"explotador_{k}", v["explotador"]) for k, v in sorted(st["explotadores"].items()) if v["explotador"]]
        res = {}
        for i, (name, spec) in enumerate(rivals):
            r = self.match(f"final_vs_{name}", final, spec, seed=500 + i)
            res[name] = r and dict(score=r["score"], wins=r["wins"], draws=r["draws"], losses=r["losses"])
        expl = [v["puntaje"] for k, v in sorted(st["explotadores"].items(), key=lambda x: int(x[0]))]
        last = str(len(expl) - 1)
        beaten = [res.get(f"explotador_{k}") for k in st["explotadores"] if k != last]
        ok_expl = None not in expl[:1] + expl[-1:] and len(expl) > 1 and expl[-1] <= expl[0] - a.robust_margin
        ok_prev = all(b and b["score"] >= 0.5 for b in beaten)
        ok_bc = bool(res.get("bc")) and res["bc"]["score"] >= a.min_bc
        st["evaluacion_final"] = res
        st["explotabilidad"] = expl
        st["veredicto_liga"] = dict(veredicto="mas_dificil_de_explotar" if ok_expl and ok_prev and ok_bc
                                    else "sin_mejora_de_robustez",
                                    explotabilidad_baja=bool(ok_expl), le_gana_a_explotadores_viejos=bool(ok_prev),
                                    fuerza_contra_bc=bool(ok_bc))
        self.save()
        self.note("evaluación final de la liga", **st["veredicto_liga"], explotabilidad=expl)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2h", nargs=2, metavar=("A", "B"), help="sólo un enfrentamiento A contra B (y salir)")
    ap.add_argument("--out", default="")
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=str(ROOT / "runs" / "x4_liga"))
    ap.add_argument("--bc", default=str(BC))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default="", help="checkpoint de PPO del que arranca el principal (y el primer objetivo)")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--segments", type=int, default=2, help="tramos del principal (explotadores: uno más)")
    ap.add_argument("--segment-updates", type=int, default=3000, help="~200M decisiones con 1024 × 64 (PLAN E3)")
    ap.add_argument("--exploiter-updates", type=int, default=600)
    ap.add_argument("--exploiter-envs", type=int, default=1024)
    ap.add_argument("--accept", type=float, default=0.55, help="puntaje del explotador para entrar al pool")
    ap.add_argument("--h2h-matches", type=int, default=64, help="partidos por lado en cada enfrentamiento")
    ap.add_argument("--h2h-minutes", type=float, default=3.0)
    ap.add_argument("--main-pool-frac", type=float, default=0.5)
    ap.add_argument("--robust-margin", type=float, default=0.05)
    ap.add_argument("--min-bc", type=float, default=0.8)
    ap.add_argument("--refs", default="", help="checkpoints de referencia para la evaluación final, separados por coma")
    ap.add_argument("--rl-extra", default="", help="argumentos para todos los entrenamientos (no --pool/--init/--updates)")
    ap.add_argument("--export-to", default=str(ROOT / "deploy" / "rs4z" / "x4_rl.onnx"))
    ap.add_argument("--confirm-seed", type=int, default=20261007)
    ap.add_argument("--cert-extra", default="")
    ap.add_argument("--preflight-extra", default="")
    a = ap.parse_args(argv)
    if a.h2h:
        r = h2h(a.h2h[0], a.h2h[1], a.matches, a.minutes, a.seed, a.device)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r), flush=True)
        return 0
    if not a.init:
        ap.error("--init es obligatorio para la liga")
    a.updates, a.extend_to = a.segments * a.segment_updates, a.segments * a.segment_updates
    return League(a).main()


if __name__ == "__main__":
    sys.exit(main())
