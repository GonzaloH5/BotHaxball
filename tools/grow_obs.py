"""Cirugía de checkpoint: agranda la entrada del modelo cuando la obs gana features nuevas, sin perder
lo aprendido.

Las entradas nuevas entran con peso CERO en la primera capa (y media 0 / varianza 1 en la normalización),
así que el modelo operado juega EXACTAMENTE igual que antes (se verifica) y aprende a usarlas a medida que
sigue entrenando. Se operan el modelo, el estado de Adam y todos los rivales guardados de la liga.

Uso típico (después de agregar features a env/haxball_env.py):
    python -m tools.grow_obs runs/multi/latest.pt --self end:10            # 10 features al final del bloque propio
    python -m tools.grow_obs runs/multi/latest.pt --self 22:3 --self end:2  # 3 en la posición 22 (vieja) y 2 al final
    python -m tools.grow_obs runs/multi/latest.pt --ent end:1               # 1 feature más por entidad
    python -m train.multitask --run multi --resume

Las posiciones son índices de la obs VIEJA: `22:3` inserta 3 entradas antes de la que era la 22.
Por defecto escribe <checkpoint>_grown.pt; con --in-place pisa el original y deja una copia .bak.
Soporta SetActorCritic (obs universal, multi-tarea) y ActorCritic (obs plana).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from train.model import build_model


def parse_spec(specs: list[str], old_dim: int, min_pos: int = 0) -> list[tuple[int, int]]:
    """["22:3", "end:2"] -> [(22, 3), (old_dim, 2)], ordenado y validado (posiciones de la obs vieja)."""
    out = []
    for s in specs:
        pos, cnt = s.split(":")
        p = old_dim if pos == "end" else int(pos)
        c = int(cnt)
        if not (min_pos <= p <= old_dim) or c <= 0:
            raise ValueError(f"inserción inválida {s!r} (posición {min_pos}..{old_dim}, cantidad > 0)")
        out.append((p, c))
    return sorted(out)


def index_map(old_len: int, ins: list[tuple[int, int]]) -> torch.Tensor:
    """Índice nuevo de cada posición vieja."""
    shift = torch.zeros(old_len, dtype=torch.long)
    for p, c in ins:
        shift[p:] += c
    return torch.arange(old_len) + shift


def insert(t: torch.Tensor, dim: int, ins: list[tuple[int, int]], fill: float = 0.0,
           offset: int = 0) -> torch.Tensor:
    """Inserta en `dim` las posiciones `ins` (relativas a `offset`) rellenas con `fill`."""
    if not ins:
        return t
    old = t.shape[dim]
    shifted = [(p + offset, c) for p, c in ins]
    new_shape = list(t.shape)
    new_shape[dim] = old + sum(c for _, c in ins)
    out = torch.full(new_shape, fill, dtype=t.dtype)
    idx = index_map(old, shifted)
    out.index_copy_(dim, idx, t)
    return out


def _plan(cfg: dict, self_ins, ent_ins):
    """(clave del state_dict, dim, inserciones, relleno, offset) + config nueva."""
    kind = cfg.get("type", "mlp")
    new = dict(cfg)
    ops = []
    if kind == "set":
        if self_ins:
            ops += [("self_norm.mean", 0, self_ins, 0.0, 0), ("self_norm.var", 0, self_ins, 1.0, 0),
                    ("pi_body.0.weight", 1, self_ins, 0.0, 0), ("v_body.0.weight", 1, self_ins, 0.0, 0)]
            if cfg.get("pooling") == "attention":
                ops.append(("q.weight", 1, self_ins, 0.0, 0))
            new["self_dim"] = cfg["self_dim"] + sum(c for _, c in self_ins)
        if ent_ins:
            # las 2 primeras (presente, es_rival) no se normalizan: ent_norm empieza en la entrada 2
            ops += [("ent_norm.mean", 0, ent_ins, 0.0, -2), ("ent_norm.var", 0, ent_ins, 1.0, -2),
                    ("ent_enc.0.weight", 1, ent_ins, 0.0, 0)]
            new["ent_dim"] = cfg["ent_dim"] + sum(c for _, c in ent_ins)
    elif kind == "mlp":
        if ent_ins:
            raise ValueError("la obs plana no tiene entidades: usar --self")
        ops += [("norm.mean", 0, self_ins, 0.0, 0), ("norm.var", 0, self_ins, 1.0, 0),
                ("pi_body.0.weight", 1, self_ins, 0.0, 0), ("v_body.0.weight", 1, self_ins, 0.0, 0)]
        new["obs_dim"] = cfg["obs_dim"] + sum(c for _, c in self_ins)
    else:
        raise ValueError(f"modelo {kind!r} no soportado (sólo 'set' y 'mlp')")
    return ops, new


def grow_state_dict(sd: dict, ops) -> dict:
    sd = dict(sd)
    for key, dim, ins, fill, off in ops:
        sd[key] = insert(sd[key], dim, ins, fill, off)
    return sd


def grow_opt_state(opt_sd: dict, cfg: dict, ops) -> dict:
    """Adam guarda sus momentos por índice de parámetro (orden de model.parameters())."""
    names = [n for n, _ in build_model(cfg).named_parameters()]
    by_name = {key: (dim, ins, off) for key, dim, ins, _, off in ops}
    state = {}
    for i, st in opt_sd["state"].items():
        st = dict(st)
        name = names[int(i)]
        if name in by_name:
            dim, ins, off = by_name[name]
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    st[k] = insert(st[k], dim, ins, 0.0, off)
        state[i] = st
    return {**opt_sd, "state": state}


def _obs_widths(cfg: dict, E: int = 3):
    if cfg.get("type") == "set":
        return cfg["self_dim"], cfg["ent_dim"], E
    return cfg["obs_dim"], 0, 0


@torch.no_grad()
def verify(old_cfg, old_sd, new_cfg, new_sd, self_ins, ent_ins, n: int = 256, seed: int = 0) -> float:
    """Mismo partido visto con la obs vieja y la nueva (valores cualquiera en las entradas nuevas):
    las salidas tienen que ser iguales. Devuelve el error máximo."""
    g = torch.Generator().manual_seed(seed)
    old_m, new_m = build_model(old_cfg).eval(), build_model(new_cfg).eval()
    old_m.load_state_dict(old_sd)
    new_m.load_state_dict(new_sd)
    S, D, E = _obs_widths(old_cfg)
    s_old = torch.randn(n, S, generator=g) * 2
    s_new = insert(s_old, 1, self_ins)
    new_idx = torch.ones(s_new.shape[1], dtype=torch.bool)
    new_idx[index_map(S, self_ins)] = False
    s_new[:, new_idx] = torch.randn(n, int(new_idx.sum()), generator=g) * 3
    parts_old, parts_new = [s_old], [s_new]
    if E:
        e_old = torch.randn(n, E, D, generator=g) * 2
        e_old[..., :2] = (torch.rand(n, E, 2, generator=g) > 0.5).float()
        e_old[:, 0, 0] = 1.0
        e_new = insert(e_old, 2, ent_ins)
        new_e = torch.ones(e_new.shape[2], dtype=torch.bool)
        new_e[index_map(D, ent_ins)] = False
        e_new[..., new_e] = torch.randn(n, E, int(new_e.sum()), generator=g) * 3
        parts_old.append(e_old.reshape(n, -1))
        parts_new.append(e_new.reshape(n, -1))
    lo, vo = old_m(torch.cat(parts_old, 1))
    ln, vn = new_m(torch.cat(parts_new, 1))
    return max((lo - ln).abs().max().item(), (vo - vn).abs().max().item())


def grow_checkpoint(ck: dict, self_specs: list[str], ent_specs: list[str]) -> tuple[dict, float]:
    old_cfg = ck["model_config"]
    S, D, _ = _obs_widths(old_cfg)
    self_ins = parse_spec(self_specs, S)
    ent_ins = parse_spec(ent_specs, D, min_pos=2) if ent_specs else []
    ops, new_cfg = _plan(old_cfg, self_ins, ent_ins)
    out = dict(ck)
    out["model"] = grow_state_dict(ck["model"], ops)
    out["model_config"] = new_cfg
    if "opt" in ck:
        out["opt"] = grow_opt_state(ck["opt"], old_cfg, ops)
    if "league" in ck:
        out["league"] = [(e[0], grow_state_dict(e[1], ops), *e[2:]) for e in ck["league"]]
    err = verify(old_cfg, ck["model"], new_cfg, out["model"], self_ins, ent_ins)
    return out, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--self", dest="self_specs", action="append", default=[], metavar="POS:N",
                    help="insertar N entradas en el bloque propio antes de POS (o 'end'); repetible")
    ap.add_argument("--ent", dest="ent_specs", action="append", default=[], metavar="POS:N",
                    help="insertar N entradas por entidad antes de POS (>= 2, o 'end'); repetible")
    ap.add_argument("--out", default=None)
    ap.add_argument("--in-place", action="store_true", help="pisar el checkpoint (deja una copia .bak)")
    args = ap.parse_args()
    if not args.self_specs and not args.ent_specs:
        ap.error("nada para insertar: usar --self y/o --ent")
    src = Path(args.ckpt)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out_ck, err = grow_checkpoint(ck, args.self_specs, args.ent_specs)
    if err > 1e-5:
        raise SystemExit(f"FALLO: el modelo operado no juega igual (error {err:.2e}); no se escribió nada")
    if args.in_place:
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, bak)
        dst = src
        print(f"copia de seguridad: {bak}")
    else:
        dst = Path(args.out) if args.out else src.with_name(src.stem + "_grown.pt")
    torch.save(out_ck, dst)
    o, n = ck["model_config"], out_ck["model_config"]
    dims = (lambda c: f"propio {c['self_dim']}, entidad {c['ent_dim']}") if o.get("type") == "set" \
        else (lambda c: f"obs {c['obs_dim']}")
    print(f"{src} -> {dst}\n  antes: {dims(o)}\n  ahora: {dims(n)}\n"
          f"  liga: {len(out_ck.get('league', []))} rivales operados | salida idéntica (error {err:.1e})")


if __name__ == "__main__":
    main()
