"""Inventario de checkpoints RS4 (PLAN_RS4.md, sección 2.1).

Recorre checkpoints locales (incluido el archivo bajado del Pod) y la lista de hashes
calculada en el Pod. Por checkpoint: hash, tamaño, tipo de modelo, pasos, estado del
programa RS4, tareas y si está disponible localmente. Nunca modifica los checkpoints.

  python -m tools.rs4_inventory --pod-hashes runs/pod_archive_20261003/pod_checkpoint_sha256.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "rs4_b1"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def describe(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model_config", {})
    program = ck.get("rs4_program_state") or {}
    row = dict(kind="bc" if "bc" in ck else ("ppo" if "opt" in ck else "modelo"),
               model_type=model.get("type"), pooling=model.get("pooling"), ent_hidden=model.get("ent_hidden"),
               memory_size=model.get("memory_size"), critic_features=model.get("critic_features", 0),
               public_signals=model.get("public_signals_version", 0), rule_observation=model.get("rule_observation"),
               steps=ck.get("steps"), iteration=ck.get("iteration"), tasks=(ck.get("env") or {}).get("tasks"),
               program_version=program.get("version"), program_relative_steps=program.get("relative_steps"),
               league_size=len(ck.get("league", [])) if isinstance(ck.get("league"), list) else None)
    if "bc" in ck:
        history = ck["bc"].get("history") or [{}]
        row.update(bc_val_acc=history[-1].get("acc"), bc_val_loss=history[-1].get("loss"),
                   bc_train_samples=ck["bc"].get("train_samples"), bc_data=ck["bc"].get("data_dir"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=["runs"])
    ap.add_argument("--pod-hashes")
    ap.add_argument("--min-bytes", type=int, default=100_000)
    a = ap.parse_args()
    rows, by_hash = [], {}
    for root in a.roots:
        for path in sorted((ROOT / root).rglob("*.pt")):
            if path.stat().st_size < a.min_bytes or "freeze" in path.parts:
                continue
            digest = sha256(path)
            row = dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), sha256=digest, bytes=path.stat().st_size,
                       local=True)
            try:
                row.update(describe(path))
            except Exception as error:  # checkpoint ilegible: se registra, no se oculta
                row["error"] = str(error)
            rows.append(row)
            by_hash.setdefault(digest, []).append(row["path"])
    pod_only = []
    if a.pod_hashes:
        for line in Path(a.pod_hashes).read_text(encoding="utf-8").splitlines():
            digest, _, path = line.partition("  ")
            if not digest:
                continue
            if digest in by_hash:
                for row in rows:
                    if row["sha256"] == digest:
                        row.setdefault("pod_paths", []).append(path)
            else:
                pod_only.append(dict(path=path, sha256=digest, local=False))
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoints=rows, pod_only=pod_only,
                  duplicates={h: paths for h, paths in by_hash.items() if len(paths) > 1})
    (OUT / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Inventario de checkpoints RS4", "",
             f"{len(rows)} checkpoints locales, {len(pod_only)} sólo en el Pod (hash registrado, no descargados).", "",
             "| Checkpoint | Tipo | Modelo | Pasos (programa) | SHA-256 |", "|---|---|---|---|---|"]
    for row in rows:
        model = f"{row.get('model_type')}/{row.get('pooling')}/{row.get('ent_hidden')}" + (
            f"+GRU{row['memory_size']}" if row.get("memory_size") else "") + (
            f"+crítico{row['critic_features']}" if row.get("critic_features") else "")
        steps = row.get("program_relative_steps") if row.get("program_relative_steps") is not None else row.get("steps")
        lines.append(f"| `{row['path']}` | {row.get('kind', '?')} | {model} | {steps} | `{row['sha256'][:12]}` |")
    if pod_only:
        lines += ["", "## Sólo en el Pod", ""] + [f"- `{r['path']}` `{r['sha256'][:12]}`" for r in pod_only]
    (OUT / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} locales, {len(pod_only)} sólo en el Pod -> {OUT / 'inventory.md'}")


if __name__ == "__main__":
    main()
