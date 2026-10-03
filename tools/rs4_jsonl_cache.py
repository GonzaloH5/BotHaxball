"""Convertir grabaciones .hbr2 a JSONL tick a tick (motor original de HaxBall) una sola vez.

Cada grabación se identifica por el SHA-256 de su contenido: copias con otro nombre
se registran como duplicadas y no se vuelven a convertir. El índice guarda nombre,
hash, estadios y estado de la conversión; nunca modifica las grabaciones.

  python -m tools.rs4_jsonl_cache --source replays_real/stadiums/rsx4 --out data/rs4_jsonl
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def open_ticks(path):
    """Abrir JSONL plano o comprimido (.jsonl.gz) como texto."""
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def compress(jsonl):
    """El JSONL tick a tick ocupa ~130 MB por partido; comprimido, ~10 veces menos."""
    jsonl = Path(jsonl)
    target = jsonl.with_suffix(".jsonl.gz")
    with jsonl.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as sink:
        shutil.copyfileobj(source, sink, 1 << 20)
    jsonl.unlink()
    return target


def header_and_stadiums(jsonl):
    stadiums, header = [], None
    with open_ticks(jsonl) as handle:
        for line in handle:
            if '"type": "tick"' in line[:20] or '"type":"tick"' in line[:20]:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "header":
                header = item
                stadiums.append(dict(frame=0, name=item.get("stadium"), file=item.get("stadiumFile")))
            elif item.get("name") == "stadium_change":
                stadiums.append(dict(frame=item.get("frame"), name=item.get("stadium"), file=item.get("stadiumFile")))
    return header, stadiums


def convert(path, out, max_minutes):
    with tempfile.TemporaryDirectory(prefix="rs4-jsonl-") as temp:
        result = subprocess.run(["node", str(ROOT / "bridge" / "replay_to_jsonl.js"), str(path), "--out", temp,
                                 "--max-minutes", str(max_minutes)], cwd=ROOT, capture_output=True, text=True,
                                timeout=15 * 60)
        produced = list(Path(temp).glob("*.jsonl"))
        if result.returncode != 0 or not produced:
            raise RuntimeError((result.stderr or result.stdout)[-400:])
        target = out / path.stem
        target.mkdir(parents=True, exist_ok=True)
        for item in Path(temp).iterdir():
            shutil.move(str(item), target / item.name)
        return compress(target / produced[0].name), result.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(ROOT / "replays_real" / "stadiums" / "rsx4"))
    ap.add_argument("--out", default=str(ROOT / "data" / "rs4_jsonl"))
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=1, help="conversiones simultáneas (procesos node)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"recordings": {}}
    by_hash = {row["sha256"]: name for name, row in index["recordings"].items() if row.get("sha256")}
    # Entradas convertidas antes de comprimir: comprimir en el lugar.
    for row in index["recordings"].values():
        if row.get("status") == "ok" and row["jsonl"].endswith(".jsonl") and (out / row["jsonl"]).exists():
            row["jsonl"] = str(compress(out / row["jsonl"]).relative_to(out))
            index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    pending = []
    for path in sorted(Path(a.source).glob("*.hbr2")):
        if path.name in index["recordings"] and index["recordings"][path.name].get("status") in ("ok", "duplicate"):
            continue
        digest = sha256(path)
        row = dict(sha256=digest, source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                   bytes=path.stat().st_size)
        if digest in by_hash and by_hash[digest] != path.name:
            row.update(status="duplicate", duplicate_of=by_hash[digest])
            index["recordings"][path.name] = row
            print(f"duplicate {path.name}", flush=True)
            continue
        by_hash[digest] = path.name  # reservado: copias posteriores se registran como duplicadas
        pending.append((path, row))
    from concurrent.futures import ThreadPoolExecutor
    def work(item):
        path, row = item
        try:
            jsonl, log = convert(path, out, a.max_minutes)
            header, stadiums = header_and_stadiums(jsonl)
            row.update(status="ok", jsonl=str(jsonl.relative_to(out)).replace("\\", "/"), log=log,
                       room=(header or {}).get("room"), stadiums=stadiums)
        except Exception as error:  # una grabación rota no detiene el lote
            row.update(status="error", error=str(error))
        return path, row
    # Hilos alcanzan: el trabajo pesado ocurre en procesos node independientes.
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for path, row in pool.map(work, pending):
            index["recordings"][path.name] = row
            index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{row['status']:9s} {path.name}", flush=True)
    counts = {}
    for row in index["recordings"].values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
