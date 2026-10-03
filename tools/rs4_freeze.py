"""Congelar la versión de código de los experimentos RS4 (PLAN_RS4.md, sección 2.1).

La huella es el SHA-256 de cada archivo que influye en simulación, entorno,
entrenamiento, evaluación y exportación, más el commit base y el estado del árbol.
Guarda un archivo .tar con esas fuentes para reproducir exactamente el experimento sin
depender de commits manuales. Un experimento registra el identificador y el lanzador
verifica que el código actual coincide antes de entrenar.

  python -m tools.rs4_freeze            # crea una congelación nueva (o reutiliza una idéntica)
  python -m tools.rs4_freeze --verify reports/rs4_b1/freeze/<id>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_DIR = ROOT / "reports" / "rs4_b1" / "freeze"
ARCHIVE_DIR = ROOT / "runs" / "rs4_b1" / "freeze"
TRACKED = ("sim/**/*.py", "env/**/*.py", "train/**/*.py", "train/**/*.yaml", "eval/**/*.py",
           "bots/**/*.py", "tools/**/*.py", "export/**/*.py", "bridge/*.js", "bridge/*.py",
           "stadiums/**/*.hbs", "stadiums/**/*.json", "train/tasks.yaml")
CRLF, LF = bytes([13, 10]), bytes([10])


def source_files():
    files = set()
    for pattern in TRACKED:
        files.update(p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def content_hash(path):
    """SHA-256 del contenido con finales de línea LF: igual en Windows (CRLF) y en el Pod."""
    return hashlib.sha256(Path(path).read_bytes().replace(CRLF, LF)).hexdigest()


def fingerprint(files=None):
    files = source_files() if files is None else files
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): content_hash(p) for p in files}
    combined = hashlib.sha256("\n".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()).hexdigest()
    return combined, hashes


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze(label=""):
    combined, hashes = fingerprint()
    identity = combined[:16]
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FREEZE_DIR / f"{identity}.json"
    if manifest_path.exists():
        return manifest_path
    archive = ARCHIVE_DIR / f"{identity}.tar"
    with tarfile.open(archive, "w") as tar:
        for relative in hashes:
            tar.add(ROOT / relative, arcname=relative)
    manifest = dict(id=identity, sha256=combined, created=time.strftime("%Y-%m-%d %H:%M:%S"), label=label,
                    git_head=(_git("rev-parse", "HEAD") or "").strip() or None,
                    git_dirty_files=[line[3:] for line in (_git("status", "--porcelain") or "").splitlines()],
                    files=hashes, archive=str(archive.relative_to(ROOT)).replace("\\", "/"),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def verify(manifest_path):
    """Lista de diferencias entre el código actual y la congelación (vacía = idéntico)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _, current = fingerprint()
    frozen = manifest["files"]
    differences = [f"cambiado: {k}" for k in frozen if k in current and current[k] != frozen[k]]
    differences += [f"falta: {k}" for k in frozen if k not in current]
    differences += [f"nuevo: {k}" for k in current if k not in frozen]
    return differences


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="")
    ap.add_argument("--verify")
    a = ap.parse_args()
    if a.verify:
        differences = verify(a.verify)
        print("código idéntico a la congelación" if not differences else "\n".join(differences))
        raise SystemExit(1 if differences else 0)
    path = freeze(a.label)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"congelación {manifest['id']} | {len(manifest['files'])} archivos | {path}")


if __name__ == "__main__":
    main()
