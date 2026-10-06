"""Sincronizar el código del repo con el Pod (sin datos ni deploy/; de checkpoints, sólo la imitación X4 versionada).

Los archivos de texto viajan con finales de línea LF: el árbol de Windows usa CRLF
(core.autocrlf) y en Linux eso aparece como cambios falsos en git. Sólo agrega o
reemplaza archivos; nunca borra nada en el Pod.

  python -m tools.pod_sync                 # código
  python -m tools.pod_sync --extra replays_real/stadiums/rsx4
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POD = os.environ.get("HAXBALL_POD", "aa368f72-ad42-43ee-b278-61d0a50706de@96.28.88.208")
PORT = os.environ.get("HAXBALL_POD_PORT", "40900")
REMOTE = os.environ.get("HAXBALL_POD_REPO", "/workspace/HaxballRL")
CODE_DIRS = ("sim", "env", "train", "eval", "bots", "tools", "tests", "export", "bridge", "stadiums",
             "learn", "docs", "reports", "runs/x4_bc/final_sangu_rsone")   # X4: código, particiones e imitación de partida
TEXT = {".py", ".yaml", ".yml", ".js", ".json", ".hbs", ".md", ".txt", ".sh", ".cfg", ".toml"}


def code_files():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    files = set(git("ls-files", "--", *CODE_DIRS)) | set(git("ls-files", "--others", "--exclude-standard", "--", *CODE_DIRS))
    files |= {f for f in ("PLAN_RS4.md",) if (ROOT / f).exists()}
    return sorted(f for f in files if "node_modules" not in f and "__pycache__" not in f and not f.endswith(".pyc")
                  and (ROOT / f).is_file())


def build(files, target):
    with tarfile.open(target, "w") as tar:
        for relative in files:
            path = ROOT / relative
            data = path.read_bytes()
            if path.suffix.lower() in TEXT:
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(relative.replace("\\", "/"))
            info.size, info.mtime, info.mode = len(data), int(path.stat().st_mtime), 0o644
            tar.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", nargs="*", default=[], help="carpetas o archivos adicionales (p. ej. grabaciones)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = code_files()
    for extra in a.extra:
        path = ROOT / extra
        files += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (path.rglob("*") if path.is_dir() else [path]) if p.is_file()]
    if a.dry_run:
        print("\n".join(files))
        return
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "sync.tar"
        build(files, archive)
        subprocess.run(["scp", "-P", PORT, str(archive), f"{POD}:/workspace/sync.tar"], check=True)
    subprocess.run(["ssh", "-p", PORT, POD, f"cd {REMOTE} && tar -xf /workspace/sync.tar && rm /workspace/sync.tar"], check=True)
    print(f"{len(files)} archivos sincronizados con {POD}:{REMOTE}")


if __name__ == "__main__":
    main()
