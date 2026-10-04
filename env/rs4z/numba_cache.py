"""Invalidación de la caché de numba por dependencias.

`@njit(cache=True)` sólo revisa el archivo de la propia función: si una función cacheada llama a otra
de otro módulo que cambió, numba sigue usando el código viejo (pasó con RS-Pro). `guard` compara la
huella de las dependencias y borra la caché del módulo cuando difiere. Llamar al importar el módulo,
antes de la primera compilación.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def guard(module_file, deps):
    module = Path(module_file).resolve()
    digest = hashlib.sha256()
    for dep in [module, *[ROOT / d for d in deps]]:
        digest.update(dep.read_bytes())
    stamp_dir = module.parent / "__pycache__"
    stamp_dir.mkdir(exist_ok=True)
    stamp = stamp_dir / f"{module.stem}.numba_deps"
    value = digest.hexdigest()
    try:
        if stamp.read_text() == value:
            return
    except OSError:
        pass
    for cached in stamp_dir.glob(f"{module.stem}.*.nb[ic]"):
        try:
            cached.unlink()
        except OSError:
            pass
    stamp.write_text(value)
