#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/HaxballRL
target=fdb186d
cd "$repo"

# Refuse to swap evaluator/trainer sources underneath a running process.
.venv/bin/python - <<'PY'
from pathlib import Path
import os
import sys
root = Path('/workspace/HaxballRL').resolve()
modules = {'tools.run_rs4_v3', 'tools.evaluate_rs4_v3', 'train.multitask',
           'tools.benchmark_multitask', 'eval.render'}
active = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        args = entry.joinpath('cmdline').read_bytes().decode().split('\0')
        cwd = entry.joinpath('cwd').resolve(strict=True)
        if modules.intersection(args) and (cwd == root or root in cwd.parents):
            active.append((entry.name, ' '.join(args)))
    except (OSError, UnicodeError):
        pass
if active:
    for pid, command in active:
        print(f'Proceso activo {pid}: {command}')
    print('La actualización está preparada. Repetir este comando cuando termine el runner actual.')
    sys.exit(75)
PY

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo 'Hay cambios locales en el código del Pod. No se sobrescribieron.'
    exit 1
fi
git cat-file -e "$target^{commit}"
mkdir -p pod_logs
before=pod_logs/collective_update_before.json
.venv/bin/python - "$before" <<'PY'
from pathlib import Path
import hashlib, json, sys
run = Path('runs/rs4_v3_public')
paths = [run/name for name in ('parent.pt', 'teacher.pt', 'champion.pt', 'ledger.json',
         'specialization.json', 'control/latest.pt', 'memory/latest.pt',
         'control/config.yaml', 'memory/config.yaml')]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()
Path(sys.argv[1]).write_text(json.dumps({str(p):digest(p) for p in paths if p.is_file()}, indent=2))
PY

git merge --ff-only "$target"
.venv/bin/python -m tools.run_rs4_v3 --run rs4_v3_public --resume --dry-run
.venv/bin/python - "$before" <<'PY'
from pathlib import Path
import hashlib, json, sys, torch, yaml
for name, expected in json.loads(Path(sys.argv[1]).read_text()).items():
    h = hashlib.sha256()
    with Path(name).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    assert h.hexdigest() == expected, f'Archivo del run modificado: {name}'
assert torch.cuda.is_available(), 'CUDA no está disponible'
print('GPU:', torch.cuda.get_device_name(0))
for branch in ('control', 'memory'):
    cfg = yaml.safe_load(Path(f'runs/rs4_v3_public/{branch}/config.yaml').read_text())
    ppo = cfg['ppo']
    assert ppo['device'] == 'cuda', f'{branch}: device no es cuda'
    assert ppo['torch_threads'] == 2 and ppo['numba_threads'] == 4, f'{branch}: revisar threads'
print('Actualización aplicada. Pesos, optimizadores, configuración y ledger conservados.')
PY
git rev-parse HEAD > pod_logs/collective_update_applied.txt
