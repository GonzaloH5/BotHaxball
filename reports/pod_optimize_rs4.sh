#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rs4_optimize_20261002
out=/workspace/HaxballRL/pod_logs/optimization_20261002
mkdir -p "$out"
exec 8>/workspace/HaxballRL/pod_logs/rs4_optimization_job.lock
flock -n 8 || { echo 'Ya existe una optimización activa'; exit 1; }
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/HaxballRL/.venv/bin/python -u \
    -m tools.optimize_rs4_pod --run-dir /workspace/HaxballRL/runs/rs4_v3_public \
    --out "$out" --wait --apply
bash /workspace/rs4_update_fdb186d/pod_apply_collective_fixes.sh
/workspace/HaxballRL/.venv/bin/python - "$out" <<'PY'
from pathlib import Path
import json, subprocess, sys
out = Path(sys.argv[1])
status = json.loads((out/'status.json').read_text())
status.update(state='ready_for_training', code_revision=subprocess.check_output(
    ['git','-C','/workspace/HaxballRL','rev-parse','HEAD'], text=True).strip())
(out/'status.json').write_text(json.dumps(status, indent=2))
print('Optimización y actualización terminadas. El entrenamiento principal queda detenido y listo para continuar.')
PY
