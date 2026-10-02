#!/usr/bin/env bash
set -euo pipefail
bash /workspace/rs4_update_fdb186d/pod_apply_collective_fixes.sh
cd /workspace/HaxballRL
log="pod_logs/rs4_collective_$(date -u +%Y%m%d_%H%M%S).log"
nohup .venv/bin/python -u -m tools.run_rs4_v3 \
    --run rs4_v3_public --resume --segment-steps 100000000 \
    >> "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > pod_logs/rs4_prueba.pid
printf '%s\n' "$log" > pod_logs/rs4_collective_latest_log.txt
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 40 "$log"
    echo 'El runner terminó durante el arranque; revisar el log.'
    exit 1
fi
echo "Runner iniciado: PID $pid"
echo "Log: /workspace/HaxballRL/$log"
echo 'Empieza con la reevaluación requerida por la nueva firma y después entrena hasta 100M pasos adicionales.'
tail -n 12 "$log"
