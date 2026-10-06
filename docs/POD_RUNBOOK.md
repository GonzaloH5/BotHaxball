# Runbook del pod: del dataset al RL

Comandos para correr en el pod (1× RTX 3060, unos 8 hilos útiles) el pipeline que se armó el 2026-10-06 en la rama `claude/funny-franklin-iwtrqe`. Todo lo pesado queda en `data/` y `runs/`, que no se versionan.

## 0. Código y dependencias

```bash
git fetch origin claude/funny-franklin-iwtrqe && git checkout claude/funny-franklin-iwtrqe
pip install numpy numba torch orjson onnx onnxruntime pytest
(cd bridge && npm ci) && (cd deploy && npm ci)
python -m pytest -q tests                 # 101 tests (1 se saltea sin el caché de §1)
node deploy/test_x4_room_state.js         # paridad sala ↔ dataset de la obs v3
export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1
```

## 1. Dataset (una sola vez, unos 15 min con 8 hilos)

```bash
python -m tools.rs4_jsonl_cache --source replays_real/stadiums/rsx4 --out data/rs4_jsonl --workers 8
python -m tools.rs4_jsonl_cache --source replays_real/stadiums/haxarg2k23 --out data/haxarg_jsonl
python -m tools.x4_ticks --out data/x4_ticks --workers 8
python -m tools.x4_ticks --cache data/haxarg_jsonl --out data/x4_ticks
python -m tools.x4_metrics --map sanguchito_rs_x4 --out reports/x4/human_metrics_sanguchito.json
```

- Las particiones ya están versionadas en `reports/x4/splits.json`.
- `tools.x4_index` solo hace falta si cambian las grabaciones. Antes de correrlo hay que generar `data/meta/` con `bridge/replay_meta.js` (ver `reports/x4/dataset_audit.md`).

## 2. Conformidad (verificación, opcional)

```bash
python -m tools.rs4z_restart_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*' --workers 8 \
    --out reports/x4/restart_conformance_sanguchito.json
python -m tools.rs4z_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*' --workers 8 \
    --out reports/x4/conformance_sanguchito.json
```

## 3. Imitación (E1)

En GPU el cuello de botella es la featurización en numba: unas 90k muestras/s con 8 hilos.

```bash
python -m learn.x4_bc --out runs/x4_bc/final --maps sanguchito_rs_x4,rs_one --delay 6:15 \
    --steps 150000 --batch 4096 --eval-every 5000 --threads 4 --device cuda
python -m learn.x4_eval --policies runs/x4_bc/final/best.pt,scripted --matches 32 --minutes 3 --record 8 \
    --out reports/x4/eval_bc_final.json
python -m export.to_onnx_x4 --ckpt runs/x4_bc/final/best.pt --out deploy/rs4z/x4_bc.onnx
```

Criterio de avance:
- En desarrollo, la imitación supera a "repetir la tecla de hace 3 ticks" en NLL y en precisión de los cambios de tecla.
- En lazo cerrado le gana al scripted.
- Sus métricas de parecido humano (`human` en el reporte de `x4_eval`) no se alejan de la referencia más que unas pocas veces el techo prueba-vs-entrenamiento de `reports/x4/human_metrics_sanguchito.json`.

## 4. RL con ancla KL (E2 + E3): la cola pre-registrada

Todo el plan de la corrida (pasos, umbrales y qué hacer en cada caso) está en `docs/PRELANZAMIENTO.md` y lo ejecuta sola
`learn/x4_queue.py`: preflight → A/B temprano del shaping (EPV contra franjas, 300 actualizaciones) → RL principal →
recuperación si se corta por deriva → una extensión si el índice de la cadena de pase sigue subiendo → certificación contra
humanos, con confirmación con otras semillas → export a ONNX. Es idempotente: si el pod se reinicia, se relanza el mismo
comando y sigue donde estaba (`runs/x4_cola/queue_state.json`).

- Veredictos en `queue_state.json`: "competitivo_en_pases" (un checkpoint aprobó la certificación y su confirmación),
  "no_competitivo" (se midió y no llegó) o "revisar" (la recuperación también se cortó, o hubo un `fallo_tecnico` o una
  certificación caída: no se llegó a medir lo planeado).
- Duración: el preflight estima 3000 actualizaciones con sus evaluaciones (si pasan de 24 h, la cola no arranca). La
  extensión o la recuperación pueden duplicarla.
- No reentrenar la imitación (§3) mientras corre la cola: compiten por la GPU y por los hilos de numba que midió el
  preflight.

```bash
tmux new -s cola
cd /workspace/HaxballRL      # o donde esté el repo
export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1
python -m learn.x4_queue --root runs/x4_cola --device cuda 2>&1 | tee -a runs/x4_cola.console.log
```

Para salir de tmux sin cortar la corrida: `Ctrl-b d`. Para volver: `tmux attach -t cola`.

Requisitos (los verifica el preflight, que es el primer paso de la cola y la detiene si algo falla):
- `data/x4_ticks/` y `data/human_metrics_sanguchito.samples.npz` (§1);
- `reports/x4/pass_chain_human.json` y `runs/x4_epv/epv.pt` (versionados);
- `NUMBA_NUM_THREADS` definido, ≥ 5 GB libres, la GPU con memoria para la actualización real (1024 × 64).

Qué mirar mientras corre (`runs/x4_cola/rl_principal/`):
- `eval.jsonl` (cada 50 actualizaciones; la fila 0 es la BC): `cadena_pase.indice` (1 = promedio humano) con su `indice_ic90`,
  `indice_sin_valor` (sin las métricas del valor de posesión, que el shaping EPV optimiza), `fallan` y `sin_datos`,
  `cadena_pase_vs_bc` (contra la BC como rival fijo), `vs_bc.score`, `human_w1_mean`, `humanos_dev`
  (NLL y KL sobre estados humanos fijos), `selfplay_kickoff_safety`, `drift`, `new_best`, `new_best_pase`,
  `pase_aprobado`, `brazo_pases_activado` / `brazo_pases_descartado` (una sola decisión, desde la actualización 1000).
- `log.jsonl` (cada actualización): `kl_bc`, `explained_var`, `grad_norm_pi`, `lambda_eff`, `dist_ball_1`, `still_frac`,
  `kick_frac`, `kickoff_frac`, `forfeits_by`, `epv_shaping_abs`, `nonfinite_skipped`.
- `evals/`: la política y los partidos de self-play de cada evaluación (para reanalizar).
- `run_meta.jsonl`: versión del código, GPU e hilos de cada lanzamiento.

Checkpoints:
- `best.pt` (+ `best.json` con `beats_bc`): mejor fuerza contra la BC entre los que cumplen el parecido humano, confirmada
  con una segunda evaluación;
- `best_pase.pt`: mejor índice de la cadena entre los que no pierden contra la BC;
- `pase_aprobado_*.pt`: aprobó el gate de pases en dos evaluaciones seguidas (candidatos a certificación);
- `last.pt` (reanudar), `snap_*.pt` (pool), `stopped.pt` / `stopped.json` (corte por deriva).

Uso manual del trainer (fuera de la cola): `python -m learn.x4_ppo --help`. `--resume` continúa una carpeta; sin
`--resume` el trainer se niega a usar una carpeta con `last.pt`; con `stopped.json` presente no reanuda salvo
`--continue-after-stop`; `--init` arranca desde otro checkpoint de PPO manteniendo la BC como ancla.

Certificación manual de un checkpoint: `python -m learn.x4_certify --ckpt <ckpt> --device cuda --out reports/x4/cert.json`.
Sale con 0 si aprueba y con 3 si no aprueba; otro código es una falla de ejecución. Para confirmar un aprobado, repetirla
con `--seed 20261007`.

## 5. Sala

```bash
# la cola deja deploy/rs4z/x4_rl.onnx (el checkpoint certificado o, si ninguno aprobó, el de mayor índice, marcado
# como no competitivo en runs/x4_cola/queue_state.json)
node deploy/rs4z/join_bots.js --join <link> --count 7 --model deploy/rs4z/x4_rl.onnx --map sanguchito_rs_x4 --trace
```

- El RL se entrena sólo en Sanguchito: el modelo del RL va sólo en esa sala (`--map sanguchito_rs_x4`). En RS ONE y 2K23 se
  sigue con la BC hasta que el RL incluya esos mapas (el trainer informa `vs_bc_rs_one` sólo como monitor de olvido).
- El bot detecta el mapa por el nombre del estadio (`SANGUCHITO`, `2K23`/`HAXARG`; si no, RS ONE). `--map <nombre>` lo fuerza (también desde `join_bots.js`).
- Cada bot usa un hilo de ONNX (`--ort-threads`, por defecto 1). Así se evita la contención que el 2026-10-06 hizo decidir cada 9–10 ticks en vez de cada 3.
- La traza (`--trace`) más la grabación del host sirven para medir latencia con `tools.rs4z_room_latency` y para conformidad.
