# Runbook del pod: del dataset al RL

Comandos para correr en el pod (1× RTX 3060, unos 8 hilos útiles) el pipeline que se armó el 2026-10-06 en la rama `claude/funny-franklin-iwtrqe`. Todo lo pesado queda en `data/` y `runs/`, que no se versionan.

## 0. Código y dependencias

```bash
git fetch origin claude/funny-franklin-iwtrqe && git checkout claude/funny-franklin-iwtrqe
pip install numpy numba torch orjson onnx onnxruntime pytest
(cd bridge && npm ci) && (cd deploy && npm ci)
python -m pytest -q tests                 # 81 tests
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

## 4. RL con ancla KL (E2 + E3)

**Antes de lanzar, el preflight** (2–5 min). Verifica GPU, caché, particiones, referencia humana y checkpoint, y corre una mini-corrida real (StateBank, rollouts, actualización, snapshot, evaluación y reanudación) con medición de RAM y memoria de GPU:

```bash
python -m learn.x4_preflight --bc runs/x4_bc/final_sangu_rsone/best.pt --device cuda --out runs/preflight.json
```

En GPU el preflight también corre una actualización sintética con el buffer de la corrida real (1024 partidos × 64 pasos, `escala_real`) y reporta el pico de RAM y de memoria de GPU. Si esa prueba falla por memoria, usar `--rollout 32`.

Si termina en `PREFLIGHT OK`, lanzar dentro de `tmux` (o con `nohup`) con un bucle que reintenta si el proceso muere:

```bash
tmux new -s rl
OUT=runs/x4_ppo/lam02
until python -m learn.x4_ppo --bc runs/x4_bc/final_sangu_rsone/best.pt --out $OUT --device cuda \
    --envs 1024 --rollout 64 --updates 3000 --lambda-dist 0.2 --lambda-decay 0.9995 --lambda-min 0.05 \
    --critic-warmup 20 --human-starts 0.4 --pool-frac 0.2 --eval-every 25 --resume; do
  [ -f $OUT/stopped.json ] && break; echo "reintento en 60 s"; sleep 60
done 2>&1 | tee -a $OUT.console.log
```

- **Reanudación:** `--resume` continúa desde `<out>/last.pt` con optimizadores, normalización del valor, pool de snapshots, `best.pt` y contadores. `last.pt` se guarda cada 10 actualizaciones y después de cada evaluación, de forma atómica (un corte durante la escritura no lo rompe; si igual no se puede leer, se reanuda desde el snapshot más reciente). Si la corrida se corta, el bucle la relanza.
- Una corrida cortada por deriva no se reanuda sola: con `stopped.json` presente, `--resume` termina con un mensaje. Para seguirla igual: `--continue-after-stop`.
- **Seguir desde otra corrida:** `--init runs/x4_ppo/<otra>/best.pt` toma política, crítico y normalización de ese checkpoint; el ancla KL sigue siendo `--bc`.
- **Corte automático** (criterio pre-registrado, `--stop-on-drift 2`): la corrida se detiene sola cuando dos evaluaciones seguidas cumplen cualquiera de:
  - `vs_bc.score` < 0,4;
  - `human_w1_mean` > 1,0;
  - `selfplay.dist_ball_1` > 200 px (la BC está en ~110 y los humanos en 84).

  Deja `stopped.pt` y `stopped.json` con el motivo. Qué hacer después: volver a `best.pt`, subir λ o bajar `--lr`.
- **Mapas:** por defecto, solo Sanguchito (`--maps sanguchito_rs_x4:1.0`). Es el único mapa con saques verificados. Sumar `rs_one`/`haxarg_2k23` recién cuando su conformidad de saques esté resuelta (`reports/x4/conformance_x4.md`).
- **Punto de partida:** la imitación entrenada en CPU esa noche está versionada en `runs/x4_bc/final_sangu_rsone/best.pt` (164M muestras, Sanguchito + RS ONE, retardo 6–15). Se puede empezar sin el paso 3, aunque conviene reentrenarla más larga en GPU: la NLL todavía bajaba y pasa poco (ver "Pases").
- **Barrido corto de λ** (plan E3): `--lambda-dist` en {0,06; 0,2; 0,4}, con y sin `--lambda-decay 0.9995`, con `--out` distintos. Por defecto se usa 0,2 (VPT), porque en dos corridas chicas de CPU con 0,06 la política se alejó de la pelota y empeoró contra la BC desde la actualización 50.
- **Penalización por saque vencido:** `--forfeit-penalty 0.1` (por defecto). Evita el equilibrio "nadie saca" (`docs/PLAN.md` E3). No hay que bajarla a 0 sin mirar `forfeits` en `log.jsonl` y `selfplay.kickoff_wait_s` en `eval.jsonl`.
- **Pases** (el objetivo de juego colectivo):
  - Referencia humana agregada (`reports/x4/pass_stats.json`): 9,2 pases/min, 0,36 de pases/(pases+pérdidas) y 0,55 pases por posesión.
  - La BC de partida está en 3,1 / 0,19 / 0,23.
  - Se sigue en `eval.jsonl` → `selfplay_pases`, que usa la misma definición que la referencia humana. En `log.jsonl`, `rollout_passes_per_min_kernel` cuenta toques sucesivos de compañeros en los rollouts: da 40–57% más que la métrica humana y solo sirve para ver la tendencia.
  - **Criterio pre-registrado** (plan E3): si a 100–300M decisiones `selfplay_pases.pases_por_min` < 6 o `pases_sobre_pases_mas_perdidas` < 0,28, se lanza el brazo de TiZero en paralelo: el mismo comando con `--pass-bonus 0.05` y otro `--out`.
- **Memoria:** con `--envs 1024 --rollout 64` el pico es de unos 3,2 GB de RAM (rollout más copia para la actualización), más ~1 GB del StateBank de estados humanos. La actualización sube el rollout entero a la GPU (~1 GB). Si falta memoria, usar `--rollout 32`. El preflight informa los picos.
- **Pool de rivales:** un snapshot cada `--snapshot-every` actualizaciones, como máximo `--pool-max 10` además de la BC. Cuando se llena, sale del pool (el archivo queda) el snapshot al que el aprendiz más le gana.
- **Qué mirar:**
  - `runs/x4_ppo/*/eval.jsonl`: `vs_bc.score` (victorias contra la BC con latencia de sala), `human_w1_mean`, `selfplay_safety` (saques iniciales que nadie ejecuta), `selfplay` (patadas, pases y goles por minuto, espera del saque inicial; humanos p50: 3,2 / 8,8 / 0,25 / 3,9 s), `selfplay_pases` y `drift`;
  - `log.jsonl`: `kl_bc`, `entropy`, `clipfrac`, `forfeits` y `rollout_passes_per_min_kernel`.
- **Shaping:** se retira para siempre cuando `vs_bc.score` ≥ 0,75 en `--retire-patience 2` evaluaciones seguidas.
- **Evaluación:** el azar de la evaluación depende solo de la semilla y de la actualización, así dos checkpoints se comparan con los mismos partidos.
- **Checkpoints:**
  - `best.pt`: el de mejor `vs_bc.score` entre los que cumplen `--human-gate`. Un candidato se confirma con una segunda evaluación de fuerza con otras semillas y se guarda con el promedio de las dos (evita quedarse con un pico de suerte);
  - `snap_*.pt`: snapshots que entran al pool;
  - `last.pt`: para reanudar.

## 5. Sala

```bash
python -m export.to_onnx_x4 --ckpt runs/x4_ppo/lam02/best.pt --out deploy/rs4z/x4_ppo.onnx
node deploy/rs4z/join_bots.js --join <link> --count 7 --model deploy/rs4z/x4_ppo.onnx --trace
```

- El bot detecta el mapa por el nombre del estadio (`SANGUCHITO`, `2K23`/`HAXARG`; si no, RS ONE). `--map <nombre>` lo fuerza (también desde `join_bots.js`).
- Cada bot usa un hilo de ONNX (`--ort-threads`, por defecto 1). Así se evita la contención que el 2026-10-06 hizo decidir cada 9–10 ticks en vez de cada 3.
- La traza (`--trace`) más la grabación del host sirven para medir latencia con `tools.rs4z_room_latency` y para conformidad.
