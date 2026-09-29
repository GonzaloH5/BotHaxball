# HaxballRL

Bot de HaxBall entrenado con aprendizaje por refuerzo: PPO, self-play, liga de rivales (PFSP) y currículo.

## Piezas

| Carpeta | Qué hace |
|---|---|
| `sim/` | Motor de física de HaxBall reimplementado con numba, en lote (~5M ticks/s en 1v1 con 1024 partidos). Validado contra partidas reales: error 0 por tick |
| `stadiums/` | `classic.hbs` = Classic real exportado de una sala; `classic_repo.hbs` = versión del repo oficial (difiere) |
| `env/` | Entorno multi-agente vectorizado: obs espejada (un solo modelo sirve para rojo y azul), 18 acciones, frame-skip 3, recompensa de gol + shaping por potencial |
| `bots/scripted.py` | Bot por reglas: baseline y primer rival |
| `train/` | PPO + self-play + liga (snapshots, PFSP, Elo) + currículo (`config.yaml`) |
| `eval/` | `arena.py` (torneo con Elo) y `render.py` (replay HTML) |
| `export/` | Checkpoint → ONNX + fixture de verificación |
| `deploy/` | Bot en Node (node-haxball + onnxruntime-node) que juega en una sala real |

## Uso

```bash
# instalar (Python 3.11+)
python -m venv .venv
.venv/Scripts/python -m pip install numpy numba gymnasium pytest pyyaml tensorboard onnx onnxruntime onnxscript
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# tests de física y velocidad
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python -m sim.benchmark

# entrenar (Ctrl+C guarda; --resume continúa)
.venv/Scripts/python -m train.ppo_selfplay --run classic_1v1
.venv/Scripts/tensorboard --logdir runs

# evaluar
.venv/Scripts/python -m eval.arena "runs/classic_1v1/ckpt_*.pt" scripted --games 128
.venv/Scripts/python -m eval.render runs/classic_1v1/latest.pt scripted --out replays/partido.html

# exportar y jugar en una sala real
.venv/Scripts/python -m export.to_onnx runs/classic_1v1/latest.pt --out deploy/model
cd deploy && npm install && npm test
HAXBALL_TOKEN=thr1.xxx node bot.js --name "Mi sala" --password 1234
```

Para unirse a una sala ajena (sin recaptcha): `node bot.js --join <roomId> [--password x] [--extrap 80]`.

### Lag (salas ajenas)

Como host, el bot no tiene lag. Como cliente, entre el frame que ve al decidir y el frame en que el host aplica su input pasan **6 a 11 ticks (mediana 8, ~130 ms)** aunque el ping sea ~0. Es estructural del cliente de HaxBall (medido en `runs/real/latency_*.json`), y en salas remotas se suma el ping. Se compensa de dos formas:

1. **En el bot:** mide en vivo ese retraso (desde `setKeyState` hasta el `onPlayerInputChange` propio), extrapola el estado esa cantidad (o lo que diga `--extrap` en ms) y decide sobre eso.
2. **En el entrenamiento:** `env.action_delay_max` sortea en cada partido un retraso de 0..N ticks entre elegir la acción y aplicarla, para el jitter que queda. Lo recomendable es un fine-tune del modelo ya entrenado:

```bash
.venv/Scripts/python -m train.ppo_selfplay --run classic_1v1 --resume --override env.action_delay_max=3
```

El token se saca a mano en https://www.haxball.com/headlesstoken y vence a los pocos minutos.

## Cómo sigue el entrenamiento

1. **Etapa 0**: 70% de los partidos contra el bot scripteado con 50% de ruido, 30% self-play.
2. **Etapa 1**, cuando le gana al bot en más del 80% de los goles: bot casi sin ruido, y entran rivales de la liga.
3. **Etapa 2, liga**: la mitad de los partidos contra snapshots pasados (se juega más contra los que más le cuestan) y el resto self-play.

El shaping baja a 0 en los primeros 300M pasos, así que al final sólo importan los goles.

Métricas a mirar en TensorBoard:
- `winrate_vs_scripted`, `winrate_vs_pool` y `elo`.
- `approx_kl`: sano entre ~0.005 y 0.02.
- `entropy`: tiene que bajar despacio, no colapsar.

Para 2v2/3v3: hacer otro run con `--override env.n_per_team=2`. La obs cambia de tamaño, así que es otro modelo.

## Multi-mapa y multi-formato (un solo modelo para todo)

`train/multitask.py` entrena **un solo modelo** que juega cualquier mapa y formato, mezclando en cada iteración partidos de todas las tareas activas, para que no se encasille en una modalidad.

- **Tareas** (`train/tasks.yaml`): mapa × formato × reglas. Hay Classic, Big, Futsal x1/x2, x3 y x4, Real Soccer x6 (Pegeche), Real Futsal x7 y HaxEleven. Las reglas pueden ser `plain`, `real` (powershot + pelotas paradas simplificadas) o `pegeche` (el script completo de la sala, ver abajo).
- **Saques simplificados `real`** (incluye `rs4_3v3`/`rs4_4v4`): los rivales no pueden patear ni acercarse a menos de 50 unidades durante la espera. En saque de arco también se mantiene una barrera de área rectangular, con proporciones RS de referencia 840/1150 del ancho y 320/600 del alto. La protección se libera con una patada del equipo que saca o al vencer el plazo: mínimo 7 segundos, ampliado cuando la distancia requiere más tiempo de traslado (estimación física más 3 segundos de margen). Los resets la limpian. Es una aproximación, no una réplica verificada del script de cada host. Este estado sólo pertenece al árbitro del entorno: no agrega features al modelo, no cambia recompensas ni el currículo y conserva `rule_observation: masked`. Física/barreras se ejecutan tick por tick mientras hay un saque pendiente, también en replays. Pruebas: `python -m pytest tests/test_simple_setpieces.py tests/test_pegeche.py -q`.
- **Scripted en saques**: elige un ejecutor elegible (el más cercano; el designado en penal Pegeche), sin mandarlo a defender ni aplicar roles de juego abierto. Laterales hacia adentro y córners hacia la cancha; el rival scripted espera quieto durante el saque central. Esta estrategia usa el estado del árbitro sólo en el baseline, nunca en la política RL. En mapas con salidas, el timeout del saque central suma el tiempo estimado de traslado al margen `kickoff_timeout`; con `0` sigue deshabilitado. Los timeouts Pegeche de pelotas paradas no cambian. Replays y arena tienen por defecto margen de saque central de 180 ticks y episodios de 7200 ticks, como la configuración de entrenamiento; el render informa `kickoff_stalls` y avisa si reinicia por espera, sin sumar goles ficticios. Pruebas end-to-end: `python -m pytest tests/test_scripted_restarts.py -q`. El rival y los plazos cambiaron: las métricas anteriores no son una comparación equivalente, y la ventana del entrenamiento puede mezclar goles antiguos hasta renovarse.
- **Reglas `pegeche`** (`env/pegeche.py`, tareas `rs_*` sobre x6 / x6_half): réplica de `Pegeche Aureus/Script Pegeche`. Incluye laterales (con validez y timeout de 7 s), córners, saques de arco, slide (mantener X 500 ms moviéndose: impulso, 4 s inmóvil y 25 s de cooldown), faltas (slide o choque cerca de la pelota), pedir la falta (la víctima mantiene X 500 ms), tiro libre / penal, tarjetas y expulsiones. Suma 15 entradas a la obs universal, con el estado de pelota parada, el slide propio y la falta para pedir. El bot real todavía no las estima en la sala: TODO en `deploy/bot.js`.
- **Obs "universal"**: posiciones normalizadas por el tamaño de la cancha, más un descriptor del mapa, distancias a las paredes (8 rayos desde el jugador y 8 desde la pelota, `env/geometry.py`) y compañeros/rivales como entidades con máscara. El modelo `SetActorCritic` acepta cualquier cantidad de jugadores.
- **Currículo por amplitud con repaso** (`train/config_multi.yaml`):
  - A: 1v1/2v2 en Classic, Big y Futsal.
  - B: se suman 3v3/4v4 y Real Soccer.
  - C: se suman 6v6 y Real Futsal 5v5/7v7.
  - Las tareas viejas nunca bajan del 4% de las muestras. Si una tarea empeora respecto de su mejor marca, recibe más peso automáticamente.
- HaxEleven (11v11) está en el catálogo pero fuera del currículo por ahora: en CPU cuesta ~4 veces más.

```bash
.venv/Scripts/python -m train.multitask --run multi                 # Ctrl+C guarda; --resume continúa
.venv/Scripts/python -m eval.matrix runs/multi/latest.pt --best     # tabla mapa × formato vs bot, con regresiones
.venv/Scripts/python -m eval.render runs/multi/latest.pt scripted --task rf_7v7 --out replays/rf7.html
.venv/Scripts/python -m export.to_onnx runs/multi/latest.pt --out deploy/model   # el bot detecta el mapa de la sala
node deploy/bot.js --join <roomId> --rules auto                     # plain | real | auto (por el nombre del mapa)
```

Seguir entrenando después de cambiar la obs (features nuevas) sin empezar de cero: `tools/grow_obs.py` agranda la entrada del modelo con pesos en cero. El modelo juega exactamente igual que antes (lo verifica) y aprende a usar lo nuevo. Opera el modelo, el optimizador y los rivales de la liga.

```bash
.venv/Scripts/python -m tools.grow_obs runs/multi/latest.pt --self end:10 --in-place   # posiciones = índices de la obs vieja
.venv/Scripts/python -m train.multitask --run multi --resume
```

Mapas nuevos:
1. Grabar una partida con `node bridge/record.js <sala>` o convertir un replay con `node bridge/replay_to_jsonl.js <archivo.hbr2>`.
2. Importar el estadio con `python -m tools.import_stadium <archivo.hbs> <nombre> [--field W H]`.
3. Validar la física con `python bridge/compare_sim.py runs/real/*.jsonl --gate`.
4. Agregar la tarea a `train/tasks.yaml`.

### Imitación de partidos reales (para jugar como humano)

Antes del RL, el modelo aprende a imitar a jugadores de primera división. Así sale con costumbres humanas (posiciones, esperar el saque, pelotas paradas), en vez de las convenciones que inventa jugando contra sí mismo.

1. **Replays** en `replays_real/stadiums/<carpeta>/*.hbr2`. Una carpeta por modalidad (bigx3, futsalx3, futsalx4, futsalx7, rfx7, rsx4, rsx6).
2. **Dataset:** `python -m tools.build_bc_dataset`.
   Para incorporar sólo recs nuevas de RS sin rehacer las anteriores: `python -m tools.build_bc_dataset --folders rsx6`.
   Para revisar cobertura real, formatos, archivos excluidos y separación por replay de validación:
   `python -m tools.audit_bc_dataset --out data/bc/coverage_report.json`.
   El auditor valida observaciones finitas, filas y etiquetas; `pending` significa que falta un shard, no que el replay necesariamente tenga muestras válidas. No confundir el nombre de la carpeta con el mapa o el tamaño de equipo efectivos.
   - Reproduce cada replay y arma, con el mismo entorno del entrenamiento, la obs de cada jugador y la tecla que apretó.
   - Usa el estadio exacto de cada replay.
   - Descarta prácticas, mapas fuera del currículo y partidos con equipos desparejos.
   - `rsx6` (sala pública) pesa la mitad.
   - Resultado en `data/bc/`.
3. **Preentrenamiento:** `python -m train.bc --run bc`. Separa replays enteros para validación y muestra el acierto por mapa.
4. **RL desde el modelo imitador:** `python -m train.multitask --run multi --init-from runs/bc/bc.pt`. Con `ppo.bc_kl_coef` penaliza alejarse del estilo humano; el peso baja con el tiempo hasta `bc_kl_final`.

Para ver cómo juega el imitador: `python -m eval.render runs/bc/bc.pt runs/bc/bc.pt --task futsal_3v3 --out replays/bc.html`.

La incorporación de recs RS del 29/09/2026 y su cobertura se documentan en
`reports/recordings_coverage_20260929.md`. El imitador nuevo se publica por separado
en `runs/bc_rsx6_20260929/bc.pt`; no reemplaza `bc2` ni el PPO existente.
Para probarlo como referencia del entrenamiento actual, sin reiniciar la política:
`python -m train.multitask --config train/config_runpod.yaml --run multi --resume --override bc_reference=runs/bc_rsx6_20260929/bc.pt`.
El acierto de imitación no es una tasa de victorias; evaluar también juego y saques.

Rendimiento medido en esta PC (sólo CPU): la red se lleva ~90% del tiempo (`python -m tools.profile_train`). 8 hilos de torch es lo óptimo.

### Entrenamiento CPU + GPU en Runpod

En una imagen de PyTorch con CUDA, crear el entorno con `python3 -m venv .venv --system-site-packages`
y activarlo con `source .venv/bin/activate`. Instalar las dependencias de arriba sin instalar el wheel
CPU de PyTorch. Verificar `python -c "import torch; print(torch.cuda.is_available())"` (debe ser `True`).

Desde la raíz del proyecto, con `runs/multi/latest.pt` y `runs/bc2/bc.pt` presentes:

```bash
python -m pytest tests/test_cuda_decisions.py tests/test_cuda_runtime.py -q
python -m tools.benchmark_multitask --config train/config_runpod.yaml --iters 5
python -m train.multitask --config train/config_runpod.yaml --run multi --resume
```

El benchmark usa una copia temporal del checkpoint. Descarta el calentamiento de Numba/CUDA y muestra
pasos/s reales, rollout, GAE/preparación, update y VRAM. Para comparar el CPU del mismo Pod, repetir
con `--device cpu`. Para medir el mejor número de hilos de física, usar `--numba-threads 4`, `8`, `12`, etc.
Más hilos no siempre ayudan cuando cada tarea tiene pocos partidos.

La configuración Runpod hereda recompensas, modelo y currículo del run multi. Usa CUDA para la red y
dos hilos de PyTorch CPU como máximo; Numba respeta el cupo de CPU del contenedor. En CUDA se reutilizan
buffers pinned, se agrupan rivales por snapshot entre tareas, y el cálculo CPU de bots coincide con la
bajada de decisiones de la GPU. Las métricas PPO se descargan una vez por update. Rollout y update
siguen alternándose para recoger muestras con la política actual, sin agregar retraso de política.

`Ctrl+C` guarda `latest.pt`. Ejecutar una sola instancia por run y conservar `/workspace` en almacenamiento
persistente. Las cifras de rendimiento deben medirse en la máquina alquilada.

#### Rollout optimizado (31 hilos en este Pod)

No cambia el checkpoint, red, PPO, recompensas, currículo ni número de muestras. Fusiona la observación
universal y los rayos en un kernel; agrupa los ticks de física sólo sin callbacks ni lag; conserva los
eventos de todos los ticks. Los resets recalculan sólo las observaciones afectadas. Los bots calculan
sólo los partidos seleccionados y conservan los sorteos de ruido del rival. Reutiliza buffers e índices
de snapshots; sube observaciones directamente a memoria pinned y agrupa los bootstrap de CUDA.
Las reglas Pegeche y el retraso de acciones siguen ejecutándose tick por tick.

Detener primero el entrenamiento con `Ctrl+C` y esperar el mensaje de guardado. Después:

```bash
git pull --ff-only
python -m pytest tests/test_rollout_optimized.py tests/test_cuda_runtime.py -q
python -m tools.benchmark_multitask --config train/config_runpod.yaml --numba-threads 31 --warmup 3 --iters 15 --baseline
python -m tools.benchmark_multitask --config train/config_runpod.yaml --numba-threads 31 --warmup 3 --iters 15 --decision-backend legacy
python -m train.multitask --config train/config_runpod.yaml --run multi --resume
```

Las dos mediciones parten del mismo archivo y descartan su aprendizaje temporal. Comparar
`pasos/s reales` y `rollout`, preferiblemente repitiendo en orden inverso; no ejecutar entrenamiento,
replays ni otros benchmarks a la vez. La primera compilación Numba puede tardar minutos y no representa
la velocidad estable. Para localizar el cuello de botella restante, agregar `--profile-rollout`:
los tiempos son inclusivos y hay solapamiento CPU/GPU, por lo que no deben sumarse.
Para volver a la ruta de referencia completa: `--override runtime.optimize_rollout=false runtime.cuda_decisions=legacy`. En otro host con menos
de 31 hilos disponibles, usar `--override ppo.numba_threads=auto` al entrenar.

#### Decisiones CUDA con menos lanzamientos desde CPU

`config_runpod.yaml` activa `runtime.cuda_decisions=auto`: combina los logits del aprendiz y de los
snapshots, captura la inferencia determinista en un CUDA Graph y lo repite durante el rollout. El
muestreo categórico queda fuera del graph y usa aleatoriedad fresca en cada decisión. No activa AMP/TF32
ni cambia recompensas, física, rivales, normalizadores, cantidad de muestras o PPO. Los valores y logp
de las filas que aprenden siguen siendo los del aprendiz. Combinar los sorteos cambia el orden del RNG:
la distribución es la misma, pero no se espera una trayectoria idéntica con la misma semilla.

El graph se libera antes de actualizar RunningNorm/Adam y se captura de nuevo en cada rollout, porque
los buffers de normalización y los rivales pueden cambiar. El coste de preparar/capturar está incluido
en el benchmark, también después del calentamiento. `auto` avisa y usa la ruta eager si la captura
no es compatible; los errores de modelo, falta de memoria o acceso ilegal no se ocultan. `graph` es
estricto para validar el Pod. El mecanismo sigue las [indicaciones de CUDA Graphs de PyTorch 2.8](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs).

Detener y guardar con `Ctrl+C`. Para comparar **sólo este retoque**, sin cambiar el rollout optimizado
anterior ni los 31 hilos, ejecutar ambas mediciones sin entrenamiento/replays concurrentes:

```bash
git pull --ff-only
python -m pytest tests/test_cuda_decisions.py tests/test_cuda_runtime.py tests/test_rollout_optimized.py -q
python -m tools.benchmark_multitask --config train/config_runpod.yaml --warmup 3 --iters 15 --decision-backend legacy
python -m tools.benchmark_multitask --config train/config_runpod.yaml --warmup 3 --iters 15 --decision-backend graph
```

Ambos benchmarks usan copias del mismo checkpoint y descartan el aprendizaje temporal. Comparar
`pasos/s reales`, no el máximo de una iteración, y repetir en orden inverso si hay mucho ruido del host.
La ruta graph debe indicar `capturas 1.0/iter | replays 128/iter` con el rollout actual. No hay una
aceleración garantizada: depende del overhead CPU/GPU y del número de snapshots de cada rollout.
`--decision-backend eager` permite separar el beneficio de combinar muestreos del beneficio del graph.

Con `--profile-rollout` se añaden eventos para H2D, inferencia, muestreo y D2H, leídos tras la espera que
el rollout ya necesita (sin otra sincronización). Los intervalos del stream incluyen huecos de
lanzamiento CPU; no prueban que la GPU esté saturada y no se suman a los tiempos CPU/inclusivos.

Si las pruebas pasan y el benchmark mejora, reanudar normalmente:

```bash
python -m train.multitask --config train/config_runpod.yaml --run multi --resume
```

Para desactivar sólo este retoque, conservando las optimizaciones anteriores y el checkpoint:
`--override runtime.cuda_decisions=legacy`. CPU y el entrenador recurrente no usan este graph.

#### Lote PPO persistente: preparación visible y menos asignaciones

`runtime.reuse_ppo_batch=true` reutiliza un buffer host pinned y un buffer CUDA por campo (`obs`,
`act`, `logp`, `adv`, `ret`). Copia cada tarea directamente al buffer host, sin concatenar y pinear
un tensor grande nuevo en cada iteración. La capacidad crece por potencias de dos y sólo se reasigna
si cambia el dtype, ancho de observación o supera la capacidad; se entregan únicamente las filas
válidas, nunca la cola sobrante. Conserva orden, dtypes, valores, PPO, rivales y reparto de tareas.

Un evento protege la memoria host y GPU antes de sobrescribirla, incluyendo consumidores pendientes
del update. Normalmente no espera porque PPO ya descarga sus métricas antes de retornar. Los buffers
no se guardan en el checkpoint. CPU mantiene su ruta anterior y el entrenador recurrente no usa este
cache. Se reserva algo más de RAM/VRAM para evitar reasignaciones pequeñas entre lotes.

La línea de entrenamiento ahora muestra `prep`, además de `rollout` y `upd`. Con `--profile-rollout`
se separan también `lote PPO` y `normalización`. El benchmark imprime espera, asignación, empaquetado
y envío del cache: el envío es tiempo host de encolado, no la duración completa de la copia GPU.
Después del calentamiento, las asignaciones deberían ser cero si el lote cabe en los buffers.

Guardar y detener con `Ctrl+C`, activar `.venv` y ejecutar desde la raíz del Pod:

```bash
git pull --ff-only
python -m pytest tests/test_ppo_batch_transfer.py tests/test_cuda_runtime.py -q
python -m tools.benchmark_multitask --config train/config_1v1_focus.yaml --warmup 3 --iters 15 --decision-backend graph --profile-rollout
```

Para una comparación justa con la preparación anterior, repetir el mismo benchmark agregando
`--no-reuse-ppo-batch` (sin cambiar graph ni 50/10). Comparar `pasos/s reales` y `preparación/GAE`, no
un pico individual. El benchmark descarta sus cambios en una copia temporal; no modifica `latest.pt`.
No se promete una aceleración sin verificar los tiempos en el Pod. Para desactivar sólo este cache
al entrenar, usar `--override runtime.reuse_ppo_batch=false`.

#### Fase manual de técnica individual y posterior cooperación

Para un checkpoint que todavía está en **etapa 0**, hay dos perfiles separados del currículo normal:

| Perfil | `x1_1v1` | Cada una de las otras cinco tareas |
|---|---:|---:|
| `train/config_1v1_focus.yaml` | 50% | 10% |
| `train/config_cooperation.yaml` | 20% | 16% |

Son porcentajes del **presupuesto de agentes del entorno**, no multiplicadores de recompensa ni
pesos extra de la pérdida PPO. El número de partidos se ajusta por tamaño de equipo. Hay pequeños
redondeos, y la fracción efectiva de muestras PPO depende también de qué jugadores aprenden
(self-play, liga o bot). Con 1152 agentes, la fase de técnica asigna 288 partidos simultáneos al 1v1:
`big_2v2:29x4, big_3v3:19x6, futsal_2v2:29x4, futsal_3v3:19x6, x1_1v1:288x2, aha_3v3:19x6`.

`fixed_weights: true` impide que los boosts por regresión alteren este reparto, incluso si el checkpoint
ya trae boosts. Las métricas y el currículo de rivales siguen activos. Los perfiles normales conservan
su control adaptativo. No se cambian PPO, recompensas, 31 hilos ni las optimizaciones CUDA.

Detener con `Ctrl+C`, esperar el guardado y comenzar la fase individual desde el mismo checkpoint:

```bash
git pull --ff-only
python -m pytest tests/test_focus_profiles.py -q
python -m train.multitask --config train/config_1v1_focus.yaml --run multi --resume
```

Estas fases **no pasan automáticamente a B/C ni cambian al 20% por tiempo**: no se ha fijado un plazo
de práctica. Cuando se quiera priorizar cooperación, volver a guardar con `Ctrl+C` y ejecutar:

```bash
python -m train.multitask --config train/config_cooperation.yaml --run multi --resume
```

Se conservan pesos, optimizador, pasos, liga, dificultad de rivales, historial y decaimiento del shaping.
Para recuperar el currículo automático original, reanudar con `train/config_runpod.yaml`; sus umbrales
usan los pasos de etapa acumulados, incluidos los entrenados en estos perfiles. No usar estos perfiles
de una sola etapa para un checkpoint que ya avanzó a etapa 1 o superior.

Uso responsable: usar el bot sólo en salas propias o con permiso. En salas públicas o competitivas contra personas es hacer trampa.
