# HaxballRL

Bot de HaxBall entrenado con aprendizaje por refuerzo: PPO, self-play, liga de rivales (PFSP) y currículo.

### RS4 v3: especialización larga

Programa nuevo de **6B pasos útiles compartidos**, dos pilotos de200M y seis fases
con evaluación independiente, saques recurrentes y compañeros congelados.
No reutilizar preparadores antiguos ni arrancar con `--init-from`.
La ruta recurrente sólo se selecciona si supera las pruebas de calidad y conserva
al menos70% del throughput medido en el Pod.
Ver [infraestructura, límites y comandos](reports/rs4_v3_implementation.md).

### Guía defensiva de JJRS 6v6

`config_multi.yaml` (heredado por RunPod) habilita **sólo en `jjrs_6v6`**
`task_reward_overrides.jjrs_6v6: {w_defense_support: 0.35, defense_shaping_floor: 0.5}`.
Cuando la pelota avanza hacia el arco propio, un potencial geométrico valora al
arquero y tres coberturas distintas entre pelota y arco. Excluye de esas coberturas
al arquero más cercano al arco y al presionante más cercano a la pelota; deja un
jugador libre. No fuerza acciones, identidades ni una formación permanente.

El premio es `peso * max(shaping_tarea, piso) * (gamma * Phi_siguiente - Phi_actual)`:
quedarse quieto no genera un bonus positivo por tick, ni un ciclo de ida y vuelta
genera retorno descontado positivo. Se anula el potencial durante saques y en
estados terminales de gol. Es una heurística para explorar defensa colectiva,
no una garantía de mejor juego; durante ataque no da guía defensiva.
No añade estados privados del host a la observación ni cambia scripted, física,
arquitectura, reparto de modalidades o recompensas de otros mapas.

Detener el entrenamiento con **Ctrl+C una vez** y esperar `guardado`, luego:

```bash
git pull --ff-only
python -m pytest tests/test_defensive_support.py tests/test_multitask.py tests/test_rollout_optimized.py tests/test_simple_setpieces.py -q
python -m train.multitask --config train/config_runpod.yaml --run multi --resume
```

Si ya se usaba `--override bc_reference=...`, conservar exactamente esa opción
al reanudar. No usar `--init-from` para este retoque. El checkpoint, optimizador,
liga y métricas se conservan; no se reinicia el aprendizaje. No incorpora las
nuevas recs ni entrena un nuevo imitador.
Comparar varios partidos/semillas de JJRS contra **el mismo rival** antes y
después, revisando coberturas, goles recibidos y capacidad de contraatacar.
Para desactivar sin perder progreso, quitar el bloque `task_reward_overrides`
o poner `w_defense_support: 0` y reanudar. Validar velocidad con el benchmark
del Pod; las pruebas locales no miden CUDA.

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

Los replays públicos de MrREPLAY se pueden buscar y descargar directamente. `--query` usa el mismo
texto que la barra de búsqueda pública; `--team-size 3` además abre cada grabación y exige al menos un
tramo 3 contra 3. `--folder` selecciona la modalidad y evita mezclar estadios incompatibles:

```bash
python -m tools.fetch_mrhost_replays --query 3v3 --team-size 3 --folder futsalx3
python -m tools.fetch_mrhost_replays --query 3v3 --team-size 3 --folder futsalx3 --stadium af_futsalx3
python -m tools.fetch_mrhost_replays --query 3v3 --team-size 3 --folder futsalx3 --max-results 2 --dry-run
```

También acepta `--min-duration`, `--max-duration` (segundos), `--country`, `--continent`, `--sort` y
`--max-results`. Las descargas válidas quedan como `.hbr2` en la carpeta elegida y
`_mrhost_manifest.jsonl` registra descargados, duplicados y descartes para poder reanudar. El catálogo
público no forma parte de la API v1 autenticada: el comando descubre y usa la interfaz interna de la
web, por lo que puede requerir ajustes si MrREPLAY cambia su implementación.
2. **Dataset:** `python -m tools.build_bc_dataset`.
   Para incorporar sólo recs nuevas de RS sin rehacer las anteriores: `python -m tools.build_bc_dataset --folders rsx6`.
   Para preparar RS4 normal y corregir shards antiguos creados con powershot:
   `python -m tools.build_bc_dataset --folders rsx4 --overwrite` (respaldar los shards anteriores antes).
   Los duplicados byte por byte se identifican con SHA-256 y se omiten, conservando
   todos los `.hbr2` originales. Esto evita crear dos shards del mismo archivo
   con nombres distintos y repartir esas copias entre entrenamiento y validación.
   Los shards duplicados que ya existieran deben apartarse de `data/bc`;
   omitir una rec no elimina automáticamente un shard antiguo.
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

### Pod sólo CPU (i5-10400F, 6 núcleos / 12 hilos)

`train/config_cpu.yaml` conserva PPO, referencia BC, recompensas, liga y currículo de
`config_runpod.yaml`, con CPU, Torch 8 y física 4. Esta combinación dio 6.876 pasos/s
reales en la medición posterior del i5 (rollout 5,734 s, prep 0,167 s, update 9,566 s).
La baseline histórica era 8/12 (~5.2k/s según el log); no son mediciones directamente
comparables ni una demostración del óptimo global. No cambia `env.agents=1152`,
rollout 128, epochs 3 ni minibatch 8192.
Este perfil conserva el currículo normal; si el run usa otro reparto/perfil o una
referencia BC distinta, conservar esa configuración y cambiar sólo dispositivo e hilos.

Guardar y detener el entrenamiento con Ctrl+C, esperar `guardado` y medir sin otros
entrenamientos, replays ni benchmarks concurrentes. Desde la raíz del Pod, en Bash:

```bash
# Baseline: copia temporal del mismo checkpoint, sin alterar latest.pt.
python -m tools.benchmark_multitask --config train/config_cpu.yaml --torch-threads 8 --numba-threads 12 --warmup 3 --iters 15

# Primero física, manteniendo Torch en 8; pruebas secuenciales.
for n in 4 6 8 10 12; do
  python -m tools.benchmark_multitask --config train/config_cpu.yaml --torch-threads 8 --numba-threads "$n" --warmup 3 --iters 15
done

# Luego Torch; sustituir 12 por la mejor física del barrido anterior.
for t in 4 6 8 10 12; do
  python -m tools.benchmark_multitask --config train/config_cpu.yaml --torch-threads "$t" --numba-threads 12 --warmup 3 --iters 15
done

# Perfil inclusivo CPU (añade overhead): usar aquí los hilos ganadores.
python -m tools.benchmark_multitask --config train/config_cpu.yaml --torch-threads 8 --numba-threads 12 --warmup 3 --iters 15 --profile-rollout

# Reanudar con 8/4, configuración adoptada tras la medición posterior.
python -m train.multitask --config train/config_cpu.yaml --run multi --resume
```

Comparar `pasos/s reales` (muestras útiles divididas por tiempo total), las muestras
por iteración y las tres fases. Cada prueba carga el mismo checkpoint y descarta
su aprendizaje temporal. Repetir baseline y finalistas en orden inverso para medir
ruido del host. El total incluye también log y trabajo fuera de las tres fases.
Rollout y update alternan: menos hilos de física no libera automáticamente CPU
durante PPO; cualquier beneficio de los pools de hilos debe comprobarse.
`--profile-rollout` sirve para multitarea CPU; `tools.profile_train` mide el entrenador
antiguo de una sola tarea. No sumar tiempos inclusivos de padres e hijos.

Cambiar cantidad de entornos queda como experimento separado: cambia las muestras
por actualización, los minibatches y los redondeos del reparto de tareas/rivales.
No cumple la comparación estricta de aprendizaje conservado. La RAM libre por sí
sola no garantiza una mejora. Los objetivos 5.7–6k o superiores se verifican en el
Pod antes de adoptar una configuración; las mediciones locales no los demuestran.

#### Referencia BC reutilizada durante el update CPU

`config_cpu.yaml` activa `runtime.cache_bc_logits_cpu=true`. La referencia BC está
congelada y en eval: se calculan sus log-probabilidades una vez por lote, en bloques
del tamaño de minibatch, y se reutilizan en las épocas PPO. El cache vive sólo durante
ese update y se reconstruye con las observaciones del siguiente lote. Se mantienen
la fórmula KL, coeficiente BC, optimizador, muestras, permutaciones y número de updates.
Las diferencias de tamaño/orden de inferencia pueden introducir redondeos de coma
flotante; no se promete una trayectoria bit a bit idéntica. CUDA no usa este flag CPU:
el perfil nocturno activa explícitamente `runtime.cache_bc_logits=true` para CUDA.
Recurrente conserva su ruta; con BC desactivada o una sola época no se crea cache.

Para 106.496 muestras y 18 acciones float32 reserva ~7,3 MiB adicionales. La medición
aportada del Pod fue 4.147 pasos/s reales, rollout 13,064 s, prep 0,149 s y update
10,109 s, con Torch 8/física 12. No se ha medido todavía la aceleración de este cache.
Guardar/detener el entrenamiento y comparar desde el mismo checkpoint, sin cambiar hilos:

```bash
python -m pytest tests/test_cpu_bc_cache.py tests/test_cpu_config.py -q
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15 --no-cache-bc-logits
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15
```

Repetir en orden inverso si hay ruido. Si mejora, reanudar con el perfil CPU habitual.
Para desactivarlo al entrenar: `--override runtime.cache_bc_logits_cpu=false`.

#### Rollout y minibatches CPU agrupados

`runtime.optimize_cpu=true` en el perfil CPU añade una pasada por snapshot para
todos sus rivales entre tareas (antes se repetía por tarea), un buffer reutilizable
para las observaciones de cada decisión y un buffer de minibatch PPO con `index_select`.
El bootstrap agrupa tareas/timeouts y calcula sólo la cabeza de valor, sin ejecutar
la cabeza de acciones. Los sorteos mantienen su orden y tamaños por tarea/rival;
las filas de log-probabilidades y valores del aprendiz se conservan. Los buffers se
reconstruyen cuando cambia el tamaño y los índices cuando se reasignan rivales.
No se cambian acciones disponibles, normalizadores, pesos del currículo, cantidades
de muestras, pérdida, épocas ni optimizador. Las llamadas agrupadas pueden introducir
redondeos numéricos; no se promete igualdad bit a bit de entrenamientos largos.
CUDA y el entrenador recurrente mantienen sus rutas de decisiones y update.

Las duraciones del entrenador usan ahora `perf_counter`: los saltos de hora del host
ya no producen valores negativos como `upd -1.9s`. El resumen añade setup, mantenimiento,
logging, reparto/avance de etapa y retorno/overhead para explicar el tiempo total.
`--profile-rollout` también desglosa el update CPU en cache BC, selección de minibatch,
forward/pérdida, backward y clipping/Adam, y muestra medias de entropía, KL y clipfrac.
El log habitual de pasos/s sigue midiendo las tres fases principales; para comparar
rendimiento de punta a punta usar `pasos/s reales` del resumen.

Con entrenamiento detenido, comparar conservando el cache BC en ambos casos:

```bash
python -m pytest tests/test_cpu_rollout.py tests/test_cpu_bc_cache.py tests/test_cpu_config.py tests/test_cpu_tuning.py tests/test_cpu_benchmark.py tests/test_cuda_runtime.py tests/test_rollout_optimized.py -q
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15 --no-optimize-cpu
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15
# Desglose: agrega overhead, usarlo separado de la comparación de velocidad.
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 5 --profile-rollout
# Barrido secuencial completo, misma copia fija del checkpoint en todos los procesos.
python -m tools.tune_cpu --config train/config_cpu.yaml --warmup 3 --iters 15
```

El barrido guarda JSON en un directorio nuevo de `reports/cpu_tuning`, mide primero
física manteniendo Torch 8, luego Torch con la mejor física, y repite el finalista
antes de volver a medir 8/12. No modifica el YAML ni reanuda el entrenamiento. Usa
`--torch-threads`/`--numba-threads` para otra baseline y `--threads` para otra lista de
candidatos. La interacción entre hilos y el ruido del host pueden requerir otro barrido;
es una búsqueda por coordenadas, no una garantía del óptimo global.
Para volver a la ruta CPU anterior al entrenar:
`--override runtime.optimize_cpu=false` (conserva el cache BC).

Medición aportada del Pod con cache BC: 4.064 → 4.419 pasos/s reales (+8,7%), mismas
106.496 muestras/iter. Sus tiempos por fase tenían saltos del reloj; el total del
benchmark ya era monotónico. En ese archivo la entropía fue ~1,9–2,05 en ambas rutas,
no 0,9. Los benchmarks entrenan copias temporales y descartan su aprendizaje.
Una caída a 0,9 en el run real debe revisarse junto con KL, clipfrac y rendimiento por
tarea; por sí sola no demuestra mejora ni colapso y no justifica cambiar PPO durante
esta optimización.

### Entrenamiento CPU + GPU en Runpod

#### Rama especializada RS4, independiente del generalista

##### Guía táctica RS4 opcional

La rama conservadora anterior sigue disponible. La guía nueva se activa
explícitamente, con el entrenamiento detenido y guardado. Si `rs4` ya existe:

```bash
python -m tools.configure_rs4_rewards --run rs4
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

Si todavía no existe, preparar una sola vez con:

```bash
python -m tools.prepare_rs4 --source runs/multi/latest.pt --run rs4 --additional-steps 500000000 --tactical-rewards
```

El configurador crea un backup único `config_before_tactics_*.yaml`. No modifica
pesos, normalizadores, Adam, liga, checkpoints, config del generalista ni
LR/entropía/BC. Rechaza el run generalista y activaciones repetidas que reinicien
el decay. La config generada es la autoridad; no reanudar con la plantilla estática.

**Qué cambia en el aprendizaje:** se desactivan acercamiento genérico a la pelota,
bonus de patada hacia delante y separación genérica (`w_near_ball`, `kick_to_goal`,
`w_spread`, `team_spread_floor`). Se mantienen gol, salidas, saques, córners y
cooperación existente. Los pases ya requieren retención/utilidad, controlan
devoluciones y tienen un cap por posesión de 0,01 por defecto; no se agregan pagos
por tocar, despejar, atajar o completar cualquier pase.

La Φ original (formation_version=1) usa cuatro roles geométricos: arquero/líbero, presión/conductor y dos
apoyos/coberturas diagonales. Las zonas se desplazan con pelota y ventaja geométrica
de control; ataque abre el equipo y defensa lo compacta. Se evalúan las 24
asignaciones: cada jugador ocupa un solo rol, sin identidades permanentes, con
una banda de tolerancia en lugar de un punto exacto. No se añaden IDs de roles a
la observación ni una red separada por posición. La guía puede ser imperfecta:
no es una formación óptima demostrada ni una rotación con memoria/histeresis.

Φ v1 combina **65% estructura + 35% balance de amenaza/peligro**, acotada en [0,1].
Amenaza usa distancia al arco, apertura angular, control geométrico y cobertura
de tres líneas de tiro. Es una heurística, NO una probabilidad de gol calibrada.
Los compañeros reciben la misma diferencia de potencial; no bonos individuales
que compitan por ser arquero o pateador. No se usa historial privado de posesión.

Reward adicional: `gamma * potencial_ponderado_siguiente - potencial_ponderado_anterior`.
El coeficiente inicial **0,12** baja a **0** en **200 millones de pasos globales
adicionales desde la activación**, no desde cero ni desde el comienzo del generalista.
El potencial anterior conserva su propio coeficiente al cambiar el nuevo; no se
recalcula con el nuevo peso. Se usa Φ terminal 0 en gol/stall; truncaciones mantienen
el potencial de la observación final para el bootstrap PPO. Saques tienen Φ 0;
se contabiliza la transición hacia/desde ese estado, no una recompensa por esperar.
No se recorta la diferencia de potencial después de calcularla: eso rompería su
telescopado. No hay porcentajes de reward total garantizados: dependen de los eventos.

La construcción sigue el criterio de [potential-based shaping de Ng et al.](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
No se promete invariancia de la política aprendida en PPO aproximado/self-play,
ni inmunidad general a exploits: existen otros bonuses no potenciales y el rival
cambia. Se prueban invariantes concretos, no calidad futbolística.

Estilos R3 existentes: **0 equilibrado, 1 agresivo, 2 conservador**. La guía activa
mantener el estilo mezclado de cada rival durante todo el partido, en lugar de
cambiarlo tras cada gol/reset PPO; se alterna al reiniciar un partido completo.
La liga y PFSP siguen intactos. No se inventaron tres nuevos bots ni se supone
que cubren todos los estilos humanos. No se agregan features privados al agente.
`rs4_4v4` es `rs_one` con saques simplificados, sin faltas/slides/tiros libres;
`glh_4v4` y `rs_4v4` son modalidades distintas del catálogo.

#### RS4 v2: córners, bloque dinámico e imitación específica

La configuración recibida el 01/10 conserva `bc_reference: runs/bc2/bc.pt`: la
rama hereda el generalista y su imitador, no selecciona automáticamente las recs RS4.
Hay una copia actualizada, sin modificar el original, en
`reports/rs4_config_v2_20261001.yaml`. Mantiene PPO, BC y el comienzo del decay.
Es preferible actualizar el config vivo del Pod, con el entrenamiento detenido:

```bash
python -m tools.upgrade_rs4 --config runs/rs4/config.yaml --renew-guide
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

El actualizador guarda un backup único de config. `--renew-guide` es explícito:
abre una fase v2 de guía **0.08 → 0 en 200M pasos desde latest**, sin reiniciar LR,
entropía, Adam, pesos o dificultad. Sin ese flag mantiene el decay anterior; si ya
llegó a cero, cambiar sólo formación no tendrá efecto. No repetir renew-guide.
La versión de métricas RS4 pasa a 2 y limpia ventanas no comparables, no la dificultad.
También aplica retención de checkpoints 5/30 minutos y 6 históricos.

**Córners/saques:** antes del plazo el rival sigue bloqueado. No se pudo reproducir
un robo a los 2–3 segundos con las reglas actuales. Se corrigió otra debilidad:
en RS One 4v4, vencer el plazo ya no deja la pelota libre para el rival, sino que
termina el episodio PPO y reinicia posiciones sin inventar un gol. Es la misma
regla en entrenamiento/evaluación/replays. En el perfil v2 se penaliza al equipo
que no ejecutó con -0.25 y se añade potencial compartido de acercamiento acotado
a 0.05. No se fuerza ninguna tecla ni se mueve al ejecutor hacia la pelota.
El bonus de córner útil sigue requiriendo ejecución hacia dentro y continuidad.
El log/TensorBoard muestra intentos, córners útiles y expiraciones por color;
la consola acumula desde el último log (ver «Córners: currículo continuo»),
no corresponde al partido que se observa. TensorBoard conserva también cada rollout.

**Bloque:** v2 combina 50% estructura/50% amenaza-peligro. Un último hombre móvil
adelanta con pelota avanzada y ventaja de acceso; al perder esa ventaja recupera
profundidad. Hay portador/presionante, apoyo de balance detrás y amenaza por delante.
Se prueban las 24 asignaciones y las dos orientaciones laterales; no hay identidades
fijas ni sesgo de banda. Sigue siendo una heurística geométrica, sin posesión real,
memoria de transiciones ni promesa de aprender presión orientada perfecta.

**Grabaciones e imitador RS4 independiente:**

```bash
python -m tools.fetch_mrhost_replays --query Rsx4 --team-size 4 --folder rsx4 --stadium rs_one --min-duration 120 --max-results 100
python -m tools.build_bc_dataset --folders rsx4 --stadium rs_one --team-size 4 --out data/bc_rs4_v2
python -m train.bc --run bc_rs4_v2_20261001 --config runs/rs4/config.yaml --data-dir data/bc_rs4_v2 --folders rsx4 --stadium rs_one --team-size 4 --init-from runs/bc2/bc.pt --epochs 4 --lr 0.0001 --threads 4
```

`--team-size` del downloader exige algún tramo 4v4, mientras que el filtro de BC
descarta **todas** las filas de otros formatos. El conversor filtra cada tramo por
`rs_one` y 4v4, incluso cuando se cambia el mapa durante el replay; el nuevo dataset
queda separado del general. El filtro BC exige `rs_one`, no sólo una carpeta llamada
rsx4; shards antiguos con powershot se excluyen.
La validación se separa por replay. El fine-tuning copia pesos/normalizadores de
BC y entrena un imitador nuevo; no toca el PPO ni sobrescribe bc2.
La descarga es resumible: aumentar max-results revisa también duplicados ya presentes.
Los archivos descargados, datasets y el nuevo imitador no se incluyen automáticamente
en un git pull. Hay que copiarlos al Pod o ejecutar este flujo allí.

Se generó localmente un imitador RS4, con mejor acierto humano validado;
eso **no demuestra** mayor winrate. Tras probarlo y disponer del archivo en el Pod,
para cambiar sólo la referencia (sin reiniciar la política PPO):

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume --override bc_reference=runs/bc_rs4_v2_20261001/bc.pt
```

El override explícito evita que preserve_bc_reference restaure bc2. No usar
`--init-from` en train.multitask para aplicar esta referencia. Volver a bc2 con el
mismo override permite retirar la referencia nueva, no deshace aprendizaje realizado.
Si se exige gate scripted R3, regenerar su informe tras cambiar haxball_env.py.

Comparar, con entrenamiento detenido, contra cada estilo y el padre congelado:

```bash
python -m tools.evaluate_rs4 --run rs4 --styles --games 128 --minutes 2 --seed 51
python -m tools.evaluate_rs4 --run rs4 --styles --games 128 --minutes 2 --seed 73
```

Son cuatro rivales, evaluados con ambos modelos y colores; puede llevar tiempo.
Medir puntos, goles, partidos sin goles, pérdidas, pases/cadenas y timeouts, no
reward de entrenamiento. Las ventanas de entrenamiento heredadas contienen datos
anteriores a la guía; no son una evaluación limpia del cambio. Los informes nuevos
sí son independientes. La evaluación no activa shaping táctico ni premia seguir
la formación propuesta.

El log/TensorBoard muestra `rs4/tactical_coef`, reward táctico medio y absoluto.
Si se quiere comparar velocidad local del Pod, usar benchmark con
`--config runs/rs4/config.yaml --checkpoint runs/rs4/latest.pt`; la primera
compilación Numba se descarta con warmup. No se ha medido rendimiento CUDA de
esta guía en la PC local. Si el config guardado exige gate R3, cambiar
`env/haxball_env.py` invalida su hash: regenerar con
`python -m eval.scripted_gate --out reports/scripted_r3_gate.json` y no omitir el gate.

Restaurar la config del backup permite retirar la guía; no revierte el aprendizaje
ya realizado. El generalista `parent.pt`/`runs/multi/latest.pt` permanece intacto.

#### Fase de adaptación RS4 con calendario LR propio

El especialista heredó el calendario global del generalista. Tras 3,000M pasos,
su LR ya es ~0.00005; ampliar total_steps no la sube. Para probar adaptación más
rápida sin tocar el run actual, preparar **una copia detenida**:

```bash
# Primero Ctrl+C en el entrenamiento y esperar "guardado en .../latest.pt".
python -m tools.prepare_rs4_adaptation --source runs/rs4/latest.pt --run rs4_adapt
python -u -m train.multitask --config runs/rs4_adapt/config.yaml --run rs4_adapt --resume
```

Por defecto, `rs4_adapt` prueba LR **0.0001 → LR efectiva anterior (~0.00005)**
durante **200M pasos adicionales**, y se detiene al completar ese presupuesto.
Sólo se modifica el LR: gamma, GAE, epochs, minibatch, clip, entropía, BC,
rewards, física, rivales, ventanas de métricas y decay de guía siguen iguales.
`latest.pt` es una copia byte a byte: conserva pesos, normalizadores, Adam,
liga, pasos/iteraciones y contadores de etapa. `parent.pt` congela el **RS4 actual**,
no el antiguo generalista. `runs/rs4` y el BC no se modifican.

El ancla está en `ppo.lr_schedule.start_steps` y no se recalcula al reanudar.
El calendario global sigue controlando entropía/BC; la guía mantiene su ancla
anterior y **no se renueva**. El log añade `lr` y TensorBoard
`training/learning_rate`/`training/lr_phase_steps`. Después de llegar al final,
ampliar el presupuesto conserva el LR final, no reinicia la fase.
El preparador rechaza runs existentes, fuentes mixtas y fuentes con una fase LR
ya activa. El trainer rechaza arrancar pesos nuevos con un ancla de continuación,
`--init-from`, un latest ausente o un checkpoint anterior al ancla.

No repetir el preparador para continuar; usar el mismo comando --resume. Para
volver a la baseline, detener rs4_adapt y reanudar runs/rs4/config.yaml en run rs4.
Esto cambia explícitamente la dinámica de aprendizaje, no es una optimización de
velocidad. No garantiza subir KL, reducir empates o resolver los córners.

Revisar aproximadamente tras 50M y 100M pasos nuevos, detenidos y guardados:

```bash
python -m tools.evaluate_rs4 --run rs4_adapt --styles --games 128 --minutes 2 --seed 51
python -m tools.evaluate_rs4 --run rs4_adapt --styles --games 128 --minutes 2 --seed 73
```

Compara la copia entrenada contra el RS4 congelado previo, con iguales rivales,
semillas y ambos colores. Evalúa puntos, victorias/empates/derrotas y replays de
saques; no juzgar sólo KL. Esta comparación detecta regresión/progreso, pero no
atribuye causalmente el cambio al LR sin un control entrenado con el LR anterior.
El JSON conserva las claves generalist/specialist por compatibilidad, mientras
que `reference_label` y consola identifican **RS4 anterior**. No hay promoción
automática, evaluación concurrente ni entrenamiento iniciado por el preparador.

#### Córners: currículo continuo y contadores específicos

Corregido un fallo: `corner_reset_prob: 0.05` se aplicaba al arranque, pero todos
los reinicios automáticos pasaban un equipo de saque explícito y omitían ese
currículo. Ahora el **5% de los reinicios sin gol** puede empezar en córner,
además del reset inicial. Incluye fin de episodio/partido y saques expirados;
no sustituye el saque tras un gol. El dueño del córner y la banda se sortean,
sin ejecutar acciones por el aprendiz ni añadir información privada a la observación.
No se cambian PPO, recompensas, LR, BC, guía, presupuesto ni checkpoints.
Sí cambia la distribución de experiencias: las trayectorias RS4 con la misma
semilla ya no serán las de la versión anterior.

Las evaluaciones y los replays no usan córners artificiales. `eval.arena` y
`eval.render` fuerzan probabilidad cero; las pruebas de saques/gate pasan
`corner_curriculum=False` a `make_env`. Los córners naturales siguen disponibles.

El log del entrenador `train.multitask` acumula **todas las iteraciones desde la
última línea impresa**, no sólo el último rollout:

```text
RS4 córners (desde último log; oportunidades/currículo/intentos/útiles/expirados) | rojo 12/4/2/1/7 azul ...
RS4 expiraciones de todos los saques (desde último log), rojo/azul ...
```

En ese ejemplo se ofrecieron 12 córners rojos, 4 de currículo; se patearon 2,
1 cumplió el criterio de ejecución útil y 7 expiraron sin ejecución. Una
oportunidad se cuenta una vez al jugar la primera decisión desde ese córner;
un córner creado al final de un paso se cuenta en el siguiente. Los contadores
pueden cruzar ventanas: un intento en una ventana puede resultar útil después.
No son tasas de éxito de una cohorte cerrada; no interpretar `útiles/intentos`
de una sola línea como una probabilidad. Azul mezcla los rivales asignados
(scripted, liga o self-play); sus éxitos no demuestran aprendizaje del rojo.

TensorBoard conserva `rs4/corner_attempts_*`, `corner_successes_*` y
`restart_timeouts_*` por rollout. Añade `corner_opportunities_*`,
`corner_curriculum_starts_*`, `corner_timeouts_*` y sus variantes
`rs4/window_<evento>_<red|blue>` acumuladas entre logs. Los JSON de evaluación
incluyen también los nuevos eventos. Las expiraciones de córner son un
subconjunto de las de todos los saques, no se deben sumar ambas.

Reanudar el mismo experimento; no ejecutar otra vez el preparador ni renovar la guía.
Después de publicar estos cambios y con el entrenamiento detenido/guardado en el Pod:

```bash
git pull --ff-only
python -m pytest tests/test_rs4_corner_curriculum.py tests/test_rs4_rollout.py -q
# Sólo si el config exige scripted_readiness.required: true: regenerar su report.
python -m eval.scripted_gate --out reports/scripted_r3_gate.json
python -u -m train.multitask --config runs/rs4_adapt/config.yaml --run rs4_adapt --resume
```

Se conserva el límite actual. El cambio garantiza práctica recurrente, **no**
que el bot aprenda a sacar córners en un número determinado de pasos. Comprobar
primero que `currículo` siga aumentando y luego que el rojo ejecute córners
útiles; el aumento de expiraciones por sí solo no es progreso. El rendimiento
durante entrenamiento incluye posiciones artificiales y no es directamente
comparable con el log anterior: para comparar fuerza usar `tools.evaluate_rs4`.

#### Menos coste de rollout RS4 v2

Con `runtime.optimize_rollout=true`, las optimizaciones RS4 se activan por defecto
(`runtime.optimize_rs4=true`). No cambian las recompensas, pesos, decay, PPO,
acciones, observaciones ni reglas de saques:

- Contactos defensivos y potencial de aproximación a saques compilados con Numba,
  float64 sin fastmath. Se conserva la referencia NumPy.
- La formación sólo se calcula en juego abierto: en saques protegidos su potencial
  ya era cero. Tras un reset sólo se refrescan las filas reiniciadas.
- Se reutiliza el buffer de posiciones al inicio de la decisión; sigue siendo una
  copia independiente de la física, con la misma antigüedad de contexto.
- La formación v2 reutiliza los costes de GK/presión entre bandas, manteniendo
  las dos orientaciones y las 24 asignaciones por orientación.

Actualizar el código con el entrenamiento detenido y reanudar **con el config
actual guardado**. No renovar la guía, ampliar el presupuesto ni cambiar BC para
aplicar esta optimización. La primera compilación Numba no mide velocidad estable.

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

Para comparar en el Pod, sin entrenamientos concurrentes y usando el mismo checkpoint:

```bash
python -m tools.benchmark_multitask --config runs/rs4/config.yaml --checkpoint runs/rs4/latest.pt --warmup 3 --iters 15 --no-optimize-rs4
python -m tools.benchmark_multitask --config runs/rs4/config.yaml --checkpoint runs/rs4/latest.pt --warmup 3 --iters 15
```

Repetir en orden inverso. Ambos descartan sus updates en copias temporales.
El flag recupera los callbacks NumPy y el recálculo de formación en lote completo;
el ahorro interno de costes comunes de roles se comparte entre ambas rutas.
`--profile-rollout` incluye formación/amenaza, aproximación a saques y contactos
defensivos RS4; sus tiempos son inclusivos, no sumarlos con entorno total.
Para volver a la ruta de referencia en entrenamiento: `--override runtime.optimize_rs4=false`.

Medición local CPU (no PPO/CUDA): 144 partidos, 4 hilos de física, R3 en ambos
equipos y guía .08; el tiempo de `env.step` pasó de 6.322 a 3.430 ms/lote respecto
al código anterior, con transiciones idénticas. No es una promesa de mejora de
steps/s del entrenamiento en el Pod. Detalles en
[`reports/rs4_rollout_optimization_20261001.md`](reports/rs4_rollout_optimization_20261001.md).

#### Salidas defensivas RS4

El perfil táctico usa `reward.rs4_defensive_out_scale=0.2`: reduce la penalización
base de salida (normalmente -0.10) a -0.02 sólo en **laterales** tras contacto único
en juego abierto, en zona propia (`x < -0.55 * goal_x` en coordenadas del equipo),
con rival a menos de `0.12 * goal_x` y contacto de hace como máximo 120 ticks.
La presión se estima con posiciones rivales al inicio de la decisión (hasta tres
ticks antes con frame_skip=3), no con una predicción de gol. Contactos posteriores
reemplazan el contexto; contactos ambiguos lo invalidan. No hay bonus por despejar.
Líneas de fondo/córners, pelotas trabadas y pérdidas sin esa evidencia mantienen
la penalización normal. No cambia física, posesión ni saque concedido.
El valor por defecto 1 conserva el comportamiento anterior y las otras modalidades.

Para una rama RS4 ya configurada, detener guardando y, tras actualizar el código,
reanudar sin volver a ejecutar el configurador ni reiniciar el decay:

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume --override reward.rs4_defensive_out_scale=0.2
```

Con `reward.rs4_defensive_out_scale=1.0` se retira sólo este cambio. La formación
táctica no se ha modificado: su sesgo conservador requiere evaluación aparte.

Esta es una rama de **entrenamiento**, no otra arquitectura ni una rama Git.
`tools.prepare_rs4` crea un run separado desde un checkpoint PPO completo y su
`config.yaml` guardado. Conserva modelo, normalizadores, Adam, liga, pasos e
iteraciones globales, dificultad/ventanas RS4, recompensas, PPO y configuración
de rivales. Sólo cambia el reparto a **100% RS4**, la etapa manual a 0 y su contador
a 0. Los entornos se reconstruyen; no es continuación exacta de los mismos partidos.
Las demás modalidades no reciben práctica en esta rama y pueden degradarse.

No modifica `runs/multi/latest.pt`, no reinicia desde BC y no inicia entrenamiento.
Rechaza destinos existentes: para reanudar una rama NO ejecutar otra vez el preparador.
Guarda en `runs/rs4/`: `latest.pt` independiente, `parent.pt` congelado (copia exacta
del generalista), `parent_config.yaml`, `config.yaml` resuelto y `specialization.json`
con hash y procedencia. Conserva todos los snapshots de la liga; sus estadísticas
heredadas no deben interpretarse como una evaluación nueva exclusivamente RS4.

En el Pod, detener/guardar el generalista con un solo Ctrl+C y esperar. Después
de actualizar el código, preparar una vez:

```bash
python -m pytest tests/test_rs4_specialist.py -q
python -m tools.prepare_rs4 --source runs/multi/latest.pt --run rs4 --additional-steps 500000000
```

500 millones es un presupuesto inicial configurable, no un umbral de calidad.
El límite global se fija a pasos del padre + ese presupuesto. El horizonte anterior
de LR/entropía/BC se conserva, incluso si la config vieja no tenía `schedule_steps`.
El preparador usa la config GUARDADA del padre, no sustituye sus hiperparámetros por
los defaults de la plantilla `train/config_rs4_specialist.yaml`.

Entrenar y reanudar usando siempre la config generada:

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

Para dejarlo en segundo plano (no lanzar también el comando foreground):

```bash
nohup .venv/bin/python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume > "runs/rs4/train-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
tail -n 50 -f "$(ls -t runs/rs4/train-*.log | head -n 1)"
```

No entrenar generalista y especialista simultáneamente en el mismo Pod al medir
rendimiento. `Ctrl+C` en tail sólo cierra tail, no el entrenamiento en segundo plano.
Los runs, TensorBoard (`runs/rs4/tb`) y checkpoints periódicos son independientes.
No se versionan automáticamente los artefactos nuevos de `runs/rs4`.

Evaluación pareada, con entrenamiento pausado/guardado para evitar competencia:

```bash
python -m tools.evaluate_rs4 --run rs4 --games 128 --minutes 2 --seed 51
python -m tools.evaluate_rs4 --run rs4 --games 128 --minutes 2 --seed 73
```

Ambos modelos juegan RS4 contra los mismos rivales: `scripted:r3` y `parent.pt`.
Cada evaluación fija semillas, alterna rojo/azul y mantiene muestreo estocástico.
Congela candidato y rivales en copias temporales, comprueba hash del padre y guarda
informes únicos en `runs/rs4/evaluations/rs4_*/summary.json`, con hashes, puntos,
G-E-P, goles y eventos. El padre enfrentándose a sí mismo es sólo una referencia,
no un test de progreso. No reemplaza automáticamente al generalista ni declara
superioridad estadística por una pequeña diferencia de puntos.
`--checkpoint runs/rs4/ckpt_XXXXXX.pt` permite evaluar un candidato histórico;
`--opponents scripted:r3 ruta/rival_congelado.pt` permite añadir rivales comunes.
Usar checkpoints locales de confianza: PyTorch carga estos archivos con pickle.

Para volver al generalista, reanudar `multi` con su perfil habitual; jamás copiar
el especialista sobre `runs/multi/latest.pt`. RS4 sigue usando el presupuesto de
agentes y las optimizaciones del padre, pero la cantidad de muestras útiles por
iteración puede variar con el reparto de rivales y el nuevo tamaño de observación.

#### Menos asignaciones CUDA y perfil del update

Última medición aportada: callbacks **45.378 → 48.374 pasos/s reales (+6,6%)**,
rollout **1,683 → 1,501 s**, mismo checkpoint y 97.920 muestras/iter. El update
osciló de 0,380 a 0,427 s; no se atribuye esa oscilación a una mejora del update.

El nuevo retoque reutiliza el stream y el pool privado de captura
(`runtime.reuse_cuda_capture_pool=true`). Conserva una captura NUEVA y tres
warmups por rollout, con pesos, normalizadores e índices de rivales actuales;
el graph anterior sólo mantiene viva su reserva de memoria y nunca se reproduce.
Las capturas/replays son secuenciales, no concurrentes. Puede reservar más VRAM
entre rollouts. No elimina el coste completo de captura ni garantiza aceleración.

`runtime.reuse_minibatch_obs=true` reúne las filas mediante `index_select` en un
buffer por update. El backward se encola antes de sobrescribirlo en el mismo
stream. Conserva permutaciones, épocas, tamaño de minibatch, descarte de cola,
normalización, gradientes, Adam, BC, recompensas y número de muestras.

La velocidad de estos dos cambios **no está medida en CUDA localmente**.
Con entrenamiento detenido/guardado, probar en el Pod:

```bash
python -m pytest tests/test_cuda_decisions.py tests/test_minibatch_obs.py -q
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15 --no-reuse-capture-pool --no-reuse-minibatch-obs
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15
```

Comparar pasos/s REALES; repetir en orden inverso. Para aislar cada cambio,
desactivar sólo uno de los dos flags. El benchmark usa copias temporales del run.
`--profile-rollout` ahora muestra también el update CUDA: tiempo host de envío y
eventos GPU para BC, gather, forward/loss, backward y clipping/Adam. No agrega
sincronización por minibatch; lee eventos después de la descarga habitual de
métricas. Los intervalos GPU incluyen huecos de envío CPU: no representan sólo
kernels ni se suman a los host. La instrumentación agrega overhead; medir velocidad
sin ella. `--json-output` incluye captura y, al perfilar, desglose del update.

Reanudar con el mismo checkpoint y perfil habitual; para desactivar sólo este
retoque: `--override runtime.reuse_cuda_capture_pool=false runtime.reuse_minibatch_obs=false`.
CPU y el entrenador recurrente conservan su comportamiento previo.

#### Rollout: callbacks compilados de saques y cooperación

La comparación aportada del Pod validó la geometría anterior: **37.286 → 43.014
pasos/s reales (+15,4%)**, rollout **2,061 → 1,746 s**, mismas 97.920 muestras/iter.
Es una comparación de dos mediciones secuenciales, no un óptimo global demostrado.
El perfil posterior midió 1,198 s de entorno total, 0,282 s sólo en RS4,
0,175 s de contactos/retención de pases y 0,113 s de callbacks de saques.
Estos últimos están incluidos en los entornos: no sumar todos esos tiempos.

`runtime.optimize_callbacks=true` (por defecto con rollout optimizado) ejecuta
protección/pre/post-tick de saques y detección/retención de pases en kernels Numba
seriales, float64 y sin fastmath. Conserva cada tick: distancia y barrera de área,
patadas bloqueadas, liberación por patada/gol/timeout y velocidad mínima de saque
de arco. Los contactos simultáneos, candidatos de pase, retención, cadenas,
devoluciones, intercepciones y límite de recompensa conservan sus reglas.

Durante `step`, los callbacks acumulan directamente en un lote de reward/eventos
del paso, evitando arrays/diccionarios temporales y cuatro sumas por evento.
Ese lote es nuevo por paso: las llamadas públicas a los helpers y los datos
devueltos no se sobrescriben en pasos posteriores. No se altera PPO, física,
currículo, modelo, referencia BC, RNG ni cantidad de muestras. La captura CUDA
continúa invalidándose después de cada rollout, evitando pesos/normalizadores viejos.

Con el código actualizado y el entrenamiento **detenido y guardado**, comparar
desde el mismo checkpoint, sin carga concurrente:

```bash
# Referencia: conserva la geometría optimizada y todas las optimizaciones CUDA.
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15 --no-callbacks
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15
```

Repetir en orden inverso si hay ruido. Ambas pruebas usan copias temporales;
`latest.pt` no recibe sus updates. Para comparar el perfil, agregar
`--profile-rollout` a **ambos** (añade overhead). Para desactivar sólo callbacks
al entrenar: `--override runtime.optimize_callbacks=false`.
Con `runtime.optimize_rollout=false` también se usa la referencia de callbacks.
El flag `--no-reward-geometry` no desactiva estos callbacks: mide otro componente.

Microbenchmarks locales de 512 pasos de entorno dieron +17,3% en Futsal 3v3,
+12,3% en 5v5, +6,4% en 7v7 y +61,8% en RS4. **No son mejoras del entrenamiento
completo ni mediciones de la GPU del Pod**. Detalle:
`reports/rollout_callbacks_20261001.md`.

#### Rollout: geometría de recompensas y diagnóstico de CPU

El log aportado del nuevo Pod se mantiene en 35–40k pasos/s, con rollout 1,9–2,2 s,
prep ~0,06 s y update ~0,4–0,5 s: el rollout ocupa aproximadamente el 80% de esas
fases. No se busca subir el porcentaje de CPU como objetivo independiente.
Las tareas se avanzan secuencialmente después de recibir decisiones de la GPU;
hay muchas llamadas pequeñas y esperas, no un trabajo paralelo continuo.

Con `runtime.optimize_rollout=true`, las distancias jugador-pelota por equipo y
el potencial de separación se calculan en kernels Numba seriales pequeños,
float64 y sin fastmath. Evitan temporales NumPy de pares de compañeros y conservan
la misma geometría/recompensa. La distancia posterior se reutiliza para potenciales
y aproximación durante saques, sin calcularla dos veces. No se omiten premios de
cooperación, separación mínima o defensa aunque el log diga `shaping 0.00`.
El bootstrap CUDA usa sólo la cabeza de valor: no calcula logits que luego descarta.
No cambia PPO, modelo, currículo, RNG del entorno, entornos, rollout ni épocas.
Reducciones en float64 pueden tener redondeos mínimos; no se promete una trayectoria
bit a bit idéntica en todas las modalidades.

Para comparar sólo la geometría nueva, con entrenamiento detenido y el mismo
checkpoint/hilos/perfil en ambos casos:

```bash
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15 --no-reward-geometry
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 15
# Diagnóstico aparte: agrega overhead; no comparar su velocidad con las dos anteriores.
python -m tools.benchmark_multitask --config train/config_gpu_overnight.yaml --numba-threads 4 --warmup 3 --iters 5 --profile-rollout
```

Repetir finalistas en orden inverso si hay ruido. El benchmark informa CPU del
proceso en **núcleos equivalentes** (tiempo CPU de todos sus hilos / tiempo real)
y porcentaje del cupo detectado por afinidad/cgroups. Es distinto del panel del host;
el porcentaje agregado no demuestra que todos los hilos individuales estén ociosos.
El desglose separa entornos por tarea, distancias, separación, pases y protección
de saques, además de física/observaciones/esperas CUDA. Son tiempos inclusivos:
no sumar padres e hijos ni tiempos CPU/GPU solapados.
Para desactivar sólo geometría al entrenar:
`--override runtime.optimize_reward_geometry=false`. Para volver a la referencia
completa de rollout, conservar las opciones de vuelta documentadas abajo.

La comparación local de 128 transiciones de entorno dio mejoras de 3,8% en
Futsal 3v3, 29,9% en 5v5, 32,3% en 7v7 y 16,7% en RS4. **No mide inferencia,
PPO ni el Pod** y no demuestra un aumento equivalente de pasos/s globales.
Detalle: `reports/rollout_geometry_20261001.md`.

#### Perfil nocturno RTX 3060 Ti + Xeon (10 hilos / 15 GB RAM)

Con el código actualizado, guardar y detener la instancia anterior con **Ctrl+C una
vez**, esperar `guardado` y activar el entorno CUDA del Pod. Desde la raíz:

```bash
python -m tools.launch_gpu_overnight --run multi
```

El lanzador mide física 4/6/8/10, limitada al cupo real. Mantiene Torch `auto` (hasta
dos hilos CPU en CUDA), mide `pasos/s reales` y confirma finalista/baseline en orden
inverso. Si la ventaja es menor del 3%, conserva física 4. Cada proceso usa la misma
copia congelada de `latest.pt` y del `config.yaml` efectivo; los updates temporales se
descartan. Guarda resultados en `reports/gpu_startup/probe_*` y luego reanuda
automáticamente **el checkpoint original**, no el entrenado en un sondeo.
Es un ajuste rápido de arranque, no una garantía del óptimo. Un fallo detiene el
lanzador con su error, sin iniciar otra sesión. `--skip-tuning` permite usar física 4
directamente. La primera compilación puede tardar minutos.

Para dejarlo al cerrar SSH, sin otra instancia entrenando, en Bash:

```bash
nohup .venv/bin/python -u -m tools.launch_gpu_overnight --run multi > "runs/multi/overnight-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
```

`train/config_gpu_overnight.yaml` conserva arquitectura, física, recompensas, PPO,
1152 agentes, rollout 128, tres épocas y minibatch 8192. Reutiliza las predicciones
de la referencia BC congelada durante las épocas también en CUDA (~7,3 MiB para
106.496 filas), además de graph/buffers existentes. Conserva la referencia BC (o su
ausencia) del `runs/multi/config.yaml`; una opción explícita `bc_reference` al usar
`train.multitask` tiene prioridad. Se mantiene el optimizador y la liga al reanudar.

El presupuesto sube de 3.000M a 5.000M pasos útiles. `ppo.schedule_steps=3_000_000_000`
conserva los decaimientos actuales de LR, entropía y BC; después se clampa a sus
valores finales. Desde 2.919M, quedarían aproximadamente 14,5 horas **si se mantienen
40k pasos/s reales**, no una duración garantizada. Se desactivan replays concurrentes.

### Retención de checkpoints (GPU / RS4)

El perfil nocturno y las nuevas ramas RS4 actualizan `latest.pt` cada **5 minutos**
y guardan un histórico `ckpt_XXXXXX.pt` cada **30 minutos**, conservando los **6 últimos**.
Los intervalos son de tiempo real por sesión, independientes de los pasos/s, y se
comprueban al finalizar cada iteración. Ctrl+C y el fin normal siguen guardando latest.
Sólo después de publicar correctamente un histórico nuevo se eliminan los antiguos
del mismo run. No se tocan `parent.pt`, checkpoints de etapa, subdirectorios, enlaces
ni copias con otros nombres. Los históricos eliminados no se recuperan sin backup externo.
Esto limita las copias automáticas; no limita el espacio de TensorBoard/evaluaciones.

Para una rama ya existente, detener guardando primero y reanudar con:

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume --override log.checkpoint_interval_seconds=300 log.checkpoint_history_interval_seconds=1800 log.checkpoint_keep=6
```

La configuración resuelta queda guardada en el run. Las configuraciones sin intervalos
en segundos conservan el comportamiento anterior de `checkpoint_every`; sin
`checkpoint_keep` no hay limpieza automática. La primera limpieza ocurre tras el
primer histórico nuevo (aproximadamente 30 minutos desde el reinicio).

Los siguientes cambios son de **entrenamiento**, no aceleraciones equivalentes:

- Futsal pasa del 65% al 75% del presupuesto de agentes en C; aumenta AF/4v4/5v5/7v7,
  baja Futsal 3v3 del 35% al 25% y mantiene 2v2 en 10%.
- R2 usa 15% de scripted, para reunir evidencia de partidos más rápido. La promoción
  sigue exigiendo 128 partidos y dos evaluaciones consecutivas con los mismos umbrales.
- En Futsal R3, con al menos 128 partidos y >=95% de puntos, se pasa a 5% scripted,
  25% self-play y 70% liga. Por debajo del 90% se restaura 15/40/45. Los porcentajes
  se aplican al terminar partidos, nunca cambiando el rival a mitad de uno.
- PFSP conserva su fórmula y recibe más peso extra para el snapshot reciente;
  no supone que éste sea siempre más fuerte. Ganar al scripted no equivale a dominar
  la liga ni demuestra que ya no haya aprendizaje.

El log añade KL, clipfrac y evidencia para promoción/dominio. La métrica auxiliar
de goles ahora se actualiza también cuando la promoción usa puntos de partidos;
antes podía seguir mostrando una ventana vieja del checkpoint. Los porcentajes de
GPU suben/bajan al alternar física/inferencia/PPO: no se intenta fijarlos al 100%.
Más RAM libre no exige aumentar entornos; hacerlo cambiaría el lote y el aprendizaje.
El nuevo reparto puede cambiar las muestras útiles/iter y los pasos/s, por lo que
no se debe atribuir toda diferencia frente al perfil anterior a optimización técnica.

Pruebas locales: continuación sobre checkpoint con Adam/liga/BC conservados,
decaimientos, promoción, histéresis, cache BC y lanzamiento aislado. CUDA se prueba
sólo donde está disponible: no se ha medido una aceleración en esta 3060 Ti desde
la máquina local. Detalle en `reports/gpu_overnight_20261001.md`.

El reparto de rivales compensa las fracciones entre rollouts: un 5% de scripted
en una tarea de 13 entornos no se redondea permanentemente a cero ni se fuerza
un entorno en todos los rollouts. El saldo por tarea se guarda en el checkpoint
y se conserva al reconstruir entornos; los checkpoints anteriores siguen siendo
compatibles. Cuando la ventana no contiene goles contra scripted, el log muestra
`sin datos(0)` y TensorBoard registra NaN, no una proporción falsa de 0%.
Pruebas: `python -m pytest tests/test_opponent_allocation.py tests/test_validation_fixes.py -q`.

El currículo RS usa desde la etapa B `rs4_4v4` (RS One normal) y `jjrs_6v6`
(JJRS de las recs), con laterales/córners/saques de arco simplificados, sin
powershot, slide ni faltas. Las tareas reducidas RS y Pegeche permanecen en el
catálogo para evaluaciones antiguas, pero no se entrenan en este perfil. Se
mantienen las tres etapas para reanudar checkpoints existentes; no se reinician
la política, el optimizador, la liga ni los pasos. Sólo se invalidan una vez las
marcas/ventana de goles de RS4 medidas con las reglas anteriores. JJRS entra como
tarea nueva, con su propio inicio del shaping. Real Futsal conserva sus reglas.

Los checkpoints multitarea se escriben a un temporal del mismo directorio, se
sincronizan y se publican por reemplazo atómico: un fallo/interrupción durante
la escritura no trunca el archivo anterior. Si un `latest.pt` antiguo está
dañado, detener todos los procesos de ese run y ejecutar, por ejemplo:
`python -m tools.restore_checkpoint --run multi --checkpoint runs/multi/ckpt_002975.pt`.
La herramienta carga primero el checkpoint completo; si falla, no cambia latest.
Si es válido, respalda los bytes del latest anterior en un archivo de nombre único
y restaura por reemplazo atómico. Retomar con `--resume` (no `--init-from`).
Pruebas: `python -m pytest tests/test_atomic_checkpoints.py tests/test_soccer_curriculum.py -q`.

El scripted de 6v6 mantiene un portero fijo, un perseguidor entre los cinco
jugadores de campo y apoyos/coberturas en carriles distintos por identidad.
El perseguidor no abandona la pelota sólo porque un rival esté más cerca; los
apoyos no comparten el punto de cobertura ni ocupan el fondo del portero.
Los saques siguen usando un único ejecutor. Esta corrección no cambia el bot
de 4v4, recompensas ni entradas del modelo. Las métricas previas de JJRS se
invalidan una vez al reanudar porque cambió el rival, no porque se reinicie PPO.
Pruebas: `python -m pytest tests/test_scripted_six.py tests/test_scripted_restarts.py -q`.

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

#### Reparto equilibrado actual y perfiles históricos de 1v1

El perfil recomendado actualmente reparte el presupuesto por igual entre las siete tareas de la
etapa A (incluidos ambos estadios futsal 3v3). `fixed_weights` evita que los boosts guardados de una
fase anterior vuelvan a sesgar el reparto hacia 1v1:

```bash
python -m train.multitask --config train/config_balanced.yaml --run multi --resume
```

Los dos perfiles siguientes se conservan solamente para reproducir fases históricas:

Para un checkpoint que todavía está en **etapa 0**, hay dos perfiles separados del currículo normal:

| Perfil | `x1_1v1` | Cada una de las otras seis tareas |
|---|---:|---:|
| `train/config_1v1_focus.yaml` | 50% | 8,33% |
| `train/config_cooperation.yaml` | 20% | 13,33% |

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
