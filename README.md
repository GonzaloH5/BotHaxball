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
   - Reproduce cada replay y arma, con el mismo entorno del entrenamiento, la obs de cada jugador y la tecla que apretó.
   - Usa el estadio exacto de cada replay.
   - Descarta prácticas, mapas fuera del currículo y partidos con equipos desparejos.
   - `rsx6` (sala pública) pesa la mitad.
   - Resultado en `data/bc/`.
3. **Preentrenamiento:** `python -m train.bc --run bc`. Separa replays enteros para validación y muestra el acierto por mapa.
4. **RL desde el modelo imitador:** `python -m train.multitask --run multi --init-from runs/bc/bc.pt`. Con `ppo.bc_kl_coef` penaliza alejarse del estilo humano; el peso baja con el tiempo hasta `bc_kl_final`.

Para ver cómo juega el imitador: `python -m eval.render runs/bc/bc.pt runs/bc/bc.pt --task futsal_3v3 --out replays/bc.html`.

Rendimiento medido en esta PC (sólo CPU): la red se lleva ~90% del tiempo (`python -m tools.profile_train`). 8 hilos de torch es lo óptimo.

### Entrenamiento CPU + GPU en Runpod

En una imagen de PyTorch con CUDA, crear el entorno con `python3 -m venv .venv --system-site-packages`
y activarlo con `source .venv/bin/activate`. Instalar las dependencias de arriba sin instalar el wheel
CPU de PyTorch. Verificar `python -c "import torch; print(torch.cuda.is_available())"` (debe ser `True`).

Desde la raíz del proyecto, con `runs/multi/latest.pt` y `runs/bc2/bc.pt` presentes:

```bash
python -m pytest tests/test_cuda_runtime.py -q
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

Uso responsable: usar el bot sólo en salas propias o con permiso. En salas públicas o competitivas contra personas es hacer trampa.
