# Plan: bot competitivo de Futsal x3

## Objetivo

Crear un bot independiente para jugar Futsal x3 en el estadio `Futsal x3 by Bazinga from HaxMaps`, usando las grabaciones de `replays_real/stadiums/x3futsal` para aprender el comportamiento humano y, después, mejorar mediante entrenamiento competitivo en simulación. El bot se desplegará como modalidad separada: sus datos, política, modelos, configuración y pruebas no deben alterar el pipeline ni los modelos de X4/RS4-Z.

El objetivo final no es solo imitar las grabaciones: es jugar partidos completos y ser competitivo contra una referencia humana de esa modalidad, con las reglas y física representadas y la cadencia de control del despliegue host-local.

## Inventario inicial

- Hay **146 archivos `.hbr2`** en `replays_real/stadiums/x3futsal`.
- El estadio proporcionado está en `stadiums/futsal3bybazinga.hbs`; declara 76 segmentos, 130 vértices, 8 discos fijos y 2 arcos.
- Sus parámetros exportados incluyen radio de pelota 6,25, masa inversa 1,5, aceleración de jugador 0,11 y fuerza de patada 5.
- Las grabaciones pueden cambiar de estadio durante una partida. Solo se usarán los ticks cuyo estadio activo corresponda al mapa objetivo; los tramos de otros estadios no entrarán al dataset ni a la evaluación.

## F0. Auditoría realizada (2026-10-27)

Herramienta: `python -m tools.x3futsal_audit --workers 2`. Reutiliza el conversor existente sin modificar su comportamiento y guarda la caché, el índice, el informe y el split en `data/x3futsal_jsonl/` (datos locales ignorados por Git).

- De 146 archivos, **144 son únicos** y **2 son duplicados exactos por SHA-256**. Los 144 únicos terminaron con el callback `fin del replay`; tres no contienen ningún tramo del mapa objetivo.
- **141 replays contienen el estadio objetivo**. El filtro compara una identidad semántica normalizada del estadio que resuelve traits y física; no depende del nombre ni de que el JSON de replay conserve la misma representación del `.hbs`.
- Se conservaron **6.592.342 ticks** (1.831,21 min, unas 30,5 h) del mapa y se excluyeron **129.876 ticks** (36,08 min) de otros estadios. Los límites de los spans son `[inicio, fin)` y los cambios intermedios quedan fuera.
- **6.350.843 ticks (96,34%) son 3v3**.
- En los spans objetivo hay 1.791 goles, 83.900 patadas y 367 eventos `game_start`.
- Split agrupado por sala generado con semilla `20261027`: **53 grupos train, 7 dev y 7 test** (141 grabaciones elegibles; distribución **80,0001% / 10% / 10%**).

## F1. Validación de física y reglas de Futsal x3 (2026-10-27)

- La muestra determinista de 12 replays comparó **418.888 transiciones 3v3**. El máximo p90 por replay fue **0,00010 px para la pelota** y **0,0001414 px para jugadores**; el máximo p99 fue **0,0001415 px** y **0,0001542 px**, respectivamente.
- **Gate F1: completado para la física estándar del estadio.**

## F2. Dataset y baseline de imitación humana (2026-10-28)

- Contrato táctico v3 (`featurize_tactical` y `featurize_tactical_team` en `learn/x3futsal_data.py`): expone **28 features propias y 10 por entidad**, incorporando distancia euclídea a la pelota, margen exacto de contacto (`ball_gap`), indicador binario de alcance de patada (`ball_in_reach`), velocidad relativa, intercepción proyectada a 6 ticks, vectores a ambos arcos (`±557,5 px`), alineación de tiro/pase, rol dinámico (`is_closest_to_ball`, `is_last_man`, `race_margin_rival`) y estado de `kick_armed`.
- Filtrado activo contra anclaje pasivo en el área (`filter_box_camping` en `X3Data.sample_indices`): excluye del muestreo uniforme las muestras en las que un jugador permanece estático (`action % 9 == 0`) dentro de su propia área (`x < -half_w + 115`) mientras la pelota circula adelantada (`ball_x > -half_w + 210`), evitando que la imitación copie comportamientos AFK/arquero estático de salas públicas.

## F3 y F4. Optimización de Throughput y Evolución Táctica Rotativa 3v3 (2026-10-28)

### 1. Diagnóstico y solución del cuello de botella de rendimiento (~969 muestras/s → 28.690 muestras/s)

**Causas raíz detectadas en la medición previa (`710–969 muestras/s`):**

1. **Ausencia de configuración de hilos de simulación y fallback escalar en `sim/physics.py`:** `learn/x3futsal_rl.py` solo invocaba `torch.set_num_threads(threads)` sin configurar los hilos del motor físico (`set_physics_threads`), y cuando `numba` no estaba activo el simulador caía en triples bucles anidados en Python puro (`for n in prange(N): for i in range(K): for s in range(S):`), consumiendo más del 92% del tiempo de CPU. Además, `_collide_env` recalculaba `math.sqrt(sx*sx + sy*sy)` en cada segmento recto y verificaba vértices duplicados en cada tick.
2. **Bucles Python seriales por partido y por jugador en `learn/x3futsal_rl.py` y `learn/x3futsal_eval.py`:**
   - `_tactical_actions` y `_scripted_actions` iteraban `for row in range(sim.N): for slot in range(3): _quantize_direction(...)` en Python puro en cada paso de decisión.
   - `action_to_sim` se aplicaba mediante comprensiones de lista por partido (`[action_to_sim(a, 1) for a in ...]`) en lugar de una tabla vectorizada (`MIRROR_X_ACTION`).
   - `_observe_team` construía `(N, 7, 4)`, lo repetía 3 veces (`np.repeat`) y recalculaba distancias 3 veces por equipo en cada paso.
3. **Cómputo redundante en la actualización PPO (`learn/x3futsal_rl.py` y `learn/x3futsal_policy.py`):**
   - El ancla congelada (`anchor`) se evaluaba **dentro** del bucle `for _ in range(epochs):` (4 veces sobre las mismas observaciones estáticas).
   - `X3CentralCritic` se evaluaba sobre las `3 * N` filas de actores en lugar de una fila canónica por equipo (`N` filas, 3x menos cómputo).
   - `X3TacticalPolicy.features` expandía y concatenaba `ctx` (`17` dims) para cada compañero y rival en cada forward pass en lugar de proyectar linealmente `F.linear(entities, W_ent) + F.linear(ctx, W_ctx, b).unsqueeze(1)`.

**Optimizaciones implementadas:**

- **Motor físico multi-backend (`sim/physics.py`):**
  1. Backend **Numba JIT paralelo** (`@njit(cache=True, parallel=True)`).
  2. Backend **C99/OpenMP nativo auto-compilado** (`ctypes`, `_load_c_kernel`) con paralelismo `omp parallel for` y normales de segmento precalculadas (`_s_nx`, `_s_ny`) más deduplicación exacta de vértices activos (`46 → 33` vértices únicos en `futsal3bybazinga.hbs`, error numérico `0,0`).
  3. Backend **NumPy vectorizado con broadphase por lote** (`_step_batch_fast_numpy`, 20x más rápido que el bucle Python y con error `0,0`).
- **Observación por equipo sin copias (`featurize_tactical_team` en `learn/x3futsal_data.py`):** calcula las observaciones `(N*3, 28)`, `(N*3, 2, 10)` y `(N*3, 3, 10)` directamente sobre `sim.pos` y `sim.vel` en buffers pre-reservados.
- **Política táctica y decodificador 100% vectorizados (`_tactical_actions`, `_scripted_actions`, `_quantize_direction_batch`, `decode_actions_np`):** cero bucles Python sobre `envs` o `slots`.
- **PPO con ancla precalculada, crítico por equipo y minibatches residentes en caché (`learn/x3futsal_rl.py`).**

**Comparativa de rendimiento medida (20 updates × `horizon=128`):**

| Configuración | Tiempo anterior | Muestras/s anterior | Tiempo optimizado | Muestras/s optimizado | Aceleración |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `envs=16, threads=2` (`40.960` muestras) | 57,7 s | 710/s | **8,25 s** | **4.963/s** | **7,0x** |
| `envs=32, threads=4` (`81.920` muestras) | 99,8 s | 821/s | **8,14 s** | **10.062/s** | **12,3x** |
| `envs=64, threads=4` (`163.840` muestras) | 180,5 s | 908/s | **9,00 s** | **18.202/s** | **20,0x** |
| `envs=64, threads=8` (`163.840` muestras) | 169,2 s | 969/s | **9,43 s** | **17.376/s** | **17,9x** |
| `envs=128, threads=8` (`327.680` muestras) | 346,0 s | 947/s | **11,42 s** | **28.690/s** | **30,3x** |

---

### 2. Diagnóstico y solución del comportamiento táctico ("jugador anclado en el área propia")

**Causas raíz detectadas:**

1. **Secuestro del `slot 0` en `ModelPolicy(assist_mode="hybrid")` y `_tactical_actions` (`learn/x3futsal_eval.py`):**
   - En la versión previa, `_tactical_actions` fijaba `anchor = 0` (`slot 0`) permanentemente y lo enviaba a `guard_x = -half_w + 34 = -516 px` siempre que `bx < 220 px` (el 70% de la cancha).
   - Además, `ModelPolicy` con `assist_mode="hybrid"` sobrescribía incondicionalmente `actions[row, 0] = tactical_acts[row, 0]` en todos los ticks del partido, impidiendo que la red neuronal entrenada con BC/RL moviera al jugador `0` fuera del área.
2. **Bug de indexación y falta de incentivos de rotación/circulación en `learn/x3futsal_rl.py`:**
   - La penalización `deep_players = sim.player_pos[:, :, 0] < (-half_w + 115.0)` evaluaba a los 6 jugadores (incluyendo a los 3 atacantes rivales azules cuando entraban al área roja) en lugar de `sim.player_pos[:, :3, 0]`.
   - No existía recompensa por pases completados, recuperación por presión alta, rotación del último hombre (`cierre`), ni potencial de línea defensiva adelantada.

**Soluciones tácticas implementadas:**

1. **Rotación circular dinámica de los 3 jugadores en `_tactical_actions` (`learn/x3futsal_eval.py`):**
   - Se eliminó el rol fijo `anchor = 0`. En cada paso, los 3 jugadores (`slots 0, 1, 2`) compiten dinámicamente por los 3 roles del triángulo de futsal:
     - **1.º Hombre — Presionador / Conductor (`striker`):** el jugador con menor coste de intercepción a la pelota proyectada (con penalización de `+65 px` si quedó pasado por delante de la pelota `px > pred_bx + 16`, lo que activa el relevo inmediato de un compañero que viene de frente).
     - **2.º Hombre — Ala / Línea de Pase (`support`)** y **3.º Hombre — Cierre Dinámico (`cierre`):** asignados entre los 2 jugadores restantes mediante emparejamiento óptimo 2×2 hacia el carril débil de pase (`open_wing_y`) y el eje de cobertura (`cover_x, cover_y`), de modo que al cambiar la pelota de banda el ala y el cierre rotan sin cruzar toda la cancha.
     - **Línea adelantada del `cierre`:** cuando la pelota está en mediocampo o ataque (`bx >= -80`), el `cierre` adelanta su posición a `guard_x = clip(bx - 210, -240, +95)` (más de 220 px fuera del área propia) para participar en la circulación y cortar contraataques; en transición (`-350 <= bx < -80`) sostiene línea en `[-395, -230]`, y solo retrocede a cubrir la boca del arco cuando la pelota entra en zona de peligro profundo (`bx < -350`).
2. **Desbloqueo del modelo en `ModelPolicy` (`learn/x3futsal_eval.py`):**
   - `assist_mode="hybrid"` ya no secuestra el `slot 0`; permite que la política neuronal controle a los 3 jugadores e interviene cuando detecta que algún jugador intenta quedarse acampando pasivamente en el área propia (`px < -half_w + 115` con `bx > -half_w + 210`).
3. **Sistema de recompensas 3v3 y currículo con Self-Play + inicios de juego abierto (`learn/x3futsal_rl.py`):**
   - Corrige `red_px = sim.player_pos[:, :3, 0]` e incorpora penalización escalonada `box_stay_ticks` contra la permanencia en el área propia.
   - Añade término de línea adelantada al potencial de suma cero (`_potential`), recompensa explícita por pase completado (`--pass-reward 0.06`), recuperación por presión en campo rival (`press_recovery_reward = 0.025`) y rotación dinámica del cierre (`rotation_reward = 0.015`).
   - Combina en cada actualización entornos contra `TacticalPolicy` rotativa (con jitter de saque), `ScriptedPolicy` (con jitter) y **Self-Play (`--selfplay-frac 0.25`)**, más **25% de reinicios en situaciones tácticas de juego abierto (`reset_open_play_tactical`)**.

**Resultados verificados en lazo cerrado (`reports/x3futsal_eval_tactical_20261028.json`, 16 partidos por par, 60 s):**

- **`tactical` (rotativo) vs `scripted`:** **12 victorias, 4 empates y 0 derrotas** (`winRate` **75,0%** frente a 25,0% anterior), **28 goles a favor y 4 en contra** (**GD +1,50/partido**, 6x superior al +0,25 anterior), **5,00 pases completados/partido** (más del doble que el 2,25 anterior), **0,2% de tiempo en área propia (`ownBoxCampingFrac = 0,002` frente a `1,000` de `scripted`)**, **6,75 rotaciones de último hombre/partido** y **0,0 `safetyKickoffs`**.
- **`tactical` (rotativo) vs `repeat`:** **16 victorias, 0 empates y 0 derrotas** (`winRate` 100%), **56 goles a favor y 0 en contra** (**GD +3,50/partido**), **0,0% de tiempo en área propia** y **0,0 `safetyKickoffs`**.

## F5. Exportación ONNX y despliegue aislado (`export/to_onnx_x3futsal.py`)

- Implementado en `export/to_onnx_x3futsal.py`: exporta checkpoints `X3TacticalPolicy` a `<out>.onnx` y `<out>.json` con el contrato `x3-futsal-observation-v3`, dimensiones `(28, 2×10, 3×10)`, umbrales del decodificador desacoplado y validación automática de paridad de probabilidades (`err_p <= 1e-5`) frente a `onnxruntime`.

## Comandos listos para ejecutar la próxima etapa

```bash
# 1. Entrenar BC táctico v3 con muestreo activo anti-camping y selección en lazo cerrado
python -m learn.x3futsal_bc --tactical --out runs/x3futsal_bc/tactical_v3 --steps 20000 --threads 8

# 2. Entrenar RL competitivo MAPPO de alto rendimiento (envs=64 o 128, threads=8)
python -m learn.x3futsal_rl --bc runs/x3futsal_bc/tactical_v3/best_competitive.pt \
    --out runs/x3futsal_rl/ppo_v2 --envs 64 --threads 8 --horizon 128 --updates 200

# 3. Evaluar en lazo cerrado (midiendo goles, pases, ownBoxCampingFrac y roleRotations)
python -m learn.x3futsal_eval --checkpoint runs/x3futsal_rl/ppo_v2/best.pt \
    --matches 8 --minutes 1.0 --threads 8 --out reports/x3futsal_eval_ppo_v2.json

# 4. Exportar el modelo certificado a ONNX (F5)
python -m export.to_onnx_x3futsal --ckpt runs/x3futsal_rl/ppo_v2/best.pt \
    --out deploy/x3futsal/x3futsal_rl.onnx
```
