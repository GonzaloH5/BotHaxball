# Reporte de la sesión autónoma del 2026-10-06

Rama: `claude/funny-franklin-iwtrqe`, basada en `reinicio`. El trabajo de esta sesión está en commits chicos y separados por tema; no se tocó `reinicio`.

## Restricciones del entorno

- **Sin acceso al pod.** El contenedor solo sale por un proxy HTTPS que corta el SSH (el pod llega a mandar su banner pero el túnel se cierra), y no hay clave privada. Todo corrió en un contenedor de 4 vCPU sin GPU. Por eso:
  - lo que necesita GPU quedó implementado, probado en miniatura y con comandos listos en `docs/POD_RUNBOOK.md`;
  - la imitación sí se entrenó acá, a ~28k muestras/s.
- **`onnxruntime-node` no se pudo instalar** (la descarga del binario se corta en el proxy). El ONNX se verificó con onnxruntime de Python y la observación de sala con el test de paridad en Node.

## Qué se hizo, en el orden de prioridad pedido

### 1. Fidelidad del simulador

- **Conformidad de saques sobre las 498 grabaciones de Sanguchito** (antes se había hecho con 7): `reports/x4/conformance_x4.md`.
- **Corrección del script de Sanguchito (kernel).** En el córner, la sala solo libera con una patada hacia la cancha también en y, y juzga la dirección con la posición del pateador al patear (antes del tick).
  - Con la regla nueva, las 2.746 patadas que liberan y las 346 ignoradas se clasifican sin errores; con la anterior había 43 errores.
  - Tiene 2 tests sintéticos y 1 de regresión con una grabación real.
- **Corrección de la herramienta de conformidad.** Los 629 laterales cedidos al rival salían como "no detectados" porque se dejaba de simular un tick antes del saque siguiente.
  - El simulador sí cede: el 69% por corrida > 270 px y casi todo el resto por entrada sin patada.
  - Ahora coinciden el 99,7% de los laterales.
- **Demora de colocación de los cobros**, medida en todo el dataset: Sanguchito no tiene demora y RS ONE coincide con los rangos del contrato. Los tests de la mecánica ya existían (`tests/rs4z/test_rs4z_script_rs.py`).
- **Convención de retardo verificada contra el kernel:** la decisión tomada en S_t con retardo D se aplica desde el tick que produce S_{t+D+1}.
  - El armado de muestras desde grabaciones tenía un desfase de un tick. A retardo 0, la feature de patada armada filtraba la tecla de la etiqueta.
  - Corregido, con test.

### 2. Observabilidad y mediciones

- **Latencia de sala:**
  - los reportes `reports/room_latency/` dan un lag (mejor desfase por bot) de 11–12 ticks en la sesión 1 (7 bots) y 9–11 en la sesión 2, entre el frame observado y el aplicado. Las dos se midieron **con** la contención de ONNX; con el arreglo puede ser menor, y hay que volver a medirlo en E4 con `--trace`;
  - en la convención del simulador, D = lag − 1, y el RL sortea D ∈ {8, 9, 10, 11, 14};
  - **arreglo de hilos de ONNX:** `--ort-threads`, 1 por defecto, con test. Con 7 bots en la misma PC cada sesión usaba un hilo por núcleo.
- **Métricas de parecido humano** comunes a grabaciones y simulación (`tools/x4_metrics.py`): pases, posesión, forma del equipo, actividad, saques y saque inicial.
  - Referencia humana y techo prueba-vs-entrenamiento en `reports/x4/human_metrics_sanguchito.json`.
- **Evaluación en simulación** (`learn/x4_eval.py`): partidos con latencia de sala, ranking Bradley–Terry y parecido humano en self-play. El pool fijo incluye un scripted (`learn/x4_scripted.py`).

### 3. Dataset humano

- Las 651 grabaciones están convertidas, indexadas y auditadas: `reports/x4/dataset_audit.md`.
  - 498 de Sanguchito (antes había 7) y 147 de RS ONE: 93 h de 4v4 en total.
  - 4 duplicados exactos; las 2 grabaciones de 2K23 son pruebas con bots.
- **Particiones por bloque de sesión** (`reports/x4/splits.json`). Sanguchito queda en 2.072/307/443 min de entrenamiento/desarrollo/prueba.
- **Caché compacto por tick** (`tools/x4_ticks.py`, 2,4 GB en `data/`): base común de métricas, imitación y arranques desde estados humanos.
- **Defecto encontrado:** el arranque de muchas grabaciones de Sanguchito tiene ~630 ticks con posiciones nulas. Ahora se excluye; antes producía NaN en el entrenamiento e inflaba la espera del saque inicial.

### 4. Scripted

- `learn/x4_scripted.py`: presiona el más cercano, apoya el segundo y cubren los otros dos.
- Es débil a propósito: sirve como piso independiente de los datos. Contra un rival quieto mete 0,5 goles por minuto.

### 5–6. Imitación y RL

- **Observación v3** (`env/rs4z/obs_v3.py`):
  - 5 decisiones pendientes (15 ticks) para la latencia de 9–12;
  - geometría y mapa como condición;
  - margen de contacto con la pelota.
  - La misma función sirve para el simulador y las grabaciones. El bot de sala la reproduce en JS, con paridad verificada en 3.088 observaciones reales (máx 7e-7).
- **Imitación** (`learn/x4_bc.py`): política de conjuntos para compañeros y rivales, con el retardo de observación como feature y sorteado por muestra.
- **RL** (`learn/x4_ppo.py`): MAPPO con KL(BC‖π), crítico asimétrico con información privilegiada, shaping CHECKPOINT retirable, arranques desde estados humanos, pool con PFSP y evaluación periódica. Está probado en miniatura en CPU; para entrenar de verdad necesita el pod.
- **Despliegue:** `deploy/bot.js` acepta modelos `x4-obs-v3`; `export/to_onnx_x4.py` exporta y verifica contra torch.

### Resultados de la imitación (CPU, esta noche)

**Sonda de retardo** (retardo de observación aleatorio 0–24, 25M muestras, Sanguchito). La NLL en desarrollo sube de forma suave con el retardo: 0,587 a 3 ticks, 0,589 a 12 y 0,603 a 24. No hay codo, así que no se puede estimar el retardo de cada humano con este modelo. Ver PLAN §2, E1.

**Ablación del margen de contacto** (12M muestras, retardo 6–15, Sanguchito):

| | NLL | NLL en cambios de tecla | Recall / precisión de patada |
|---|---|---|---|
| A, sin margen | 0,599 | 2,804 | 0,569 / 0,718 |
| B, con margen | 0,601 | 2,812 | 0,561 / 0,713 |

No hay diferencia en lazo abierto. Se conserva el margen porque no empeora nada y su efecto esperado es en lazo cerrado.

**En lazo cerrado** (latencia de sala, partidos de 2 min), las imitaciones son pasivas:

| | Sonda (25M) | B (12M) | Humanos (p50) |
|---|---|---|---|
| Patadas por minuto y jugador | 1,5 | 0,55 | 3,2 |
| Pases por minuto | 1,6 | 1,7 | 8,8 |
| Distancia a la pelota del 1.º (px) | 123 | 118 | 84 |
| Contra el scripted (16 partidos) | 0 G, 15 E, 1 P (1–2 en goles) | 0 G, 5 E, 11 P (0–20) | |

- En B, los saques de arco se liberan por seguridad (599 ticks) y hubo saques iniciales que nadie ejecutaba.
- **Diagnóstico.** En lazo abierto la probabilidad de patada está calibrada: 0,028 contra 0,026 de los humanos, y 0,09–0,10 contra 0,104 con la pelota al alcance. La pasividad no es del modelo sino de la deriva de estados de la imitación pura (HR-PPO lo reporta): los bots se alejan de la pelota y nunca llegan a situaciones de patada.
- Es exactamente lo que E3 (RL con ancla KL) tiene que corregir.
- Mientras tanto, la imitación sola **no sirve para llevar a la sala**.

### RL en miniatura en CPU (prueba de mecánica, no de aprendizaje)

**Primera corrida** (desde B, λ=0,06, 32 partidos en paralelo): a las 50–60 actualizaciones **dejó de ejecutar el saque inicial**. El 83% de las acciones era "quieto" y los partidos de evaluación quedaban 1,5 min enteros en el saque inicial.
- Causa: equilibrio degenerado del entorno de entrenamiento. El reloj está congelado en el saque inicial (como en la sala) y el plazo de entrenamiento le pasa el saque al rival, que es la misma política. Quedarse quieto vale exactamente 0.
- Arreglo [inferencia]: penalización de suma cero de 0,1 por dejar vencer un saque o el saque inicial (`--forfeit-penalty`), con tests.
- También se corrigió la evaluación: un saque inicial abandonado ahora se libera por seguridad y se cuenta.

**Segunda corrida** (con penalización, 300 actualizaciones, ~307k decisiones):

| Actualización | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| Contra la BC (G-E-P, 16 partidos) | 1-15-0 | 1-14-1 | 0-14-2 | 1-14-1 | 1-14-1 | 1-15-0 |
| Patadas/min en self-play | 0,18 | 0,24 | 0,39 | 0,38 | 0,13 | 0,33 |
| Distancia a la pelota del 1.º (px) | 186 | 241 | 186 | 199 | 267 | 239 |
| Espera del saque inicial (s) | 13,5 | 2,7 | 20,0 | 4,7 | 9,7 | 10,1 |

- La mecánica es estable: KL(BC‖π) de 0,05–0,12, `clipfrac` de 0,02–0,05 y ningún colapso.
- No mejora contra la BC: a esta escala casi no hay goles en los rollouts (0–1 cada 10 actualizaciones).
- **Se aleja de la pelota** más que la propia BC (118 px) y patea menos. En el pod hay que vigilar `selfplay.dist_ball_1` y `kicks_per_min` desde la primera evaluación. Si la deriva sigue con millones de decisiones, el primer brazo a probar es un λ mayor (0,2, VPT).

### Imitación final (CPU, 164M muestras, Sanguchito + RS ONE con el mapa como condición, retardo 6–15)

- **Checkpoint versionado:** `runs/x4_bc/final_sangu_rsone/best.pt`. Es el punto de partida del RL en el pod.
- **Entrenamiento:** 40000 pasos con lote de 4096, unos 75 min a ~37k muestras/s.
- **En desarrollo** (Sanguchito + RS ONE): NLL 0,617; en los cambios de tecla, NLL 2,49 y precisión 5,0%; patada con precisión 0,70 y recall 0,50. Se estabiliza cerca de los 150M muestras.

**En lazo cerrado** (`reports/x4/eval_bc_final.json`; latencia de sala, partidos de 2 min):

| | Final | B (12M, solo Sanguchito) | Scripted |
|---|---|---|---|
| Rating Bradley–Terry | **+145** | −217 | +72 |
| Contra B | 12 G, 4 E, 0 P (16–0) | | |
| Contra el scripted | 5 G, 10 E, 1 P (5–1) | 0 G, 4 E, 12 P (0–17) | |

**Self-play contra humanos** (p50):

| | Final | Humanos | W1 normalizada | Techo |
|---|---|---|---|---|
| Patadas por minuto | 2,72 | 3,2 | 0,85 | 0,14 |
| Pases por minuto | 4,3 | 8,8 | 1,16 | 0,22 |
| Posesión (s) | 3,35 | 2,45 | 0,21 | 0,02 |
| Goles por minuto (por tramo) | 0,23 | 0,25 | 0,38 | 0,09 |
| Distancia a la pelota del 1.º (px) | 111 | 84 | 0,33 | 0,02 |
| Profundidad del equipo (px) | 530 | 464 | 0,30 | 0,05 |
| Jugadores parados | 3% | 2,6% | 0,33 | 0,12 |
| Cambios de tecla/s | 3,4 | 3,0 | 0,67 | 0,20 |
| Espera del saque inicial (s) | 5,1 | 3,9 | 0,38 | 0,10 |

**Sondas de coordinación** desde estados humanos de prueba (`reports/x4/probes_bc_final.json`):

| | Final | B |
|---|---|---|
| 2v1: pase | 27% | 6% |
| 2v1: gol | 9,4% | 0% |
| 3v2: pase | 25% | 10% |
| 3v2: gol | 5,5% | 2% |

Para B la muestra era chica (48 sondas).

- **Conclusión:** la pasividad de las imitaciones chicas era sobre todo falta de datos y de entrenamiento: de 12M a 164M muestras y sumando RS ONE, las patadas pasan de 0,6 a 2,7 por minuto. Solo una parte se explica por la deriva de la imitación pura.
- La final todavía está a 2–20 veces el techo humano según la métrica, pero casi siempre a menos de un desvío. Es un ancla razonable para E3.
- Conviene reentrenarla más larga en GPU: la NLL seguía bajando lento.

### RL desde la imitación final: λ = 0,06 contra λ = 0,2 (CPU, 32 partidos, 150 actualizaciones cada una)

| Actualización | 50 | 100 | 150 |
|---|---|---|---|
| **λ = 0,06** contra BC (G-E-P) | 0-13-3 | 0-10-6 | 0-8-8 |
| λ = 0,06 W1 humana media / patadas por min / distancia del 1.º a la pelota | 1,16 / 1,12 / 382 | 1,08 / 0,94 / 162 | 2,67 / 0,18 / 482 |
| **λ = 0,2** contra BC (G-E-P) | 1-14-1 | 2-11-3 | 3-7-6 |
| λ = 0,2 W1 humana media / patadas por min / distancia del 1.º a la pelota | 0,48 / 2,48 / 136 | 0,49 / 2,32 / 124 | 1,11 / 1,61 / 186 |

- Con 0,06 la política empeora de forma sostenida: pierde contra la BC y se aleja de la pelota. Es la misma deriva que la corrida desde B.
- Con 0,2 se mantiene cerca de la BC hasta las 100 actualizaciones y empieza a derivar a las 150.
- No hubo goles en los rollouts de entrenamiento.
- **No encontré un bug.** Revisé la atribución de recompensas por equipo, el espejado, el GAE y el KL; hay tests del shaping, de la penalización y de las acciones del crítico.
- **Lectura más probable:** con lotes de 7k muestras y sin goles, las ventajas son casi ruido, y cada actualización mueve la política (limitada por el clip) en una caminata aleatoria que el ancla solo frena. En el pod el lote es ~70 veces mayor (1024 partidos × 64 decisiones × 8), que es la recomendación de MAPPO.
- **Decisiones:** λ = 0,2 por defecto, decaimiento opcional `--lambda-decay` (VPT) y un criterio de corte pre-registrado en el runbook.

## Hipótesis anteriores corregidas

| Antes | Ahora | Evidencia |
|---|---|---|
| Latencia de sala ~15 ticks | lag 9–12 ticks (D = 8–11), medido antes del arreglo de hilos de ONNX | `reports/room_latency/` (14 trazas de bots × grabación del host) |
| 2K23 tiene 2 grabaciones humanas | son pruebas con 7 bots | nombres `RL-*` en los metadatos |
| Córner de Sanguchito: basta x hacia la cancha | también y, juzgado al patear | 3.092 patadas clasificadas sin error |
| El simulador no cedía laterales | sí los cede; era la herramienta | traza tick a tick |
| Estimar el retardo de cada humano por verosimilitud (plan E1) | no se pudo con este modelo: la NLL sube de forma monótona, sin codo | `reports/x4/human_delay_probe.json` |

## Qué queda (en orden)

1. **En el pod:**
   - E3 desde `runs/x4_bc/final_sangu_rsone/best.pt` con λ=0,2 y lote grande (`docs/POD_RUNBOOK.md` §4), mirando el criterio de corte desde la primera evaluación;
   - en paralelo, reentrenar la imitación más larga en GPU (§3): la NLL todavía bajaba.
2. **B1 (2K23):** grabar partidos humanos en esa sala y ajustar el script. Con las 2 pruebas con bots no coincide.
3. **RS ONE:** resolver los saques (v2 contra v2_lateral; el lateral se desvía 3,4 px al liberar) antes de volver a sumarlo al RL.
4. **Sondas 2v1/3v2** sobre los snapshots del RL. Ya existen para la BC: 27% y 25% de pase.
5. **E4 en la sala:** solo cuando E3 pase sus criterios, con `--trace` para volver a medir la latencia ya sin contención de ONNX.

## Archivos principales de esta sesión

| Qué | Dónde |
|---|---|
| Plan revisado | `docs/PLAN.md` |
| Comandos para el pod | `docs/POD_RUNBOOK.md` |
| Auditoría del dataset | `reports/x4/dataset_audit.md`, `index.json`, `splits.json` |
| Conformidad | `reports/x4/conformance_x4.md` y los JSON `restart_conformance_*`, `physics_conformance_*` |
| Referencia humana | `reports/x4/human_metrics_sanguchito.json` |
| Evaluaciones de la imitación | `reports/x4/eval_bc_final.json`, `probes_bc_final.json`, `eval_bc_B.json` |
| Código nuevo | `tools/x4_ticks.py`, `x4_index.py`, `x4_metrics.py`, `bridge/replay_meta.js`, `env/rs4z/obs_v3.py`, `learn/` (`x4_data`, `x4_policy`, `x4_bc`, `x4_ppo`, `x4_eval`, `x4_probes`, `x4_scripted`, `x4_delay_probe`), `export/to_onnx_x4.py`, `export/x4_room_fixture.py`, `deploy/rs4z/obs_v3.js` |
| Tests nuevos | `tests/rs4z/test_obs_v3.py`, `test_x4_ppo_parts.py`, `test_x4_real_conformance.py`, 3 nuevos en `test_rs4z_sangu.py`, `deploy/test_x4_room_state.js`, `deploy/test_ort_threads.js`. En total pasan 75 de Python y 10 de Node; los de Node que necesitan `onnxruntime-node` o `ws` no se pudieron instalar acá. |
