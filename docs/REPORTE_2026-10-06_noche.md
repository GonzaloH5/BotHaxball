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

FINAL_BC

## Hipótesis anteriores corregidas

| Antes | Ahora | Evidencia |
|---|---|---|
| Latencia de sala ~15 ticks | lag 9–12 ticks (D = 8–11), medido antes del arreglo de hilos de ONNX | `reports/room_latency/` (14 trazas de bots × grabación del host) |
| 2K23 tiene 2 grabaciones humanas | son pruebas con 7 bots | nombres `RL-*` en los metadatos |
| Córner de Sanguchito: basta x hacia la cancha | también y, juzgado al patear | 3.092 patadas clasificadas sin error |
| El simulador no cedía laterales | sí los cede; era la herramienta | traza tick a tick |
| Estimar el retardo de cada humano por verosimilitud (plan E1) | no se pudo con este modelo: la NLL sube de forma monótona, sin codo | `reports/x4/human_delay_probe.json` |

## Qué queda para el pod (en orden)

Ver `docs/POD_RUNBOOK.md` y `docs/PLAN.md` §5.
