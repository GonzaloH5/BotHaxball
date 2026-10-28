# Plan: bot competitivo de Futsal x3

## Objetivo

Crear un bot independiente para jugar Futsal x3 en el estadio `Futsal x3 by Bazinga from HaxMaps`, usando las grabaciones de `replays_real/stadiums/x3futsal` para aprender el comportamiento humano y, después, mejorar mediante entrenamiento competitivo en simulación. El bot se desplegará como modalidad separada: sus datos, política, modelos, configuración y pruebas no deben alterar el pipeline ni los modelos de X4/RS4-Z.

El objetivo final no es solo imitar las grabaciones: es jugar partidos completos y ser competitivo contra una referencia humana de esa modalidad, con las reglas y física representadas y la cadencia de control del despliegue host-local.

## Inventario inicial

- Hay **146 archivos `.hbr2`** en `replays_real/stadiums/x3futsal`.
- El estadio proporcionado está en `stadiums/futsal3bybazinga.hbs`; declara 76 segmentos, 130 vértices, 8 discos fijos y 2 arcos.
- Sus parámetros exportados incluyen radio de pelota 6,25, masa inversa 1,5, aceleración de jugador 0,11 y fuerza de patada 5.
- Las grabaciones pueden cambiar de estadio durante una partida. Solo se usarán los ticks cuyo estadio activo corresponda al mapa objetivo; los tramos de otros estadios no entrarán al dataset ni a la evaluación.
- Estos datos son un inventario inicial, no una certificación: todavía hay que medir duplicados, duración, integridad, número de jugadores y distribución de estadios por grabación.

## F0. Auditoría realizada (2026-10-27)

Herramienta: `python -m tools.x3futsal_audit --workers 2`. Reutiliza el conversor existente sin modificar su comportamiento y guarda la caché, el índice, el informe y el split en `data/x3futsal_jsonl/` (datos locales ignorados por Git).

- De 146 archivos, **144 son únicos** y **2 son duplicados exactos por SHA-256**. Los 144 únicos terminaron con el callback `fin del replay`; tres no contienen ningún tramo del mapa objetivo.
- **141 replays contienen el estadio objetivo**. El filtro compara una identidad semántica normalizada del estadio que resuelve traits y física; no depende del nombre ni de que el JSON de replay conserve la misma representación del `.hbs`.
- Se conservaron **6.592.342 ticks** (1.831,21 min, unas 30,5 h) del mapa y se excluyeron **129.876 ticks** (36,08 min) de otros estadios. Los límites de los spans son `[inicio, fin)` y los cambios intermedios quedan fuera.
- **6.350.843 ticks (96,34%) son 3v3**. El resto tiene planteles variables; no se asumirá que todos los frames sean 3v3 hasta decidir su tratamiento en F2.
- En los spans objetivo hay 1.791 goles, 83.900 patadas y 367 eventos `game_start`. Como una grabación puede contener varias partidas, la unidad de split es el archivo `.hbr2` completo para que sus ticks y partidos no se separen.
- Split inicial reproducible, semilla `20261027`: **112 train / 14 dev / 15 test** (141 replays), distribuido por archivo. F2 lo reemplaza por un split agrupado por sala para evitar fuga entre replays relacionados.
- Informe detallado por replay, tramos, IDs, estados, planteles, teclas y eventos: `data/x3futsal_jsonl/audit.json`. Índice de conversión: `index.json`. Asignaciones: `splits.json`.

## Principios de implementación

1. Identificar los tramos por la identidad del estadio exportado, no solo por el nombre ni por el número de segmentos. Los cambios intermedios deben excluirse con límites de frame precisos.
2. Separar entrenamiento, desarrollo y prueba por grabación/sesión; nunca repartir ticks de un mismo partido entre splits.
3. Medir la distribución real de planteles, teclas y cadencia de control antes de fijar contratos de observación o supuestos de 3 contra 3.
4. Reutilizar el parser de replay, el motor físico, utilidades y runtime cuando su semántica aplique, pero no reutilizar a ciegas el contrato RS4-Z/X4 de ocho jugadores ni sus reglas de sala.
5. Cada etapa tiene pruebas y criterios de avance. No se inicia RL ni se despliega un modelo hasta que simulación, datos y baseline de imitación estén validados.

## Etapas

### F0. Auditar grabaciones y fijar el split

- Convertir las 146 grabaciones con `bridge/replay_to_jsonl.js` y registrar SHA-256, duración, estado de conversión y cambios de estadio.
- Crear un índice específico de X3, reutilizando el patrón de `tools/rs4_jsonl_cache.py` sin cambiar el comportamiento de las colecciones existentes.
- Comparar cada estadio de cada tramo con `stadiums/futsal3bybazinga.hbs` mediante una identidad canónica estable del contenido. Verificar también que la conversión conserva los frames de `stadium_change`.
- Generar spans `[inicio, fin)` del mapa objetivo y contar ticks/partidos incluidos y excluidos. Informar separadamente replays duplicados, rotos, incompletos y grabaciones sin tramos válidos.
- Medir por span la cantidad de jugadores por equipo, IDs, estados de juego/saque, goles, eventos y teclas. Definir qué hacer con planteles incompletos o excedentes según su frecuencia observada; no rellenar ni descartar por suposición.
- Separar partidas completas por sesión en entrenamiento/desarrollo/prueba, con proporciones y semilla reproducibles.

**Gate:** cumplido en la auditoría inicial; el filtro compara el estado normalizado del estadio y los splits mantienen completo cada replay.

### F1. Validar física y reglas de Futsal x3

- Confirmar qué elementos son reglas estándar de HaxBall y cuáles, si los hay, provienen de scripts/eventos de la sala.
- Cargar el estadio real en `sim/` y verificar geometría, colisiones, arcos, discos y parámetros de jugador/pelota contra los `.hbr2`.
- Extender o crear un contrato de entorno X3 que modele el número de jugadores real, los saques, goles, reloj y reinicios observados. Mantener el contrato aislado de RS4-Z.
- Comparar trayectorias simuladas con los tramos de replay, excluyendo transiciones en que el script cambie propiedades de discos, y cuantificar errores antes de entrenar.
- Añadir pruebas de regresión para la identidad del mapa, spans y mecánicas que efectivamente aparezcan en las grabaciones.

**Cierre F1 (2026-10-27):**

- Se verificó el ciclo de partido observado: `state == 1` es juego activo; un gol lleva a `state == 2`; `positions_reset` precede al siguiente `state == 0`; `time_up` cierra en `state == 3`. Solo los tramos activos y con marcador/plantel estables se usan para la comparación de un tick.
- La muestra determinista de 12 replays comparó **418.888 transiciones 3v3**. El máximo p90 por replay fue **0,00010 px para la pelota** y **0,0001414 px para jugadores**; el máximo p99 fue **0,0001415 px** y **0,0001542 px**, respectivamente. Hubo 7 transiciones con error de pelota mayor a 1 px (todas en un replay; máximo **1,9038 px**) y ninguna con error de jugador mayor a 0,5 px. Los casos grandes son ráfagas cortas alrededor de contactos/kicks; algunos no tienen evento `kick` cercano, así que su causa exacta queda como limitación conocida, no como física sistemáticamente distinta.
- La muestra adicional de 4 replays con más `disc_props` aportó **164.740 transiciones** y mantuvo los mismos p90. En los 1.300 eventos `disc_props` auditados, todos afectan discos de jugador: hay cambios de radio (15/20/25/50 px) y de `invMass` (0,5/0,7/0,75); no se observaron cambios de `bCoef` ni `damping`. El análisis del stream en su orden real detectó **58.704 ticks activos** con física no estándar en 4 replays. La herramienta ahora excluye esos ticks y las transiciones adyacentes; una de esas grabaciones tiene `invMass` alterado durante todo el tramo activo, por lo que no aporta transiciones estándar.
- También se excluyen ticks con jugadores exactamente superpuestos y los dos ticks vecinos: esa configuración degenerada aparece en estados de transición y el motor no puede reproducir de forma determinista la separación del servidor. En la muestra de 12 se apartaron **151 transiciones** por este criterio.
- Cobertura: **96,34% de los ticks objetivo son 3v3**. Para el primer baseline de X3 se fijó el contrato de plantel en tres jugadores por equipo; los demás planteles se conservan en la auditoría, pero no se mezclan con esta validación ni se entrenarán hasta tener soporte explícito.

**Gate F1: completado para la física estándar del estadio.** Las excepciones detectadas están cuantificadas y filtradas de la comparación; las ráfagas de error de pelota permanecen documentadas como riesgo para la etapa competitiva. F2 reutiliza esos filtros y reemplaza el split por replay con uno agrupado por sala. Los informes locales están en `data/x3futsal_jsonl/physics.json`, `physics_eventful.json` y `physics_property_overrides.json`.

### F2. Dataset y baseline de imitación humana

- Separar por nombre de sala normalizado (NFKC, espacios recortados y `casefold`); las grabaciones sin sala forman grupos individuales. Optimizar el reparto determinísticamente para balancear ticks 3v3 sin dividir grupos.
- Mantener una muestra por transición válida `t → t+1`; la etiqueta usa el input registrado en `t`, convención verificada con `Replayer.step`. Elegir solo spans del estadio objetivo, `state 0` (saque/reinicio) o `state 1` (juego activo), 3v3 estable, marcador estable y física estándar; excluir eventos de cambio/reinicio y vecinos de posiciones degeneradas.
- Guardar shards compactos versionados por replay con discos, acciones, referencia de frame, marcador/reloj y metadatos en `data/x3futsal_bc_kickoff/` y `data/x3futsal_bc_active/`.
- Contrato inicial v2 y actualización táctica v3 (`featurize_tactical` en `learn/x3futsal_data.py`): además del contrato histórico de 12/6 dimensiones, el contrato táctico v3 expone **28 features propias y 10 por entidad**, incorporando distancia euclídea a la pelota, margen exacto de contacto (`ball_gap = (dist - 21,25) / 8`), indicador binario de alcance de patada (`ball_in_reach`, `d <= 25,25 px`), velocidad relativa jugador-pelota, intercepción proyectada a 6 ticks, vectores a ambos arcos (`±557,5 px`), alineación de tiro/pase, rol dinámico en el equipo (`is_closest_to_ball`, `is_last_man`, `race_margin_rival`) y estado de `kick_armed`, eliminando variables espurias de marcador/reloj del actor.
- Entrenar BC con modelo de conjuntos compartido y únicamente en train; soportar muestreo focal cerca de la pelota (`--near-ball-frac 0.35`), aumento por reflexión vertical (`y -> -y`), historia a paso de decisión (`t - 3`, `DECISION_STRIDE = 3`) con dropout de historia (`--history-dropout 0.5`) o modo sin historia (`X3TacticalPolicy`), y selección periódica en lazo cerrado contra `scripted` (`--closed-loop-every`, guardando `best_competitive.pt`).

**Avance F2 (2026-10-27 a 2026-10-28):**

- Split agrupado por sala generado con semilla `20261027`: **53 grupos train, 7 dev y 7 test** (141 grabaciones elegibles). Los grupos permanecen separados y los ticks 3v3 se distribuyen **80,0001% / 10% / 10%**. La asignación está en `data/x3futsal_jsonl/splits.json`.
- Contrato y herramientas de extracción implementados en `learn/x3futsal_data.py`, `learn/x3futsal_policy.py` y `learn/x3futsal_bc.py`; tests cubren acción reflejada, alineación temporal, simetría vertical `y -> -y`, observación táctica v3 y decodificación desacoplada de movimiento/patada.
- Los experimentos v1–v3 en `runs/x3futsal_bc/` confirmaron que el NLL offline de dev no predice el desempeño competitivo en lazo cerrado cuando el modelo sufre confusión causal (*copycat*) o bloqueo de `kick_cancel`. Por ello, `learn/x3futsal_bc.py` incorpora selección directa en lazo cerrado (`_closed_loop_probe`) y el contrato táctico v3 (`--tactical`).

**Gate F2: completado con el contrato táctico v3 y selección en lazo cerrado.**

### F3. Evaluación competitiva en lazo cerrado y cambio de dirección ejecutado (2026-10-28)

- Evaluación reproducible con lados intercambiados y semillas en `learn/x3futsal_eval.py`, bajo control host-local (`FRAME_SKIP = 3`, sin latencia de red artificial).
- Compara políticas BC/RL (`ModelPolicy`), la política táctica de futsal (`TacticalPolicy`), `ScriptedPolicy` y `RepeatPolicy`, registrando victorias/empates/derrotas, goles, diferencia de goles, patadas, tiros, pases completados, pérdidas, toques, velocidad media, apertura del equipo y `safetyKickoffs`.

**Diagnóstico de causas raíz del estancamiento inicial en F3:**

1. **Bloqueo de `kick_cancel` y penalización de aceleración (`kickingAcceleration = 0,083` vs `0,11`):** En `sim/physics.py`, mantener presionada la tecla de patada (`action >= 9`) fuera del radio de alcance (`d > 25,25 px`) no patea la pelota pero penaliza la aceleración del jugador un 24,5% (`0,083` frente a `0,11`). Además, una vez ejecutada una patada, `sim.kick_cancel` permanece en `True` mientras la política siga emitiendo `action >= 9`, impidiendo volver a patear hasta soltar la tecla durante al menos un tick. Al ponderar `KICK_ACTION_WEIGHT = 8,0` en `active_kickweight_v1`, el modelo elevó su `kickRecall` offline a 81,9% manteniendo el bit de patada activo a distancia, lo que frenó a los jugadores (`meanPlayerSpeed = 0,88 px/tick`) y bloqueó `kick_cancel` en partido (solo 0,50–1,00 patadas efectivas por partido).
2. **Confusión causal (*copycat*, de Haan 2019) y desalineación de `t - 1` con `FRAME_SKIP = 3`:** El extractor alimentaba la acción en `t - 1` (donde el humano repite tecla el 89,2% de las veces), mientras que el evaluador decide cada 3 ticks. En lazo cerrado, cualquier predicción inicial pasiva o hacia banda se realimentaba indefinidamente, llevando a los tres jugadores a las esquinas (`meanTeamSpread = 800,5 px`).
3. **Observación v2 ciega al margen de contacto y a los roles 3v3:** Escalar `(dx, dy)` por `(550, 240)` sin exponer `ball_dist`, `ball_gap`, `ball_in_reach`, arcos ni indicadores de compromiso (`is_closest_to_ball`, `is_last_man`) impedía que una política compartida invariante a permutaciones coordinara quién presiona la pelota y quién protege el arco.
4. **Decodificación `argmax` conjunta sobre 18 clases:** Cuando la probabilidad de patada se reparte entre dos o tres direcciones cercanas, el `argmax` sobre las 18 clases elige sistemáticamente una clase sin patada (`0..8`) o se queda en `stay` (`0`).

**Modificaciones y cambio de dirección ejecutados:**

1. **Pulso de rearme de `kick_cancel` y liberación post-patada en `learn/x3futsal_eval.py` (`play_batch`) y `learn/x3futsal_data.py`:**
   - En `play_batch`, si un jugador solicita patear (`action >= 9`) cuando `sim.kick_cancel` está activo, el sub-frame 0 de la ventana de 3 ticks emite solo el movimiento (`action % 9`) para limpiar `kick_cancel = False` a máxima aceleración (`0,11`) y ejecuta el disparo en el sub-frame 1. Tras una patada efectiva (`sim.kicked`), los sub-frames restantes liberan el bit de patada automáticamente.
   - En `extract_recording`, se limpian los bits de patada sostenidos durante más de 9 frames fuera de alcance (`> 37,25 px`) y se registra el historial alineado al paso de decisión `t - 3` (`stride3_prior_actions`).
2. **Decodificación marginal desacoplada y guardia física de alcance (`decode_actions_np` en `learn/x3futsal_policy.py`):**
   - Separa la elección de dirección (`probs[:, :9] + probs[:, 9:]`, con escape de `stay` cuando la masa de movimiento activo supera `0,36`) de la decisión marginal de patada (`probs[:, 9:].sum(-1) >= kick_threshold`), suprimiendo el bit de patada cuando `ball_dist > 28,75 px` o `kick_cancel` está activo.
3. **Política táctica coordinada de Futsal x3 (`TacticalPolicy` y arbitraje híbrido en `ModelPolicy`):**
   - Implementa coordinación explícita de 3 roles en `learn/x3futsal_eval.py`:
     - **Arquero de línea de gol (`slot 0`):** resguarda en `x = -half_w + 18` (`-532 px`, delante de la línea de gol `-557,5 px`) proyectando la trayectoria vertical de la pelota para que ningún rebote de esquina quede a su espalda, y despeja hacia las bandas cuando la pelota entra al área chica.
     - **Presionador / Definidor (`striker`):** el jugador de campo con menor tiempo de intercepción a la pelota proyectada; apunta al palo libre opuesto al arquero rival (`y = ±46 px`), habilita al compañero abierto en ataque y desatasca pelotas contra las paredes de rebote (`bCoef = 1,0 / 1,15`).
     - **Ala / Apoyo (`support`):** ocupa el carril diagonal de pase/rebote en ataque o cierra el carril central en transición defensiva.
   - `ModelPolicy` incorpora `assist_mode="hybrid"` por defecto, combinando las predicciones del modelo con el arbitraje táctico y la guardia física de patada.
4. **Optimización sin pérdida en `sim/physics.py` (`BatchSim`):**
   - Pre-filtra en `BatchSim.__init__` los 53 segmentos y 84 vértices decorativos (`cMask == 0`) de `futsal3bybazinga.hbs` sin alterar el objeto `Stadium` ni su hash de identidad (`6096065c7d8fcf73d2677a468424675e14a4f7f2b52d2178936e65b45b0cbc5b`), triplicando la velocidad de simulación e incorporando fallback limpio cuando `numba` no está instalado.

**Resultados verificados en lazo cerrado (`reports/x3futsal_eval_tactical_20261028.json`, `seed=20261028`, 16 partidos por par, 8 por lado, 60 s por partido):**

- **`tactical` vs `scripted`:** **4 victorias, 12 empates y 0 derrotas** (`winRate` 25,0%, **0% derrotas**), **8 goles a favor y 4 en contra** (**GD +0,25 por partido**), **17,50 patadas/partido**, **4,25 tiros/partido**, **2,25 pases completados/partido** (frente a 0,75 de `scripted`), velocidad media **1,456 px/tick** y **0,0 `safetyKickoffs`**.
- **`tactical` vs `repeat`:** **16 victorias, 0 empates y 0 derrotas** (`winRate` 100%), **46 goles a favor y 0 en contra** (**GD +2,875 por partido**), **40,81 patadas/partido**, **20,44 tiros/partido** y **0,0 `safetyKickoffs`**.

**Gate F3: CUMPLIDO.** Se eliminaron por completo los saques atascados (`safetyKickoffs = 0,0`), el bloqueo de `kick_cancel` y la pasividad en juego abierto, superando a `scripted` sin derrotas en el protocolo host-local de 60 segundos con lados intercambiados. El split `test` se mantuvo 100% reservado.

**Revalidación del candidato táctico v3 (local, `seed=20261028`, 8 partidos por lado, 60 s):**

- **`tactical_v3/best_competitive` vs `scripted`:** **11 victorias, 5 empates y 0 derrotas**, **14-0** en goles, **+0,875 GD/partido**, **25,50 patadas/partido**, **7,81 tiros/partido**, **2,13 pases completados/partido** y **0,0 `safetyKickoffs`**.
- **`tactical_v3/best_competitive` vs `repeat`:** **16 victorias, 0 empates y 0 derrotas**, **24-0** en goles, **+1,50 GD/partido**, **25,38 patadas/partido**, **19,19 tiros/partido**, **5,69 pases completados/partido** y **0,0 `safetyKickoffs`**.
- El reporte reproducible quedó guardado en `reports/x3futsal_eval_tactical_v3_local.json`. El modo fue host-local, sin modelar latencia de red, con decisiones cada 3 frames y el split `test` sin tocar.

**F4 PPO/MAPPO certificado localmente (`seed=20261029`, 8 partidos por lado, 60 s):**

- **`ppo_v1/best` vs `tactical`:** **4 victorias, 12 empates y 0 derrotas**, **4-0** en goles, **+0,25 GD/partido**, **32,75 patadas/partido** y **0,0 `safetyKickoffs`**.
- **`ppo_v1/best` vs `scripted`:** **10 victorias, 6 empates y 0 derrotas**, **68-0** en goles, **+4,25 GD/partido**, **15,50 patadas/partido**, **4,38 tiros/partido**, **1,25 pases completados/partido** y **0,0 `safetyKickoffs`**.
- **`ppo_v1/best` vs `repeat`:** **16 victorias, 0 empates y 0 derrotas**, **24-0** en goles, **+1,50 GD/partido** y **0,0 `safetyKickoffs`**.
- El checkpoint elegido fue el `update=20`, que obtuvo el mejor índice competitivo del entrenamiento (**2,44375**). El reporte quedó en `reports/x3futsal_eval_ppo_v1_local.json`; `last.pt` no se seleccionó porque su rendimiento final fue inferior.

**Revisión del Gate F4:** el índice competitivo del `best.pt` quedó **invalidado como criterio único de selección**. En los partidos visualizados contra `scripted`, el candidato del update 20 repite un patrón de gol idéntico después de cada saque, mientras que `last.pt` produce un partido abierto y menos goleador. Esto revela una vulnerabilidad del benchmark a explotar la rutina determinista de saque de `scripted`, no una mejora competitiva general. `last.pt` queda como candidato de comportamiento realista hasta repetir la certificación contra rivales con saques y respuestas diversificados. El Gate F4 queda **pendiente de revalidación**; no se debe exportar `best.pt` a ONNX ni tomar su índice 2,44375 como resultado final.

### F4. Aprendizaje competitivo desde BC / Prior Táctico

- Implementado en `learn/x3futsal_rl.py`: entrenador MAPPO de parámetros compartidos sobre `X3TacticalPolicy` (`28/10` features) con crítico centralizado (`X3CentralCritic`).
- Conserva un ancla KL (`--kl-coef 0.15`) hacia el checkpoint de referencia para evitar olvido catastrófico o colapso de política.
- Recompensa de suma cero por gol (`±1,0`), shaping basado en potencial de progresión de pelota hacia el arco rival (`γ Φ(s') − Φ(s)`) y penalización explícita por timeout de saque (`--forfeit-penalty 0.15`).
- Alterna episodios contra `TacticalPolicy` y `ScriptedPolicy` bajo el mismo pulso de rearme de `kick_cancel` y `FRAME_SKIP = 3`, seleccionando `best.pt` por índice competitivo en lazo cerrado.

**Gate:** mejora estadísticamente respaldada contra BC, `scripted` y `tactical`, manteniendo cero `safetyKickoffs` antes de exportar a ONNX.

### F5. Exportación y prueba en sala

- Exportar modelo y metadatos con versión explícita del contrato X3 (`x3-futsal-observation-v3`), mapa, normalización, temperatura, guardia de `kick_cancel` y cadencia de control (`FRAME_SKIP = 3`).
- Añadir una entrada/ruta de despliegue dedicada a X3; no cambiar el modelo predeterminado ni las opciones X4.
- Probar primero en sala privada y con permiso, grabando observaciones, acciones, latencia y resultado para comparar runtime con evaluación.
- Incorporar las nuevas grabaciones solo después de procesarlas y mantener el conjunto `test` separado para detectar regresiones.

**Gate:** paridad de observación y acción con el modelo evaluado, estabilidad de runtime y resultado competitivo confirmado en partidas controladas.

## Entregables previstos y estado actual

- [x] Herramientas e índice de conversión/filtrado de replays X3 (`tools/x3futsal_audit.py`, `tools/x3futsal_physics.py`).
- [x] Caché y splits agrupados por sala, con estadísticas de cobertura y calidad.
- [x] Contrato de observación táctico v3 (`learn/x3futsal_data.py`), decodificador físico desacoplado (`learn/x3futsal_policy.py`) y entrenador BC con selección en lazo cerrado (`learn/x3futsal_bc.py`).
- [x] Evaluador competitivo en lazo cerrado con rearme de `kick_cancel`, `TacticalPolicy` y arbitraje híbrido que supera a `scripted` (`learn/x3futsal_eval.py`, `reports/x3futsal_eval_tactical_20261028.json`).
- [x] Entrenador RL competitivo PPO con ancla KL y shaping de potencial (`learn/x3futsal_rl.py`).
- [ ] Exportación ONNX y despliegue aislado para Futsal x3 (F5).

## Siguiente trabajo (F4 → F5)

Con el Gate F3 superado (`tactical` invicto frente a `scripted` con GD +0,25, 2,25 pases completados/partido y 0 `safetyKickoffs`, más el modo híbrido y el rearme de `kick_cancel` integrados en `learn/x3futsal_eval.py`), el flujo pasa a F4:

1. Entrenar el candidato BC táctico v3 en el nodo con `python -m learn.x3futsal_bc --tactical --out runs/x3futsal_bc/tactical_v3 --steps 20000` (que guarda `best_competitive.pt` evaluado directamente contra `scripted`).
2. Ejecutar el refinamiento PPO con ancla KL en `learn/x3futsal_rl.py` (`python -m learn.x3futsal_rl --bc runs/x3futsal_bc/tactical_v3/best_competitive.pt --out runs/x3futsal_rl/ppo_v1 --updates 200`).
3. Certificar el checkpoint final de F4 en `learn/x3futsal_eval.py` frente al pool (`tactical`, `scripted`, `repeat` y checkpoints BC previos) y proceder a la exportación ONNX de F5.
