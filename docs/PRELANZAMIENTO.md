# Revisión pre-lanzamiento del RL X4 (2026-10-06)

Qué tiene que pasar antes, durante y después de la corrida larga en el pod, y con qué evidencia se llega a ella. El objetivo
del usuario es explícito: **la cadena de pase completa del agente tiene que llegar, como mínimo, al nivel del humano promedio
de la modalidad**, medida objetivamente contra grabaciones reales, antes de considerarlo competitivo.

> Este documento es la referencia de la corrida: los umbrales y las acciones de la cola (`learn/x4_queue.py`) salen de acá.
> Lo marcado **[inferencia]** es razonamiento propio sin verificación experimental.

## 1. La vara: la cadena de pase contra humanos

`tools/x4_pass_chain.py` mide 39 métricas con **el mismo código** sobre grabaciones humanas y sobre partidos simulados
(muestreo cada 3 ticks, estado antes del paso, mismas definiciones de toque, pase, pérdida y posesión que
`tools/x4_metrics.py`). Todo en el marco del equipo que tiene la pelota. El valor de cada acción sale de un modelo de valor
de posesión aprendido de humanos (`learn/x4_epv.py`: P(gol propio en 10 s) − P(gol rival); AUC 0,873 y logloss skill 0,28
en la partición de prueba, `reports/x4/epv.json`).

| Eslabón del usuario | Métricas (tipo) |
|---|---|
| posicionarse, generar líneas | compañeros con línea abierta (cono de intercepción), línea progresiva disponible, apoyo cercano no adelantado, control del espacio (eficacia); separación de la marca (estilo) |
| desmarcarse, ofrecer apoyo | desmarques que abren línea por minuto de posesión, desmarques que terminan en recepción, reofrecerse tras pasar (eficacia) |
| elegir y ejecutar el pase | pases/min, precisión (completados / intentos dirigidos), precisión bajo presión, progresivos, rompe líneas, al espacio, cambios de orientación, paredes, salidas de presión, asistencias, valor por intento (eficacia); fracción hacia atrás, largo y velocidad del pase (estilo) |
| anticipar, recibir, controlar, orientarse | retención 2 s tras recibir, retención bajo presión, control orientado hacia adelante, anticipación del receptor (eficacia); de primera, tiempo con la pelota (estilo) |
| aprovechar la ventaja | avance de la pelota en 3 s tras un pase progresivo o que rompe líneas, tiro en 5 s (eficacia); devolución inmediata hacia atrás (estilo) |
| posesión y valor de la secuencia | valor por posesión, pases por posesión, tiros/min (eficacia); circulación inútil (≥ 3 pases sin avanzar ni sumar valor; menos es mejor) |
| cuándo pasar y cuándo no | valor de cada decisión del portador (pase, conducción, aguantar, tiro, pérdida), pérdida sin pase bajo presión (menos es mejor), pasar cuando hay línea progresiva (estilo) |
| contexto defensivo | presión sobre el portador (estilo): evita que el pase "mejore" porque la defensa dejó de defender |

**Referencias humanas** (`reports/x4/pass_chain_human.json`):
- `sanguchito_test`: 77 grabaciones de prueba de la sala Sanguchito (376 min). Es la referencia del gate: mismo mapa,
  misma sala y jugadores que el agente no vio.
- `liga_rs_one`: 116 grabaciones de la liga HAXARG (equipos de torneo; RS ONE, misma cancha y física). Es la vara de
  "jugador competente": más pases (16,1 contra 12,8 por minuto de juego abierto), más precisión (0,73 contra 0,69), más
  paredes (1,9 contra 0,8) y defensas más cerradas (presión al portador 77 contra 98 px).

**Gate** (`tools.x4_pass_chain.gate`, pre-registrado y calibrado con humanos reales). Sobre las métricas con datos
suficientes (≥ 20 eventos esperados al ritmo humano, o ≥ 20 observaciones), aprueba si se cumplen las cinco condiciones
(umbral de las evaluaciones del entrenamiento / de la certificación):
1. **global:** índice de la cadena ≥ 0,93 / 0,95 (media geométrica de agente/humano en las métricas de eficacia, cada
   razón acotada a [0,25; 2]; 1 = promedio humano);
2. **cada eslabón:** el índice de cada etapa de la tabla ≥ 0,80 / 0,85 (ninguna parte del proceso queda claramente por
   debajo);
3. **ninguna métrica de eficacia muy por debajo:** cada una ≥ 0,65 / 0,70 veces la humana (en las de "menos es mejor",
   ≤ 1/0,65 / 1/0,70 veces);
4. **estilo:** cada métrica de estilo dentro del p10–p90 humano entre grabaciones;
5. **datos:** el 80% de las métricas juzgables en las evaluaciones del entrenamiento; el 100% en la certificación.

Los umbrales salen de la calibración con humanos (`tools/x4_gate_calibration.py`, `reports/x4/gate_calibracion_humanos.json`):
muestras de grabaciones humanas de entrenamiento del tamaño de cada evaluación, juzgadas contra la referencia de prueba,
aprueban 87 de 100 (55 min, evaluaciones del entrenamiento) y 95 de 100 (150 min, certificación). La BC (índice ~0,5)
queda lejos.

**Qué certifica y qué no.** El gate certifica "indistinguible del humano promedio de Sanguchito con esta muestra", no
"≥ promedio": en las 100 muestras humanas de 150 min el índice va de 0,935 a 1,047 (media 0,991), así que un agente con
un índice verdadero algo menor que el humano (~0,93–0,95) puede aprobar una certificación suelta por ruido. Por eso la
cola pide una confirmación con otras semillas (§4). Además, el "humano promedio" es el de la sala pública Sanguchito:
la liga HAXARG de RS ONE (más pases y más precisión) queda como vara informativa de "jugador competente".

Por qué no "≥ promedio en cada una de las 30 métricas": un agente exactamente promedio quedaría por debajo en la mitad por
puro ruido de muestreo; esa regla exigiría ser mejor que el promedio en todo (la primera versión reprobaba a 29 de 30
muestras humanas de 150 min). La regla de arriba exige nivel promedio en el conjunto y en cada eslabón, y no tolera ningún
déficit claro.

**Informativo, fuera del veredicto:** `indice_sin_valor`, el índice sin las tres métricas que salen del valor de posesión
(`epv_por_intento`, `epv_por_posesion`, `epv_por_decision`). Con el shaping EPV el RL optimiza justamente Δφ, así que esas
tres métricas no son evidencia independiente de la recompensa (§6).

## 2. Dónde estamos: la imitación (BC) de partida

Self-play de la BC con la latencia de sala: **índice ~0,57** [0,55; 0,62] con el índice actual (`reports/x4/ab_cpu_reeval.json`,
24 partidos). `reports/x4/pass_chain_bc_final.json` da 0,45 porque se calculó con la primera versión del índice, antes de
exigir datos mínimos por métrica: una métrica rara sin eventos lo hundía. Las cifras de abajo son de ese reporte y las de la
reevaluación van entre paréntesis.
- Al nivel humano: líneas disponibles, apoyo, línea progresiva, anticipación del receptor, presión defensiva.
- Muy por debajo: pases/min 5,7 (6,5) contra 12,8; precisión 0,49 (0,52) contra 0,69; retención tras recibir 0,40 (0,43)
  contra 0,58; tiempo con la pelota 0,10 (0,05) contra 0,40 s; paredes 0,12 (0,18) contra 0,77; valor por intento ~0
  (0,004) contra 0,022.

Diagnóstico (`reports/x4/bc_diagnose.json`, `pass_chain_bc_final_lowdelay.json`):
- En estados humanos (lazo abierto), la probabilidad de patear de la BC está **calibrada** con la humana en todas las
  distancias a la pelota.
- En sus propios partidos (lazo cerrado), alguien está al alcance de la pelota el 13% del tiempo contra el 29% humano: la
  BC toca la pelota y la suelta, no la controla. Con latencia baja (3–5 ticks) mejora poco (0,53), así que no es sobre todo
  la latencia. Bajar la temperatura empeora (0,42 a 0,7; 0,29 a 0,5).
- [inferencia] El déficit es de control en lazo cerrado (errores que se acumulan, intención que la imitación por decisión
  no sostiene), no de "no saber cuándo pasar". Eso es justo lo que el RL corrige bien: perder la pelota baja el valor de
  posesión en el paso siguiente.

## 3. Cómo empuja el entrenamiento hacia cada eslabón

| Mecanismo (learn/x4_ppo.py) | Qué empuja |
|---|---|
| Ancla KL a la BC (λ 0,2 → 0,05) | conserva lo que la BC ya hace como humano (posicionamiento, líneas, apoyo, estilo de pase) mientras el RL corrige el control |
| Shaping de potencial con el EPV, F = γ·φ(s') − φ(s) (suma cero) | cada pérdida de pelota se paga en el paso; cada pase que acerca al gol o rompe una línea suma; una posición sin pelota que los humanos asocian a goles sube φ. No cambia la política óptima (Ng 1999; equilibrios de Nash en juegos: Devlin y Kudenko 2011), así que no se puede explotar a la larga |
| Gol ±1 | el objetivo real |
| Arranques desde estados humanos (40%) | situaciones reales de posesión, presión y saque desde el primer minuto |
| Pool con PFSP (BC + snapshots) | defensas distintas; la BC es una defensa humano-símil |
| Brazo de pases (una sola decisión, en la primera evaluación desde la actualización 1000) | si el índice está < 0,85, +0,05 por pase de las posesiones que terminan en gol (TiZero, verificado en el paper: 0,05 por pase exitoso antes de un gol); no premia circulación inútil |
| Penalización por espera en saques | evita el equilibrio "nadie saca" (sólo esperas; el mal lateral ya cuesta la pelota) |

## 4. Plan de corrida (cola pre-registrada, `learn/x4_queue.py`)

Una sola orden en el pod (`docs/POD_RUNBOOK.md` §4). Cada paso tiene su condición y su acción; nada espera a una persona.

| Paso | Qué hace | Si sale bien | Si sale mal |
|---|---|---|---|
| 1. Preflight | GPU, hilos, disco, datos, referencias humanas, EPV, BC y su export a ONNX; mini-corrida con evaluación 0 y reanudación; una actualización y una evaluación de tamaño real con su tiempo y memoria | sigue | la cola se detiene sin gastar GPU (`preflight.json` dice qué falló) |
| 2. A/B temprano del shaping | `rl_epv` (potencial con el EPV) y `rl_checkpoint` (franjas de GRF) en paralelo hasta la actualización 300, con 512 partidos y la mitad de los hilos cada uno | gana el de mayor índice de la cadena (media de sus últimas 3 evaluaciones), salvo que en la última pierda contra la BC (< 0,45) o tenga deriva; con diferencia < 0,03, el EPV | un brazo cortado o sin evaluaciones no es candidato; si ninguno es válido, EPV |
| 3. RL principal | el ganador sigue en su carpeta hasta 3000 actualizaciones × 1024 partidos × 64 decisiones (~187M decisiones con el A/B) | certificación | corte por deriva → paso 4; se cae sin llegar a 3000 → fallo técnico (abajo) |
| 4. Recuperación (sólo si hubo corte) | desde `best_pase.pt` o `best.pt` del principal (o la BC), λ 0,4 y lr 1e-4; con el brazo de pases activo si el principal lo había activado | certificación | si también se corta: veredicto "revisar" y fin (no se gasta más) |
| 5. Extensión (una sola vez) | si la corrida llegó a 3000, ningún checkpoint aprobó el gate de pases y el índice de la cadena sube en las últimas 8 evaluaciones (pendiente > 0), sigue hasta 6000 | certificación | — |
| 6. Certificación | `learn/x4_certify.py` sobre el último `pase_aprobado_*`, `best_pase.pt` y `best.pt`, en ese orden: 64 partidos de self-play de 3 min, 64 por lado contra la BC, sondas; el que aprueba se certifica otra vez con otras semillas (`--confirm-seed`) | el primero que aprueba las dos veces se exporta a `deploy/rs4z/x4_rl.onnx`, veredicto "competitivo_en_pases" | se exporta el de mayor índice, veredicto "no_competitivo" (sirve para probar, no para competir) o "revisar" si hubo un fallo técnico |

**Fallas técnicas** (no son resultados): si un entrenamiento termina con error después de sus reintentos sin llegar a su
última actualización y sin corte por deriva, queda `fallo_tecnico` en `queue_state.json` con la actualización a la que
llegó, no se extiende y, si nada aprueba, el veredicto es "revisar". Una certificación que se cae (código distinto de 0,
aprobado, y 3, no aprobado) se reintenta una vez; si vuelve a caer queda en `certificacion_con_error` y no cuenta como
rechazo. Las partes informativas de la certificación (sondas, liga, cadena contra la BC) no pueden cambiar el veredicto:
si fallan quedan en `errores_informativos`.

**Duración** [inferencia]: la estimación real de 3000 actualizaciones con sus evaluaciones la da el preflight
(`actualizacion_real.horas_3000_con_evaluaciones`; si supera 24 h, la cola no arranca). La extensión o la recuperación
pueden duplicarla, y cada certificación con su confirmación agrega del orden de media hora a una hora.

**Dentro de la corrida** (cada 50 actualizaciones hay una evaluación; la evaluación 0 es la BC con el mismo protocolo):
- **Corte por deriva** (dos evaluaciones seguidas con alguna de estas condiciones; deja `stopped.json` con el motivo):
  - pierde contra la BC (`vs_bc.score` < 0,4);
  - se aleja del parecido humano (W1 media > máx(1; la inicial + 0,5));
  - se aleja de la pelota (distancia del más cercano > máx(200 px; 1,6 × la inicial));
  - deja de sacar (saques iniciales sin ejecutar por partido > máx(1; 3 × los iniciales));
  - la cadena de pase cae (índice < 0,8 × el inicial y todo su IC90 por debajo del inicial);
  - la defensa deja de presionar (presión al portador p50 > 1,3 × el p90 humano).
- **Brazo de pases**: una sola decisión, en la primera evaluación desde la actualización 1000 (~56M decisiones con el A/B).
  Si el índice de la cadena es < 0,85, se activa +0,05 por pase de las posesiones que terminan en gol, desde ahí hasta el
  final (`eval.jsonl` → `brazo_pases_activado`); si no, queda descartado aunque el índice baje después
  (`brazo_pases_descartado`).
- **Checkpoints:** `best.pt` (fuerza con parecido humano, confirmada con otra evaluación; `best.json` dice si le gana a la BC
  con margen), `best_pase.pt` (mejor índice de la cadena sin perder contra la BC), `pase_aprobado_*.pt` (gate aprobado en dos
  evaluaciones seguidas), `last.pt` cada 10 actualizaciones y después de cada evaluación, `evals/` con la política y los
  partidos de cada evaluación.

**Qué debería verse en cada tramo** (señales tempranas; si no aparecen, la corrida igual sigue hasta un corte o el final,
pero quedan registradas para decidir la próxima):

| Tramo | Debería pasar | Señal en los logs | Alarma |
|---|---|---|---|
| 0–20 (crítico solo) | la política es la BC; el crítico aprende el valor | `explained_var` sube desde ~0; `kl_bc` = 0 | `explained_var` negativa o `nonfinite_skipped` > 0 |
| 20–300 (~10M, A/B con 512 partidos por brazo) | menos pérdidas tontas: la BC suelta la pelota; el shaping de valor castiga cada pérdida | `retencion_tras_recibir` y `precision_pase` suben; `tiempo_con_pelota_p50` sube hacia 0,4 s; `kl_bc` 0,01–0,05 | `presion_al_portador_p50` sube (la defensa se afloja); `dist_ball_1` sube; `kickoff_frac` sube |
| 300–1000 (~56M) | más pases que sirven: progresivos, que rompen líneas, salidas de presión; pool con rivales distintos | índice de la cadena hacia 0,7–0,9; `epv_por_intento` positivo; `vs_bc.score` > 0,5 | el índice no sube desde la evaluación 0 → brazo de pases en 1000 |
| 1000–3000 (~187M) | λ baja hasta 0,05 (llega cerca de la actualización 2770): más libertad; combinaciones (paredes, al espacio) y aprovechar la ventaja | índice ≥ 0,93 y etapas ≥ 0,80 (gate del entrenamiento); `pase_aprobado_*`; `indice_sin_valor` cerca de `indice` | `human_w1_mean` sube (estilo artificial); `humanos_dev.nll` sube mucho (se olvida lo humano); `indice` muy por encima de `indice_sin_valor` (el valor sube por el shaping, no por el juego) |

[inferencia] Las cifras de cada tramo son expectativas, no garantías: salen del diagnóstico de la BC y del A/B de CPU (§7), a
una escala 30 veces menor. Los millones de decisiones cuentan que las primeras 300 actualizaciones son del A/B, con la
mitad de los partidos.

## 5. Problemas encontrados y corregidos

Ninguno de estos estaba a la vista con "el comando corre". Cada uno tiene test o evidencia en el repo.

**Simulador**
- **Lateral que rebota sin fin** (`env/rs4z/kernel.py`). Un arranque desde un estado humano grabado en el primer tick de un
  lateral, con la pelota todavía adentro de la línea, disparaba "entrada sin patada" en cada decisión: el lateral pasaba de
  un equipo al otro todo el partido. Afectaba a ~12% de los arranques humanos con lateral. Ahora "entrada sin patada" exige
  que la pelota haya estado colocada afuera, y el StateBank coloca la pelota donde la pone el script
  (`test_lateral_started_with_ball_inside_does_not_ping_pong`).
- **Causa de cada saque perdido** (`EV_FWHY`): la penalización por saque vencido caía en un 97% sobre laterales mal
  ejecutados (regla de la sala), no sobre esperas. Ahora sólo se penalizan las esperas; el mal lateral ya cuesta la pelota.

**Medición**
- **No había forma de medir la cadena de pase contra humanos.** Ahora: `tools/x4_pass_chain.py`, el EPV y las dos
  referencias humanas.
- **El gate reprobaba a humanos reales.** La primera versión ("≥ promedio en cada métrica" o "ninguna significativamente
  peor") reprobaba a 29 de 30 muestras humanas de 150 min. Recalibrado con humanos (`tools/x4_gate_calibration.py`,
  `reports/x4/gate_calibracion_humanos.json`): aprueba al 95% de las muestras humanas del tamaño de la certificación y al
  87% de las del entrenamiento; la BC (índice ~0,5) queda lejos.
- **Índice ruidoso con evaluaciones chicas.** Una métrica rara con 0 eventos hundía el índice y disparaba cortes en falso
  (lo mostró el A/B de CPU). Ahora cada métrica exige datos suficientes y el corte por pases exige que todo el IC90 del índice
  quede por debajo del valor inicial.
- **Grabador de simulación desalineado.** Tomaba el estado después del paso, con la pelota ya fuera del pie; las
  grabaciones toman el estado antes. Ahora están alineados.
- **El self-play podía inflar los pases con una defensa blanda.** Se agregaron la presión sobre el portador (banda humana y
  corte por deriva) y la cadena de pase contra la BC como rival fijo.

**Entrenamiento**
- **Retardo informado:** el bot de la sala le informa siempre 10 a la política, pero en el entrenamiento veía el retardo
  verdadero. Ahora la mitad de los partidos ven 10 y el resto el verdadero ± 2; el retardo sorteado incluye 6–7.
- **Crítico:** la "acción próxima" que veía era la de una decisión antes (corregido, con test); ahora ve el tipo de rival y
  el λ efectivo.
- **Shaping:** el CHECKPOINT no era Markov para el crítico (franjas cobradas invisibles) y en una corrida de CPU vino con la
  defensa sin presión. Por defecto, shaping de potencial con el EPV.
- **Guardas de no finitos:** una actualización con NaN se descarta y vuelven los pesos anteriores; nunca se guarda un
  checkpoint no finito.

**Operación**
- **Decisiones que esperaban a una persona** (brazo de pases "a mano entre 100 y 300M", qué hacer tras un corte, cuándo
  extender, qué checkpoint exportar): ahora todo lo decide la cola con reglas escritas.
- **Reanudación:** `best.pt` podía quedar pisado por uno peor al reanudar; los checkpoints no eran atómicos; el pool no
  tenía tope; nada relanzaba el proceso. Corregido (guardado atómico después de cada evaluación, pool con reindexado, bucle
  de reintentos en la cola).
- **Trazabilidad:** `run_meta.jsonl` (versión del código, GPU, hilos), `evals/` con la política y los partidos de cada
  evaluación, `log.jsonl` con varianza explicada, normas de gradiente y conducta (distancia a la pelota, quietos, patadas,
  saques iniciales).
- **Cortes relativos a la evaluación 0** (antes eran absolutos y calibrados con otra BC) y nuevos: "nadie saca", defensa
  que deja de presionar, caída de la cadena de pase.

**En la revisión pre-lanzamiento** (cada uno con test en `tests/rs4z/test_x4_queue.py`, `test_x4_ppo_parts.py` o
`test_x4_pass_chain.py`, o reproducido antes de corregirlo; §7)
- **El brazo de pases no era una sola decisión.** Se activaba en cualquier evaluación posterior a la 1000 con el índice
  < 0,85, por ejemplo en una baja pasajera en la 2950: la recompensa podía cambiar casi al final. Ahora se decide una vez
  y la decisión sobrevive a la reanudación (`test_pass_arm_is_decided_once`).
- **La cola no distinguía una corrida caída de una terminada.** Si el RL principal fallaba en los 5 reintentos (p. ej. un
  error reproducible en la actualización 1200), la cola seguía a la extensión y a la certificación y terminaba en
  "no_competitivo", como si se hubiera medido. Ahora queda `fallo_tecnico`, no se extiende y el veredicto es "revisar"
  (`test_crashed_main_run_is_a_technical_failure`).
- **Una certificación que se caía contaba como rechazo.** Python sale con 1 ante un error, el mismo código que "no
  aprueba", y el reporte se escribía al final: una falla en una parte informativa (p. ej. las sondas) descartaba un
  checkpoint aprobado sin dejar rastro. Ahora "no aprueba" es 3, las fallas se reintentan y las partes informativas no
  cambian el veredicto (`errores_informativos`; `test_crashed_certification_is_not_a_rejection`).
- **Certificación no reproducible.** Las políticas muestreaban con el generador global de torch sin semilla. Ahora dos
  certificaciones con la misma semilla dan lo mismo.
- **Varias chances de aprobar por ruido.** Se certificaban hasta 3 candidatos por corrida, una vez cada uno, y ganaba el
  primero que aprobaba. Ahora el que aprueba se confirma con otras semillas
  (`test_certification_needs_a_confirmation_with_other_seeds`). [inferencia] Con confirmaciones independientes, un agente
  de nivel humano promedio aprueba ~90% de las veces (0,95²) y uno con índice verdadero ~0,93 pasa de ~20% a ~4% por
  candidato.
- **A/B decidido con una sola evaluación.** El índice de una evaluación tiene un IC90 de ±0,04–0,06
  (`reports/x4/ab_cpu_reeval.json`), más que el margen de 0,03 de la regla. Ahora se compara la media de las últimas 3
  (`test_ab_uses_the_mean_of_the_last_evaluations`).
- **Reanudación:** el récord de `best_pase.pt` se perdía si el proceso caía entre guardarlo y guardar `last.pt` (una
  evaluación peor lo pisaba), y esa evaluación quedaba repetida en `eval.jsonl`, que usa la regla de extensión. Corregido
  (`test_resume_keeps_the_best_pass_record`, `test_evaluations_are_deduplicated_by_update`).
- **Recuperación con otra recompensa.** Si el principal había activado el brazo de pases, la recuperación seguía desde su
  checkpoint sin el brazo. Ahora lo hereda (`test_recovery_keeps_the_pass_arm_of_the_main_run`).
- **El export a ONNX no se probaba hasta el final.** Sin `onnx` u `onnxruntime` en el pod, la cola fallaba después de
  horas de corrida. Ahora el preflight exporta la BC y la verifica.
- **Documentación desalineada con el código:** el gate de §1 (decía índice ≥ 1, etapas ≥ 0,9 y la regla del IC90; el
  código usa los umbrales calibrados), el índice de la BC en §2 (0,45 era de la primera versión del índice; con la actual,
  ~0,57), el paso A/B que faltaba en §4, las decisiones de cada tramo y los docstrings de la certificación, de la cola y
  del trainer.

## 6. Riesgos que siguen

Ordenados por cuánto pueden cambiar la conclusión de la corrida. Ninguno impide lanzar; cada uno dice qué mirar.

1. **La medida no es independiente de la recompensa.** Con el shaping EPV el RL optimiza Δφ, y tres métricas del gate son
   Δφ (`epv_por_intento`, `epv_por_posesion`, `epv_por_decision`). Pesan 3 de 30 en el índice global, pero son la mitad
   de la etapa "decisión" y un cuarto de "posesión y valor". Además, el RL puede visitar estados poco humanos donde φ se
   equivoca: el potencial no cambia la política óptima, pero en el corto plazo la señal puede engañar. **Mirar:**
   `indice_sin_valor` contra `indice` en `eval.jsonl` y en la certificación. Si el gate aprueba sólo gracias a las
   métricas de valor, no es evidencia de juego de pase.
2. **El gate sólo mide self-play.** Una defensa propia blanda puede inflar los pases del mismo agente. Lo frenan la banda
   humana de presión al portador y el corte por deriva. La cadena de pase contra la BC como rival fijo se informa pero no
   se exige, porque no está calibrada con humanos. **Mirar:** `cadena_pase_vs_bc.indice` (BC contra BC ≈ 0,55) en
   `eval.jsonl` e `indice_vs_bc` en la certificación.
3. **"Humano promedio" es el promedio de Sanguchito, con tolerancia.** El gate certifica que el agente es indistinguible
   del humano promedio de esa sala pública, no que lo supera (§1). La confirmación reduce los aprobados por ruido, pero no
   la tolerancia. La liga RS ONE (16,1 pases/min contra 12,8; precisión 0,73 contra 0,69) queda como vara informativa.
4. **Los pases se detectan por trayectoria.** Un cambio de dueño dentro del equipo con ≥ 40 px de recorrido cuenta como
   pase, aunque sea un rebote o un choque. Las reglas son las mismas para humanos y agente, pero un estilo distinto puede
   cambiar la mezcla: la BC suelta la pelota a los 0,05 s y los humanos a los 0,4 s. No hubo una auditoría manual del
   detector. La biblioteca de conocimiento lo advierte (`docs/conocimiento/real_soccer_x4/09_entrenamiento_y_replays.md`:
   la trayectoria sola no revela intención). **Antes de llamar competitivo a un checkpoint:** revisar a mano unos 30 pases
   detectados del agente (`evals/ep_*.npz`) y otros 30 humanos.
5. **Lo que el gate no mide.** Defensa tras pérdida (pérdidas peligrosas, ocasiones concedidas), segundas pelotas, roles y
   reinicios. Hoy sólo aparecen de forma indirecta: en la fuerza contra la BC y en las sondas 2v1/3v2, que son
   informativas.
6. **El A/B ve sólo el comienzo.** 300 actualizaciones con la mitad de los partidos (~10M decisiones por brazo) miden
   efectos tempranos. Con la media de 3 evaluaciones la regla sigue cerca del ruido; en empate gana el EPV.
7. **Latencia de la sala.** El RL entrena con D ∈ {6, 7, 8, 9, 10, 11, 14} y el bot le informa siempre 10. El lag real
   después del arreglo de hilos de ONNX no se volvió a medir (E4 con `--trace`). La certificación usa D ∈ {8–11}; las
   evaluaciones del entrenamiento, la distribución completa.
8. **Fidelidad que sigue supuesta.** El plazo del lateral de Sanguchito (599 ticks) no se verificó: ningún lateral grabado
   pasó de 301 ticks. Los arranques desde estados humanos empiezan con las acciones pendientes en "quieto" durante hasta
   14 ticks (menos del 0,1% de las decisiones).
9. **Operación.** Un reinicio del pod mata la sesión de tmux: hay que relanzar a mano el mismo comando, y la cola sigue
   donde estaba. La imitación no se reentrena en paralelo con la cola, porque compite por la GPU y por los hilos que midió
   el preflight. La duración total no está acotada: el preflight estima sólo 3000 actualizaciones, y la extensión o la
   recuperación pueden duplicarla.

## 7. Evidencia

**De la revisión** (rama `claude/funny-franklin-iwtrqe`, 2026-10-06). El workflow automático de revisión se cortó por
límite de uso; la revisión se rehízo completa en una sesión local, sin GPU y sin el caché `data/x4_ticks`.
- **Tests.** Antes de los cambios: 90 pasan, 1 saltado (la conformidad con grabaciones reales, que necesita el caché).
  Después: 100 pasan, 1 saltado (`python -m pytest -q tests`).
- **Hallazgos reproducidos con el código sin cambios** (script de verificación):
  - brazo de pases: con índice 0,90 en la evaluación 1000 y 0,80 en la 1050, se activaba en la 1050;
  - cola: con el principal fallando en todos los reintentos, seguía a la certificación sin marcar ninguna falla;
  - certificación: ante un error salía con código 1 y sin reporte, y la cola lo contaba como rechazo.
- **Correcciones verificadas de punta a punta.** Una certificación chica (4 partidos) sale con 3, "no aprueba". Registra
  la falla de las sondas, que acá no tienen datos, sin cambiar el veredicto, y repetida con la misma semilla da los mismos
  números. El export a ONNX que ahora corre el preflight funciona: diferencia máxima con torch de 2,3e-5.
- **Ruido del índice** (`reports/x4/ab_cpu_reeval.json`, 24 partidos de self-play): BC 0,575 [0,548; 0,623], EPV 0,583
  [0,552; 0,629], franjas 0,640 [0,582; 0,689].
- **Calibración del gate** (`reports/x4/gate_calibracion_humanos.json`): aprueban 95 de 100 muestras humanas de 150 min y
  87 de 100 de 55 min. El índice de las de 150 min va de 0,935 a 1,047.
- **Citas verificadas:** TiZero (arXiv 2302.07515) da 0,05 por pase exitoso antes de un gol; VPT (arXiv 2206.11795) usa
  un coeficiente KL de 0,2 que decae ×0,9995 por iteración.
- **Revisado sin defectos:**
  - el shaping de potencial: suma cero, φ(s') = 0 al terminar el partido, mismo γ que el GAE, y el crítico ve φ en el marco
    de cada equipo;
  - el GAE con fin de partido, los partidos contra el pool y las máscaras del aprendiz;
  - el KL(BC‖π) y la reanudación del pool;
  - los cortes por deriva, frente a §4;
  - la alineación del grabador de simulación;
  - las tasas de un solo equipo contra la BC (cuentan media duración);
  - el export a ONNX contra torch.
- **No verificado localmente** (lo cubren el preflight y la cola en el pod): todo lo que necesita `data/x4_ticks` o la
  GPU (tiempos, memoria, referencia humana, sondas) y el preflight completo, que usa `resource` (sólo Linux).

**De la sesión anterior:** `docs/REPORTE_2026-10-06_noche.md` y los reportes de `reports/x4/` citados en §1–§4.
