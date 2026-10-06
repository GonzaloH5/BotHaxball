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

**Gate** (`tools.x4_pass_chain.gate`, pre-registrado). Sobre las métricas con datos suficientes (≥ 20 eventos esperados
al ritmo humano, o ≥ 20 observaciones), aprueba si se cumplen las cinco condiciones:
1. **global:** índice de la cadena ≥ 1 (media geométrica de agente/humano en las métricas de eficacia; 1 = promedio humano);
2. **cada eslabón:** el índice de cada etapa de la tabla ≥ 0,9 (ninguna parte del proceso queda claramente por debajo);
3. **ninguna métrica de eficacia significativamente peor** que el promedio humano (todo su IC90 por debajo);
4. **estilo:** cada métrica de estilo dentro del p10–p90 humano entre grabaciones;
5. **datos:** el 80% de las métricas juzgables en las evaluaciones del entrenamiento; el 100% en la certificación.

Por qué no "≥ promedio en cada una de las 30 métricas": un agente exactamente promedio quedaría por debajo en la mitad por
puro ruido de muestreo; esa regla exigiría ser mejor que el promedio en todo. La regla de arriba exige nivel promedio en el
conjunto y en cada eslabón, y no tolera ningún déficit claro.

## 2. Dónde estamos: la imitación (BC) de partida

Self-play de la BC con la latencia de sala (`reports/x4/pass_chain_bc_final.json`): **índice 0,45**.
- Al nivel humano: líneas disponibles, apoyo, línea progresiva, anticipación del receptor, presión defensiva.
- Muy por debajo: pases/min 5,7 contra 12,8; precisión 0,49 contra 0,69; retención tras recibir 0,40 contra 0,58;
  tiempo con la pelota 0,10 contra 0,40 s; paredes 0,12 contra 0,77; valor por intento ~0 contra 0,022.

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
| Brazo de pases (actualización 1000) | si el índice sigue < 0,85, +0,05 por pase de las posesiones que terminan en gol (TiZero); no premia circulación inútil |
| Penalización por espera en saques | evita el equilibrio "nadie saca" (sólo esperas; el mal lateral ya cuesta la pelota) |

## 4. Plan de corrida (cola pre-registrada, `learn/x4_queue.py`)

Una sola orden en el pod (`docs/POD_RUNBOOK.md` §4). Cada paso tiene su condición y su acción; nada espera a una persona.

| Paso | Qué hace | Si sale bien | Si sale mal |
|---|---|---|---|
| 1. Preflight | GPU, hilos, disco, datos, referencias humanas, EPV, BC; mini-corrida con evaluación 0 y reanudación; una actualización y una evaluación de tamaño real con su tiempo y memoria | sigue | la cola se detiene sin gastar GPU (`preflight.json` dice qué falló) |
| 2. RL principal (`rl_principal`) | 3000 actualizaciones × 1024 partidos × 64 decisiones (~196M decisiones) | certificación | corte por deriva → paso 3 |
| 3. Recuperación (sólo si hubo corte) | desde `best_pase.pt` o `best.pt` del principal (o la BC), λ 0,4 y lr 1e-4 | certificación | si también se corta: veredicto "revisar" y fin (no se gasta más) |
| 4. Extensión (una sola vez) | si ningún checkpoint aprobó el gate de pases y el índice de la cadena sube en las últimas 8 evaluaciones (pendiente > 0), sigue hasta 6000 | certificación | — |
| 5. Certificación | `learn/x4_certify.py` sobre el último `pase_aprobado_*`, `best_pase.pt` y `best.pt`, en ese orden: 64 partidos de self-play de 3 min, 64 por lado contra la BC, sondas | el primero que aprueba se exporta a `deploy/rs4z/x4_rl.onnx`, veredicto "competitivo_en_pases" | se exporta el de mayor índice, veredicto "no_competitivo" (sirve para probar, no para competir) |

**Dentro de la corrida** (cada 50 actualizaciones hay una evaluación; la evaluación 0 es la BC con el mismo protocolo):
- **Corte por deriva** (dos evaluaciones seguidas con alguna de estas condiciones; deja `stopped.json` con el motivo):
  - pierde contra la BC (`vs_bc.score` < 0,4);
  - se aleja del parecido humano (W1 media > máx(1; la inicial + 0,5));
  - se aleja de la pelota (distancia del más cercano > máx(200 px; 1,6 × la inicial));
  - deja de sacar (saques iniciales sin ejecutar por partido > máx(1; 3 × los iniciales));
  - la cadena de pase cae (índice < 0,8 × el inicial y todo su IC90 por debajo del inicial);
  - la defensa deja de presionar (presión al portador p50 > 1,3 × el p90 humano).
- **Brazo de pases** en la actualización 1000 (~65M decisiones): si el índice de la cadena es < 0,85, se activa +0,05 por pase
  de las posesiones que terminan en gol, desde ahí hasta el final (se registra en `eval.jsonl` → `brazo_pases_activado`).
- **Checkpoints:** `best.pt` (fuerza con parecido humano, confirmada con otra evaluación; `best.json` dice si le gana a la BC
  con margen), `best_pase.pt` (mejor índice de la cadena sin perder contra la BC), `pase_aprobado_*.pt` (gate aprobado en dos
  evaluaciones seguidas), `last.pt` cada 10 actualizaciones y después de cada evaluación, `evals/` con la política y los
  partidos de cada evaluación.

**Qué debería verse en cada tramo** (señales tempranas; si no aparecen, la corrida igual sigue hasta un corte o el final,
pero quedan registradas para decidir la próxima):

| Tramo | Debería pasar | Señal en los logs | Alarma |
|---|---|---|---|
| 0–20 (crítico solo) | la política es la BC; el crítico aprende el valor | `explained_var` sube desde ~0; `kl_bc` = 0 | `explained_var` negativa o `nonfinite_skipped` > 0 |
| 20–250 (~16M) | menos pérdidas tontas: la BC suelta la pelota; el shaping de valor castiga cada pérdida | `retencion_tras_recibir` y `precision_pase` suben; `tiempo_con_pelota_p50` sube hacia 0,4 s; `kl_bc` 0,01–0,05 | `presion_al_portador_p50` sube (la defensa se afloja); `dist_ball_1` sube; `kickoff_frac` sube |
| 250–1000 (~65M) | más pases que sirven: progresivos, que rompen líneas, salidas de presión; pool con rivales distintos | índice de la cadena hacia 0,7–0,9; `epv_por_intento` positivo; `vs_bc.score` > 0,5 | el índice no sube desde la evaluación 0 → brazo de pases en 1000 |
| 1000–3000 (~196M) | λ baja a 0,05: más libertad; combinaciones (paredes, al espacio) y aprovechar la ventaja | índice ≥ 1 y etapas ≥ 0,9; `pase_aprobado_*` | `human_w1_mean` sube (estilo artificial); `humanos_dev.nll` sube mucho (se olvida lo humano) |

[inferencia] Las cifras de cada tramo son expectativas, no garantías: salen del diagnóstico de la BC y del A/B de CPU (§7), a
una escala 30 veces menor.

## 5. Problemas encontrados y corregidos

(se completa con la revisión)

## 6. Riesgos que siguen

(se completa con la revisión)

## 7. Evidencia

(se completa con la revisión)
