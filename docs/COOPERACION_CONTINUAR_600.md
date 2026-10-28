# Continuación desde el refinamiento 600 — 2026-10-08

La corrida `x4_coop_refine_200_20261007` terminó por su límite de 600 actualizaciones adicionales. No se cayó. Empezó en el modelo conservado de 200; sus updates 0–600 son nuevos, no los números del primer entrenamiento.

Los pases mejoraron en el selfplay del trainer: precisión 43,24% → 55,53%, retención 42,71% → 50,65%, retención bajo presión 38,98% → 42,65%. La actividad ofensiva empeoró: tiros/min 1,2525 → 0,36 y goles/min 0,338 → 0,121. Estas comparaciones de selfplay usan políticas/rivales que cambian y sirven como diagnóstico.

La evaluación final ya realizada contra el checkpoint fijo de 200 dio **3 victorias, 23 empates y 38 derrotas** (64 partidos de 3 minutos, ambos lados). Fracción de puntos 0,2266, IC90 [0,1641; 0,2891]. No hay mejora de fuerza confirmada; el aumento de precisión de pase no compensa la pérdida ofensiva. BC y scripted no representan rivales humanos.

El currículo pasó de etapa 1 a 2 en el update 113. En los últimos 100 updates, los ejercicios principales de salida bajo presión tuvieron ~36,47% de éxitos. La etapa 3 requería dos ventanas disjuntas de 512 episodios con límite inferior Wilson ≥40% y al menos 100 actualizaciones del actor en la etapa. El repaso de etapa 1 no aporta evidencia de dominio de etapa 2.

Se encontró un conflicto concreto: los ejercicios de etapas 1/2 rechazaban cualquier gol, incluso después de un pase ejecutado y recibido por el agente. Esos goles terminaban con recompensa −1 mientras los partidos completos premian los goles. Es una explicación plausible del retroceso ofensivo, no una demostración causal aislada.

## Cambio y presupuesto

La continuación usa el actor, crítico, normalización, Adam y liga del checkpoint 600. Conserva el checkpoint 600 completo como referencia de sólo lectura. El modelo original de 200 también sigue conservado y tiene copia local en `runs/x4_coop_preserved/curriculum_update_00200.pt`.

- 50% de filas: partidos 4v4 completos.
- 25%: salida bajo presión, etapa principal 2.
- 12,5%: repaso de pase/recepción, etapa 1.
- 12,5%: definición 3v2 tras una recepción humana en el último tercio; un defensor presiona al receptor y otro se elige por su proximidad al arco rival, conservando posiciones humanas.

Un gol propio después de un pase válido cuenta como éxito en etapas 1/2. En definición se exige una patada propia posterior a la cola inicial de órdenes humanas; un gol heredado no da éxito. Un gol propio sin el requisito válido termina con recompensa neutra. Las recompensas de los partidos completos no cambian.

La práctica de definición no promueve la etapa principal. Se reinician sus ventanas, racha y reloj mínimo porque cambió el criterio. Los contadores históricos se mantienen, separados de la evidencia nueva de dominio. Si la etapa principal avanza, pasa a definición con 25% de sus ejercicios dedicado al repaso anterior.

Presupuesto: **200 updates adicionales**, 600→800, con evaluación DEV en 700 y 800 y límite de 2 horas. El control observa pases, precisión, retención, tiros, goles y resultados contra la referencia fija 600. Dos evaluaciones consecutivas con deterioro material detienen la corrida. Las comparaciones fijas contra 200 y el original permiten evaluar recuperación. Esto es exploración con DEV: no se reutiliza el test 600 como una evaluación final nueva.

## Ubicación en el pod

Código aislado: `/workspace/HaxballRL_coop_refine_20261007`.

Corrida nueva: `/workspace/HaxballRL_coop_refine_20261007/runs/x4_coop_finish_600_20261008`.

Log accesible mediante el enlace de la carpeta principal:

```bash
tail -f /workspace/HaxballRL/runs/x4_coop_finish_600_20261008/curriculum.out
```

`run_state.json` registra cada entrenamiento/evaluación y sus PID. `manifest.json` conserva hashes, comandos y presupuesto. `curriculum/log.jsonl` mide entrenamiento; `evaluation_dev_00700.json` y `evaluation_dev_00800.json` comparan partidos completos y sondas con referencias fijas. `stage3_successes` registra los goles válidos de los ejercicios de definición; `cooperative_goals` registra goles cooperativos de etapas 1/2.
