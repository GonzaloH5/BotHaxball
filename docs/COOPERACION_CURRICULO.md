# Experimento cooperativo de Sanguchito RS X4

Autorizado el 7 de octubre de 2026. Objetivo: aprender apoyo, recepción y continuidad de posesión que se transfieran a partidos completos. La auditoría anterior no confirmó mejora de fuerza ni de cooperación: [mediciones](AUDITORIA_POD_2026-10-07.md).

## Qué se entrena

Los ejemplos se extraen exclusivamente de pases reales del split **train**. Se rechazan intercepciones, cambios de jugadores, cortes, saques y rebotes de pared; se exige vuelo de al menos 18 ticks y recorrido de 40 px. Esto limita el banco inicial y no representa todos los pases válidos de RS.

| Etapa | Situación | Éxito |
|---|---|---|
| 0 | Dos atacantes reciben un pase humano en vuelo | Recepción y dos segundos de control físico del equipo |
| 1 | Dos atacantes empiezan antes de la patada humana | Pase ejecutado por la política, recepción y conservación |
| 2 | Dos o tres atacantes frente a uno o dos defensores | Pase propio, conservación, avance controlado y salida de presión |
| 3 | Ventaja 3v2 | Gol propio, sin requisito artificial de pase |
| 4 | Transferencia | Predominan los partidos completos; continúan ejercicios 3v2 |

Todos los atacantes activos aprenden: el receptor y los apoyos reciben la misma recompensa terminal que el portador. Los defensores alternan entre BC, modelo inicial congelado y política programada. La recompensa del ejercicio es +1/−1 por equipo al terminar; sustituye la recompensa normal en esa fila, sin pagar cada pase. Pérdida, salida o tiempo agotado terminan los ejercicios. Las etapas 0–2 no pueden aprobarse con un gol que evite la secuencia.

Se reconstruye la cola de teclas con inputs humanos originales y latencias de 8–11 ticks. Desde la etapa 1, la cola comienza suficientemente antes de la patada registrada: una acción humana heredada no debe completar el ejercicio. La etapa 0 sí contiene un vuelo humano y comandos pendientes; su preflight compara STILL con la política inicial para medir cuánto éxito ya entrega ese arranque. Aprobar esa etapa no demuestra aprender a pasar.

En etapa 2 se puede acercar un defensor a 60–70 px del portador. La posición debe quedar dentro del campo y separada de todos los jugadores activos y de la pelota; si no hay punto legal se conserva la posición original. Esta alteración es sintética y queda distinguida de la grabación.

## Mezcla y avance

Se reserva la mitad de las filas para ejercicios, conservando al menos la mitad para 4v4. La reserva persiste al reiniciar: sortear en cada episodio diluiría los ejercicios porque duran menos que los partidos. En transferencia queda 12,5% de ejercicios con esta configuración. Se registran las decisiones y muestras de actores de cada tipo: la igualdad de decisiones de entorno no implica igualdad de muestras de jugadores, porque los planteles son distintos.

Cada etapa requiere al menos 256 episodios y que el límite inferior de Wilson, unilateral al 95%, supere 15% de éxito en una ventana de 512. Es un umbral experimental de factibilidad, elegido para este piloto; no deriva de un resultado publicado ni certifica fuerza 4v4. No se avanza por presupuesto agotado.

El actor conserva su observación pública. El crítico recibe cuatro campos adicionales de tarea, inicializados en cero al cargar un crítico anterior. No describen todo el historial del ejercicio: tiempo, pase pendiente e identidad de objetivo introducen información oculta para el valor. Es una limitación de eficiencia que debe reconsiderarse si el piloto muestra varianza o estancamiento. Reanudar conserva el progreso y exige los mismos checkpoints de defensores; no restaura episodios parciales.

## Comparación previamente fijada

Dos brazos parten de `runs/x4_estilo/rl_control/best.pt`: **control** y **curriculum**. Ambos usan macro de pase listo, seguimiento de teclas a través de la latencia, entropía 0,002, gamma 0,999, GAE 0,98, ancla KL 0,05, 50% de rivales congelados/snapshots y 40% de arranques humanos en sus partidos completos. No usan bonus de pase ni shaping. Sólo cambia la presencia del currículo.

Presupuesto por brazo: 512 entornos × 128 decisiones × 1500 actualizaciones = **98.304.000 decisiones**, incluyendo las primeras 100 del piloto. Ambos corren en paralelo; límite total de cuatro horas, incluida evaluación. La carpeta de salida debe estar vacía y el driver no exporta modelos a la sala.

El driver copia BC e inicial a `references/` antes de arrancar, con sus hashes: cambios concurrentes en los archivos originales no alteran rivales ni evaluación. Los comandos de ambas fases conservan esas copias.

El piloto debe terminar exactamente en actualización 100, sin actualizaciones no finitas, con partidos BC completos, puntaje BC ≥65% y caída menor de 15 puntos frente al inicio. El currículo debe registrar muestras de aprendizaje y al menos 32 ejercicios completos con un éxito. Estos requisitos sólo comprueban que el experimento funciona. Se preserva un diagnóstico separado de **dev** con 4 partidos por lado y 32 estados por sonda; no permite declarar mejora. Durante entrenamiento, la referencia del índice de pase es `sanguchito_train`.

Al finalizar se evalúan los últimos checkpoints frente a BC, defensa programada, control y original: 32 partidos por lado de tres minutos, ambos equipos y D=8–11. Las sondas de **test**, usadas una sola vez, comparten estados entre políticas y rivales. Se informan retención, pérdida bajo presión, progreso tras pase, desmarques y precisión; no se premia ninguna de ellas durante esa evaluación.

Para registrar mejora de fuerza simulada: todos los partidos deben terminar, el puntaje contra control debe alcanzar 55%, su IC bootstrap 90% debe quedar por encima de 50% y el puntaje contra original debe ser ≥50%. La cooperación se analiza por sus métricas y soportes, por separado. Una sola semilla de entrenamiento no mide variación entre corridas. Ningún resultado simulado certifica competitividad contra humanos.

## Fundamento y artefactos

[TiZero](https://arxiv.org/html/2302.07515) estudia asignación de crédito a jugadores sin pelota, currículo y diversidad de rivales. El [trabajo de fútbol con robots](https://arxiv.org/html/2505.13834v1) separa habilidades y estrategia e incluye situaciones 2v1. Usar ejercicios breves, mezcla con partidos y defensas congeladas en Haxball es una **inferencia experimental**; esas publicaciones no validan los umbrales ni garantizan su transferencia a RS.

Implementación: `learn/x4_coop_curriculum.py`, integración en `learn/x4_ppo.py`, atribución con `learn/x4_macro_trace.py`, evaluación en `learn/x4_coop_eval.py`, preflight en `learn/x4_coop_preflight.py` y driver `learn/x4_coop_run.py`.

La ejecución aislada del pod vive en `/workspace/HaxballRL_coop_20261007`, rama `codex/coop-curriculum-20261007`, basada en `b6d1213`, con macro congelada de `a2028ab`. Conserva la corrida anterior y permite trabajo concurrente en el checkout principal. `runs/x4_coop_20261007/manifest.json` registra hashes y comandos; `run_state.json`, `pilot_gate.json`, `pilot_evaluation_dev.json`, `evaluation_test.json` y `verdict.json` dejan estado y evidencia.

## Puesta en marcha comprobada

El pod aprobó **91 pruebas** de integración, currículo, evaluación, macros y compatibilidad. Un humo real ejecutó aprendizaje del actor y del crítico, y una prueba de capacidad ejecutó ambos brazos con 512×128: aproximadamente 14–15 mil decisiones por segundo por brazo, sin actualizaciones no finitas.

El [preflight real](../reports/x4/coop_preflight_pod_20261007.json) minó 73 pases de ocho grabaciones train. En 64 recepciones pareadas, STILL conservó 0 y el original 29 (45,3%). En otros 64 ejercicios de ejecución, STILL tuvo cero pases propios y cero éxitos. Pasaron la geometría de los resets, la finitud y la suma cero de recompensas. La muestra pequeña y fuentes repetidas limitan la interpretación del intervalo.

Se lanzó el A/B con commit remoto `ef77ff9e7936e45aedf8536c1325eda8a03666d9`, driver PID 3259, y banco de **592 pases** de 80 grabaciones train. El directorio aparece también mediante enlace en `/workspace/HaxballRL/runs/x4_coop_20261007`. Las referencias se copiaron y congelaron antes del lanzamiento. Copias locales: [manifest](../reports/x4/coop_manifest_pod_20261007.json), [estado al verificar el arranque](../reports/x4/coop_run_state_pod_20261007.json).

Los primeros avances 0→1 y 1→2 ocurrieron durante las 20 actualizaciones de preparación del crítico: el actor seguía siendo el original. Demuestran habilidades ya presentes para los ejercicios seleccionados, no una mejora nueva. Las etapas 2→3 y 3→4 avanzaron en actualizaciones 32 y 35, con 21,68% de éxito en ventanas de 512 y límite de Wilson 18,84%; son resultados de ejercicios train.

El [piloto de 100 actualizaciones](../reports/x4/coop_pilot_gate_pod_20261007.json) aprobó: 6.553.600 decisiones por brazo, cero actualizaciones no finitas y **84,4% contra BC en ambos**, con 32/32 partidos terminados. El currículo registró 18.781 ejercicios completos, 3.251 éxitos y 4.368.298 muestras de actores de ejercicios.

El [diagnóstico dev](../reports/x4/coop_pilot_dev_pod_20261007.json) terminó correctamente. Con sólo ocho partidos de un minuto por cruce, currículo ganó 3 y empató 5 contra control (68,75%), pero empató 7 y perdió 1 contra original (43,75%). Las sondas usaron 32 estados por situación de 13 grabaciones: frente a BC, en 2v1 tuvo más pérdidas que control (65,6% vs 50%); en 3v2 conservó 6/8 recepciones contra 8/15 del control, pero produjo menos pases y goles. Los soportes son pequeños y los resultados mixtos: no hay mejora colectiva confirmada.

Ambos brazos reanudaron correctamente en actualización **101**, conservando el currículo en etapa 4. Estado verificado el 7 de octubre a las 20:00:48 Uruguay: control 108 y currículo 105, con entrenamiento activo y sin errores. Sigue el presupuesto hasta 1500, incluida la evaluación final test. Este apartado es una fotografía del piloto, no un monitor actualizado.

El control físico continuo de dos segundos es una habilidad de recepción de estos ejercicios; puede penalizar cadenas rápidas con vuelo libre entre compañeros. Su utilidad depende de la transferencia medida fuera del ejercicio. El criterio no se aplica a la recompensa ni a la selección de acciones de los partidos completos.

## Relanzamiento con controlador corregido

El usuario pidió incorporar la corrección completa el 7 de octubre. La primera corrida quedó conservada y marcada `superseded`: últimas actualizaciones registradas, control 327 y currículo 298. No se mezclan sus resultados ni sus gradientes con el nuevo experimento.

La nueva corrida usa el controlador y sus pruebas congelados del commit `01c9e6045cfda4db78190c07a9675cf47557f1ae`: punto de rodeo detrás de la pelota, costo de acercamiento basado en el recorrido completo, restricciones de contexto y contacto para laterales propios, giros amplios y compromiso máximo de tres segundos. Se verifica el hash tanto en el manifest como en los metadatos de los procesos.

Salida nueva: `/workspace/HaxballRL_coop_20261007/runs/x4_coop_corrected_20261007`, visible también en `/workspace/HaxballRL/runs/x4_coop_corrected_20261007`. Ambos brazos arrancan desde las copias congeladas de BC e inicial de la primera corrida, con igual presupuesto y configuración; sólo cambia la mezcla de ejercicios entre brazos. Se repite el piloto de 100 antes de continuar hasta 1500.

Se conserva `--pass-ready`: sólo se ofrecen planes que predicen una patada próxima. Esto limita el uso de planes que requieren rodear o acercarse a la pelota; incorporar el controlador completo no reproduce automáticamente los porcentajes de las pruebas del controlador libre. Los resultados nuevos deberán medirse, especialmente las cesiones de laterales y la retención tras pase.

Validación del relanzamiento: **40 pruebas aprobadas localmente y 40 en el pod**. Ambos brazos ejecutaron además dos actualizaciones con 512×128, aprendizaje real del actor y del crítico, y cero actualizaciones no finitas. El commit remoto es `205b4812dd4b1a2f5520c0d1dba1ef15027d3465`; driver PID 3904, control 3906 y currículo 3907 al arrancar el piloto. El manifest y los metadatos de ambos procesos coinciden con SHA256 `1e15b9d9fc460467dc44ec5fb9354ee5a2d098f8c43e96d216d27e997e8ee4f5` del controlador corregido. El currículo cargó los mismos 592 pases humanos de train. [Registro del lanzamiento](../reports/x4/coop_corrected_launch_pod_20261007.json).
