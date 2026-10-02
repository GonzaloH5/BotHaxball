# Auditoría del objetivo RS4 4v4 — 2026-10-02

La evidencia confirma aprendizaje de saques y capacidad de vencer al scripted, pero no acredita buen juego colectivo transferible. El sistema de evaluación tiene un gate imposible en este run, mezcla situaciones que debería separar y permite que un resultado global bueno oculte una regresión importante jugando con tres compañeros externos. Consumir más presupuesto sin corregir estos instrumentos no garantiza resolverlo.

## Alcance y evidencia

- Checkpoint inspeccionado: `runs/rs4_v3_public/control/latest.pt`, SHA256 `85997b64d86ad30f90a7ae36f368e85d00888fcbabc40b8333527d1671111cf6`.
- 1.942.311.810 muestras PPO propias; 4.540.765.440 pasos heredados. Los 2B son decisiones de jugadores que participaron en PPO, no 2B situaciones independientes ni 2B demostraciones humanas. Jugadores y decisiones consecutivas comparten partido y están correlacionados.
- Rama feedforward `set`, mean/max pooling, encoder de entidades de 32 unidades, cuerpo de 256×2, 18 acciones, sin GRU. Señales públicas v1 activas, normalizadores congelados. La proyección pública tiene norma 3,057: no sigue en su inicialización nula.
- 19 evaluaciones guardadas, de 100M a 1.900M. A terminó anticipadamente a 400M; B terminó por presupuesto a 1.400M; lleva 542,3M en C, aproximadamente 45% de su duración nominal. Deudas actuales: defensa y ataque.
- Auditoría estática de currículo, entrenador, PPO, liga, recompensas, escenarios, evaluación, selección de campeón, migración de señales, BC y despliegue. Probes locales sin PPO y suite focalizada: **73 tests aprobados**.
- No se dispone del ledger/config/log actuales del Pod ni del resultado completo del piloto de memoria. La copia local permite conclusiones sobre ese checkpoint; no prueba el estado posterior del entrenamiento remoto. No se alteraron pesos, configuración de entrenamiento ni fases.

Reproducción: `.venv/Scripts/python.exe -m tools.audit_rs4_objective`. Resultados y filas originales resumidas en `reports/rs4_objective_audit_20261002.json`.

## Hallazgos prioritarios

### P1 — Defensa es matemáticamente imposible de aprobar en este run

`train/rs4_program.py:344` exige `base_conceded > 0` y una reducción del 20%. La referencia tiene `defense_conceded = 0` en las 19 evaluaciones. Incluso cuando el candidato también concede cero (1.000M y 1.500M), falla. Esto invalida interpretar la deuda como diagnóstico deportivo fiable.

El efecto supera el logging: `settings()` agrega ejercicios por deuda y la consolidación reserva recuperación. En C, con defensa y ataque pendientes, el reparto efectivo es exit 26,67%, attack 43,33%, transition 13,33%, defense 16,67% de los ejercicios. Con una cuota de ejercicios del 30%, defensa ocupa aproximadamente 5% de entornos por esta deuda, aunque el gate no puede retirarla. Las proporciones de muestras útiles pueden diferir por controladores congelados.

Corrección propuesta: distinguir saturación/no regresión de mejora demostrada; medir despeje/recuperación segura, ocasiones concedidas y goles en escenarios defensivos más exigentes. Si la referencia da cero, no inventar una mejora relativa ni bloquear para siempre. Versionar el criterio, reevaluar referencia y candidato, y reconciliar deudas conservando pasos e historial.

### P1 — La evaluación global oculta el caso de un bot con tres compañeros

La evaluación larga a 1.400M (128 partidos por celda, 2 minutos) contiene:

| Celda | Puntos candidato | Observación |
|---|---:|---|
| 4 aprendices vs R3 estilo 0 | 96,81% | 768 partidos entre colores y semillas |
| 4 aprendices vs R3 estilo 1 | 97,33% | 768 partidos |
| 4 aprendices vs R3 estilo 2 | 96,16% | 768 partidos |
| 4 aprendices vs parent | 57,23% | 768 partidos |
| 4 aprendices vs anchor 0 / 1 | 58,46% / 54,10% | Ventaja mucho menor contra redes |
| 1 aprendiz + 3 BC, semilla 51, estilo 0 | **24,80%** | 16 victorias, 95 empates, 145 derrotas; goles 38–218 |
| 2 aprendices + 2 BC, semilla 73, estilo 1 | 63,67% | Otra composición y otro rival |
| 3 aprendices + 1 BC, semilla 91, estilo 2 | 91,21% | Otra composición y otro rival |

En la primera celda mixta la referencia obtenía **38,48%**: el candidato retrocedió **13,67 puntos porcentuales**, aunque el promedio mixto mejoró de 50,72% a 59,90% y el global de 63,45% a 74,28%. Es una regresión concreta que el objetivo agregado tolera.

`eval/rs4_v3.py:191` liga cantidad de aprendices, semilla y estilo rival. No permite atribuir causalmente toda la diferencia a la cantidad de compañeros. Además, con seis rivales, las celdas mixtas pesan sólo 1/7 del promedio global. El gate de compañeros usa éxito de ejercicios de ataque y seguridad del promedio global, sin exigir no regresión en partidos de un solo aprendiz.

Corrección propuesta: cruzar 1/2/3/4 aprendices × estilos × colores × varias semillas/posiciones, incluir proxies diversos y exigir un mínimo por celda crítica. El caso de una sola instancia debe tener evaluación y aceptación propias si ése es el uso real.

### P1 — «Ataque colectivo» se aprueba sin demostrar colaboración ni gol

`env/rs4_v3.py:439` considera peligro pelota más allá de 60% del semiancho, dentro de la franja central y dos compañeros a menos de 40% del semiancho. Basta último toque propio; no exige pase recibido, desmarque aprovechado, tiro, conservación posterior ni gol. `exit` exige cruzar -15% del semiancho con último toque propio; ni siquiera exige allí la condición de dos apoyos. El éxito queda acumulado hasta terminar, salvo gol contrario.

A 1.900M, el escenario attack alcanza **96,88%**, mientras exit alcanza **76,04%** y la mezcla que publica `attack_success` queda en **86,46%**. La referencia de esa mezcla es 79,69%; el gate pide 89,69%. El escenario de ataque está casi saturado mientras siguen faltando salida y transferencia a compañeros. `integrated_success` es el promedio de ejercicios, no juego integrado de partidos.

Corrección propuesta: separar salida, creación y finalización; medir recepción bajo presión, mantenimiento después del pase, avance con control, tiros y goles. Usar métricas deportivas como evaluación antes de convertirlas en premios: sumar bonus indiscriminadamente crearía otros atajos.

### P2 — Las transiciones ofensivas y defensivas pierden su identidad

`train/rs4_program.py:287` convierte ambos nombres a `transition`. `env/rs4_v3.py:369` elige después ataque/defensa al azar, 50/50. Por tanto, el bloque anunciado como 30% de transiciones defensivas en B sólo dedica en expectativa la mitad a ese sentido; lo mismo sucede con las ofensivas de C. La métrica `defense_conceded` promedia defensa y todas las transiciones, incluidas las ofensivas.

Corrección propuesta: preservar el sentido en la configuración/etiqueta del escenario, y agregar sólo transiciones defensivas a defensa. Reevaluar referencias con el nuevo contrato.

### P2 — Los controles de 30 segundos y los promedios suavizan el estancamiento

Los controles cada 100M usan 16 partidos por celda de **30 segundos**, frente a 128 de **2 minutos** en cierres. El salto de 61,24% a 74,28% entre 1.300M y 1.400M no es una mejora comparable: cambió la duración y el tamaño de muestra. Tampoco la vuelta a 59,45% a 1.500M demuestra por sí sola regresión.

A 1.900M, contra las tres referencias neuronales hay **281 empates en 288 partidos** (93+96+92). Casi no discrimina diferencias de nivel a esa duración. Entre 1.000M y 1.900M los controles globales oscilan cerca de 58–62%; ataque no supera el gate en ninguna evaluación de C. Hay progreso inicial, no evidencia de un salto sostenido reciente.

La evaluación funcional permanece en 16 partidas por escenario/color/semilla incluso durante evaluación “full”: el runner no pasa `--functional-games`. Las semillas 51/73/91 se reutilizan para selección y aceptación; falta una batería reservada independiente.

Corrección propuesta: duración fija para comparar tendencia, errores/incertidumbre por celda, holdout de semillas/escenarios, y aumentar también funcionales en cierres. No leer puntos como porcentaje de victorias.

### P2 — Campeón y aceptación no protegen las habilidades críticas

`tools/run_rs4_v3.py:246` considera campeón sólo en evaluaciones full y ordena por `(puntos_globales, éxito_funcional)` de forma lexicográfica. Una mejora mínima de puntos puede ganar aunque empeore una habilidad relevante. Los checkpoints entre cierres no compiten por campeón bajo ese camino; la historia rotatoria tampoco conserva necesariamente todos.

La comparación con campeones anteriores no verifica en esa selección igualdad del fingerprint de evaluador. El historial de este checkpoint contiene tres fingerprints. El evaluador sí separa cachés de referencia por fuente; esa protección no alcanza por sí sola la comparación con el score de un campeón previo.

Corrección propuesta: candidatos elegibles por no regresión en celdas críticas; luego selección sobre una suite común. Revalidar campeón tras cambios de contrato y guardar candidatos con evidencia reproducible. La aceptación final tampoco debe reducirse a promedios mejores que parent.

## Recompensas, aprendizaje y liga

- El contador usa máscaras de aprendices, excluye controladores congelados/relleno y acota presupuesto. No encontré evidencia de que los 1.942B sean un contador inflado por esos conceptos. No significa independencia estadística.
- PPO usa tres épocas como máximo, KL al final para control, normalización congelada y bootstrap de truncaciones. Las 25 KL guardadas están entre 0,00109 y 0,00303, media 0,00188; LR 1e-4. No indican explosión de actualización ni justifican aumentar LR a ciegas. Falta el log para diagnosticar gradientes, entropía y pérdida de valor a lo largo de todo el run.
- La guía geométrica y los goles son compartidos; el crédito de pase a participantes es pequeño (0,0005, dentro de un tope por posesión de 0,01). Esto no asigna explícitamente el crédito al desmarque o a una secuencia táctica. Es una limitación plausible, no una causa aislada probada.
- Con gamma 0,998 a 20 decisiones/s, la semivida del descuento es ~17,3 s. Con lambda 0,97, la semivida de los residuos GAE es ~1,07 s; el rollout dura 6,4 s. El valor transmite información más lejana, de modo que esto no es un límite absoluto de memoria o crédito, pero merece una ablación si falla el juego secuencial.
- En C sólo 20% de asignaciones incorpora compañeros congelados; dentro de esas asignaciones el número de aprendices se sortea entre 1–3. El caso de un único aprendiz ronda 6,7% de asignaciones, antes de dividir por tipo de compañero. Los porcentajes no equivalen exactamente a muestras ni a minutos. Si se despliega una sola instancia, el currículo dedica relativamente poco a ese uso.
- La liga contiene snapshots y anclas del propio linaje; a 1.942B la tasa acumulada de puntos contra varias instantáneas está por debajo de 50%. Es evidencia descriptiva, no ranking actual: acumula partidas de distintos momentos/composiciones. Dominar R3 no equivale a dominar jugadores humanos.
- La referencia BC proviene de ~2,76M decisiones humanas RS4, no de los 2B PPO. El informe previo sitúa su acierto global en 48,91% y al cambiar acción en 27,11%. Es una referencia débil; replicarla o mezclarla no acredita coordinación humana. El loader enmascara su bloque privado durante entrenamiento; no encontré aquí el supuesto bug de pasar señales públicas crudas como estados privados al teacher KL.
- La decisión de descartar GRU/atención se tomó en un piloto de 200M de saques. Eso no demuestra que feedforward sea suficiente para ataque colectivo más tarde. Sin los dos informes del piloto no se puede auditar la selección efectiva ni recomendar cambiar arquitectura por intuición.

## Simulador y despliegue

Hay paridad física local de una muestra previa usando el mapa exacto grabado, pero el propio informe dice `catalog_fidelity_verified: false`. Detectó kickStrength 5,75 en un replay frente a 5,85 del catálogo. No prueba que esa pequeña diferencia explique el bajo nivel. El árbitro simplificado reinicia tras timeout de saque; las salas pueden comportarse distinto. Hace falta comparar la sala concreta y sus saques, no extrapolar la prueba de movimiento libre.

Entrenamiento y evaluación muestrean acciones; el híbrido y el bot por defecto usan argmax. Es una diferencia real de contrato, probada abajo con el mismo checkpoint. No corresponde culpar automáticamente a greedy ni cambiar temperatura sin evidencia. El informe del híbrido deja pendiente la aceptación en una sala real: fidelidad de observaciones, retrasos, jugadores externos y reglas siguen siendo un frente separado.

## Probes nuevos sobre 1.942B

Mismo checkpoint, simulador local actual, semilla 51, ambos colores, 16 partidos por celda y 2 minutos. No se ejecutó entrenamiento:

| Configuración | V–E–D | Puntos | Goles |
|---|---:|---:|---:|
| 4 aprendices, acciones muestreadas, tres estilos R3 | 83–13–0 | 93,23% | 169–1 |
| 4 aprendices, argmax, tres estilos R3 | 96–0–0 | 100% | 256–0 |
| 1 aprendiz + 3 BC, muestreo, R3 estilo 0 | 3–20–9 | 40,63% | 9–18 |
| 1 aprendiz argmax + 3 BC muestreados, mismo R3 | 6–24–2 | 56,25% | 10–5 |

Los partidos greedy con inicio y rivales deterministas repiten trayectorias: 96 victorias no son 96 pruebas independientes. Las cuatro celdas mixtas usan el slot 0 fijo, mientras la evaluación histórica rota slots. Por ello estos números no establecen una tendencia frente a 1.400M ni una comparación con humanos. Sí muestran que el caso de cuatro copias puede dar una impresión muy superior a una instancia entre compañeros externos, y que greedy no produjo el supuesto deterioro en este probe.

También se probó un equipo inmóvil: concedió en 50% de ejercicios defense y 53,13% de transition (32 ensayos por color). La prueba defensiva sí distingue un agente inútil; el fallo confirmado es el gate con referencia cero, no que cualquier agente apruebe defensa. Estos contraejemplos evitan atribuirle al sistema un defecto que no se reprodujo.

## Orden propuesto de corrección

1. Versionar gates y evaluación: defensa con baseline cero, transiciones separadas, métricas ofensivas no saturadas y celdas de una sola instancia. Conservar historial original.
2. Reevaluar parent, campeón y latest con la misma suite y modo de acción del despliegue. Añadir holdout y separar resultados de 4 bots y de un bot con compañeros.
3. Reconciliar deudas desde esas evaluaciones, sin reiniciar pesos, Adam ni presupuesto. La deuda actual de defensa no debe decidir consumo de recuperación indefinidamente.
4. Sólo entonces ajustar currículo: mayor exposición relevante a una instancia y compañeros diversos, dificultad ofensiva/defensiva gradual y rivales que ya no estén saturados.
5. Hacer una continuación acotada comparada con referencia congelada antes de gastar el resto. Cambios de arquitectura, horizonte o recompensas necesitan ablation y una mejora observable por celda; más pasos por sí solos no son criterio de aceptación.

No se aplicaron estas modificaciones al entrenamiento durante la auditoría.
