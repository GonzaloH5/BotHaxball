# Auditoría del pod: avance y cooperación en Real Soccer

Fecha: 7 de octubre de 2026 (Uruguay).

**Resultado:** el último A/B no demuestra una mejora de fuerza ni de juego colectivo. Ambos brazos terminaron en 1500 actualizaciones y 98.304.000 decisiones de partido por brazo. Al terminar la revisión no había entrenamiento activo; la A4000 estaba al 0%.

## Fuerza medida

Puntaje = (victorias + 0,5 × empates) / partidos. Cada cruce incluye 256 partidos de tres minutos, ambos lados y latencia de 8–11 ticks.

| Cruce | G–E–P | Puntaje | IC 95% aproximado |
|---|---|---:|---:|
| Pase vs control | 84–108–64 | 53,91% | 49,27–58,54% |
| Pase vs punto de partida | 75–91–90 | 47,07% | 42,17–51,97% |
| Control vs punto de partida | 74–109–73 | 50,20% | 45,55–54,84% |
| Pase vs imitación (BC) | 186–66–4 | 85,55% | 82,57–88,53% |
| Control vs imitación (BC) | 183–70–3 | 85,16% | 82,20–88,11% |

El pase no alcanzó el criterio predefinido de 55% contra el control. Los intervalos se calculan sobre la varianza de los resultados 0/0,5/1 y suponen partidos independientes; no incluyen variación entre semillas de entrenamiento. El 53,91% no demuestra una ventaja, y el 47,07% tampoco demuestra por sí solo una regresión.

## Juego colectivo

Evaluación final de self-play, comparada con la referencia humana del evaluador. Son simulaciones y grabaciones, no partidos del bot contra humanos. Estas métricas provienen de cadena_pase: juego abierto y tasas agregadas. No deben mezclarse con tasas por reloj total ni con la referencia de otra partición del dataset.

| Métrica | Control final | Pase final | Humanos de referencia |
|---|---:|---:|---:|
| Pases/minuto | 5,57 | 6,10 | 12,76 |
| Precisión de pase | 42,62% | 44,99% | 68,62% |
| Pases/posesión | 0,181 | 0,181 | 0,585 |
| Paredes/minuto | 0,150 | 0,200 | 0,768 |
| Desmarques útiles | 2,38% | 1,41% | 10,06% |
| Retención tras recibir | 42,31% | 38,69% | 58,29% |
| Retención bajo presión | 35,06% | 24,68% | 47,58% |
| Pasa cuando hay línea progresiva | 39,30% | 31,72% | 60,38% |
| Pérdida sin pase bajo presión | 65,01% | 68,47% | 51,84% |

La rama pase mejoró ligeramente algunas tasas frente al control en esta evaluación, pero no la continuidad de posesión. Ningún brazo aprobó el criterio colectivo. El índice final fue 0,691 para control y 0,636 para pase; la media de las últimas tres evaluaciones fue 0,678 y 0,671 respectivamente. No es evidencia de una caída inequívoca de un brazo respecto al otro.

## Abandono de la macro

Los primeros 100 rollouts de pase registraron 29.466 inicios de macro, unos 295 por actualización. Los últimos 100 registraron 1.053, unos 11 por actualización: caída del 96,4%. Las patadas registradas mientras estaba activa la macro fueron 40,3% de los inicios al principio y 23,1% al final.

Esta proporción **no es precisión de pase ni una tasa exacta de fracaso**: con latencia, una patada emitida puede ejecutarse después de que la macro se cancele. Para atribuir el resultado haría falta seguir sus teclas pendientes hasta su aplicación.

Una sonda adicional de 16 partidos de un minuto por checkpoint confirmó el desuso. El modelo final tenía una probabilidad media de elegir macro de 0,082% entre decisiones con al menos una ofrecida. Había opciones ofrecidas en el 4,69% de las decisiones. En el checkpoint 100 esa probabilidad era 0,155%. La cabeza se inicializa con 4,74% antes de aprender. La sonda final registró siete inicios y ninguna patada mientras seguía activa. Es una sonda de uso con una sola semilla, no una nueva prueba de fuerza.

La corrida usó pase completo, sin --pass-ready. Esa alternativa está en el código actual, pero este A/B no la evalúa. El controlador acomoda al jugador y apunta; el compromiso termina si desaparece la opción, si patea o al segundo de duración. No coordina una intención de recepción ni un plan compartido con los compañeros.

## Por qué puede estancarse

**Verificado:** la recompensa se comparte entre los jugadores del equipo. Este A/B premia goles y penaliza saques cedidos por las causas configuradas; no están activos el estilo, el bonus de pase ni el shaping. El 80% del entrenamiento es self-play y el 20% usa la BC y snapshots propios. El mejor checkpoint se selecciona principalmente por fuerza contra la BC, con un filtro general de parecido humano; se guarda otro checkpoint para el índice de pase.

**Lectura de la evidencia:** ganar ampliamente a una BC débil no identifica todos los problemas que encontraría un equipo humano. Los rivales siguen concentrados en la misma familia de políticas. Los datos tampoco prueban que conducir sea gratis: la sonda anterior de [PASE_MACRO](PASE_MACRO.md) mostró que los defensores del agente castigan la conducción. La brecha incluye decisiones de patada, uso del apoyo, recepción y conservación.

**Hipótesis técnica:** en partido completo, descubrir al mismo tiempo el pase del portador, el desmarque del compañero y la continuidad tras recibir requiere cambios coordinados. Una macro del portador simplifica sólo una parte. El uso inicial escaso, sin bonus de entropía (--ent 0), puede facilitar que el aprendizaje la abandone antes de encontrar esas combinaciones. Esto es compatible con los datos, pero no demuestra una causa única.

**Hipótesis de asignación de crédito:** se decide cada 0,05 s, con gamma 0,997, GAE 0,95 y rollout de 64 decisiones (3,2 s). El descuento del retorno tiene una semivida aproximada de 11,5 s; el peso directo de un error futuro en GAE tiene una semivida de 0,64 s. El crítico puede propagar valor más allá: esto no significa que el agente sólo vea 0,64 s. Justifica estudiar si una secuencia colectiva de varios segundos recibe suficiente señal.

## Próximo experimento recomendado

No extender esta configuración esperando que más tiempo produzca cooperación.

1. Validar pase y recepción como secuencia completa desde estados humanos: trayectoria correcta, compañero que se ofrece, recepción y conservación. Seguir la acción a través de la latencia para distinguir cancelación, patada tardía, intercepción y recepción.
2. Entrenar situaciones cooperativas progresivas: conservar con apoyo, salir de presión, resolver un 2v1, devolución y cobertura tras pérdida. Enseñar también al jugador sin pelota. Comparar con un control de igual presupuesto; no convertir un número arbitrario de pases en objetivo.
3. Evaluar frente a defensas de comportamiento distinto y frente al punto de partida, además de la BC. Confirmar en sala contra humanos antes de declarar competitividad.

Son propuestas para HaxballRL. [TiZero](https://arxiv.org/html/2302.07515) documenta dificultades de crédito de los jugadores sin pelota y usa un currículo progresivo antes de enfrentar un pool diverso. Un [trabajo de fútbol con robots](https://arxiv.org/html/2505.13834v1) separa habilidades de control y decisión estratégica y muestra cooperación en 2v1; no prueba que esa arquitectura resuelva RS X4.

## Documentación y alcance

- [Datos de esta auditoría](../reports/x4/pod_audit_2026-10-07.json).
- [Pase como macro y sondas anteriores](PASE_MACRO.md).
- [Ataque colectivo](conocimiento/real_soccer_x4/04_ataque_colectivo.md).
- [Aplicación de conceptos al proyecto](conocimiento/real_soccer_x4/12_aplicacion_a_haxballrl.md).

Se inspeccionaron logs, código y checkpoints, y se ejecutó la sonda de evaluación. No se modificaron modelos, recompensas ni corridas. Las mediciones automáticas necesitan validación visual para interpretar intenciones como pase, rebote o despeje.

El run_meta registra el commit 4a2037f; el checkout remoto inspeccionado era b6d1213 y el local a2028ab. Se consultó el código remoto para no asumir que coincidía con el local. La sonda usa el evaluador actual del pod; las métricas históricas pertenecen al evaluador que produjo cada archivo.
