# Correcciones del objetivo RS4 — contrato RS4-v4-impact-1

Aplicadas al código sobre el programa RS4 v3 existente. No hay migración de arquitectura ni reinicio de pesos, Adam, fases o presupuesto. El contrato deportivo cambia; las notas anteriores siguen siendo evidencia histórica y no se comparan directamente con las nuevas.

## Cambios

- Defensa permite mantener cero goles cuando la referencia da cero, pero exige recuperación segura >=60% y no aumentar la exposición a peligro más de 2pp. Cuando la referencia concede, se pide una reducción de 20%, con un suelo de resolución del 2%. No basta quedarse quieto.
- Salida requiere pase confirmado y un segundo de control continuo con apoyo después de superar la zona defensiva, o gol. Ataque requiere pase progresivo, medio segundo de creación central con control y una patada dirigida al arco posterior a esa creación, o gol. Se publican recepción, control, tiros, goles y peligro. El tiro es un proxy de trayectoria, no una certificación de ocasión de gol. Estas métricas no añaden premios PPO ni features privadas.
- `defensive_transition` y `offensive_transition` conservan el sentido en muestreo, colocación y evaluación. Defensa sólo agrega el sentido defensivo. `transition` se conserva para callers antiguos, sin usarlo en la nueva suite deportiva.
- La matriz cruza cada semilla, color, cantidad de aprendices 1/2/3 y cada estilo R3, con compañeros scripted de dos estilos y BC cuando existe. Cuatro aprendices conservan la matriz de R3 y referencias históricas. Slots rotan entre semillas y colores. Los inicios tienen un pequeño jitter reproducible para evitar copias idénticas con argmax.
- Los cuatro tamaños de equipo aprendiz pesan igual en `balanced_points`. Cada celda exige no perder más de 5pp respecto de la referencia, con al menos 32 partidos agregados por celda; la evaluación habitual 3×16 llega a 48. Se publican conteos e intervalos descriptivos. Una mejora global no compensa una celda crítica que retrocede.
- Selección de campeón requiere no regresión funcional y por celda, primero contra parent y después contra el campeón. Ambos se miden con la suite exacta del candidato; no se usan sus scores antiguos. Se prioriza la peor celda de una instancia, luego puntos equilibrados y éxito funcional. Los controles de cada 100M también pueden promover candidatos.
- Controles y cierres duran dos minutos por defecto. Cierres aumentan también funcionales: 64 por escenario/color/semilla frente a 16 en controles. Modo greedy por defecto, configurable a sampled para comparar con el modo usado en sala. Referencia y proxies neuronales usan ese mismo modo.
- Semillas reservadas 137/211/307 sólo se usan al cerrar el programa sobre el campeón seleccionado. Nunca actualizan deudas, rachas o selección. La aceptación requiere habilidades, no regresión por celda y mejora de partidos en al menos dos semillas reservadas.
- El runner reevaluará parent, latest y campeón antes de PPO cuando detecte el contrato antiguo, otro scorer, modo o duración. Guarda `objective_before_v4.pt` una sola vez; añade una reconciliación al historial, cambia las deudas según evidencia nueva y pone la racha a cero. Reevaluar no avanza de fase ni consume muestras PPO. Los antiguos checkpoints v3 cargan con campos nuevos opcionales.
- Después de reconciliar, los compañeros congelados pasan como mínimo a 35/45/50/50% en C/D/E/F. Dentro de ese grupo, 1/2/3 aprendices tienen probabilidades 50/30/20%. En C, una instancia recibe aproximadamente 17,5% de asignaciones, frente al ~6,7% previo. No equivale a porcentaje de muestras o minutos. La dificultad de ejercicios sube gradualmente por fase; la suite de evaluación usa el nivel final constante.
- La CLI continúa en segmentos de 100M por defecto y vuelve al usuario con evaluación y deudas. `--segment-steps 0` pide explícitamente completar todo el presupuesto restante. El presupuesto nominal de fases conserva el comportamiento de avance con deuda; una fase nueva sigue sin certificar dominio.

## Uso en el Pod

Actualizar estos archivos en el Pod y detener el runner anterior limpiamente antes de ejecutar la versión nueva. Usar el run completo existente, que contiene config, parent, teacher, ledger, referencias y ambas ramas.

Entrega portable: `reports/rs4_objective_v4_20261002.patch`, que incluye sólo estos cambios y sus pruebas/documentación. Copiarlo al repositorio del Pod y verificar `git apply --check reports/rs4_objective_v4_20261002.patch` antes de `git apply reports/rs4_objective_v4_20261002.patch`. Si la revisión del Pod difiere, resolver los conflictos de código antes de reanudar; el patch no contiene checkpoints ni cambios ajenos a este objetivo.

```bash
cd /root/HaxballRL
.venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume --evaluate-only
.venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume --segment-steps 100000000
```

El primer comando sólo evalúa y reconcilia. El segundo entrena un segmento acotado y evalúa después. No repetir prepare, upgrade, --init-from ni restaurar un checkpoint antiguo para aplicar estas correcciones.

La nueva matriz es más costosa: con seis rivales, BC y 3 semillas hay 198 celdas de partido (3×(12+54)). Los controles con 16 partidos por celda son 3.168 partidas simuladas y los cierres con 128 son 25.344. Los rivales se mantienen constantes; la GPU de PPO está detenida mientras corre la evaluación en CPU. La duración es deliberadamente constante para que la tendencia pueda compararse.

## Verificación y alcance

Las regresiones cubren defensa con referencia cero, rechazo de inactividad, dirección de transiciones, salida con recepción/control, matriz cruzada, rechazo de regresiones de una instancia, rechazo de suites distintas, carga de checkpoints antiguos y reconciliación preservando pesos/Adam/contadores. La suite focalizada inicial aprobó 88 pruebas. La suite completa aprobó **716 pruebas, con 43 omitidas**; tras el ajuste final de validación del hash y aislamiento de la suite reservada, **58 regresiones focalizadas** adicionales pasaron. Avisos: exportador ONNX legado, no fallos de entrenamiento.

Se ejecutó una evaluación técnica breve del checkpoint real de 1.942B, sin PPO ni reemplazarlo: un seed, 2 partidos por celda, 6 segundos por partido y 4 funcionales por celda. Sirve para comprobar ejecución, nuevos campos y métricas; está por debajo de la evidencia mínima para elegir campeón o acreditar nivel. Archivo: `reports/rs4_objective_v4_smoke_20261002.json`.

La copia local sólo tiene `control/latest.pt`; faltan config, ledger, parent y las referencias exactas del programa. La reevaluación deportiva y reconciliación reales deben ejecutarse sobre el run completo del Pod. El código queda preparado para hacerlo, pero no se inventaron esas referencias ni se cambiaron las deudas del checkpoint local basándose en la prueba breve.

Los umbrales siguen siendo criterios de ingeniería y necesitan contraste con el uso real. No se cambian LR, horizonte, arquitectura ni física del mapa a partir de hipótesis no probadas. La fidelidad de una sala concreta y la aceptación del híbrido siguen pendientes de evidencia de esa sala.
