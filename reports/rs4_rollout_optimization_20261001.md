# Optimización de rollout RS4 v2 — 2026-10-01

## Alcance

Se mantiene la formación v2 y todas las recompensas nuevas. No se modifica PPO,
la física, los pasos útiles, la referencia BC, los checkpoints, el currículo,
el presupuesto, el coeficiente táctico ni su calendario de decay.

El log del Pod muestra ~0.6 s de rollout antes frente a ~0.9–1.0 s después;
prep ~0.04 s y update ~0.5 s se mantienen. La caída ya aparece usando BC2,
antes de cargar el imitador RS4 nuevo. Parte del coste procede de reactivar la
guía, que había llegado a cero; no atribuir toda la diferencia sólo a formación v2.

## Implementación

- `env/rs4_rollout.py`: seguimiento de contacto defensivo y potencial de saque
  compilados con Numba, float64 sin fastmath, sin RNG. Preservan contactos
  ambiguos, antigüedad, snapshot anterior, zona/umbral, propietario del saque
  y reparto del potencial al equipo ejecutor.
- `HaxballEnv._rs4_potential(indices=None)`: en la ruta rápida sólo procesa juego
  abierto. En kickoff y set-piece el resultado ya era cero. Permite refrescar
  únicamente los partidos reiniciados, sin recalcular todos los vecinos.
- Reutilización del buffer de posiciones de presión sin alias con el simulador.
- Formación v2: calcula una vez los costes de GK/presión compartidos entre
  ambas bandas y elimina la construcción inútil de objetivos v1. Conserva los
  costes de balance/profundidad, pesos, operaciones y orden de sumas; aún evalúa
  ambas orientaciones y las 24 asignaciones de cada una.
- `runtime.optimize_rs4` habilitado por defecto cuando optimize_rollout lo está;
  false recupera callbacks NumPy y potencial/refresco en lote completo. No cambia
  semántica. El kernel de formación mejorado es compartido entre ambas rutas.
- El profiler añade los tres componentes RS4. El benchmark permite A/B con
  `--no-optimize-rs4` sin desactivar las otras optimizaciones.

## Medición local, no entrenamiento del Pod

Referencia de código anterior:
`593ef04759c355521bba1140dba10180d89749c6`.
Comparación contra sus implementaciones originales de HaxballEnv y components,
cargadas en memoria desde Git, sin modificar archivos ni checkpoints.

Windows, Python 3.13.5, Numba 0.67.0, Torch 2.14.0+cpu; CUDA no disponible.
144 partidos 4v4, frame_skip=3, 4 hilos de física, semilla 51, random_reset=.3,
max_ticks=7200, kickoff_timeout=180, observación universal con 7 entidades.
Acciones R3 en ambos equipos, calculadas fuera del tiempo medido. Rewards v2:
guía .08, aproximación .05, expiración .25 y salidas defensivas .2.

320 decisiones de calentamiento, luego 3 bloques de 320. El orden de ejecución
anterior/nuevo se alternó por bloque. Estadística: mediana del tiempo medio de
env.step de cada bloque. Ambos recibieron exactamente las mismas acciones.

| Medida | Anterior | Optimizado |
| --- | ---: | ---: |
| env.step, ms por lote | 6.3220 | 3.4297 |
| components v2, ms por 144 partidos aleatorios | 0.6239 | 0.5857 |

Reducción de tiempo de entorno **45.75%** (+84.33% en lotes de entorno/s).
Los bloques medidos acumularon 8,418 filas con saque protegido y 37 resets.
El microbenchmark components usa mediana de 5 bloques de 100 llamadas.

Se comprobaron igualdad exacta de observaciones/recompensas/terminales,
contexto defensivo y RNG durante las 1,280 decisiones, y de components v1/v2
en 512 partidos aleatorios. Esto no incluye inferencia neuronal, transferencias,
preparación/update PPO, evaluación ni guardado. No extrapolar esos porcentajes
a steps/s finales ni prometer volver a 90k sin medir en el Pod.

## Validación

Las pruebas nuevas cubren contactos repetidos/ambiguos, snapshot/fallback,
umbrales exactos y adyacentes, potencial de saques, outputs independientes,
subset vacío/desordenado, omisión de filas protegidas, equivalencia de las dos
asignaciones completas, refresco de un único reset, decay hasta cero y
transiciones con goles, salidas, retrasos, expiraciones, eventos y RNG.
Los callbacks compilados y de referencia se verifican por separado.

Suite completa: **544 passed, 42 skipped**, con avisos ONNX preexistentes.
Las pruebas CUDA no pudieron ejecutarse en esta PC. Los artefactos deploy que
regeneran las pruebas se restauraron desde un backup previo; no se entregan cambios
de modelo ni de fixture.

## Aplicación y medición en el Pod

Detener guardando con Ctrl+C, actualizar código y reanudar con el config guardado.
No repetir `tools.upgrade_rs4 --renew-guide`, no usar --init-from y no añadir otro
presupuesto de 500M por aplicar este cambio. El config guardado del run actual ya
contiene la referencia BC y el límite de pasos del último comando.

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

Opcional, con entrenamiento detenido:

```bash
python -m tools.benchmark_multitask --config runs/rs4/config.yaml --checkpoint runs/rs4/latest.pt --warmup 3 --iters 15 --no-optimize-rs4
python -m tools.benchmark_multitask --config runs/rs4/config.yaml --checkpoint runs/rs4/latest.pt --warmup 3 --iters 15
```

Repetir en orden inverso; comparar pasos/s reales, muestras útiles y rollout.
Añadir --profile-rollout a ambos para localizar costes (añade overhead).
Si el config exige scripted_readiness, el cambio de haxball_env invalida el hash:
regenerar `python -m eval.scripted_gate --out reports/scripted_r3_gate.json`,
sin desactivar el gate como atajo.
