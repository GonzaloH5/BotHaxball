# Optimización CPU — 2026-10-01

## Medición enviada desde el Pod

Mismo checkpoint (paso 2.922.658.944), Torch 8, Numba 12, warmup 3 e iters 15:

| Ruta | Pasos útiles/s reales | Segundos/iter | Muestras/iter |
|---|---:|---:|---:|
| Sin cache BC | 4.064 | 26,205 | 106.496 |
| Con cache BC | 4.419 | 24,100 | 106.496 |

Mejora observada: 8,7%. Los tiempos por fase incluyen saltos del reloj del sistema,
incluso update negativo; no usar esos promedios para atribuir precisamente la ganancia.
El total ya usaba `perf_counter`. La entropía registrada en ambas rutas ronda 1,9–2,05.

## Cambios nuevos

- Agrupación de inferencia de cada snapshot entre tareas CPU, manteniendo los sorteos
  por tarea/rival en el orden original. Un forward por snapshot por decisión.
- Buffer reutilizable de observaciones de inferencia y de minibatch PPO.
- Bootstrap CPU agrupado y cabeza de valor sin ejecutar la cabeza de política.
- Reloj monotónico para fases y desglose del trabajo exterior; perfil por fases de PPO.
- Barrido de hilos en procesos separados desde una copia fija del checkpoint,
  con resultados JSON y confirmación inversa. No cambia el YAML automáticamente.

No cambia arquitectura, recompensa, normalización, PPO, épocas, minibatch, agentes,
rollout_len, referencia BC ni checkpoint. Puede haber redondeos numéricos por agrupar
inferencias. El cache anterior sigue activo y se compara por separado.

## Verificación local de rendimiento

Host distinto del Pod: Ryzen 7 5700U/Windows, Torch 4, Numba 4. Checkpoint local
en paso 2.912.822.912, con 105.856 muestras/iter. Warmup 1 e iters 2, con profiling.
Prueba corta exploratoria, no una predicción del rendimiento del i5.

Orden: referencia inicial, optimizado, referencia repetida. Sólo una corrida optimizada.

| Ruta | Pasos/s | Rollout s | Preparación s | Update s |
|---|---:|---:|---:|---:|
| Referencia inicial | 3.747 | 12,649 | 0,250 | 15,305 |
| Optimizado | 4.563 | 9,371 | 0,228 | 13,547 |
| Referencia repetida | 3.721 | 13,069 | 0,293 | 15,026 |

Frente a la referencia repetida: +22,6% pasos/s, con ruido del host sin cuantificar.
Decisiones: 10,817 → 7,250 s/iter. Bootstrap: 0,034 → 0,011 s/iter.
No atribuir toda la diferencia de update al buffer: también varía el coste de backward,
que este cambio no reescribe. No hubo cambio en las métricas medias impresas:
entropy 1,91014, approx_kl 0,00048, clipfrac 0,00329, bc_kl 1,62508.
Los checkpoints fuente no fueron modificados.

El perfil actualizado incluye además `step_frames_with_touches`, ausente en la
instrumentación anterior. Las cifras antiguas de física eran parciales; el tiempo
inclusivo de entorno sí incluía esa ruta. El mayor coste restante del update local
es backward (~7,95 s), seguido del forward/pérdida (~4,44 s); clipping/Adam ~0,17 s.

## Validación en el Pod

Detener/guardar el entrenamiento. Comparar desde el mismo checkpoint sin procesos
competidores y repetir en orden inverso si hay ruido:

```bash
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15 --no-optimize-cpu
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 15
python -m tools.benchmark_multitask --config train/config_cpu.yaml --warmup 3 --iters 5 --profile-rollout
python -m tools.tune_cpu --config train/config_cpu.yaml --warmup 3 --iters 15
```

Los hilos 8/12 siguen siendo baseline hasta completar el barrido en el i5. No hay
evidencia suficiente para cambiar afinidad, políticas de espera de OpenMP, precisión
o tamaño de los lotes. Física, métricas y logging no se omiten para inflar pasos/s.
