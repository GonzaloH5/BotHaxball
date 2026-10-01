# Optimización del rollout: geometría de recompensas

## Evidencia recibida

El nuevo log avanza de it 35250 a 35775 y de ~3.633M a ~3.684M pasos:
no está congelado. Imprime cada 25 iteraciones en entrenamiento real,
aproximadamente una vez por minuto a esta velocidad; los sondeos imprimen
cada iteración. Futsal 5v5 promueve a R3 en ese intervalo, sin forzar la promoción.
Rendimiento observado: ~35–40k/s, rollout 1,9–2,2 s, prep 0,06 s, update 0,4–0,5 s.
Las cifras del log excluyen parte del mantenimiento: comparar benchmark real.

Con 2,1 + 0,06 + 0,5 s, el rollout representa ~79% del tiempo de las fases.
El panel muestra CPU baja, pero no hay un perfil del Pod que identifique si
predomina física, callbacks/reglas, Python, inferencia, transferencias o esperas.
No se atribuye el límite entero a una sola causa sin esa medición.

## Implementación

- `env/reward_geometry.py`: dos kernels Numba seriales, sin fastmath, con entrada
  y salida float64. Calculan la misma distancia mínima por equipo a la pelota y
  la media de distancia al compañero más cercano, con el mismo tope de separación.
  No materializan diferencias/distancias de todos los pares en arrays NumPy.
  No modifican estado, RNG, reglas, física, reparto ni pesos de recompensa.
- `HaxballEnv.step`: la distancia posterior usada en potenciales también sirve
  para el premio de aproximación al saque, evitando un cálculo duplicado.
  Conserva la referencia NumPy con `optimize_reward_geometry=false` y cuando
  `optimize_rollout=false`. No elimina ninguna recompensa con shaping cero:
  existe cooperación y un piso de separación independiente del shaping global.
- `MultiTrainer.values`: CUDA con rollout optimizado usa `value_only` para
  bootstrap normal y timeouts. Evita el actor descartado; conserva las filas,
  orden, valores del mismo crítico y semántica de truncamiento.
- Benchmark: opción `--no-reward-geometry` para comparación aislada. Informa
  tiempo CPU agregado del proceso, núcleos equivalentes y porcentaje del cupo,
  también en JSON. El perfil añade desglose por tarea y por geometría/pases/saques.
  Es instrumentación opcional inclusiva y añade overhead; no sumar padres e hijos.

Se conservan PPO y el perfil de aprendizaje actual (incluidos rivales/currículo).
No se amplía el lote para llenar RAM/VRAM, no se activa AMP/TF32 ni se retrasa
la política con actores asíncronos. No cambia arquitectura/checkpoint/optimización
del modelo. Reducciones de coma flotante pueden introducir redondeos mínimos;
se verifica equivalencia, no identidad universal de trayectorias largas.

## Microbenchmark local del entorno (no del entrenamiento CUDA)

PC local sin CUDA, Numba 4 hilos. Para cada tarea, dos mediciones NumPy y dos
optimizadas en orden referencia/optimizada/optimizada/referencia, reconstruyendo
el entorno con semilla 51 en cada medición. Shaping 0, max_entities 13,
20 transiciones de calentamiento y 128 medidas, mismas acciones uniformes
generadas con semilla 4. Incluye física/observaciones/recompensas/resets del entorno,
no red, transferencias, PPO ni escritura de buffers del rollout del entrenador.

| Tarea / partidos | NumPy, s/128 | Optimizada, s/128 | Aumento de transiciones/s |
|---|---:|---:|---:|
| Futsal 3v3 / 48 | 0,1291 | 0,1243 | 3,8% |
| Futsal 5v5 / 12 | 0,1062 | 0,0818 | 29,9% |
| Futsal 7v7 / 8 | 0,1116 | 0,0843 | 32,3% |
| RS4 4v4 / 22 | 0,2468 | 0,2115 | 16,7% |

Son microbenchmarks cortos y dependen del host y trayectorias. No se promete
esa misma mejora del rollout completo ni de los pasos/s del Pod. El bootstrap
CUDA nuevo no pudo ejecutarse localmente; sus casos se omiten cuando no hay CUDA.

## Verificación y uso en el Pod

Suite completa: **449 passed, 34 skipped** en 54,95 s. Los casos CUDA se omiten
en esta PC sin GPU; seis avisos de deprecación ONNX ya existentes.

Las pruebas comparan geometría contra NumPy en 1v1/2v2/3v3/4v4/5v5/6v6/7v7/11v11,
con 1/5/48 partidos, distancias cero y jugadores superpuestos. Comparan también
transiciones completas de seis tareas, shaping 0 y 0,7, recompensas, observaciones,
eventos, goles/timeouts y RNG. Se conserva la suite existente de física/restarts,
cooperación y rollout de referencia. El bootstrap comprueba valores/orden y que
la cabeza del actor no se ejecuta, con variantes CPU/CUDA.

Detener y guardar la instancia real antes de medir; Ctrl+C en `tail` sólo termina
la visualización, no la instancia de `nohup`. No lanzar un segundo entrenamiento
ni comparar con carga concurrente. Los benchmarks usan copias temporales de
checkpoint/config y no publican su aprendizaje en el run real.

Comandos de comparación y perfil en README. Después, reanudar el mismo run con
`python -m tools.launch_gpu_overnight --run multi --skip-tuning` (física 4 del
perfil, elegida por el sondeo recibido) o ejecutar el ajuste automático completo.
En el inicio, Numba puede recompilar estos kernels: descartar calentamiento.
