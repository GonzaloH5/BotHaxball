# Callbacks de rollout: saques y cooperación, 01/10/2026

## Medición aportada del Pod antes de este cambio

Mismo checkpoint, Torch 2, Numba 4, 3 iteraciones de warmup y 15 medidas.
Geometría NumPy: 37.286 pasos/s reales, rollout 2,061 s, prep 0,063 s,
update 0,466 s, CPU 2,92 núcleos equivalentes de 10.
Geometría compilada: 43.014 pasos/s reales, rollout 1,746 s, prep 0,064 s,
update 0,429 s, CPU 2,85 núcleos equivalentes de 10. Ambas produjeron
97.920 muestras/iter y las mismas medias impresas de entropía/KL/clipfrac/BC.
Ganancia observada de throughput +15,4%, reducción del rollout 15,3%.
La comparación no se repitió en orden inverso: parte del cambio de tiempos
puede ser ruido/deriva del host; no se atribuye el descenso de update a geometría.

Perfil posterior (3 warmup + 5 medidas, con overhead de instrumentación):
rollout 1,812 s, entorno total 1,198 s, decisiones 0,543 s,
física 0,136 s, RS4 0,282 s, retención de pases 0,079 s,
contactos/pases 0,096 s, protección de saques 0,050 s,
finalización de saques 0,063 s. Son tiempos inclusivos; los callbacks
están dentro del entorno y RS4 está dentro del total de entornos.
La captura CUDA permanece en ~0,172 s/rollout.

## Cambio implementado

`env/rollout_callbacks.py` contiene kernels seriales Numba sin fastmath:

- Protección de rivales: mantiene la barrera rectangular en saque de arco y la
  distancia circular alrededor de la pelota, incluido el caso coincidente.
  Preserva orden de proyección y anulación de velocidad.
- Pre-tick: aplica protección y quita la tecla de patada sólo a los rivales
  del equipo que saca; devuelve acciones independientes, sin modificar entrada.
- Post-tick: mantiene ticks/owner, fuerza mínima de saque de arco, release por
  patada/gol/timeout, pelota inmóvil mientras espera y protección posterior.
  No fusiona ticks a través de eventos ni cambia el criterio de timeout.
- Contactos: respeta contactos múltiples ambiguos, conducción del mismo jugador,
  distancia/fase elegible, candidato de pase y corte de cadenas por intercepción.
- Retención: misma utilidad, espacio relativo, devolución inmediata, cadena,
  eventos y cap de recompensa; sin pagos anticipados ni recompensas nuevas.

En la ruta optimizada de `step`, los kernels escriben en el reward/eventos del
paso directamente en el mismo orden cronológico. Evitan resultados temporales
y su posterior suma repetida. El bloque de eventos es contiguo, con vistas para
el mismo diccionario público de eventos. Se asigna de nuevo en cada `step`;
no recicla datos devueltos anteriormente ni guarda buffers en checkpoint.
Las llamadas públicas a los helpers sin acumulador siguen devolviendo resultados
independientes. No altera RNG, observaciones, PPO, currículo, modelo o hilos.

La referencia NumPy/Python se conserva con `runtime.optimize_callbacks=false`
o `runtime.optimize_rollout=false`. Benchmark `--no-callbacks` sólo desactiva
este cambio; conserva la geometría anterior, BC cache, transferencias y CUDA Graph.
No se modifica la captura CUDA en esta pasada: reutilizarla sin invalidar
pesos/normalizadores y asignaciones de rivales exige otra validación en GPU.

## Microbenchmark local (sin inferencia/PPO/CUDA)

Numba 4 hilos, geometría compilada en ambos casos. Para cada tarea, cuatro
mediciones en orden referencia/optimizada/optimizada/referencia, reconstruyendo
el entorno con semilla 51, shaping 0, max_entities 13. Acciones uniformes
generadas con semilla 4; 128 transiciones de calentamiento y 512 medidas.
Los tiempos incluyen entorno completo: física, observaciones, rewards y resets.

| Tarea / partidos | Referencia, s/512 | Compilada, s/512 | Aumento de transiciones/s |
|---|---:|---:|---:|
| Futsal 3v3 / 48 | 0,4862 | 0,4146 | 17,3% |
| Futsal 5v5 / 12 | 0,3856 | 0,3432 | 12,3% |
| Futsal 7v7 / 8 | 0,3683 | 0,3461 | 6,4% |
| RS4 4v4 / 22 | 0,8879 | 0,5488 | 61,8% |

No se promete esa misma mejora de rollout ni steps/s del Pod. Requiere comparación
de punta a punta; los tests locales no validan velocidad CUDA ni carga del host.

## Verificación

Suite completa local: **462 passed, 34 skipped** en 57,78 s; seis avisos
de deprecación ONNX existentes. CUDA no está disponible en esta PC.
`git diff --check` sin errores de espacios.

Pruebas específicas comparan callbacks con referencia, estados internos, eventos
y RNG: saque mixto con kick/timeout/gol, invasión del área, coincidencia, velocidades,
contactos aleatorios simples/múltiples, pases retenidos, caps y cadenas. Transiciones
completas cubren RS4, Futsal 3v3/7v7 y JJRS, con/sin lag, goles, salidas, resets
y truncamientos. Se comprueba el acumulador y que los resultados no se sobrescriban.
La suite previa de física, saques, cooperación, PPO y CUDA permanece aplicable.

Comparación en el Pod: comandos en README, entrenamiento detenido y guardado,
mismo checkpoint/hilos, sin procesos concurrentes, repetir en orden inverso.
Primera compilación Numba descartada mediante warmup. No se desplegó ni se
reinició el run remoto desde la PC local.
