# RS4: fase experimental de adaptación LR — 2026-10-01

## Motivo y límites de la evidencia

El run tiene más de 4,300M pasos globales. El calendario heredado termina a
3,000M, dejando LR ~0.00005, entropía .002 y BC .003. El log reciente muestra
KL pequeño y aumento de empates. Esto justifica **probar** un LR moderadamente
mayor, pero no demuestra que el LR sea la causa de los empates o de no sacar.

No se dispone localmente del latest RS4 actual del Pod. Se implementó y verificó
la infraestructura en checkpoints de prueba; no se inició un entrenamiento real
ni se midió mejora de winrate/KL/córners. La fase debe prepararse en el Pod sobre
su checkpoint más reciente, una vez detenido y guardado.

## Experimento aislado

`tools.prepare_rs4_adaptation` crea `rs4_adapt` desde un run exclusivo RS4 con PPO
completo, modelo Set universal sin memoria, Adam y liga. Copia sin resalvar
latest.pt y parent.pt, ambos inicialmente idénticos byte a byte al checkpoint
fuente. También guarda parent_config.yaml y specialization.json con SHA-256,
ancla global, LR previo, límite nuevo y etiqueta RS4 anterior.

Por defecto modifica únicamente:

- Nombre del run: rs4_adapt.
- Límite de ejecución: pasos actuales +200M.
- Calendario LR: .0001 → LR efectiva anterior (~.00005) en esos 200M.
- Si no existía schedule_steps, lo fija al total_steps anterior para preservar
  exactamente el calendario global de entropía y BC.

Todos los otros campos del config se conservan, incluidos BC_reference,
rs4_tactics, task_metric_versions, retención, PPO, recompensas y rivales.
No reinicia pesos, normalizadores, Adam, dificultad, ventanas de métricas,
pasos globales ni contadores de etapa. Los partidos se reconstruyen al reanudar,
como ocurre en cualquier --resume; no es un snapshot del estado físico vivo.

`train.runtime.learning_rate` usa `ppo.lr_schedule` sólo cuando está presente.
Sin él, conserva el cálculo anterior. Entropía y BC siguen usando
annealing_fraction global. La guía sigue usando rs4_tactics.start_steps: no se
renueva ni se vincula al nuevo calendario LR. El trainer común y recurrente
utilizan el helper para no ignorar silenciosamente este campo.

Se registra LR efectivo en consola y TensorBoard. El LR es el usado para esa
actualización; el contador de fase se muestra tras sumar sus muestras útiles.

## Protección frente a reinicios accidentales

- Runs existentes y nombres con rutas no se sobrescriben.
- Fuentes mixtas, incompletas o con fase LR propia se rechazan.
- La referencia BC actual debe existir antes de preparar.
- Se comprueban hashes/config durante la copia; requiere la fuente detenida.
- Una fase anclada a pasos previos requiere --resume, su latest.pt y pasos >= ancla.
- No permite arrancar desde BC o pesos frescos mediante --init-from.
- Reanudar conserva start_steps; ampliar total_steps no renueva la fase.
- El final del presupuesto sigue guardando latest.pt y deteniendo el proceso.
- El run original permanece intacto; no hay promoción automática del experimento.

## Uso en el Pod

Detener rs4 con Ctrl+C y esperar el guardado. Actualizar el código publicado y:

```bash
python -m tools.prepare_rs4_adaptation --source runs/rs4/latest.pt --run rs4_adapt
python -u -m train.multitask --config runs/rs4_adapt/config.yaml --run rs4_adapt --resume
```

El log debe mostrar al inicio LR cercano a `1.00e-04` (ya decae antes del primer
log periódico). No volver a ejecutar el preparador,
upgrade --renew-guide o el comando que suma 500M para esta fase. Si se interrumpe,
reanudar el mismo run con el mismo config.

En 50M nuevos pasos el LR es ~.0000875; en 100M, ~.000075; en 200M, ~.00005.
Revisar en 50M y 100M: si cae de forma persistente el rendimiento contra las
mismas referencias, detener y conservar la baseline, no insistir sólo por KL.

Con el experimento detenido y guardado:

```bash
python -m tools.evaluate_rs4 --run rs4_adapt --styles --games 128 --minutes 2 --seed 51
python -m tools.evaluate_rs4 --run rs4_adapt --styles --games 128 --minutes 2 --seed 73
```

Se compara RS4 anterior congelado contra el candidato actual. Las claves JSON
generalist/specialist se mantienen por compatibilidad; reference_label diferencia
el tipo de baseline. La evaluación pareada detecta progreso/regresión, pero sin
un control entrenado con LR anterior no demuestra superioridad causal del nuevo LR.
El contador de córners de consola sigue muestreando sólo el rollout mostrado,
no todas las oportunidades; revisar replays además de victorias/empates/derrotas.

Para volver al estado original, detener rs4_adapt y:

```bash
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

No copiar los pesos del experimento encima de rs4 como atajo.

## Validación local

Suite completa: **571 passed, 42 skipped**, 6 avisos ONNX preexistentes.
Las 27 pruebas nuevas cubren calendario legacy, ancla fija, clamp, extensión de
presupuesto, validación de parámetros, igualdad byte a byte, preservación del
config/BC/guía, estado real de Adam, actualización con LR nuevo, reanudación a
mitad de fase, rechazo de sobrescritura/reanclaje y protección contra pesos
nuevos/checkpoint ausente o anterior. Los modelos de prueba son pequeños y CPU;
no son el agente del Pod. No se afirma una mejora de rendimiento deportivo.
