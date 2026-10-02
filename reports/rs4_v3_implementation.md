# RS4 v3: especialización completa

Ruta nueva opt-in: conserva PPO, goles compartidos ±1 y física RS4 4v4, pero
cambia deliberadamente el currículo y la experiencia con compañeros. Los runs
anteriores permanecen intactos. No se entrenó remotamente ni se publicó un modelo.

## Presupuesto y seguridad

El preparador copia una fuente detenida, congela `parent.pt`, registra hashes y
crea control feedforward y GRU residual32 + atención residual. Los logits y
valores iniciales coinciden con el origen; Adam se transfiere por nombre/forma.
Los normalizadores quedan congelados. `--dry-run` no crea archivos y un destino
existente se rechaza.

6B muestras útiles **compartidas**, no 6B por candidata: piloto de 200M para cada
una, después la ganadora hasta 5.8B propios. Los pasos heredados no cuentan.
No hay ampliaciones automáticas; ni relleno ni controladores congelados cuentan.
El ledger cobra también el PPO de los benchmarks, incluido warmup, aunque sus
copias se descarten: `diagnostic_ppo_steps` se resta de consolidación. La ganadora
llega como máximo a 5.8B menos ese coste; se mantienen 200M finales sin ayudas.

| Fase | Pasos propios nominales | Ejercicios/partidos | Guía | Compañeros congelados |
|---|---:|---:|---:|---:|
| A: saques | .8B | 40/60 | .08 | 10% |
| B: defensa/transición | 1B | 30/70 | .06 | 15% |
| C: ataque colectivo | 1.2B | 30/70 | .05 | 20% |
| D: integrado | 1.2B | 20/80 | .035 | 25% |
| E: compañeros | 1B | 20/80 | .02 | 35% |
| F: consolidación | .6B | 5/95 | 0 | 35% |

El piloto de la elegida está dentro de A. Evaluación cada100M, avance temprano
tras media fase y dos aprobaciones, deudas de habilidades y hasta400M de
recuperación dentro de F. Al menos200M finales sin guía ni bonus de ejecución.
Tiempo nominal ahorrado pasa a consolidación sin ampliar el presupuesto.

Memoria se elige sólo si conserva70% del throughput del control, mejora ≥5pp
funcional, no pierde >3pp de puntos de partidos y repite dirección en2/3 semillas.
La comparación necesita datos reales del Pod; las pruebas CPU no acreditan eso.

## Comandos del Pod

Primero actualizar los archivos y detener limpiamente el entrenamiento fuente.
Esta implementación local **no hace git push**: `git pull` no la recibe hasta
que los cambios estén publicados.

```bash
cd /root/HaxballRL
source .venv/bin/activate

# Regenerar el gate R3 con el código actual; no saltarlo.
python -m eval.scripted_gate --out reports/scripted_r3_gate.json --threads 2

python -m tools.prepare_rs4_v3 --source runs/rs4_adapt/latest.pt --run rs4_v3 --dry-run
python -m tools.prepare_rs4_v3 --source runs/rs4_adapt/latest.pt --run rs4_v3
python -m tools.run_rs4_v3 --run rs4_v3 --dry-run

nohup .venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3 --resume \
  >> runs/rs4_v3/overnight.log 2>&1 &
echo $! > runs/rs4_v3/runner.pid
tail -n 80 -f runs/rs4_v3/overnight.log
```

Reanudar con el mismo `tools.run_rs4_v3 --run rs4_v3 --resume`; no repetir el
preparador, `--init-from` ni renovar anclas. El runner entrena y evalúa en
secuencia, nunca compitiendo ambos por la GPU. Un lock evita duplicados.
Ctrl+C en primer plano guarda el checkpoint. Para detener el proceso en segundo
plano usar `kill -TERM "$(< runs/rs4_v3/runner.pid)"`: el runner envía SIGINT a su
hijo y espera el guardado antes de salir. No matar solamente el hijo, porque
el runner podría intentar continuar; evitar SIGKILL habitual.

`ledger.json` registra gasto, selección, evaluaciones y fallos. `latest` cada5min,
seis históricos cada30min; referencia inicial y campeones protegidos acotados.
Se comprueba espacio antes del reemplazo atómico.
Tras las evaluaciones de cierre se genera un HTML por fase en
`runs/rs4_v3/replays/`, con PPO detenido y el campeón correspondiente. No se
regraba un archivo válido ni se publica automáticamente.

## Benchmark, evaluación y exportación explícitos

Detener PPO antes de estos comandos. La comparación alterna AB/BA/AB, cuenta
filas útiles y mide rollout, preparación, update y mantenimiento.

```bash
python -m tools.benchmark_rs4_v3 \
  --control-config runs/rs4_v3/control/config.yaml \
  --control-checkpoint runs/rs4_v3/control/latest.pt \
  --memory-config runs/rs4_v3/memory/config.yaml \
  --memory-checkpoint runs/rs4_v3/memory/latest.pt \
  --output runs/rs4_v3/benchmark_manual_01 \
  --warmup 3 --iters 8 --repetitions 3 --companion-fraction 0.1 \
  --max-memory-gib 7 --device cuda

python -m tools.evaluate_rs4_v3 \
  --checkpoint runs/rs4_v3/control/latest.pt \
  --reference runs/rs4_v3/parent.pt \
  --teammate-reference runs/rs4_v3/teacher.pt \
  --games 128 --functional-games 32 --minutes 2 --seeds 51 73 91 \
  --out runs/rs4_v3/evaluations/manual_control.json

python -m export.to_onnx runs/rs4_v3/champion.pt \
  --out runs/rs4_v3/export/model
```

El benchmark usa copias descartables. No activa AMP, TF32 ni reduce trabajo PPO
silenciosamente; registra flags reales de precisión. Memoria residente en GPU,
estado independiente por jugador/controlador, RNG fuera de CUDA Graphs y
recaptura tras actualizar pesos. Capacidades agrupadas acotan capturas cuando
cambian los grupos: el relleno añade trabajo de inferencia, nunca muestras PPO.
El perfil inicial requiere `action_delay_max=0` para registrar acciones ejecutadas.

Los saques se miden por oportunidades **resueltas del mismo cohorte**, no mezclando
intentos y éxitos entre ventanas. Ejercicios fuera de puntos/PFSP; evaluación
normal sin currículo artificial ni guía. Liga por victoria1/empate.5/derrota0;
Elo por goles sólo diagnóstico. Campeón seleccionado por evaluación independiente.

## Datos y fidelidad

```bash
python -m tools.build_rs4_sequences \
  --source replays_real/stadiums/rsx4 --out data/rs4_v3_sequences
python -m tools.audit_rs4_simulator \
  --source replays_real/stadiums/rsx4 \
  --out reports/rs4_v3_simulator_audit.json --limit 3 --max-samples 2000
```

Cadencia3ticks, separación por replay/roster/mapa y shards acotados. Situaciones
ambiguas excluidas de métricas específicas. No se requieren descargas. El BC
actual sigue congelado hasta validar un reemplazo; nunca sustituye pesos PPO.

Entrenamiento opcional del imitador, con shards y validación separada:

```bash
python -m tools.train_rs4_imitator --manifest data/rs4_v3_sequences/manifest.json \
  --source runs/bc_rs4_v2_20261001/bc.pt --run bc_rs4_v3_candidate --device cuda
```

No activa nada automáticamente. Antes de preparar un programa nuevo puede
añadirse `--bc-reference runs/bc_rs4_v3_candidate/bc.pt --bc-validation
runs/bc_rs4_v3_candidate/validation.json` al preparador: exige gate aprobado,
hash exacto y arquitectura feedforward compatible. El teacher, nuevo o anterior,
se copia congelado dentro del programa. No cambiarlo en un programa en marcha.

Auditoría local de muestra:100 transiciones abiertas y100 kickoff de un replay,
error p90 de posición ≈0.0001 con **su mapa**, no una certificación de la sala.
Detectó patada5.75 en ese replay frente a5.85 del catálogo; faltan versiones
adicionales antes de cambiar dinámica global. Las mutaciones de script se
excluyen; protección y árbitro no deducibles quedan documentados.
Evidencia: `rs4_v3_simulator_audit_sample.json`.

La teoría de potenciales no garantiza buen fútbol ni la invariancia bajo
cualquier calendario: [Ng, Harada y Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
Los imitadores/scripted/snapshots son proxies de compañeros humanos, no prueba
con personas. Consumir6B, reward, KL, entropía o Elo no acredita aceptación
deportiva. Si falla, el ledger conserva campeón, evidencia y habilidades pendientes.

## Verificación local

Suite completa (revisión 2026-10-02): **682 aprobadas, 43 omitidas**, sin errores. También aprobaron
las pruebas JavaScript de memoria por sesión y de reinicio del cliente.
Tras el último ajuste de replays: 31 pruebas del programa y la nueva regresión
de acciones ejecutadas aprobaron por separado.
Los checkpoints anteriores y los modelos desplegados permanecen intactos.

Las pruebas CUDA se omiten en equipos sin CUDA. La aceptación técnica de
throughput y la deportiva quedan pendientes de ejecutar el programa en el Pod.
No se inició entrenamiento remoto, no se publicó ni se hizo commit/push.

## Revisión del log del Pod — 2026-10-02

La ejecución aportada confirma reanudación de `control` (feedforward), fase B,
guía 0.06, tres épocas y conservación de los pasos tras SIGTERM. No faltaba una
implementación principal por completar cuando se interrumpió la respuesta.

En 11 iteraciones distintas del extracto: media 43,135 pasos útiles/s, rollout
1.40s, preparación 0.04s y update 0.55s. La mayor diferencia frente al perfil
anterior está en rollout, no en update. La mezcla de controladores, escenarios,
shaping y proporción de filas que aprenden impide atribuir toda la diferencia a
un único componente sin perfil. El gate de 70% compara GRU con feedforward bajo
el mismo perfil v3; no certifica conservar 70% del antiguo entrenamiento RS4.

El último `324-65-11` equivale a 81% de victorias y 89.1% de puntos sobre 400
partidos de la ventana, no 89% de victorias. El extracto baja de ~93% a ~89% de
puntos; no demuestra por sí solo una mejora ni una regresión contra rivales y
compañeros constantes. La deuda `defense` sigue pendiente. Las tasas de córners
deben salir de cohortes resueltas, no dividir los contadores de ventanas.

Correcciones de diagnóstico, sin cambiar entrenamiento:

- Separar en log/TensorBoard pasos hasta la pausa de presupuesto propio restante.
- Mostrar filas útiles/simuladas y cantidad de políticas neuronales.
- Perfilar también las funciones efectivamente usadas por v3: bots, transferencia,
  inferencia agrupada host y seguimiento de saques. Los tiempos son inclusivos,
  con solapamiento CPU/GPU; no se suman ni equivalen a duración GPU sincronizada.
- Reanudar con el intérprete explícito de la venv y añadir al log (`>>`), evitando
  `python: No such file` y perder los logs de los pilotos/evaluaciones.

Para revisar la selección y calidad sin nuevos pasos de entrenamiento, consultar
`runs/rs4_v3/ledger.json`, `benchmarks/control_summary.json`,
`benchmarks/memory_summary.json` y los informes en `evaluations/`.
Los cambios de esta revisión no requieren repetir el preparador ni reiniciar
pesos, fases, anclas o el programa. No se modificaron los checkpoints del usuario.
