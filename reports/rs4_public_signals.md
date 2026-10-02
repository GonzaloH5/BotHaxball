# RS4: señales públicas y continuación del programa

## Qué cambia

La política opt-in `public_signals_version: 1` recibe señales que puede leer el
cliente: color de la pelota, color de discos/segmentos de barrera configurados,
posición del balón e historial **estimado** de proximidad de jugadores. No recibe
dueño interno del saque, identidad del ejecutor, `last_touch` verdadero ni tipo
de ejercicio. La posesión no se conoce con certeza por estar cerca de la pelota.

Las 15 posiciones anteriormente reservadas a reglas privadas, índices 56–70,
transportan un paquete público versionado. Esas reglas continúan enmascaradas.
Una proyección residual nueva se inicializa en cero: logits y valores iniciales
son exactamente iguales al checkpoint original. Los parámetros existentes y
sus momentos de Adam se conservan por nombre y forma; sólo la proyección es nueva.
Los normalizadores permanecen congelados y estas señales acotadas no dependen
de sus estadísticas antiguas. No cambia PPO, rewards, física, reparto ni hilos.
No se activa la GRU descartada: el control feedforward recibe un historial pequeño
calculado por el adaptador, no una memoria recurrente nueva.

Se reconocen por defecto rojo `FF0000`/`E56E56` y azul `0000FF`/`5689E5`.
Un balón coloreado en movimiento no activa por sí solo el indicador de saque.
La geometría estima lateral/córner/saque de arco; no es una etiqueta del árbitro.
Color desconocido o señales contradictorias dejan dueño desconocido.
Si la sala mantiene el balón coloreado incluso estando parado durante juego
abierto, habrá que verificar esa convención: el adaptador no puede garantizar
que toda pelota parada coloreada sea un saque sin otra evidencia pública.

El historial se muestrea a la cadencia de decisiones, con posiciones no
extrapoladas en el cliente. Cercanía de un solo equipo produce confianza máxima
0.6, ambos equipos cerca producen desconocido, y el recuerdo decae en 600 ticks.
Se borra tras reposicionamientos grandes, reinicios y cambios de jugadores/control.
Un contacto entre muestras puede no verse; no se inventa un último tocador exacto.

## Migrar en el Pod

Publicar/transferir primero estos cambios: no se hizo commit/push ni se inició
entrenamiento remoto automáticamente. `git pull` sólo los recibe una vez publicados.

1. Detener el runner original con **Ctrl+C una vez** y esperar su guardado/salida.
   En segundo plano, si conservás el PID correcto del runner:
   `kill -TERM "$(< runs/rs4_v3/runner.pid)"`. Esperar que también termine el hijo;
   no usar SIGKILL ni migrar mientras otro entrenador escribe ese run.
2. Actualizar el código y activar el entorno. Ejecutar:

```bash
cd /root/HaxballRL
source .venv/bin/activate
python -m pytest tests/test_public_signals.py -q
python -m tools.upgrade_rs4_public_signals --source-run rs4_v3 --run rs4_v3_public --dry-run
python -m tools.upgrade_rs4_public_signals --source-run rs4_v3 --run rs4_v3_public
python -m tools.run_rs4_v3 --run rs4_v3_public --dry-run
python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume
```

El preparador requiere selección de arquitectura terminada y programa incompleto.
Copia todo el programa, rechaza destinos existentes y comprueba espacio disponible.
Se conserva `public_source.pt` con la política inmediatamente anterior y se modifica
sólo `latest.pt` de la candidata elegida. Los runs anteriores permanecen intactos.
No repetir `prepare_rs4_v3`, usar `--init-from`, renovar guía ni añadir otros 6B:
fase, pasos relativos, LR, deudas y diagnósticos gastados se heredan sin reiniciar.
Las evaluaciones nuevas se guardan en `evaluations_public_v1`; la evidencia y los
campeones anteriores no se reemplazan por decreto. Un campeón antiguo puede seguir
sin estas señales hasta que una política nueva lo supere en evaluación independiente.

Reanudaciones posteriores usan únicamente el último comando. Alternativa con log:

Si una copia creada antes de la corrección conserva `run_name: rs4_v3/control`,
el runner actualizado usa la carpeta seleccionada (`rs4_v3_public/control`) como
destino autoritativo. No recrear el programa ni modificar checkpoints para reparar
el error `Public signal contract differs from checkpoint`: actualizar el código
y reanudar la copia existente. El preparador nuevo también corrige ambos nombres
en los YAML al copiar.

```bash
nohup .venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume \
  >> runs/rs4_v3_public/overnight.log 2>&1 &
echo $! > runs/rs4_v3_public/runner.pid
tail -n 80 -f runs/rs4_v3_public/overnight.log
```

## Barrera real y exportación

### Descubrimiento automático de joints RS ONE

Ya se identificó la topología en `stadiums/rs_one.hbs` y se auditó en dos
grabaciones reales con el motor original, **sin conectarse a una sala**.
El cliente descubre dos pares rojo/azul en bandas opuestas, con ocho discos
auxiliares independientes, radio cero y sin colisiones. No fija los índices:
si cambia el orden, vuelve a descubrirlos. Una topología ambigua o desconocida
queda deshabilitada.

Las líneas inactivas tienen extremos coincidentes. Sólo se acepta una línea
extendida, horizontal, próxima a la banda donde está la pelota quieta; se ignoran
líneas colapsadas, adornos, extremos de jugadores y las líneas del inicio de
partido con balón en el centro. Dos equipos visibles producen conflicto, no
una asignación arbitraria. Colores observados: pelota roja `FF3F34`, azul `0FBCF9`;
joints rojo `EC7458`, azul `48BEF9`. Las paletas anteriores siguen admitidas.

En `argbol1t` se detectaron 2,879 ticks con barrera y en `argbra1tfinal`, 2,862:
todas coincidieron con el equipo indicado por el color público del balón.
Esto **no es una tasa de éxito de saques** ni una validación deportiva.
Hay saques con pelota pintada sin línea, incluidos córners/saques de arco, y
el despliegue de un lateral puede demorarse algunos ticks.
Evidencia reproducible: `rs4_joint_audit_argbol.json` y `rs4_joint_audit_argbra.json`.

El simulador añade únicamente un renderer cosmético: línea en laterales,
color del balón en los demás saques. No añade fuerzas ni cambia rewards/reglas.
Es una aproximación respaldada por esas grabaciones, no una réplica garantizada
de los tiempos de protección ni de scripts de otras salas.

### Activar en el run público existente

**No repetir `upgrade_rs4_public_signals` ni recrear pilotos.** Con el código
actualizado en el Pod, detener el runner y esperar su guardado antes de configurar:

```bash
kill -TERM "$(cat runs/rs4_v3_public/runner.pid)"
tail -n 80 -f runs/rs4_v3_public/overnight.log
```

Cuando termine el proceso (salir de `tail` con Ctrl+C), continuar:

```bash
python -m tools.configure_rs4_public_joints --run rs4_v3_public --dry-run
python -m tools.configure_rs4_public_joints --run rs4_v3_public
nohup .venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume \
  >> runs/rs4_v3_public/overnight.log 2>&1 &
echo $! > runs/rs4_v3_public/runner.pid
tail -n 80 -f runs/rs4_v3_public/overnight.log
```

El configurador conserva checkpoint y ledger byte a byte, Adam, LR, anclas,
deudas, fase, selección y presupuesto. Sólo cambia YAML/manifiesto; deja backup
`control/config_before_public_joints.yaml` (o `memory/…`). Rechaza un runner que
tenga el lock activo. La metadata de sensores se incorpora al próximo guardado
normal. El log de arranque indica `barreras laterales por joints`.
La evaluación usa un directorio nuevo y baselines versionadas por código para
no reutilizar resultados anteriores como si fueran mediciones equivalentes.

Auditoría offline, opcional:

```bash
node bridge/audit_public_joints.js replays_real/stadiums/rsx4/argbol1t__70f72c3802b7.hbr2 --max-frames 10000
```

### Otros mapas y cliente

Si no se reconoce la topología, la barrera queda desconocida. No se escanean
adornos para decidir posesión. El color del balón funciona independientemente.
Si ya conocés los índices públicos
correctos, agregá `--barrier-discs ID...` o `--barrier-segments ID...` **a ambos**
comandos del preparador. Disco 0 es el balón y está prohibido como barrera.
Los índices se guardan en configuración/checkpoint/metadatos de exportación.

Exportación explícita, después de entrenar y evaluar (no publica ni conecta):

```bash
python -m export.to_onnx runs/rs4_v3_public/control/latest.pt --out deploy/rs4_public/model
node deploy/test_obs.js deploy/rs4_public
```

Si el ledger seleccionó `memory`, sustituir `control` por `memory`.
Después de un guardado del run configurado, reexportar explícitamente: el ONNX y
su JSON anterior no se actualizan por modificar el YAML de entrenamiento.
El cliente `deploy/bot.js` admite `--rs4-auto-joints` (también heredado del JSON),
`--rs4-barrier-joints 0,1,2,3`, `--rs4-barrier-discs 12,13`,
`--rs4-barrier-segments 40,41`, `--rs4-red-colors FF0000,E56E56` y
`--rs4-blue-colors 0000FF,5689E5`. Son ejemplos de sintaxis, **no IDs de tu sala**.
`--debug-public-signals` muestra cambios de colores y candidatos públicos de barrera
del mapa como diagnóstico. Con `auto_joints`, sólo los candidatos de la topología
RS ONE pasan a detección dinámica. Observar saques de ambos colores al usar otra sala.
Si se usan barreras en producción, preparar la copia con esa señal disponible
para no introducir una entrada que nunca vio durante entrenamiento.
La API pública expone colores de discos y segmentos en sus
[tipos oficiales de node-haxball](https://github.com/wxyz-abcd/node-haxball/blob/master/src/index.d.ts).
No se ha conectado el bot a tu sala para verificar su paleta/IDs ni sus tiempos de protección.

## Verificación y límites

Pruebas de igualdad inicial FF/GRU, Adam, gradiente de la proyección, conservación
de programa/ledger, reanudación PPO desechable, ausencia de fuga de estados privados,
simetría por color/banda, conflictos, contacto ambiguo y paridad Python/JS/ONNX.
El cliente real se prueba con una sala simulada: cadencia, reinicios y seguimiento
mientras inferencia está ocupada. Los checkpoints de usuario no se usan para entrenar.

No garantiza sacar mejor inmediatamente: la proyección arranca en cero y debe
aprender a utilizar la nueva información durante el currículo que resta.
No se afirma ganancia de velocidad ni throughput CUDA sin medir en el Pod.
La ruta añade un pequeño tracker compilado y una proyección; no cambia tamaño de
observación ni transfiere historiales completos entre CPU/GPU.

Validación local: suite general 691 aprobadas / 43 omitidas (incluye pruebas CUDA
no disponibles), más pruebas Node de contrato, memoria y señales públicas.
Actualización de joints: suite completa **698 aprobadas / 43 omitidas**, más
pruebas Node del cliente y memoria; las exportaciones previas fueron restauradas
byte a byte después de los tests. Checkpoints de usuario y entrenamiento remoto
no se modificaron durante la implementación.
Microbenchmark sólo del entorno: 144 partidos, 15 decisiones de warmup y seis
repeticiones alternadas de 80 decisiones; mediana 1.337 ms sin señales y 1.432 ms
con señales (+7.1%). No incluye red/PPO, no usa la GPU del Pod ni predice sus pasos/s.
