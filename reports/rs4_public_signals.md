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

```bash
nohup .venv/bin/python -u -m tools.run_rs4_v3 --run rs4_v3_public --resume \
  >> runs/rs4_v3_public/overnight.log 2>&1 &
echo $! > runs/rs4_v3_public/runner.pid
tail -n 80 -f runs/rs4_v3_public/overnight.log
```

## Barrera real y exportación

Sin identificar su geometría real, la barrera queda desconocida tanto en
entrenamiento como en evaluación. No se escanean adornos para decidir posesión.
El color del balón funciona independientemente. Si ya conocés los índices públicos
correctos, agregá `--barrier-discs ID...` o `--barrier-segments ID...` **a ambos**
comandos del preparador. Disco 0 es el balón y está prohibido como barrera.
Los índices se guardan en configuración/checkpoint/metadatos de exportación.

Exportación explícita, después de entrenar y evaluar (no publica ni conecta):

```bash
python -m export.to_onnx runs/rs4_v3_public/control/latest.pt --out deploy/rs4_public/model
node deploy/test_obs.js deploy/rs4_public
```

Si el ledger seleccionó `memory`, sustituir `control` por `memory`.
El cliente `deploy/bot.js` admite `--rs4-barrier-discs 12,13`,
`--rs4-barrier-segments 40,41`, `--rs4-red-colors FF0000,E56E56` y
`--rs4-blue-colors 0000FF,5689E5`. Son ejemplos de sintaxis, **no IDs de tu sala**.
`--debug-public-signals` muestra cambios de colores y candidatos públicos de barrera
del mapa como diagnóstico; los candidatos no se activan automáticamente. Observar
saques de ambos colores para confirmar qué objetos cambian antes de configurarlos.
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
Microbenchmark sólo del entorno: 144 partidos, 15 decisiones de warmup y seis
repeticiones alternadas de 80 decisiones; mediana 1.337 ms sin señales y 1.432 ms
con señales (+7.1%). No incluye red/PPO, no usa la GPU del Pod ni predice sus pasos/s.
