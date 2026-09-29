# Memoria recurrente opcional

`train/config_memory.yaml` activa una GRU de 64 unidades sobre el modelo universal.
Conserva una memoria por jugador, alimentada por la observación actual y la acción
anterior enviada por ese jugador. No contiene contadores exactos de slide ni accede
al estado privado del script: debe aprender las dependencias temporales.

Los entrenamientos existentes siguen sin memoria. No cambia `train/bc.py`, el
constructor del dataset, sus muestras ni los checkpoints de imitación en curso.
El nuevo modelo usa las mismas dimensiones de observación y puede importar sus
pesos. Sus salidas adicionales se inicializan en cero: antes del ajuste conserva
la política importada sobre las mismas entradas, y aprende gradualmente a utilizar
la memoria. Las entradas privadas de reglas continúan enmascaradas.

## Entrenar después de terminar BC

```powershell
.venv/Scripts/python -m train.multitask --config train/config_memory.yaml --run futsal_memory --init-from runs/bc/bc.pt
```

Esto inicia un run independiente de Futsal 3v3; no convierte ni sobrescribe BC.
Con `--init-from` también se conserva la regularización hacia el imitador sin
memoria. Para importar un checkpoint recurrente como punto de partida, configurar
`ppo.bc_kl_coef=0`: la referencia de imitación KL admite modelos sin memoria.

```powershell
.venv/Scripts/python -m train.multitask --config train/config_memory.yaml --run futsal_memory --resume
```

Al reanudar se recuperan pesos, optimizador, liga, pasos y la referencia BC guardada.
Como se crean entornos nuevos, sus memorias comienzan en cero. No se pretende
restaurar una partida física interrumpida.

## Qué aprende y qué no está demostrado

PPO recibe segmentos ordenados de 32 decisiones (1,6 segundos a 20 decisiones/s).
Se barajan segmentos, no observaciones individuales. El estado inicial de cada
segmento se toma del rollout; la propagación del gradiente se trunca en sus límites.
La memoria persiste entre segmentos y rollouts durante el mismo episodio, por lo
que 32 decisiones no son un límite duro de retención. Tampoco garantizan que aprenda
un cooldown de 25 segundos: para tiempos conocidos siguen siendo preferibles
contadores explícitos a partir de eventos observables.

La memoria se reinicia al terminar un episodio, incluyendo goles y timeouts;
al reconstruir entornos; y para los jugadores controlados por un rival que cambia
de identidad. Cada snapshot de la liga mantiene su propio estado. El bootstrap
de timeouts consulta el valor con la memoria anterior al reset, sin avanzar la
memoria viva una segunda vez. La normalización queda congelada en los valores del
BC importado, o en media cero/varianza uno si se entrena desde cero con las entradas
ya escaladas del entorno. Esto mantiene consistente el ratio PPO y el estado oculto.

Agregar esta capacidad no certifica mejor juego, reconocimiento de faltas ni
dominio de Pegeche. El piloto acotado permite comparar primero el modelo con y sin
memoria; después hacen falta entrenamiento y evaluación específicos con scripts.

## Evaluar, exportar y jugar

Los comandos existentes de `eval.matrix`, `eval.benchmark` y `eval.render` detectan
el modelo recurrente y conservan estados separados por jugador y partida. Para
comparar versiones, usar el mismo protocolo y una referencia congelada como explica
`VALIDATION.md`.

```powershell
.venv/Scripts/python -m eval.benchmark runs/futsal_memory/ckpt_000025.pt --reference runs/bc/bc.pt --out runs/validation/memory_000025.json
.venv/Scripts/python -m export.to_onnx runs/futsal_memory/ckpt_000025.pt --out runs/validation/export_memory/model
node deploy/test_obs.js runs/validation/export_memory
node deploy/bot.js --model runs/validation/export_memory/model.onnx --join ROOM_ID --rules plain
```

El ONNX recurrente recibe `obs`, `memory` y `previous_action`; devuelve `logits` y
`memory_out`. El bot actualiza la memoria después de enviar la decisión. Reinicia
el estado al comenzar/terminar una partida, marcar un gol, resetear posiciones,
cambiar mapa o plantel y dejar de estar activo. Descarta resultados pendientes de
una inferencia anterior a esos eventos. Los ONNX sin memoria mantienen su interfaz.

La utilidad requiere entrenamiento adicional. La cirugía `tools/grow_obs.py`
todavía no admite checkpoints `recurrent_set`; rechaza ese tipo explícitamente.

## Costo y comprobaciones

Con la configuración del piloto, el modelo pasa de 239.155 a 294.899 parámetros
(+23,3%). La memoria viva por jugador ocupa 256 bytes, además de buffers de PPO.
Una medición breve del actor-crítico con dos hilos en esta PC dio aproximadamente
1,6 ms sin memoria y 2,1 ms con memoria para un jugador; con 192 jugadores, 4,0 y
5,3 ms. Son mediciones de PyTorch bajo la carga actual de la máquina, no garantías
de latencia del bot ni una medición del entrenamiento completo.

Las pruebas cubren dependencia del historial, aprendizaje de una señal temporal
sintética, gradientes entre decisiones, resets, separación de jugadores y rivales,
ratio PPO antes de actualizar, bootstrap, BC → PPO → checkpoint → resume y paridad
ONNX. Los tests de Node verifican además que un gol o cambio de equipo invalide
una inferencia pendiente. Ninguna prueba se conecta a una sala pública.
