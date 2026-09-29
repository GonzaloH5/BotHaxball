# Validación antes de ampliar modalidades

El trabajo de imitación sigue usando `train/bc.py`, sus datasets y `runs/bc`.
Esta validación usa otro run, mantiene la arquitectura compatible con BC y no
modifica esos archivos ni reanuda procesos existentes.

## Observaciones disponibles en la sala

Los nuevos runs de `train.multitask` usan `model.rule_observation: masked`.
Los 15 datos privados de Pegeche (faltas, tarjetas, estado del slide y pelotas
paradas) se reemplazan por cero **dentro del modelo**, antes de normalizar.
Así PPO, el crítico, la referencia de imitación y el ONNX no dependen de datos
que el cliente real no recibe. El simulador sigue aplicando las reglas completas.
No cambia el ancho del dataset ni se inventan estimaciones de faltas ocultas.
Las pistas físicas de powershot conservan el tratamiento que ya tenían.

Los checkpoints antiguos se cargan en modo `full` para preservar exactamente
su política. Un resume con otro modo falla con una indicación explícita: para
adaptar un checkpoint, iniciar otro run con `--init-from`; requiere entrenamiento,
no basta con volver a exportar. El bot rechaza políticas `full` en mapas detectados
como `real`, porque no puede garantizar su contrato de observación allí. En mapas
sin script siguen siendo compatibles. La detección de reglas por nombre sigue
siendo heurística; usar `--rules real` o `--rules plain` según la sala.

## Una modalidad hasta tener evidencia

Después de terminar el BC, iniciar el piloto separado (Futsal 3v3 sin script):

```powershell
.venv/Scripts/python -m train.multitask --config train/config_validation.yaml --run futsal_validation --init-from runs/bc/bc.pt
```

El piloto usa 192 agentes, dos hilos de PyTorch, retraso de acciones de hasta
3 ticks y un único estadio/formato. No tiene etapas que amplíen modalidades.
El límite de 60 millones de muestras es un presupuesto inicial, no un criterio
de calidad. No se lanzó este entrenamiento como parte de la corrección.

## Métricas y protocolo fijo

`goal_share_vs_scripted`, `goal_share_vs_pool` y `training/goal_elo` son señales
internas calculadas por goles. Sustituyen los antiguos nombres `winrate`/`elo`
de TensorBoard, sin reinterpretar ni borrar los eventos anteriores. Los nombres
internos de los checkpoints de liga se conservan por compatibilidad.

Para decidir si mejoró, usar partidos completos: 64 por rival, tres minutos,
ambos colores, cuatro semillas y una referencia congelada. El score es
`(victorias + 0.5 * empates) / partidos`. La referencia debe mantenerse idéntica
entre evaluaciones. Usar un checkpoint numerado que el entrenamiento ya no
reescriba; para el primer piloto puede ser el BC terminado, si no se vuelve a
entrenar encima de ese archivo.

```powershell
.venv/Scripts/python -m eval.benchmark runs/futsal_validation/ckpt_000025.pt --reference runs/bc/bc.pt --out runs/validation/futsal_000025.json
.venv/Scripts/python -m eval.benchmark runs/futsal_validation/ckpt_000050.pt --reference runs/bc/bc.pt --compare runs/validation/futsal_000025.json --out runs/validation/futsal_000050.json
```

Los reportes registran la huella SHA-256 del candidato, referencia, mapas y
código de evaluación/simulación. Cambiar las semillas, duración, modo greedy,
reglas o contenido de un rival produce otro protocolo y no se compara contra
la marca anterior. Los reportes se crean sin sobrescribir archivos existentes.
`eval.matrix --best` también separa las marcas por protocolo, y `--seed` controla
el muestreo de la política además del entorno.

El comando de benchmark devuelve 0 sólo si pasa los umbrales del YAML:
al menos 60% de puntos frente al bot, 50% frente a la referencia y no más del
25% de partidos sin goles. Si se proporciona un reporte anterior, también
rechaza caídas superiores a 10 puntos porcentuales. Sin referencia o con una
muestra insuficiente, devuelve 1 y explica el motivo en el reporte.
Son criterios operativos de simulación, no una prueba de significancia estadística
ni una certificación de nivel competitivo.

## Comprobación en sala antes de ampliar

Después de pasar el benchmark, exportar a una carpeta nueva para no reemplazar
el modelo desplegado por otra sesión:

```powershell
.venv/Scripts/python -m export.to_onnx runs/futsal_validation/ckpt_000050.pt --out runs/validation/export_futsal/model
node deploy/test_obs.js runs/validation/export_futsal
node deploy/bot.js --model runs/validation/export_futsal/model.onnx --join ROOM_ID --rules plain
```

En una sala de prueba con el mismo mapa, comprobar saques, ambos colores,
latencia y resultado de partidos completos frente a un rival fijo; guardar las
grabaciones y condiciones. Los reportes mantienen `room_validation: pending`
porque aprobar en el simulador no demuestra que esa prueba ocurrió. Sólo tras
esa comprobación ampliar deliberadamente a otra modalidad y validar de nuevo.

## Regresiones del currículo

Cada tarea conserva una ventana y mejor marca ligadas a `(opp_stage, scripted_eps)`.
Al cambiar esa dificultad se reinician ventana, marca y multiplicador de muestras.
Reconstruir entornos o guardar/reanudar conserva la ventana si el contexto coincide.
Las marcas antiguas sin contexto verificable se reinician, sin alterar pesos
del modelo, pasos entrenados ni la liga.
