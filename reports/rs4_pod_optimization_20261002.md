# Optimización del Pod i5-11400F / RTX 3090

Se implementó `tools.optimize_rs4_pod` para comparar perfiles sin modificar los parámetros de aprendizaje. El trabajo se ejecuta en `/workspace/rs4_optimize_20261002`, con copias del modelo actual, padre y teacher. Espera a que el runner activo libere `.runner.lock` y mantiene ese bloqueo durante toda la medición y aplicación del perfil.

## Comparación

- Entrenamiento CUDA: Torch/física 2/4 (referencia), 1/2, 1/4, 2/2, 2/6 y 4/4. Tres repeticiones alternando el orden, tres iteraciones de calentamiento y diez medidas por repetición. Se conservan entornos, rollout, minibatches, épocas, precisión y configuración de aprendizaje.
- Evaluación: CPU 2/4, 1/2, 1/4, 2/2 y CUDA 1/2, 1/4. Partidos frente a R3, frente al padre y con un aprendiz y compañeros BC; 128 partidas por lote, ambos colores y semillas 51/73/91. Se compara el hash de todas las acciones y los resultados de cada lote.
- Sólo se selecciona una mejora mediana de al menos 5% sin empeorar ninguna repetición frente a la referencia. Una ruta con acciones diferentes se descarta. Una ruta alternativa seleccionada debe superar además la comprobación de equivalencia en lotes completos de dos minutos; si falla se mantiene CPU 2/4.

La medición no prueba equivalencia universal para todos los estados futuros ni una mejora deportiva del agente. El soporte CUDA recurrente sí conserva memoria, acciones ejecutadas y resets por entorno en las pruebas. La evaluación sigue siendo secuencial; no se implementó un evaluador multiproceso en esta entrega.

## Resultados y aplicación

Resultados reales pendientes de ejecución. No sustituir estas comparaciones por una velocidad supuesta. El trabajo escribe `status.json`, `progress.json`, logs por ensayo y `result.json` en `/workspace/HaxballRL/pod_logs/optimization_20261002`.

El perfil seleccionado se aplica a la rama elegida, con copia de su configuración anterior. El reparto de hilos/dispositivo de evaluación se guarda en `runs/rs4_v3_public/evaluation_runtime.json`; el runner lo transmite al evaluador. El dispositivo forma parte de la identidad de la suite y de las cachés para no mezclar evidencia entre CPU y CUDA.

Los modelos y el optimizador fuente se conservan y se comprueban con hashes. Los benchmarks descartados cuentan como trabajo PPO: cada ensayo reserva una cota superior antes de empezar, la reconcilia con su conteo real al terminar y conserva la reserva si falla. `hardware_diagnostic_work.json` registra el coste, que la nueva versión del runner resta del presupuesto futuro sin cambiar el coste inmutable del piloto. No se modifica el checkpoint para fingir que esos pasos entrenaron al agente principal.

Después de optimizar, `pod_optimize_rs4.sh` aplica la revisión preparada cuando el proyecto está detenido y lo deja listo; no arranca otro tramo de entrenamiento principal. El comando de continuación preparado sigue siendo:

```bash
bash /workspace/rs4_update_fdb186d/pod_resume_collective_fixes.sh
```

Validación previa: 57 pruebas locales aprobadas, una omitida; 24 aprobadas en el Pod, una omitida, incluyendo soporte recurrente CUDA. La validación de rendimiento y equivalencia del checkpoint real pertenece al trabajo pendiente, no a esas cifras de tests.
