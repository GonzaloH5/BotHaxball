# Tiempo de evaluación RS4 en el Pod RTX 3090

Revisión del 2 de octubre de 2026. Proceso inspeccionado por SSH en `/workspace/HaxballRL`: runner PID 4899, evaluador PID 4911. Última muestra: 43 minutos de ejecución, estado Rl, 301% CPU y 2:10:08 de tiempo CPU acumulado. Sigue trabajando; no hay evidencia de bloqueo. El runner espera a su hijo. No se interrumpió ni reinició el entrenamiento/evaluación ni se cambiaron sus pesos, configuración o código.

## Carga real

La evaluación usa CPU, Torch 2 y física 4. La RTX 3090 se usa para PPO según las configuraciones de las ramas, pero este evaluador independiente no usa CUDA.

Cada modelo tiene 198 lotes secuenciales de 128 partidos de dos minutos simulados: 25.344 partidos, además de 66 lotes de ejercicios funcionales de 64 intentos. Son 36 lotes de equipos completos y 162 de equipos mixtos. La consola no informa el avance de cada lote.

## Mediciones

Se midieron tres lotes completos, separados del proceso principal, con los pesos guardados y sin actualizaciones PPO. Tiempos: 10,98 s frente a script, 14,04 s frente al padre y 11,71 s para un aprendiz con compañeros BC. Extrapolar 198 lotes da unos 36–46 minutos por modelo, más ejercicios. Son muestras representativas durante la evaluación activa, no un tiempo garantizado para toda la matriz ni una estimación precisa del tiempo restante.

Un perfil corto atribuye aproximadamente 47–60% del tiempo a llamadas de agentes neuronales y 6–8% a física. El resto incluye entorno, observaciones, scripts y coordinación. No hay fundamento para atribuir la demora principalmente a la física.

Datos: `evaluation_full_batch_timings_20261002.json` y `evaluation_profile_20261002.json` en esta carpeta; copias en `runs/rs4_v3_public/hardware_checks` del Pod.

## Repetición evitable de la referencia

`tools/run_rs4_v3.py::_evaluate` incorpora `full` o `control` al nombre de caché, además de una firma que ya identifica todos los parámetros efectivos de la suite. `_consider_champion` reevalúa el campeón con `full=False` pero conserva los tamaños de la evaluación completa del candidato.

Así, una misma referencia y una misma suite pueden solicitar primero `reference_full_21aa1472444a_d32691d63b79.json` y luego `reference_control_21aa1472444a_d32691d63b79.json`. Ambas cachés estaban ausentes en el Pod. Con campeón existente, este primer ciclo puede ejecutar cuatro matrices: candidato, padre, campeón y padre otra vez. La cuarta es redundante. Estimación orientativa: del orden de 2–3 horas para la reevaluación inicial completa, potencialmente más según los ejercicios y filas lentas.

La corrección propuesta es compartir la caché por firma efectiva de suite, conservando la validación de hashes y parámetros. No se aplicó en esta revisión. Cambiar la función del runner en disco tampoco cambiaría la que ya cargó el proceso activo. No conviene cancelar el cálculo acumulado para introducir este arreglo.

Los 15–17 minutos estimados antes para 100M corresponden únicamente al tramo PPO; no incluyen estas evaluaciones previas ni posteriores.

## Instrumentación

Se intentó leer la pila activa con py-spy instalado únicamente en `/tmp/rs4-profile-tools`; el Pod denegó ptrace. No se modificaron permisos del kernel. Se usaron perfiles independientes breves como alternativa. Estos diagnósticos no constituyen una evaluación de promoción ni cierran los pendientes de calidad del informe original.
