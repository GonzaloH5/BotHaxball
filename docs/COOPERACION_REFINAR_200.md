# Refinamiento desde el checkpoint cooperativo de 200

Pedido del usuario el 7 de octubre de 2026: conservar el momento de mejor pase observado y empezar desde allí con un criterio mejor. La corrida anterior se conserva, detenida en control 700 y currículo 602. El checkpoint seleccionado es `curriculum/snap_00200.pt`; se comprobó que `best_pase.pt` también corresponde a 200. La selección se basa en las evaluaciones observadas hasta 500, no en test ni en juego humano.

La copia protegida del pod es `/workspace/HaxballRL_coop_20261007/runs/x4_coop_corrected_20261007/preserved/curriculum_update_00200.pt`, SHA256 `da60585005b724c93ebbcea8218e3c5971156b594a44ae26cb9d48835b79fb22`. Conserva actor, crítico, normalización y optimizadores del snapshot original. El entrenamiento nuevo carga actor, crítico y normalización, pero empieza con optimizadores y progreso curricular nuevos; no altera esa copia.

## Criterio nuevo

- Arranca en ejecución y conservación de pase, etapa 1. No hereda el aprobado de etapa 4 de la corrida anterior.
- Mantiene 50% de los entornos en ejercicios y 50% en partidos 4v4, también en transferencia.
- Desde etapa 2, la mitad de las filas de ejercicios repasa habilidades anteriores. En etapas 3 y 4, ese repaso alterna ejecución y conservación de pase, y salida de presión mediante pase. Se reservan filas, no se sortean tipos después de cada episodio: los ejercicios cortos no se diluyen por duración.
- Cada avance exige 100 actualizaciones de aprendizaje del actor en la etapa y dos ventanas distintas y consecutivas de 512 episodios cuyo límite inferior de Wilson unilateral al 95% alcance 40%. Un fallo rompe la racha; dos lecturas de los mismos episodios no cuentan dos veces. El calentamiento del crítico no cuenta como aprendizaje del actor.
- Tras una recepción propia válida, la conservación admite vuelos posteriores entre compañeros. Para aprobar exige volver a tener control físico propio después de dos segundos sin toque rival ni salida; una pelota suelta por sí sola no aprueba. No paga cada pase ni modifica las recompensas de los partidos completos.

Estos umbrales son hipótesis experimentales. Si la tarea no alcanza el criterio, permanece practicándola; no se avanza por haber agotado tiempo. Etapas 3 y 4 siguen aceptando goles legítimos sin un número obligatorio de pases, pero mantienen las filas de repaso anteriores.

Se conserva el modo de macro «pase listo» y el controlador corregido completo de `01c9e60`, así como KL 0,05, entropía 0,002, gamma 0,999 y GAE 0,98. Cambia el criterio curricular y se reinicia el estado del optimizador; no es una comparación que aísle exclusivamente el umbral.

## Ejecución y evaluación

Una sola corrida nueva, aislada en `/workspace/HaxballRL_coop_refine_20261007`, rama `codex/coop-refine-200-20261007`, salida `runs/x4_coop_refine_200_20261007`. El driver es `learn/x4_coop_refine.py`. Los logs empiezan en actualización local 0: **100 nuevas equivalen a 300 contando el origen**, no a haber perdido las primeras 200.

Presupuesto: hasta 600 actualizaciones adicionales, 512×128 decisiones por actualización, 39.321.600 decisiones nuevas, con límite de tres horas incluidas las evaluaciones. Diez actualizaciones iniciales preparan el crítico con el actor congelado para las nuevas tareas. Las copias de referencias se congelan con hashes: modelo de 200, original, control de 200 y BC. El pool incluye BC, el modelo de 200, el original y snapshots posteriores.

Cada 100 actualizaciones nuevas se evalúan partidos completos y cooperación. Además, el diagnóstico dev juega contra las referencias congeladas, con ocho partidos por lado de dos minutos y 64 estados por sonda de 30 grabaciones. Los resultados son simulados; ganar contra BC no es el criterio de cooperación ni valida juego contra humanos. Al terminar el presupuesto se evalúa test una sola vez, con 32 partidos por lado y 128 estados.

Si al menos dos de estas métricas caen materialmente frente al nuevo baseline en dos evaluaciones consecutivas, el driver detiene la corrida y conserva los checkpoints: pases/min menor al 80% del baseline; precisión cinco puntos porcentuales por debajo; retención general ocho puntos por debajo; retención bajo presión diez puntos por debajo. Este control de deterioro es conservador y aproximado, no una prueba de significación ni una garantía de mejora. La copia de 200 siempre queda disponible.

Estado, comandos y hashes: `run_state.json` y `manifest.json` en la salida nueva. El nuevo directorio también se enlaza en `/workspace/HaxballRL/runs/x4_coop_refine_200_20261007`.
