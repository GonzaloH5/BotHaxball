# Extensión desde 800 — 2026-10-08

Autorizada por el usuario después de terminar el tramo 600–800 y su evaluación DEV. El modelo de 800 obtuvo 4 victorias, 12 empates y 0 derrotas contra el de 600 en 16 partidos. Frente al de 200 tuvo 0 victorias, 12 empates y 4 derrotas. Son resultados DEV simulados; todavía no recuperó el nivel de la referencia de 200.

Continúa desde 800 hasta **1200**, con comparaciones DEV en 900, 1000, 1100 y 1200, y límite de dos horas. Si el control detecta deterioro material en dos evaluaciones consecutivas, se detiene y conserva los checkpoints.

Se mantienen las recompensas, los macros corregidos, las cuotas de tareas y los hiperparámetros. Se preservan actor, crítico, Adam, normalización, liga y evidencia de dominio del currículo. La etapa principal permanece en 2, con reloj de aprendizaje 590 y sin promoción forzada. La extensión no reinicia ese reloj ni exige empezar a acumular las mismas habilidades de nuevo. Se inicia una nueva serie de evaluaciones y se mantienen además los controles acumulativos contra el baseline 600.

Las referencias de 600 y 800 se copian como checkpoints completos de sólo lectura en la corrida nueva. BC, 200 y original mantienen sus rutas exactas para cumplir las reglas de reanudación del trainer. Los partidos se comparan contra 600, 800, 200, original, BC y scripted. El control observa tanto el nuevo baseline como el de 600; extender varias veces no debe ocultar una pérdida acumulativa de pases.

Mientras siga en etapa 2, la distribución es: 50% partidos completos, 25% salida de presión, 12,5% repaso de pase y recepción y 12,5% definición tras recepción humana. El manifiesto registra también la distribución que corresponde a otras etapas, por si cumple el criterio de avance. Las ventanas de maestría de la etapa principal no incluyen los éxitos de los ejercicios de definición.

La alerta que se sigue observando es concreta: la salida de presión quedó cerca del 37% de éxitos y el repaso de pases bajó. La definición mejoró hasta aproximadamente 12% de éxitos, pero se estabilizó al final del tramo. La continuación examina si esa recuperación puede trasladarse al juego completo y sostener los pases; avanzar más updates no garantiza lograrlo.

Código en el pod: `/workspace/HaxballRL_coop_refine_20261007`.

Corrida nueva: `runs/x4_coop_finish_800_20261008`.

```bash
tail -f /workspace/HaxballRL/runs/x4_coop_finish_800_20261008/curriculum.out
```

El directorio conserva `manifest.json`, `run_state.json`, `curriculum/log.jsonl`, `curriculum/eval.jsonl`, referencias congeladas y los informes `evaluation_dev_00900.json` hasta `evaluation_dev_01200.json`. La corrida anterior queda terminada y conserva sus registros. No se reutiliza TEST para decidir esta extensión ni se exporta o despliega automáticamente.
