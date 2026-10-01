# Arranque nocturno GPU, 01/10/2026

## Datos aportados y límites

Pod: RTX 3060 Ti de 8 GB, Xeon E5-2630 v4 con 10 hilos asignados,
15 GB RAM y 30 GB disco. El log aportado muestra 36–42k pasos/s de las
tres fases principales, rollout 2,1–2,3 s, prep ~0,07 s, update 0,3–0,5 s,
etapa C y ~2.919 millones de pasos. No es un benchmark de punta a punta.
La GPU alcanza 100% por momentos y vuelve a ~20%; CPU ~30%. Esta alternancia
no demuestra que sea seguro ampliar el lote ni que haya una ganancia concreta.

La medición anterior CPU con Torch 8/física 4 dio 6.876 pasos/s reales,
106.496 filas/iter, rollout 5,734 s, prep 0,167 s, update 9,566 s. Se adopta
8/4 en `config_cpu.yaml`, pero no se extrapola como óptimo del Xeon/CUDA.

## Optimización técnica y operación

`runtime.cache_bc_logits` permite reutilizar log-probabilidades de la referencia
BC congelada durante las épocas PPO también en CUDA. Sólo vive durante un update;
se reconstruye con cada lote, sin gradientes en el profesor. Es la misma fórmula
KL y los mismos parámetros PPO; posibles redondeos no implican identidad bit a bit.
Para 106.496 filas x 18 acciones float32 son ~7,3 MiB de VRAM adicional. Conserva
graph, transferencias pinned y buffers PPO ya existentes. Sin BC o con una época
no crea cache. `--no-cache-bc-logits` desactiva ambos flags al comparar.

El perfil se configura para guardar cada 250 iteraciones, no cada 25, y omite
replays automáticos que competían con entrenamiento. Los snapshots internos
de liga siguen cada 25 iteraciones. No borra checkpoints previos ni cambia
garantías del guardado atómico; ante una caída puede perderse más trabajo reciente.

`tools.launch_gpu_overnight` comprueba CUDA y que exista `latest.pt`, congela
checkpoint/config en un directorio temporal, mide 4/6/8/10 hilos dentro del cupo
y confirma finalista/baseline en orden inverso. Conserva baseline si la mejora
confirmada no llega al 3%. Sondeos cortos (2 warmup + 3 iteraciones por defecto):
selección práctica, no un óptimo demostrado. La copia temporal se descarta y
se continúa el original con `--resume`. Un error de sondeo impide lanzar entrenamiento.
No usa AMP/TF32, no aumenta entornos ni cambia tamaño de minibatch/épocas.

## Cambios deliberados de entrenamiento

Los logs de Futsal 2v2/3v3 muestran 100% de puntos en R3; 3v3 tiene evidencia
de >=128 partidos, mientras otros mapas aún tienen pocas evaluaciones o no dominan
R2. No se fuerza el ascenso con 6–9 victorias ni se interpreta un Elo por goles
como calidad contra humanos.

En C, se asigna 75% del presupuesto de agentes a Futsal (antes 65%):
2v2 10%, 3v3 25%, AF 10%, 4v4 10%, 5v5 10%, 7v7 10%.
Big mantiene 5%, X1/AHA pasan a 2,5% cada uno y RS4 a 15%; JJRS sigue excluido.
El 75% es presupuesto de agentes, no necesariamente 75% de filas de pérdida.

R2 eleva scripted de 5% a 15% con el mismo rival sin ruido para acumular partidos;
mantiene promoción por 128 partidos y dos evaluaciones. Futsal R3 con >=95% de
puntos y >=128 partidos reduce scripted a 5%, self-play a 25% y liga a 70%.
La histéresis restaura 15/40/45 si cae por debajo de 90%. La ventana de partidos
y el dominio se conservan en checkpoint y se invalidan si cambia el rival/contexto.
Los rivales ya asignados permanecen hasta que termina el partido.

La liga mantiene PFSP y aumenta su bonus relativo al miembro más reciente de
0,25 a 0,50. No asegura que ese rival sea más fuerte: el objetivo es ampliar
exposición al juego del aprendiz y evitar gastar tantos partidos en un scripted fácil.
Se mantiene muestreo estocástico, no oponentes con información privilegiada.

Se corrige la ventana auxiliar de goles, que antes no recibía nuevos goles en
la rama de promoción por puntos y parecía congelada. El log añade KL, clipfrac,
partidos necesarios y estado de dominio; no cambia las reglas del partido.

## Continuación y duración

`total_steps` sube a 5.000M. Nuevo `schedule_steps=3.000M` desacopla el límite
de ejecución del horizonte de LR/entropía/BC, para no subir estos coeficientes
al extender el entrenamiento. Pasado el horizonte se usan sus valores finales,
sin extrapolaciones negativas. También soportado por el entrenador recurrente.
La referencia BC efectiva del config guardado se conserva, incluida su ausencia;
un override explícito al ejecutar `train.multitask` puede reemplazarla.

Desde 2.919M quedarían ~14,5 h a 40k pasos/s **reales** constantes. El nuevo
reparto usa equipos mayores y más rivales de liga; esto cambia costo de física y
filas útiles por iteración. Por tanto ni esa duración ni >40k/s están garantizados.
Tampoco se garantiza mejora táctica simplemente por entrenar más horas.

## Verificación

Suite completa local: `python -m pytest tests -q`: **413 passed, 29 skipped**
en 46,09 s; seis avisos de deprecación del exportador ONNX existente.
Las omisiones corresponden a casos CUDA no disponibles en esta PC.
`git diff --check` sin errores de espacios y ayuda del lanzador verificada.

Las pruebas cubren equivalencia de pérdida/parámetros con y sin cache BC,
refresh del profesor por lote, continuación del modelo y Adam exactos,
restauración de liga y BC personalizados/ausentes, avance más allá de 3.000M,
límites de los decaimientos, gates e histéresis y rivales estables por partido.
El lanzador se prueba con procesos simulados: cupo de hilos, copia de config,
confirmación contra ruido, descarte del temporal y fallo sin arranque accidental.
Incluye casos CUDA que se omiten automáticamente en la PC sin GPU.

No hay acceso a ejecutar/medir en el Pod: las mediciones reales las realiza
el lanzador allí al arrancar. Estas pruebas no constituyen validación de velocidad
ni del funcionamiento CUDA de esta GPU concreta.
