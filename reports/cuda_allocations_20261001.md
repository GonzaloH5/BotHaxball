# CUDA: reutilización de asignaciones, 01/10/2026

## Punto de partida del Pod

Último benchmark aportado, mismo checkpoint, Torch 2, Numba 4, warmup 3,
15 iteraciones medidas y 97.920 muestras por iteración:

| Ruta | Pasos/s reales | Rollout | Prep | Update | Captura |
|---|---:|---:|---:|---:|---:|
| Callbacks referencia | 45.378 | 1,683 s | 0,063 s | 0,380 s | 0,158 s |
| Callbacks compilados | 48.374 | 1,501 s | 0,061 s | 0,427 s | 0,175 s |

Ganancia observada +6,6%, rollout -10,8%. Comparación secuencial, no invertida.
Las medias impresas de PPO coinciden; el update empeoró en la segunda medición,
compatible con ruido pero no demostrado. El rollout ocupa ~74% del total.

## Cambios de esta petición

1. Cada rollout sigue recapturando su forward determinista. Mantener el graph
   anterior como propietario del pool evita liberar/recrear esa reserva; el nuevo
   graph captura en el mismo stream y pool. El anterior jamás se ejecuta otra vez.
   Los tres warmups siguen ocurriendo en cada captura. `reset` doble preserva el
   propietario hasta sustituirlo. El fallback auto libera el propietario; errores
   graves siguen propagándose. Buffers de RunningNorm pueden seguir reemplazándose.
2. Observaciones de minibatch CUDA: `index_select(..., out=buffer)` por update,
   sin modificar el orden de filas ni permutaciones. Sólo para entradas sin
   gradiente. El backward y siguiente gather se ejecutan en el mismo stream, en
   ese orden. CPU mantiene la optimización anterior. No se captura Adam/backward.
3. Perfil opcional de update con eventos GPU, además de tiempos host. No añade
   synchronize por minibatch. Lee eventos tras la descarga ya existente de las
   métricas. Eventos se crean sólo al perfilar; tiempos GPU pueden incluir huecos
   entre lanzamientos. Estadísticas posteriores a Adam no pertenecen a esas cinco
   fases; el tiempo total de update sigue siendo la referencia.

Referencia de seguridad del pool: documentación oficial de PyTorch,
[Sharing memory across captures](https://docs.pytorch.org/docs/stable/notes/cuda.html#sharing-memory-across-captures).
No hay replays concurrentes, y ningún consumidor conserva los outputs del graph
para el siguiente rollout. Se puede mantener más VRAM reservada entre iteraciones.

## Verificación y límites

Tests locales cubren la conservación de stream/pool y la renovación del graph,
doble reset, warmup sin omitir, PPO de MLP/Set con y sin BC, lotes con cola y
menores que minibatch, dos updates, pérdidas, parámetros, estado Adam, RNG e
inmutabilidad del lote. Pruebas CUDA estrictas adicionales comprueban cambios
de pesos/normalizadores/rivales y captura con tamaños variables.

Esta PC no dispone de CUDA: tests GPU se omiten aquí. No se declara mejora de
velocidad del Pod para estos cambios. Los comandos de validación y comparación
están en README. Ambos cambios se pueden desactivar por separado; no se alteró
PPO, modelo, física, recompensas, currículo, muestras ni checkpoints. No se
desplegó ni reinició el entrenamiento remoto.

Suite completa local: **470 passed, 42 skipped**, seis avisos ONNX preexistentes,
51,69 s. Las pruebas CUDA omitidas no se consideran validadas.
