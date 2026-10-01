# RS4: saques, bloque dinámico y referencia humana específica

## Alcance y configuración recibida

La configuración recibida usa `runs/bc2/bc.pt`, una referencia generalista, y
PPO final LR 5e-5, tres épocas, gamma 0.995 y 1152 agentes. Se preservan estos
parámetros, pesos PPO, Adam, liga, dificultad y presupuesto de pasos.
El archivo original `Downloads/config.yaml` no se modificó.

- SHA256 config original: `F5344FC046343C2065670E13265186AD2792B167A219E92C0F3BFAE1BDB74452`.
- SHA256 bc2 original: `1A37EF7383E8D3524BBA2E55CC65D985151C900BFC596CB41A8B4DAB16B4460A`.
- Preview actualizado: `reports/rs4_config_v2_20261001.yaml`.

## 1. Córners y saques

No se reprodujo un robo a los 2–3 segundos en las reglas actuales. Sí se encontró
que el plazo vencido liberaba la pelota al rival sin terminar el episodio.
RS One 4v4 ahora termina el episodio al expirar un saque, mantiene la protección
durante los ticks restantes de la decisión y luego reinicia posiciones. No se
inventa un gol. La regla vale también para evaluación/replays sin shaping.

La configuración v2 añade -0.25 al equipo que no ejecutó y un potencial de
acercamiento de equipo, acotado a 0.05, usando al compañero más cercano. Gol y
fallo absorben el potencial; la truncación mantiene bootstrap. El bonus de córner
útil sigue requiriendo ejecución hacia dentro y continuidad. No se fuerza al
agente a correr/patear y no se añade estado privado a sus observaciones.

Las pruebas colocan al rival junto a la pelota intentando patear durante tres
segundos, verifican ambos colores/rutas y también el tick exacto de expiración.
Los logs añaden intentos/útiles/expiraciones por color para observar aprendizaje,
no sólo si ganó el partido.

## 2. Datos y referencia BC

Búsqueda ejecutada: `Rsx4`, mínimo 120 s, 30 candidatos, estadio `rs_one`, algún
tramo 4v4. Resultado: **22 descargados, 2 duplicados, 5 mapas incompatibles,
1 sin 4v4**. La primera conversión incremental añadió 921,332 muestras; no todas
las filas de una grabación tienen necesariamente el mismo formato.

Por eso se generó un dataset independiente `data/bc_rs4_v2`, filtrando cada tick
por estadio efectivo y 4v4, incluso si el administrador cambia el mapa:

- 60 replays / **2,763,832 decisiones humanas**.
- Entrenamiento: 50 replays, **2,333,880 muestras**.
- Validación: 10 replays, **429,952 muestras**.
- No se sobrescribió el dataset general ni se cargaron otros mapas como RS One.

BC específico se afina desde bc2, manteniendo arquitectura/normalizadores, con
LR 1e-4, cuatro épocas, batch 4096, act_lag6 y peso de keyframes 4.
Esto escribe un modelo BC separado, no el checkpoint PPO.

La referencia sólo debe cambiarse después de disponer del archivo en el Pod.
El `bc_reference` del preview se mantiene en bc2; la activación es explícita.
No usar `--init-from` en train.multitask para cargar la nueva referencia.

Artefacto final: `runs/bc_rs4_v2_20261001/bc.pt` (SHA256
`4C48D59DAAA59E34F2CDF8BFD05FAD5C689BA3E296FB3872F80BF7B1A3FCF55F`).
Comparación sobre las mismas 429,952 filas de validación del dataset limpio:

| Métrica | bc2 | BC RS4 final |
| --- | ---: | ---: |
| Loss | 1.52485 | 1.45728 |
| Acierto acción | 46.26% | 48.91% |
| Top 3 | 79.89% | 81.80% |
| Dirección | 46.83% | 49.51% |
| Patada | 98.14% | 98.15% |
| Acierto al cambiar decisión | 25.91% | 27.11% |

La mejora es principalmente movimiento, no patada. Esto no es un resultado de
winrate: el PPO de RS4 del Pod no se evaluó ni se sobrescribió. El artefacto
`bc_rs4_20261001` fue un sondeo previo sobre shards existentes; utilizar el final
`bc_rs4_v2_20261001`, generado con el dataset separado y filtros por tramo.

## 3. Roles v2

Funciones intercambiables: último hombre móvil, portador/presión, apoyo de balance
y amenaza de profundidad. GK sube con pelota avanzada y ventaja geométrica de
acceso; sin esa ventaja retrocede. El apoyo queda detrás y el delantero por
delante, con dos orientaciones laterales simétricas. Ningún jugador puede cubrir
dos funciones en la asignación. No hay IDs de rol ni puestos permanentes.

Estructura baja de 65% a 50% del potencial; amenaza/peligro sube de 35% a 50%.
Todavía es una heurística: no mide posesión real ni implementa memoria de
transiciones, presión orientada perfecta o feeds largos específicos del host.
Las pruebas demuestran simetrías, límites y preferencia por balance/profundidad
frente a que todos persigan la pelota, no calidad futbolística.

## Aplicación segura en el Pod

Detener guardando, actualizar el código y ejecutar:

```bash
python -m tools.upgrade_rs4 --config runs/rs4/config.yaml --renew-guide
python -u -m train.multitask --config runs/rs4/config.yaml --run rs4 --resume
```

El upgrade respalda config y activa formación v2, saques, salida defensiva reducida
y retención de checkpoints 5/30 min con 6 históricos. `--renew-guide` abre una fase
explícita 0.08→0 en 200M pasos desde latest: no renueva LR/entropía/Adam. Sin el
flag se mantiene el decay anterior, que vence en 4,026,734,336 pasos en el config
recibido. Si el presupuesto restante es menor a 200M, se avisa y no se amplía.
La versión de métricas RS4 cambia a 2 y limpia ventanas no comparables, no su r3.

Para copiar/recrear el imitador y activarlo, consultar los comandos completos del
README. Replays, datasets y pesos BC generados están ignorados por Git: un pull no
los transporta. Conservar parent.pt y comparar especialista/padre en semillas
comunes. Mejorar acierto de imitación no demuestra mayor winrate contra rivales.

## Verificación

Se ejecutó la suite completa y pruebas específicas de continuación, saques,
filtros de formato/mapa y guardado separado de BC. No se validó throughput CUDA
ni el juego del PPO especializado en el Pod, porque no hay acceso a ese runtime.
Resultado final: **532 pasadas, 42 omitidas**, seis warnings existentes del
exportador ONNX. Se verificó que los hashes de config original y bc2 no cambiaron.
