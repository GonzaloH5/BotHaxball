# Auditoría del dataset X4 (grabaciones humanas 4v4)

2026-10-06. Dataset de la rama `reinicio` (commit `c42f905`, "Add real X4 replays"): 649 archivos en `replays_real/stadiums/rsx4` y 2 en `haxarg2k23`.

## Cómo se procesó

| Paso | Herramienta | Salida |
|---|---|---|
| Conversión tick a tick con el motor original (node-haxball) | `tools.rs4_jsonl_cache` | `data/rs4_jsonl/` (no versionado), 645 OK, 4 duplicados exactos por SHA-256 |
| Metadatos: sala, estadios, planteles con nombres, goles | `bridge/replay_meta.js` | `data/meta/` (no versionado) |
| Caché compacto por tick: pelota, jugadores, entradas, saques, patadas, saque inicial, masa | `tools.x4_ticks` | `data/x4_ticks/*.npz`, 2,4 GB (no versionado) |
| Índice, duplicados, partidos con bots, sesiones y particiones | `tools.x4_index` | `reports/x4/index.json`, `reports/x4/splits.json` |
| Métricas humanas y techo prueba-vs-entrenamiento | `tools.x4_metrics` | `reports/x4/human_metrics_sanguchito.json` |

Para regenerar todo desde las grabaciones, en este orden:

```bash
python -m tools.rs4_jsonl_cache --source replays_real/stadiums/rsx4 --out data/rs4_jsonl --workers 4
python -m tools.rs4_jsonl_cache --source replays_real/stadiums/haxarg2k23 --out data/haxarg_jsonl
ls replays_real/stadiums/*/*.hbr2 | tr '\n' '\0' | xargs -0 node bridge/replay_meta.js > data/meta/shard_00.jsonl
python -m tools.x4_ticks --out data/x4_ticks --workers 4
python -m tools.x4_ticks --cache data/haxarg_jsonl --out data/x4_ticks
python -m tools.x4_index --out reports/x4
python -m tools.x4_metrics --map sanguchito_rs_x4 --out reports/x4/human_metrics_sanguchito.json
```

## Contenido

| Familia de sala | Grabaciones | Minutos 4v4 | Goles | Jugadores distintos |
|---|---|---|---|---|
| SANGUCHITO RS X4 (sala pública, mapa objetivo) | 498 | 2.822 en SANGUCHITO RS X4 | 1.693 | 942 |
| HAXARG (liga, Real Soccer ONE) | 123 (+2 pruebas con bots en 2K23) | 2.332 en RS ONE | 720 | 756 |
| MrHOST (liga) | 22 | 392 en RS ONE | 117 | 186 |
| Otras | 2 | 21 en RS ONE | 12 | 34 |

- En total son unas 93 horas de 4v4 humano y 1.750 nombres de jugador distintos.
- Los nombres se guardan solo en `data/x4_player_names.json`. El índice versionado usa claves `p_<sha1>`.
- **Mapa objetivo:** antes había 7 grabaciones de Sanguchito y ahora hay 498. Sanguchito pasa a ser la fuente principal de imitación del mapa objetivo.
- **Variantes:**
  - Sanguchito: siempre patada 5,85 y pelota 8,325.
  - RS ONE: patada 5,85 en 125 grabaciones y 5,75 en 22; pelota 8,325 en 110 y 8 en 37.
  - 2K23: patada 5,65 y pelota 9.
- **Excluidos:**
  - 4 duplicados exactos (copias "(1)").
  - 1 grabación de Sanguchito sin tramos en el mapa: es una tanda de penales.
  - Los tramos en Classic, Real Soccer Revolution y Training RSR.
  - Las 2 grabaciones de HAXARG 2K23 del 2026-10-06, que son las pruebas con 7 bots (`RL-*`). Sirven para conformidad del script y para latencia, no como datos humanos.
- **Duplicados del mismo partido grabado por otra persona** (misma secuencia de goles y ≥ 6 nombres en común): no hay ninguno. Los pares de `HBReplay` con la misma hora son partidos simultáneos de la liga, con goles distintos.

## Particiones (`reports/x4/splits.json`)

- Las particiones son por bloque de sesión: misma sala y mismo día, con menos de 45 min entre grabaciones y cortado en bloques de hasta 2 h.
- La asignación es estable por hash y por familia: ~15% prueba, ~10% desarrollo y el resto entrenamiento.

| | Entrenamiento | Desarrollo | Prueba |
|---|---|---|---|
| Sanguchito: grabaciones / min 4v4 | 362 / 2.072 | 59 / 307 | 77 / 443 |
| RS ONE (HAXARG + MrHOST + otras) | 105 / 2.001 | 17 / 331 | 25 / 412 |

- La sala es pública y los habituales se repiten: 311 de los 504 jugadores de prueba también aparecen en entrenamiento.
- Las particiones miden generalización a partidos nuevos, no a jugadores nuevos.

## Referencia humana en Sanguchito (`reports/x4/human_metrics_sanguchito.json`)

Medida en tramos 4v4, en juego abierto, muestreando cada 3 ticks. Split de entrenamiento: 1.801 min de tramos de al menos 9 s. Se excluyen los ticks con posiciones nulas del arranque de algunos replays (~630 ticks al principio de muchas grabaciones de Sanguchito).

| Métrica | p10 | p50 | p90 |
|---|---|---|---|
| Pases por minuto (por tramo) | 2,8 | 8,8 | 13,5 |
| Largo de pase (px) | 142 | 335 | 632 |
| Duración de la posesión (s) | 0,2 | 2,45 | 7,4 |
| Pases por posesión | 0 | 0 | 2 |
| Profundidad del equipo (px) | 275 | 464 | 826 |
| Ancho del equipo (px) | 182 | 305 | 478 |
| Distancia a la pelota del 1.º / 4.º más cercano (px) | 24 / 299 | 84 / 474 | 214 / 852 |
| Fracción de jugadores parados (por tramo) | 0,7% | 2,6% | 10% |
| Cambios de tecla por segundo y jugador | 1,4 | 3,0 | 3,7 |
| Espera del saque inicial (s) | 2,75 | 3,85 | 7,3 |
| Duración del lateral, desde la colocación hasta la liberación (s) | 1,4 | 2,4 | 4,3 |

- **Techo** (W1 normalizada por el desvío, prueba contra entrenamiento): ≤ 0,05 en forma de equipo, posesión y largo de pase; 0,09 en goles/min; 0,10 en la espera del saque inicial; 0,12–0,13 en fracción de parados y sin dirección; 0,20–0,22 en cambios de tecla por segundo y pases/min (son métricas por tramo, con pocas muestras).
- Una política "humana" en simulación tendría que quedar cerca de esos valores con latencia de sala.

## Demora de colocación de los saques (`pending` en el caché)

Ticks entre la salida de la pelota y la colocación del script:

| Mapa | Lateral | Córner | Saque de arco | Contrato (`env/rs4z/contract.py`) |
|---|---|---|---|---|
| Sanguchito | 0 en el 91% | 0 en el 100% | 0 en el 100% | sin demora |
| RS ONE | p10 1, p50 2, p90 8 | p10 6, p50 28, p90 60 | p10 7, p50 32, p90 62 | uniforme [0,8], [5,60] y [6,62] |
| 2K23 (n = 12/3/1, pruebas con bots) | p50 38 | p50 24 | 21 | [15,60], [5,60], [6,62] |

El contrato coincide con lo medido. En RS ONE, la demora real del córner se concentra algo más abajo que la uniforme (mediana 28 contra 32,5). Es una mejora menor que no hace falta para el mapa objetivo.
