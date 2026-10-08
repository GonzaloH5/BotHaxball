# S1: estadio y física base de HaxEleven (versión actual)

Fecha: 7 de octubre de 2026. Etapa S1 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md).
- Corrió en local, sin pod.
- Datos completos: [`s1_fisica.json`](s1_fisica.json).

```
node bridge/hx11_ticks.js <grabación.hbr2> --out data/hx11/ticks --max-minutes 3 [--skip-minutes 4]
python -m tools.hx11_conformance fisica data/hx11/ticks/*.jsonl --align t --out reports/hx11/s1_fisica.json
python -m tools.hx11_conformance bench
```

## Estadio: `stadiums/haxeleven_x11.hbs`

- **Origen:** la exportación `5fe7e10f9c62` de "HaxEleven RSx11 [1800, 1200]", que es la de las grabaciones del 18 de septiembre al 5 de octubre (91 partidos).
- **La anterior** (`1048c51a0c1f`, 286 partidos) es idéntica salvo por la masa del jugador, que no estaba declarada y valía la de HaxBall, 0,5.
- **Otras tres variantes cortas** (2 a 18 partidos) cambian la aceleración, el damping y la patada del jugador. No se usan.

| | |
|---|---|
| Líneas de la cancha | x = ±1800, y = ±1200. Son solo dibujo: la salida la cobra el script |
| Arcos | línea en x = ±1810, medio arco 170. Postes en (±1800, ±172) con radio 5 y bCoef 1,1. Red hasta x = ±1908 |
| Límites de la pelota | planos en y = ±1238,14 y x = ±2300 |
| Límites del jugador | planos en x = ±2150 e y = ±1459,86 (y −1659,86 arriba) |
| Área grande (dibujo) | de x = ±1535 a ±1800, \|y\| ≤ 350 |
| Zonas de barrera c0 y c1 | de x = ±1050 a ±1800, \|y\| ≤ 675, con arco. El script las prende en el cGroup de los rivales, probablemente en los saques de arco |
| Línea c2 y c3 | x = ±1785, \|y\| ≤ 174. Solo choca con el disco 1 |
| Saque inicial | línea central y círculo de radio 335 (redKO/blueKO) |
| Pelota | radio 8, bCoef 0,2, invMass 1,65 (el script la cambia con la carga), damping 0,99 |
| Jugador | radio 15, invMass 0,69, bCoef 0,05, aceleración 0,119, aceleración al patear 0,1, damping 0,96, patada 5 |
| Puntos de saque | 12 por equipo, en x ≈ ±900. El 12.º está en (±150, 1300), fuera de la cancha. El script después lleva al arquero a x = ±1760 (S2) |

### Discos del mapa

| Disco | Qué es | En el simulador |
|---|---|---|
| 1 | barrera de la pelota parada. Arranca en (0, 9999) con radio 50; el script le da radio 373 y la limita a un equipo (cMask rojo o azul) | se conserva (`keep_discs`) |
| 2 | disco del punto del saque, radio 17,5; choca con los dos equipos | se conserva (`keep_discs`) |
| 3–18 | postes, esquinas de la red y banderines (estos solo chocan con la pelota) | sí |
| 19–22 | invisibles, sin choque | no |
| 23–27 | adornos atados a la pelota con joints (masa nula). El **26** es el anillo que cambia de color con la carga del powershot | no |
| 28 | **barra de fuerza**: disco sin fricción que rebota entre el 29 y el 30 (y = ±32). Su posición al patear define la potencia | no (`ignore_discs`); se modela en S3 |
| 29–46 | extremos y marcas de la barra, de radio 0 | no |

`sim/stadium.py` acepta ahora `"haxballrl": {"keep_discs": [...]}`, para discos que arrancan fuera de la cancha y el script mete en juego. `tools/import_stadium.py` acepta `--keep-discs` e `--ignore-discs`.

## Física en juego abierto

**Prueba:**
- 8 grabaciones de la versión actual, de sesiones distintas de entrenamiento y de distintos momentos del partido: 3 min cada una, 11v11.
- Se carga el estado grabado: posiciones, velocidades y **las propiedades que dejó el script** en cada disco (masa de la pelota por la carga, masa, bCoef, cMask y cGroup de los jugadores, barreras, gravedad de la pelota).
- Se avanza con las entradas grabadas y se aplican, tick a tick, las propiedades grabadas.
- Se excluyen las transiciones en que el script coloca un disco (494), los goles y cambios de plantel (12) y los tramos con gravedad en un jugador.

| Horizonte | Tramos | Pelota p50 / p90 / máx (px) | Jugadores p50 / p90 / máx (px) |
|---|---|---|---|
| 1 tick | 74.523 | 0,0000 / 0,0001 / 0,000 | 0,0001 / 0,0001 / 0,001 |
| 10 ticks | 24.117 | 0,0003 / 0,0006 / 0,050 | 0,0005 / 0,0006 / 0,100 |
| 30 ticks | 6.827 | 0,0010 / 0,0026 / 0,894 | 0,0011 / 0,0012 / 1,808 |
| 60 ticks | 3.173 | 0,0021 / **0,0140** / 2,969 | 0,0015 / **0,0030** / 1,554 |
| Saque inicial, 60 ticks | 306 | 0,0000 / 0,0000 / 0,015 | 0,0014 / 0,0018 / 0,129 |

- **Criterio del plan cumplido:** p90 a 60 ticks ≤ 0,02 px, del orden de RS (0,009–0,012).
- **1 tick:** el error es el redondeo a 4 decimales del archivo.
- **Máximos de 30 y 60 ticks:** ese redondeo amplificado en choques, que es caos y no una regla que falte.
- **Propiedades del script:** el 19% de las transiciones (14.440) tiene un cambio, y se reproducen aplicándolas.

### Dos detalles que hicieron falta

1. **Las propiedades del tick t valen para la transición t→t+1.** El script actúa al final de cada tick y la física siguiente ya usa lo que dejó. Con la otra alineación, la pelota se desvía hasta 12 px en un tick justo cuando cambia su masa (la carga del powershot).
2. **Los choques dependen del orden de los discos.** HaxBall resuelve los choques de a pares en el orden de creación de los discos. Con tres jugadores en contacto (cuerpear), ordenar los rojos primero daba hasta 0,07 px por tick. El replay usa ahora el orden del juego.
   - **[inferencia]** En el entrenamiento cualquier orden fijo es válido: cambia el detalle del choque, no su distribución.

## Velocidad del motor (esta PC, 16 hilos, 256 partidos, 600 ticks)

| | Ticks de partido/s | Ticks de jugador/s |
|---|---|---|
| 11v11 en `haxeleven_x11` | 415.535 | 9,14 M |
| 4v4 en `sanguchito_rs_x4` | 1.133.574 | 9,07 M |

- Por partido, 11v11 es 2,7 veces más lento que 4v4. Por jugador cuesta lo mismo.
- La estimación del plan ("unas 7 veces más pares de colisión") era pesimista para el costo por decisión.
- Es solo física: faltan el árbitro, las observaciones y la política.

## Lo que el simulador todavía no hace (S2–S3)

- **El script.** La conformidad usa las propiedades y colocaciones grabadas. El simulador tiene que generarlas:
  - carga del powershot (masa y color de la pelota);
  - masa 999999 y cGroup de los jugadores en los saques;
  - barreras (disco 1 de radio 373 contra un equipo, zonas c0 y c1);
  - colocación de la pelota y de los jugadores;
  - gravedad de la pelota (comba) y de los jugadores (unas 15 veces por minuto).
- **La barra de fuerza:** el oscilador de los discos 28–30, con extremos de radio 0, o un modelo de su período.
- **Puntos de saque:** el 12.º jugador y la colocación del arquero en el saque inicial.
