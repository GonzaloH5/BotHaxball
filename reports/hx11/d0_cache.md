# D0: caché por tick de HaxEleven

Etapa D0 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con [`tools/hx11_d0.py`](../../tools/hx11_d0.py). Estadio "HaxEleven RSx11 [1800, 1200]": 399 partidos, 88.4 h. Un archivo por partido en `data/hx11/d0/` (no versionado, sin nombres en el resumen).

- **Precisión de la cuantización:** posiciones a 0,05 px y velocidades a 0,0005 px/tick.
- **11 contra 11 completo,** parte de los ticks por partido: p10 0.848, p50 0.988, p90 1.0.
- **Arquero declarado con `!gk`** (el peor de los dos equipos), parte de los ticks: p10 0.0, p50 0.0, p90 1.0. 149 partidos no tienen ningún `!gk` (versiones del script sin ese comando).
- **Arquero inferido** donde no hay declarado: el más cercano a su arco, el más frecuente en bloques de 1 min. Coincide con el declarado en el 98.8% de los ticks que tienen declarado. Arquero conocido (declarado o inferido), el peor equipo: p10 1.0, p50 1.0.

| Parte | Partidos | Horas | Horas 11v11 | Decisiones de la sala | Patadas |
|---|---|---|---|---|---|
| dev | 41 | 8.4 | 7.7 | 1785 | 11554 |
| test | 53 | 11.5 | 11.0 | 2412 | 16382 |
| test_grupos | 12 | 2.6 | 2.0 | 517 | 3985 |
| train | 293 | 65.9 | 61.2 | 14097 | 92904 |

**Fases (parte de los ticks):** juego abierto 73.4%, salida pendiente 1.5%, pelota parada 19.8%, congelamiento 0.1%, gol 0.5%, saque inicial 4.8%
