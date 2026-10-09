# Referencias y marcas en HaxEleven

Pedido del usuario (2026-10-08), en D2 y D3 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con el método de [`tools/hx11_marcas.py`](../../tools/hx11_marcas.py). Son 394 partidos y 2309232 muestras con la pelota del rival. Los datos están en [`d2_marcas.json`](d2_marcas.json).

## Cómo marcan los humanos

- **Posición del que marca** = γo·atacante + γb·pelota + γh·arco propio. Por equipo y partido, p50: γo 0.945, γb -0.009, γh 0.066 (p10–p90 de γo: 0.883–1.009). En básquet, Franks et al. encontraron cerca de 0,6 / 0,1 / 0,3.
- **Modelo de carrera:** reacción 15 ticks y vmax 2.86 px/tick, del ajuste de D3.

| Línea del que marca | Minutos defendiendo | Del lado del arco | Espalda ganada (habilitado) | Marcas perdidas cada 10 min | Margen p10 / p50 (ticks) | Misma marca (s, p50) |
|---|---|---|---|---|---|---|
| ARQ | 58.1 | 80.4% | 27.70% | 34.763 | -47.4 / 33.6 | 3.9 |
| DEF | 14297.8 | 86.5% | 19.39% | 41.007 | -20.5 / 35.5 | 4.3 |
| DEL | 7325.3 | 66.5% | 45.72% | 56.585 | -68.2 / 5.3 | 5.1 |
| MCD | 2408.6 | 88.2% | 17.72% | 31.615 | -17.6 / 44.0 | 3.7 |
| MED | 13912.7 | 79.4% | 29.66% | 42.08 | -44.7 / 27.2 | 4.4 |
| R0 | 23.6 | 73.3% | 36.91% | 61.755 | -69.4 / 20.8 | 3.7 |
| R1 | 32.3 | 79.4% | 31.56% | 31.248 | -77.1 / 29.9 | 4.4 |
| R2 | 27.4 | 74.2% | 35.17% | 38.707 | -71.6 / 24.3 | 4.3 |
| R3 | 26.5 | 71.7% | 38.42% | 46.722 | -68.3 / 18.0 | 4.2 |
| R4 | 27.0 | 70.9% | 40.66% | 35.512 | -75.7 / 11.6 | 4.5 |
| R5 | 25.4 | 62.3% | 48.89% | 41.404 | -94.3 / 0.9 | 4.7 |
| R6 | 22.1 | 76.9% | 30.96% | 49.661 | -50.5 / 22.3 | 4.1 |
| R7 | 27.5 | 80.6% | 28.06% | 30.898 | -55.4 / 32.4 | 4.2 |
| R8 | 24.2 | 71.2% | 37.54% | 37.18 | -81.5 / 15.0 | 4.4 |
| R9 | 11.5 | 75.2% | 36.60% | 86.806 | -64.2 / 22.9 | 3.8 |

## ¿Marca personal o zonal?

El HMM elige casi siempre al atacante más cercano (γo alto), así que no alcanza para esto. Se fija la referencia al comienzo de ventanas de 3 s y se mide, respecto del resto del bloque, cuánto la acompaña el defensor: dd = βa·da + βb·db. βa ≈ 1 es personal y βa ≈ 0 es zonal. Las tres columnas de la derecha cuentan, cuando la referencia se movió al menos 150 px respecto del bloque, quién es al final el defensor más cercano a ella: el mismo, otro, o ninguno a 150 px (libre).

- **Por equipo y partido** (788): βa p10 / p50 / p90 = 0.211 / 0.283 / 0.351.

| Línea | Ventanas | βa (todos) | βa por jugador p10 / p50 / p90 | βb (pelota) | Referencia movida | Sigue con ella | La toma otro | Libre |
|---|---|---|---|---|---|---|---|---|
| ARQ | 1503 | 0.197 | -0.078 / 0.166 / 0.478 | -0.004 | 745 | 13.8% | 35.3% | 50.9% |
| DEF | 338770 | 0.309 | 0.135 / 0.305 / 0.481 | -0.016 | 129652 | 24.8% | 33.6% | 41.6% |
| DEL | 174167 | 0.284 | 0.113 / 0.286 / 0.471 | 0.004 | 71317 | 24.2% | 21.7% | 54.0% |
| MCD | 56604 | 0.223 | 0.073 / 0.205 / 0.407 | 0.027 | 19980 | 19.0% | 41.0% | 40.0% |
| MED | 330107 | 0.251 | 0.083 / 0.241 / 0.432 | 0.015 | 130392 | 21.1% | 29.0% | 49.9% |
| R0 | 637 | 0.193 | 0.154 / 0.167 / 0.291 | -0.003 | 335 | 17.9% | 23.6% | 58.5% |
| R1 | 859 | 0.244 | 0.155 / 0.229 / 0.401 | 0.027 | 469 | 17.9% | 26.7% | 55.4% |
| R2 | 720 | 0.212 | 0.123 / 0.285 / 0.367 | -0.014 | 364 | 18.1% | 26.4% | 55.5% |
| R3 | 709 | 0.263 | 0.139 / 0.202 / 0.412 | -0.001 | 381 | 16.3% | 28.1% | 55.6% |
| R4 | 716 | 0.272 | 0.096 / 0.286 / 0.389 | 0.014 | 374 | 18.7% | 25.1% | 56.1% |
| R5 | 660 | 0.187 | 0.061 / 0.146 / 0.323 | 0.037 | 332 | 16.3% | 28.6% | 55.1% |
| R6 | 566 | 0.291 | 0.086 / 0.303 / 0.372 | 0.032 | 301 | 20.9% | 25.9% | 53.2% |
| R7 | 714 | 0.254 | 0.131 / 0.207 / 0.483 | 0.025 | 351 | 15.4% | 33.3% | 51.3% |
| R8 | 615 | 0.461 | 0.307 / 0.44 / 0.556 | -0.028 | 335 | 13.1% | 23.6% | 63.3% |
| R9 | 267 | 0.334 | — | -0.044 | 122 | 16.4% | 24.6% | 59.0% |

## ¿La marca perdida anticipa peligro?

- **Marcas perdidas:** 167506. En los 5 s siguientes: gol 1.4%, remate (sin gol) 4.8%, el atacante recibe a la espalda 3.4%.
- **Muestras al azar** de la posesión rival (31520): gol 1.1%, remate 3.7%.
- **Remate o gol:** 6.2% contra 4.7% al azar (× 1.3).
- **Quién habilita:** la línea (otro defensor más atrasado) 153772, el mismo que marca 13734.
- **xG para el peligro:** logística de distancia y ángulo con 2463 remates de train (tasa de gol 0.1567).
