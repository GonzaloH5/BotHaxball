# E4: potencia estadística

Auditoría del 2026-10-08 (§9.2, E4), con [`tools/hx11_potencia.py`](../../tools/hx11_potencia.py) sobre 394 partidos aptos.

## Goles

- **Desvío de la diferencia de gol por partido:** 1.33.
- **Partidos por brazo** para no-inferioridad con margen δ (α = 0,05 a una cola, potencia 0,8):

| δ (goles por partido) | Partidos por brazo |
|---|---|
| 0.1 | 2179 |
| 0.2 | 545 |
| 0.3 | 243 |
| 0.5 | 88 |

Por goles, "no es carga" solo se puede medir en el simulador, y con cientos o miles de partidos por celda. En la sala se mide con métricas de proceso.

## Métricas de proceso por línea del puesto

**Partidos del bot:** los que hacen falta para que el IC 90% de su media mida a lo sumo 1/4 del rango humano p10–p90. El ruido se toma de la variación de un mismo jugador entre partidos (mediana del desvío, jugadores con ≥ 6 partidos).

| Línea | Métrica | Jugadores-partido | p10 / p50 / p90 | Desvío del mismo jugador | Partidos del bot |
|---|---|---|---|---|---|
| ARQ | con_pelota_10min | 775 | 8.709 / 15.143 / 23.327 | 5.226 | 23 |
| ARQ | pases_10min | 775 | 4.21 / 7.967 / 12.829 | 3.093 | 23 |
| ARQ | perdidas_10min | 775 | 0.0 / 0.0 / 2.195 | 0.994 | 36 |
| ARQ | patadas_min | 775 | 0.698 / 1.191 / 1.805 | 0.402 | 23 |
| ARQ | pase_completo | 657 | 0.333 / 0.588 / 0.833 | 0.186 | 25 |
| DEF | con_pelota_10min | 2895 | 9.795 / 18.652 / 31.825 | 6.722 | 17 |
| DEF | pases_10min | 2895 | 6.549 / 12.565 / 20.954 | 4.376 | 16 |
| DEF | perdidas_10min | 2895 | 0.0 / 2.062 / 5.901 | 2.311 | 27 |
| DEF | patadas_min | 2895 | 0.742 / 1.398 / 2.33 | 0.504 | 18 |
| DEF | pase_completo | 2773 | 0.267 / 0.5 / 0.695 | 0.151 | 22 |
| DEL | con_pelota_10min | 1486 | 8.389 / 16.879 / 28.241 | 6.3 | 18 |
| DEL | pases_10min | 1486 | 4.576 / 9.881 / 16.807 | 3.895 | 18 |
| DEL | perdidas_10min | 1486 | 0.0 / 2.929 / 7.55 | 2.726 | 23 |
| DEL | patadas_min | 1486 | 0.531 / 1.077 / 1.797 | 0.404 | 18 |
| DEL | pase_completo | 1314 | 0.2 / 0.429 / 0.667 | 0.159 | 21 |
| MCD | con_pelota_10min | 488 | 14.051 / 24.219 / 37.825 | 7.176 | 16 |
| MCD | pases_10min | 488 | 9.053 / 16.464 / 24.931 | 5.598 | 22 |
| MCD | perdidas_10min | 488 | 1.064 / 3.579 / 8.682 | 2.617 | 21 |
| MCD | patadas_min | 488 | 0.949 / 1.68 / 2.585 | 0.526 | 18 |
| MCD | pase_completo | 477 | 0.3 / 0.5 / 0.667 | 0.11 | 16 |
| MED | con_pelota_10min | 2812 | 12.148 / 21.905 / 34.78 | 7.145 | 18 |
| MED | pases_10min | 2812 | 7.391 / 13.84 / 22.442 | 4.708 | 17 |
| MED | perdidas_10min | 2812 | 0.984 / 3.259 / 8.593 | 3.197 | 31 |
| MED | patadas_min | 2812 | 0.794 / 1.469 / 2.364 | 0.484 | 17 |
| MED | pase_completo | 2728 | 0.25 / 0.455 / 0.667 | 0.146 | 22 |
