# L0: brazos comparados en lazo cerrado

> **Advertencia (2026-10-08):** este reporte usa la medida vieja de "Patadas/min": el promedio de las tasas por ventana. Las ventanas cortadas por una salida la inflan. La medida correcta es el total sobre el tiempo total, con patadas efectivas: ver `l0_escala_parcial.md` y `docs/TRASPASO_HAXELEVEN_D0_D3.md`. El orden cualitativo de los brazos se mantiene.

Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura 1.0. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de desarrollo.

## No reactivo: 640 ventanas de 5 s desde el jugador más cercano a la pelota

| Brazo | Patadas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |
|---|---|---|---|---|---|---|
| sinhist | 7.846 | 1.391 | 1.701 | 272.149 | 219.007 | 14.794 |
| sinhist_cerca | 4.34 | 1.506 | 1.718 | 272.681 | 228.075 | 14.667 |
| **humano** | 6.03 | 6.497 | 2.32 | 279.339 | 214.74 | 3.715 |

## Reactivo: 4 juegos de 60 s con los 22 lugares jugando el modelo

| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la pelota (p50) | Largo del equipo | Juego abierto |
|---|---|---|---|---|---|---|---|
| sinhist | 0.034 | 0.013 | 0.25 | 0.961 | 175.948 | 606.603 | 0.687 |
| sinhist_cerca | 0.068 | 0.006 | 0.5 | 0.697 | 235.412 | 816.425 | 0.583 |
| **humano** | 0.72 | 0.539 | 12.253 | 2.0 | 58.746 | 896.012 | 0.746 |
