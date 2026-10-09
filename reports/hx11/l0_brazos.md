# L0: brazos comparados en lazo cerrado

> **Advertencia (2026-10-08):** este reporte usa la medida vieja de "Patadas/min": el promedio de las tasas por ventana. Las ventanas cortadas por una salida la inflan. La medida correcta es el total sobre el tiempo total, con patadas efectivas: ver `l0_escala_parcial.md` y `docs/TRASPASO_HAXELEVEN_D0_D3.md`. El orden cualitativo de los brazos se mantiene.

Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura 1.0. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de desarrollo.

## No reactivo: 640 ventanas de 5 s desde el jugador más cercano a la pelota

| Brazo | Patadas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |
|---|---|---|---|---|---|---|
| conhist | 0.999 | 1.883 | 2.16 | 291.908 | 250.874 | 3.759 |
| sinhist | 2.483 | 1.497 | 1.691 | 271.575 | 220.442 | 14.802 |
| drophist | 0.575 | 1.711 | 2.202 | 288.261 | 256.407 | 3.908 |
| v2drop | 1.073 | 1.905 | 2.247 | 294.368 | 260.88 | 3.802 |
| mini | 1.681 | 1.98 | 2.168 | 287.777 | 250.719 | 3.86 |
| **humano** | 5.672 | 6.286 | 2.327 | 283.025 | 214.566 | 3.726 |

## Reactivo: 8 juegos de 60 s con los 22 lugares jugando el modelo

| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la pelota (p50) | Largo del equipo | Juego abierto |
|---|---|---|---|---|---|---|---|
| conhist | 0.011 | 0.036 | 1.375 | 1.586 | 225.16 | 970.419 | 0.965 |
| sinhist | 0.034 | 0.013 | 0.375 | 0.655 | 217.066 | 659.07 | 0.812 |
| drophist | 0.011 | 0.01 | 0.5 | 1.856 | 278.913 | 1002.579 | 0.808 |
| v2drop | 0.029 | 0.034 | 0.75 | 1.864 | 189.287 | 955.085 | 0.855 |
| mini | 0.023 | 0.022 | 0.625 | 1.361 | 190.638 | 1019.68 | 0.836 |
| **humano** | 0.65 | 0.628 | 10.753 | 2.03 | 56.012 | 885.778 | 0.753 |
