# L0: brazos comparados en lazo cerrado

Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura 1.0. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de desarrollo.

## No reactivo: 640 ventanas de 5 s desde el jugador más cercano a la pelota

| Brazo | Patadas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |
|---|---|---|---|---|---|---|
| conhist | 0.782 | 1.555 | 2.162 | 290.515 | 246.127 | 3.814 |
| sinhist | 1.776 | 1.165 | 1.688 | 268.361 | 217.501 | 14.903 |
| sinhist_hold | 0.39 | 1.432 | 2.189 | 282.435 | 254.141 | 2.161 |
| **humano** | 5.532 | 6.016 | 2.327 | 281.431 | 214.674 | 3.698 |

## Reactivo: 4 juegos de 60 s con los 22 lugares jugando el modelo

| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la pelota (p50) | Largo del equipo | Juego abierto |
|---|---|---|---|---|---|---|---|
| conhist | 0.034 | 0.05 | 1.0 | 1.676 | 149.678 | 921.413 | 0.897 |
| sinhist | 0.057 | 0.007 | 0.0 | 0.635 | 272.303 | 620.165 | 0.964 |
| sinhist_hold | 0.034 | 0.041 | 1.5 | 1.527 | 303.53 | 859.436 | 0.963 |
| **humano** | 0.72 | 0.539 | 12.253 | 2.0 | 58.746 | 896.012 | 0.746 |
