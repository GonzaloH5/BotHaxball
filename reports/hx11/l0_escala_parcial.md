# L0: brazos comparados en lazo cerrado

Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura 1.0. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de desarrollo. Las tasas son totales sobre el tiempo total; las patadas son efectivas (la física las ejecuta).

## No reactivo: 640 ventanas de 5 s desde el jugador más cercano a la pelota

| Brazo | Patadas efectivas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |
|---|---|---|---|---|---|---|
| sinhist | 1.884 | 1.107 | 1.707 | 275.144 | 217.36 | 14.785 |
| full_sinhist_hold | 3.07 | 0.834 | 1.705 | 289.718 | 221.465 | 14.8 |
| **humano** | 6.741 | 6.002 | 2.326 | 278.641 | 215.064 | 3.701 |

## Reactivo: 4 juegos de 60 s con los 22 lugares jugando el modelo

| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la pelota (p50) | Largo del equipo | Juego abierto |
|---|---|---|---|---|---|---|---|
| sinhist | 0.023 | 0.006 | 0.0 | 0.733 | 158.047 | 627.666 | 0.831 |
| full_sinhist_hold | 0.057 | 0.008 | 0.25 | 0.877 | 239.253 | 875.89 | 0.632 |
| **humano** | 0.72 | 0.539 | 12.253 | 2.0 | 58.746 | 896.012 | 0.746 |
