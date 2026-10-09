# L0: brazos comparados en lazo cerrado

Con [`learn/hx11_l0_compare.py`](../../learn/hx11_l0_compare.py). Decisión cada 3 ticks, temperatura 1.0. Diagnóstico de la puerta P3 (revisión de la auditoría), sobre partidos de prueba. Las tasas son totales sobre el tiempo total; las patadas son efectivas (la física las ejecuta).

## No reactivo: 640 ventanas de 5 s desde el jugador más cercano a la pelota

| Brazo | Patadas efectivas/min | Toques/min | Velocidad (px/tick) | Dist. a la pelota | Dist. al ancla | Cambios de tecla/s |
|---|---|---|---|---|---|---|
| sinhist | 2.189 | 1.3 | 1.691 | 269.15 | 206.297 | 14.773 |
| full_sinhist_hold | 3.2 | 0.881 | 1.682 | 284.413 | 205.938 | 14.903 |
| l0_sinhist_full | 6.416 | 2.145 | 1.952 | 275.927 | 217.66 | 13.274 |
| **humano** | 7.16 | 4.627 | 2.343 | 287.583 | 195.785 | 4.057 |

## Reactivo: 4 juegos de 60 s con los 22 lugares jugando el modelo

| Brazo | Patadas/jugador/min | Toques/jugador/min | Cambios de posesión/min | Velocidad | Más cercano a la pelota (p50) | Largo del equipo | Juego abierto |
|---|---|---|---|---|---|---|---|
| sinhist | 0.023 | 0.022 | 1.0 | 0.544 | 450.486 | 692.612 | 0.929 |
| full_sinhist_hold | 0.034 | 0.018 | 1.0 | 0.719 | 312.916 | 841.79 | 0.973 |
| l0_sinhist_full | 0.455 | 0.113 | 9.003 | 1.433 | 113.907 | 809.81 | 0.599 |
| **humano** | 0.716 | 0.402 | 10.753 | 1.941 | 79.465 | 843.5 | 0.708 |
