# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_memoria_s2/model.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.89 | 7.047 | 0.694 | 0.637–0.755 | sí |
| DEF | 552 | toques_min | 6.761 | 4.633 | 1.459 | 1.263–1.705 | sí |
| DEF | 552 | vel | 2.163 | 2.287 | 0.945 | 0.934–0.957 | sí |
| DEF | 552 | dist_pelota | 235.415 | 277.621 | 0.848 | 0.816–0.88 | sí |
| MCD | 92 | patadas_efectivas_min | 4.412 | 6.067 | 0.727 | 0.571–0.914 | sí |
| MCD | 92 | toques_min | 3.707 | 2.245 | 1.651 | 0.982–2.763 | sí |
| MCD | 92 | vel | 2.119 | 2.275 | 0.931 | 0.903–0.962 | sí |
| MCD | 92 | dist_pelota | 247.276 | 297.079 | 0.832 | 0.731–0.936 | sí |
| MED | 604 | patadas_efectivas_min | 4.806 | 7.093 | 0.678 | 0.624–0.735 | **no** |
| MED | 604 | toques_min | 5.851 | 6.766 | 0.865 | 0.769–0.982 | sí |
| MED | 604 | vel | 2.154 | 2.32 | 0.928 | 0.917–0.94 | sí |
| MED | 604 | dist_pelota | 228.499 | 267.709 | 0.854 | 0.819–0.885 | sí |
| DEL | 351 | patadas_efectivas_min | 3.551 | 4.878 | 0.728 | 0.63–0.848 | sí |
| DEL | 351 | toques_min | 5.212 | 6.377 | 0.817 | 0.691–0.961 | sí |
| DEL | 351 | vel | 2.257 | 2.412 | 0.936 | 0.923–0.95 | sí |
| DEL | 351 | dist_pelota | 246.329 | 284.894 | 0.865 | 0.825–0.904 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 287.758 | 281.572 | 1.022 | 0.982–1.062 | sí |
| ARQ | 123 | dist_pelota | 1079.035 | 1065.23 | 1.013 | 0.993–1.036 | sí |
| ARQ | 123 | vel | 1.588 | 1.76 | 0.903 | 0.851–0.96 | sí |
| ARQ | 123 | toques_min | 0.118 | 0.292 | 0.406 | 0.061–6.111 | sí |
| DEF | 544 | dist_ancla | 188.973 | 175.869 | 1.075 | 1.048–1.101 | sí |
| DEF | 544 | dist_pelota | 571.353 | 573.234 | 0.997 | 0.982–1.012 | sí |
| DEF | 544 | vel | 2.18 | 2.227 | 0.979 | 0.965–0.994 | sí |
| DEF | 544 | toques_min | 0.535 | 0.509 | 1.051 | 0.7–1.561 | sí |
| MCD | 73 | dist_ancla | 193.894 | 180.971 | 1.071 | 1.016–1.135 | sí |
| MCD | 73 | dist_pelota | 454.825 | 441.983 | 1.029 | 0.99–1.065 | sí |
| MCD | 73 | vel | 2.29 | 2.344 | 0.977 | 0.953–1.003 | sí |
| MCD | 73 | toques_min | 0.232 | 0.368 | 0.631 | 0.121–6.766 | sí |
| MED | 545 | dist_ancla | 189.027 | 177.815 | 1.063 | 1.038–1.096 | sí |
| MED | 545 | dist_pelota | 528.291 | 542.525 | 0.974 | 0.96–0.986 | sí |
| MED | 545 | vel | 2.243 | 2.33 | 0.963 | 0.949–0.975 | sí |
| MED | 545 | toques_min | 0.975 | 0.719 | 1.357 | 0.845–2.079 | sí |
| DEL | 315 | dist_ancla | 201.821 | 189.82 | 1.063 | 1.032–1.098 | sí |
| DEL | 315 | dist_pelota | 542.669 | 553.16 | 0.981 | 0.961–1.001 | sí |
| DEL | 315 | vel | 2.274 | 2.332 | 0.975 | 0.959–0.993 | sí |
| DEL | 315 | toques_min | 0.432 | 0.568 | 0.76 | 0.368–1.833 | sí |

