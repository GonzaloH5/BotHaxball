# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.841 | 7.32 | 0.661 | 0.604–0.723 | **no** |
| DEF | 552 | toques_min | 6.734 | 4.656 | 1.446 | 1.265–1.673 | sí |
| DEF | 552 | vel | 2.17 | 2.284 | 0.95 | 0.937–0.963 | sí |
| DEF | 552 | dist_pelota | 229.14 | 280.07 | 0.818 | 0.784–0.851 | sí |
| MCD | 92 | patadas_efectivas_min | 3.387 | 5.825 | 0.581 | 0.42–0.75 | sí |
| MCD | 92 | toques_min | 3.441 | 2.249 | 1.53 | 1.079–2.198 | sí |
| MCD | 92 | vel | 2.121 | 2.299 | 0.923 | 0.896–0.952 | sí |
| MCD | 92 | dist_pelota | 241.433 | 301.373 | 0.801 | 0.704–0.905 | sí |
| MED | 604 | patadas_efectivas_min | 4.762 | 7.057 | 0.675 | 0.622–0.735 | **no** |
| MED | 604 | toques_min | 6.056 | 6.813 | 0.889 | 0.774–1.016 | sí |
| MED | 604 | vel | 2.14 | 2.318 | 0.923 | 0.911–0.936 | sí |
| MED | 604 | dist_pelota | 225.894 | 263.653 | 0.857 | 0.825–0.891 | sí |
| DEL | 351 | patadas_efectivas_min | 4.555 | 4.736 | 0.962 | 0.848–1.097 | sí |
| DEL | 351 | toques_min | 5.601 | 6.168 | 0.908 | 0.782–1.06 | sí |
| DEL | 351 | vel | 2.268 | 2.407 | 0.942 | 0.929–0.956 | sí |
| DEL | 351 | dist_pelota | 246.994 | 283.2 | 0.872 | 0.831–0.919 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 293.134 | 281.572 | 1.041 | 1.001–1.084 | sí |
| ARQ | 123 | dist_pelota | 1079.925 | 1065.23 | 1.014 | 0.995–1.038 | sí |
| ARQ | 123 | vel | 1.671 | 1.76 | 0.949 | 0.894–1.006 | sí |
| ARQ | 123 | toques_min | 0.2 | 0.292 | 0.686 | 0.301–6.156 | sí |
| DEF | 542 | dist_ancla | 185.078 | 172.999 | 1.07 | 1.041–1.098 | sí |
| DEF | 542 | dist_pelota | 564.448 | 569.017 | 0.992 | 0.978–1.006 | sí |
| DEF | 542 | vel | 2.181 | 2.226 | 0.98 | 0.966–0.993 | sí |
| DEF | 542 | toques_min | 0.557 | 0.513 | 1.086 | 0.738–1.747 | sí |
| MCD | 73 | dist_ancla | 192.935 | 180.971 | 1.066 | 1.009–1.133 | sí |
| MCD | 73 | dist_pelota | 453.474 | 441.983 | 1.026 | 0.986–1.068 | sí |
| MCD | 73 | vel | 2.289 | 2.344 | 0.977 | 0.952–1.002 | sí |
| MCD | 73 | toques_min | 0.264 | 0.368 | 0.715 | 0.038–8.368 | sí |
| MED | 546 | dist_ancla | 189.999 | 179.04 | 1.061 | 1.034–1.091 | sí |
| MED | 546 | dist_pelota | 533.02 | 542.625 | 0.982 | 0.969–0.997 | sí |
| MED | 546 | vel | 2.236 | 2.325 | 0.962 | 0.946–0.975 | sí |
| MED | 546 | toques_min | 0.77 | 0.745 | 1.033 | 0.689–1.528 | sí |
| DEL | 316 | dist_ancla | 203.501 | 189.199 | 1.076 | 1.043–1.113 | sí |
| DEL | 316 | dist_pelota | 544.388 | 554.912 | 0.981 | 0.961–1.003 | sí |
| DEL | 316 | vel | 2.236 | 2.334 | 0.958 | 0.939–0.978 | sí |
| DEL | 316 | toques_min | 0.419 | 0.619 | 0.678 | 0.393–1.509 | sí |

