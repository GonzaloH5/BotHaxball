# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.086 | 7.167 | 0.71 | 0.645–0.782 | sí |
| DEF | 552 | toques_min | 6.234 | 4.608 | 1.353 | 1.15–1.565 | sí |
| DEF | 552 | vel | 2.166 | 2.287 | 0.947 | 0.935–0.959 | sí |
| DEF | 552 | dist_pelota | 233.276 | 279.748 | 0.834 | 0.8–0.868 | sí |
| MCD | 92 | patadas_efectivas_min | 3.544 | 5.861 | 0.605 | 0.444–0.781 | sí |
| MCD | 92 | toques_min | 4.234 | 2.262 | 1.871 | 1.264–2.927 | sí |
| MCD | 92 | vel | 2.126 | 2.291 | 0.928 | 0.9–0.958 | sí |
| MCD | 92 | dist_pelota | 240.635 | 299.539 | 0.803 | 0.704–0.906 | sí |
| MED | 604 | patadas_efectivas_min | 5.289 | 7.101 | 0.745 | 0.694–0.81 | sí |
| MED | 604 | toques_min | 5.856 | 6.779 | 0.864 | 0.762–0.964 | sí |
| MED | 604 | vel | 2.143 | 2.319 | 0.924 | 0.913–0.937 | sí |
| MED | 604 | dist_pelota | 225.828 | 265.261 | 0.851 | 0.818–0.886 | sí |
| DEL | 351 | patadas_efectivas_min | 4.579 | 4.76 | 0.962 | 0.848–1.088 | sí |
| DEL | 351 | toques_min | 5.198 | 6.22 | 0.836 | 0.704–0.991 | sí |
| DEL | 351 | vel | 2.287 | 2.408 | 0.95 | 0.937–0.964 | sí |
| DEL | 351 | dist_pelota | 248.192 | 285.164 | 0.87 | 0.835–0.914 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 292.93 | 281.572 | 1.04 | 1.0–1.08 | sí |
| ARQ | 123 | dist_pelota | 1079.739 | 1065.23 | 1.014 | 0.994–1.035 | sí |
| ARQ | 123 | vel | 1.636 | 1.76 | 0.93 | 0.871–0.992 | sí |
| ARQ | 123 | toques_min | 0.418 | 0.292 | 1.434 | 0.314–18.415 | sí |
| DEF | 542 | dist_ancla | 186.917 | 172.974 | 1.081 | 1.052–1.111 | sí |
| DEF | 542 | dist_pelota | 559.228 | 568.986 | 0.983 | 0.969–0.998 | sí |
| DEF | 542 | vel | 2.181 | 2.227 | 0.98 | 0.966–0.995 | sí |
| DEF | 542 | toques_min | 0.706 | 0.513 | 1.375 | 0.924–2.106 | sí |
| MCD | 73 | dist_ancla | 198.847 | 181.248 | 1.097 | 1.046–1.157 | sí |
| MCD | 73 | dist_pelota | 452.83 | 441.903 | 1.025 | 0.985–1.065 | sí |
| MCD | 73 | vel | 2.292 | 2.344 | 0.978 | 0.954–1.006 | sí |
| MCD | 73 | toques_min | 0.35 | 0.37 | 0.946 | 0.202–7.895 | sí |
| MED | 546 | dist_ancla | 191.208 | 179.044 | 1.068 | 1.041–1.102 | sí |
| MED | 546 | dist_pelota | 531.429 | 543.199 | 0.978 | 0.965–0.992 | sí |
| MED | 546 | vel | 2.252 | 2.327 | 0.968 | 0.954–0.982 | sí |
| MED | 546 | toques_min | 0.782 | 0.757 | 1.033 | 0.729–1.505 | sí |
| DEL | 316 | dist_ancla | 207.856 | 189.464 | 1.097 | 1.063–1.136 | sí |
| DEL | 316 | dist_pelota | 537.343 | 554.561 | 0.969 | 0.95–0.991 | sí |
| DEL | 316 | vel | 2.258 | 2.335 | 0.967 | 0.948–0.987 | sí |
| DEL | 316 | toques_min | 0.506 | 0.614 | 0.824 | 0.491–1.88 | sí |

