# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_sinhist_ancho/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.82 | 7.215 | 0.807 | 0.748–0.869 | sí |
| DEF | 552 | toques_min | 4.86 | 4.521 | 1.075 | 0.914–1.251 | sí |
| DEF | 552 | vel | 2.115 | 2.292 | 0.923 | 0.912–0.935 | sí |
| DEF | 552 | dist_pelota | 237.374 | 276.714 | 0.858 | 0.824–0.894 | sí |
| MCD | 92 | patadas_efectivas_min | 4.692 | 6.033 | 0.778 | 0.6–0.959 | sí |
| MCD | 92 | toques_min | 3.852 | 2.398 | 1.607 | 1.056–2.525 | sí |
| MCD | 92 | vel | 2.099 | 2.301 | 0.912 | 0.888–0.938 | sí |
| MCD | 92 | dist_pelota | 252.251 | 299.853 | 0.841 | 0.745–0.941 | sí |
| MED | 604 | patadas_efectivas_min | 5.357 | 7.057 | 0.759 | 0.706–0.828 | sí |
| MED | 604 | toques_min | 5.889 | 6.761 | 0.871 | 0.765–0.992 | sí |
| MED | 604 | vel | 2.095 | 2.318 | 0.904 | 0.892–0.917 | sí |
| MED | 604 | dist_pelota | 218.766 | 265.144 | 0.825 | 0.792–0.861 | sí |
| DEL | 351 | patadas_efectivas_min | 4.265 | 4.739 | 0.9 | 0.786–1.032 | sí |
| DEL | 351 | toques_min | 5.57 | 5.961 | 0.934 | 0.793–1.116 | sí |
| DEL | 351 | vel | 2.217 | 2.414 | 0.918 | 0.906–0.933 | sí |
| DEL | 351 | dist_pelota | 240.451 | 284.499 | 0.845 | 0.806–0.884 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 288.313 | 281.572 | 1.024 | 0.985–1.063 | sí |
| ARQ | 123 | dist_pelota | 1078.33 | 1065.23 | 1.012 | 0.994–1.035 | sí |
| ARQ | 123 | vel | 1.526 | 1.76 | 0.867 | 0.82–0.919 | sí |
| ARQ | 123 | toques_min | 0.315 | 0.292 | 1.08 | 0.189–15.01 | sí |
| DEF | 542 | dist_ancla | 177.19 | 173.031 | 1.024 | 0.999–1.05 | sí |
| DEF | 542 | dist_pelota | 563.17 | 568.121 | 0.991 | 0.978–1.006 | sí |
| DEF | 542 | vel | 2.079 | 2.228 | 0.933 | 0.921–0.946 | sí |
| DEF | 542 | toques_min | 0.426 | 0.494 | 0.863 | 0.634–1.175 | sí |
| MCD | 73 | dist_ancla | 194.118 | 180.971 | 1.073 | 1.015–1.132 | sí |
| MCD | 73 | dist_pelota | 447.338 | 441.983 | 1.012 | 0.977–1.045 | sí |
| MCD | 73 | vel | 2.25 | 2.344 | 0.96 | 0.936–0.986 | sí |
| MCD | 73 | toques_min | 0.122 | 0.368 | 0.331 | 0.025–3.237 | sí |
| MED | 546 | dist_ancla | 183.612 | 178.736 | 1.027 | 1.001–1.059 | sí |
| MED | 546 | dist_pelota | 528.569 | 542.299 | 0.975 | 0.962–0.988 | sí |
| MED | 546 | vel | 2.188 | 2.327 | 0.94 | 0.926–0.954 | sí |
| MED | 546 | toques_min | 0.592 | 0.772 | 0.767 | 0.597–1.066 | sí |
| DEL | 316 | dist_ancla | 197.643 | 189.257 | 1.044 | 1.014–1.077 | sí |
| DEL | 316 | dist_pelota | 541.362 | 554.181 | 0.977 | 0.958–0.997 | sí |
| DEL | 316 | vel | 2.184 | 2.334 | 0.936 | 0.918–0.954 | sí |
| DEL | 316 | toques_min | 0.354 | 0.62 | 0.571 | 0.367–1.117 | sí |

