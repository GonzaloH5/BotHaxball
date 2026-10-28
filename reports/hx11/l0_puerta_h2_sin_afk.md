# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_h2_sin_afk/model.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.77 | 7.167 | 0.666 | 0.606–0.727 | **no** |
| DEF | 552 | toques_min | 6.409 | 4.62 | 1.387 | 1.191–1.629 | sí |
| DEF | 552 | vel | 2.152 | 2.288 | 0.941 | 0.929–0.952 | sí |
| DEF | 552 | dist_pelota | 232.032 | 279.923 | 0.829 | 0.793–0.863 | sí |
| MCD | 92 | patadas_efectivas_min | 3.803 | 5.976 | 0.636 | 0.486–0.811 | sí |
| MCD | 92 | toques_min | 3.88 | 2.255 | 1.721 | 1.046–2.862 | sí |
| MCD | 92 | vel | 2.106 | 2.29 | 0.92 | 0.895–0.947 | sí |
| MCD | 92 | dist_pelota | 255.095 | 300.444 | 0.849 | 0.759–0.943 | sí |
| MED | 604 | patadas_efectivas_min | 5.038 | 7.023 | 0.717 | 0.663–0.778 | sí |
| MED | 604 | toques_min | 4.926 | 6.743 | 0.731 | 0.646–0.838 | sí |
| MED | 604 | vel | 2.171 | 2.316 | 0.937 | 0.925–0.949 | sí |
| MED | 604 | dist_pelota | 233.816 | 264.674 | 0.883 | 0.849–0.918 | sí |
| DEL | 351 | patadas_efectivas_min | 4.489 | 4.888 | 0.919 | 0.803–1.041 | sí |
| DEL | 351 | toques_min | 5.185 | 6.377 | 0.813 | 0.693–0.952 | sí |
| DEL | 351 | vel | 2.289 | 2.41 | 0.95 | 0.936–0.964 | sí |
| DEL | 351 | dist_pelota | 253.841 | 283.746 | 0.895 | 0.854–0.942 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 291.214 | 281.572 | 1.034 | 0.991–1.079 | sí |
| ARQ | 123 | dist_pelota | 1078.357 | 1065.23 | 1.012 | 0.992–1.038 | sí |
| ARQ | 123 | vel | 1.676 | 1.76 | 0.952 | 0.893–1.008 | sí |
| ARQ | 123 | toques_min | 0.422 | 0.292 | 1.446 | 0.521–12.991 | sí |
| DEF | 544 | dist_ancla | 189.011 | 176.029 | 1.074 | 1.047–1.098 | sí |
| DEF | 544 | dist_pelota | 568.326 | 573.201 | 0.991 | 0.976–1.006 | sí |
| DEF | 544 | vel | 2.177 | 2.228 | 0.977 | 0.963–0.991 | sí |
| DEF | 544 | toques_min | 0.569 | 0.487 | 1.169 | 0.77–1.801 | sí |
| MCD | 73 | dist_ancla | 202.502 | 181.204 | 1.118 | 1.067–1.179 | sí |
| MCD | 73 | dist_pelota | 456.372 | 441.864 | 1.033 | 0.996–1.069 | sí |
| MCD | 73 | vel | 2.249 | 2.344 | 0.96 | 0.928–0.991 | sí |
| MCD | 73 | toques_min | 0.179 | 0.369 | 0.485 | 0.09–4.629 | sí |
| MED | 545 | dist_ancla | 190.477 | 177.703 | 1.072 | 1.041–1.104 | sí |
| MED | 545 | dist_pelota | 529.474 | 542.307 | 0.976 | 0.963–0.99 | sí |
| MED | 545 | vel | 2.249 | 2.327 | 0.966 | 0.953–0.98 | sí |
| MED | 545 | toques_min | 0.824 | 0.732 | 1.127 | 0.774–1.762 | sí |
| DEL | 315 | dist_ancla | 202.151 | 189.486 | 1.067 | 1.033–1.101 | sí |
| DEL | 315 | dist_pelota | 540.916 | 554.782 | 0.975 | 0.956–0.994 | sí |
| DEL | 315 | vel | 2.264 | 2.337 | 0.969 | 0.95–0.989 | sí |
| DEL | 315 | toques_min | 0.44 | 0.621 | 0.709 | 0.435–1.48 | sí |

