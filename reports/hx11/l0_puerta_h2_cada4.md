# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, temperatura 0.7, decisión cada 4 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.432 | 7.19 | 0.616 | 0.558–0.676 | **no** |
| DEF | 552 | toques_min | 6.306 | 4.659 | 1.354 | 1.17–1.592 | sí |
| DEF | 552 | vel | 2.159 | 2.285 | 0.945 | 0.933–0.958 | sí |
| DEF | 552 | dist_pelota | 229.252 | 280.339 | 0.818 | 0.784–0.849 | sí |
| MCD | 92 | patadas_efectivas_min | 3.165 | 6.055 | 0.523 | 0.367–0.693 | **no** |
| MCD | 92 | toques_min | 4.495 | 2.284 | 1.968 | 1.242–3.156 | sí |
| MCD | 92 | vel | 2.099 | 2.287 | 0.918 | 0.891–0.944 | sí |
| MCD | 92 | dist_pelota | 238.542 | 295.856 | 0.806 | 0.712–0.901 | sí |
| MED | 604 | patadas_efectivas_min | 4.603 | 7.063 | 0.652 | 0.597–0.714 | **no** |
| MED | 604 | toques_min | 6.384 | 6.824 | 0.936 | 0.827–1.055 | sí |
| MED | 604 | vel | 2.144 | 2.318 | 0.925 | 0.913–0.937 | sí |
| MED | 604 | dist_pelota | 218.543 | 267.457 | 0.817 | 0.784–0.85 | sí |
| DEL | 351 | patadas_efectivas_min | 3.453 | 4.82 | 0.716 | 0.629–0.829 | sí |
| DEL | 351 | toques_min | 5.692 | 6.226 | 0.914 | 0.774–1.075 | sí |
| DEL | 351 | vel | 2.276 | 2.413 | 0.943 | 0.931–0.956 | sí |
| DEL | 351 | dist_pelota | 239.956 | 285.613 | 0.84 | 0.801–0.888 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 292.268 | 281.572 | 1.038 | 0.997–1.08 | sí |
| ARQ | 123 | dist_pelota | 1077.39 | 1065.23 | 1.011 | 0.991–1.037 | sí |
| ARQ | 123 | vel | 1.667 | 1.76 | 0.948 | 0.885–1.014 | sí |
| ARQ | 123 | toques_min | 0.167 | 0.292 | 0.571 | 0.064–7.458 | sí |
| DEF | 544 | dist_ancla | 189.45 | 175.908 | 1.077 | 1.05–1.101 | sí |
| DEF | 544 | dist_pelota | 568.969 | 572.859 | 0.993 | 0.979–1.008 | sí |
| DEF | 544 | vel | 2.194 | 2.226 | 0.986 | 0.972–1.001 | sí |
| DEF | 544 | toques_min | 0.659 | 0.508 | 1.296 | 0.869–1.961 | sí |
| MCD | 73 | dist_ancla | 200.533 | 180.971 | 1.108 | 1.044–1.179 | sí |
| MCD | 73 | dist_pelota | 457.978 | 441.983 | 1.036 | 0.998–1.078 | sí |
| MCD | 73 | vel | 2.263 | 2.344 | 0.966 | 0.93–0.999 | sí |
| MCD | 73 | toques_min | 0.4 | 0.368 | 1.085 | 0.054–10.852 | sí |
| MED | 545 | dist_ancla | 191.738 | 177.629 | 1.079 | 1.052–1.112 | sí |
| MED | 545 | dist_pelota | 530.992 | 542.737 | 0.978 | 0.966–0.991 | sí |
| MED | 545 | vel | 2.286 | 2.332 | 0.98 | 0.966–0.993 | sí |
| MED | 545 | toques_min | 0.712 | 0.706 | 1.008 | 0.672–1.583 | sí |
| DEL | 315 | dist_ancla | 206.648 | 189.758 | 1.089 | 1.057–1.125 | sí |
| DEL | 315 | dist_pelota | 542.01 | 554.775 | 0.977 | 0.958–0.996 | sí |
| DEL | 315 | vel | 2.281 | 2.332 | 0.978 | 0.96–0.996 | sí |
| DEL | 315 | toques_min | 0.504 | 0.621 | 0.813 | 0.451–1.952 | sí |

