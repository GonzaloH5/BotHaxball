# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_sinhist_ancho/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 7.005 | 7.269 | 0.964 | 0.898–1.036 | sí |
| DEF | 552 | toques_min | 3.015 | 4.651 | 0.648 | 0.556–0.754 | sí |
| DEF | 552 | vel | 2.144 | 2.285 | 0.938 | 0.927–0.951 | sí |
| DEF | 552 | dist_pelota | 256.138 | 275.421 | 0.93 | 0.897–0.96 | sí |
| MCD | 92 | patadas_efectivas_min | 5.254 | 5.807 | 0.905 | 0.711–1.158 | sí |
| MCD | 92 | toques_min | 1.673 | 2.023 | 0.827 | 0.561–1.209 | sí |
| MCD | 92 | vel | 2.043 | 2.281 | 0.896 | 0.868–0.921 | sí |
| MCD | 92 | dist_pelota | 262.978 | 297.929 | 0.883 | 0.805–0.969 | sí |
| MED | 604 | patadas_efectivas_min | 7.038 | 7.081 | 0.994 | 0.931–1.07 | sí |
| MED | 604 | toques_min | 3.448 | 6.751 | 0.511 | 0.453–0.567 | **no** |
| MED | 604 | vel | 2.166 | 2.319 | 0.934 | 0.923–0.946 | sí |
| MED | 604 | dist_pelota | 258.076 | 263.454 | 0.98 | 0.948–1.015 | sí |
| DEL | 351 | patadas_efectivas_min | 5.805 | 4.776 | 1.215 | 1.103–1.361 | sí |
| DEL | 351 | toques_min | 3.575 | 6.211 | 0.576 | 0.497–0.674 | **no** |
| DEL | 351 | vel | 2.261 | 2.413 | 0.937 | 0.925–0.95 | sí |
| DEL | 351 | dist_pelota | 267.101 | 281.733 | 0.948 | 0.907–0.989 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 286.931 | 281.572 | 1.019 | 0.986–1.054 | sí |
| ARQ | 123 | dist_pelota | 1085.124 | 1065.23 | 1.019 | 1.0–1.042 | sí |
| ARQ | 123 | vel | 1.518 | 1.76 | 0.863 | 0.818–0.912 | sí |
| ARQ | 123 | toques_min | 0.153 | 0.292 | 0.526 | 0.314–3.446 | sí |
| DEF | 542 | dist_ancla | 177.564 | 172.793 | 1.028 | 1.001–1.054 | sí |
| DEF | 542 | dist_pelota | 564.452 | 568.288 | 0.993 | 0.979–1.008 | sí |
| DEF | 542 | vel | 2.104 | 2.228 | 0.945 | 0.931–0.957 | sí |
| DEF | 542 | toques_min | 0.238 | 0.44 | 0.54 | 0.345–0.832 | sí |
| MCD | 73 | dist_ancla | 193.045 | 181.218 | 1.065 | 1.016–1.125 | sí |
| MCD | 73 | dist_pelota | 456.822 | 441.868 | 1.034 | 0.997–1.066 | sí |
| MCD | 73 | vel | 2.211 | 2.344 | 0.943 | 0.918–0.973 | sí |
| MCD | 73 | toques_min | 0.091 | 0.37 | 0.246 | 0.014–2.876 | sí |
| MED | 546 | dist_ancla | 185.497 | 179.162 | 1.035 | 1.009–1.064 | sí |
| MED | 546 | dist_pelota | 533.882 | 542.707 | 0.984 | 0.971–0.996 | sí |
| MED | 546 | vel | 2.183 | 2.325 | 0.939 | 0.926–0.951 | sí |
| MED | 546 | toques_min | 0.344 | 0.743 | 0.463 | 0.306–0.761 | sí |
| DEL | 316 | dist_ancla | 196.493 | 189.528 | 1.037 | 1.006–1.07 | sí |
| DEL | 316 | dist_pelota | 539.106 | 552.929 | 0.975 | 0.957–0.994 | sí |
| DEL | 316 | vel | 2.18 | 2.333 | 0.935 | 0.919–0.952 | sí |
| DEL | 316 | toques_min | 0.306 | 0.619 | 0.494 | 0.294–1.074 | sí |

