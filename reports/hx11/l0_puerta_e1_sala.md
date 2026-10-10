# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_quienva_e1/model.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.724 | 7.052 | 0.812 | 0.75–0.88 | sí |
| DEF | 552 | toques_min | 5.802 | 4.697 | 1.235 | 1.05–1.448 | sí |
| DEF | 552 | vel | 2.174 | 2.291 | 0.949 | 0.939–0.961 | sí |
| DEF | 552 | dist_pelota | 235.323 | 274.603 | 0.857 | 0.823–0.894 | sí |
| MCD | 92 | patadas_efectivas_min | 3.738 | 5.815 | 0.643 | 0.469–0.838 | sí |
| MCD | 92 | toques_min | 4.024 | 1.991 | 2.021 | 1.373–2.928 | **no** |
| MCD | 92 | vel | 2.094 | 2.282 | 0.917 | 0.889–0.948 | sí |
| MCD | 92 | dist_pelota | 251.275 | 298.534 | 0.842 | 0.743–0.949 | sí |
| MED | 604 | patadas_efectivas_min | 5.64 | 7.104 | 0.794 | 0.731–0.861 | sí |
| MED | 604 | toques_min | 5.545 | 6.882 | 0.806 | 0.718–0.904 | sí |
| MED | 604 | vel | 2.166 | 2.319 | 0.934 | 0.923–0.945 | sí |
| MED | 604 | dist_pelota | 223.963 | 264.838 | 0.846 | 0.811–0.877 | sí |
| DEL | 351 | patadas_efectivas_min | 3.862 | 4.837 | 0.799 | 0.685–0.919 | sí |
| DEL | 351 | toques_min | 5.305 | 6.402 | 0.829 | 0.71–0.978 | sí |
| DEL | 351 | vel | 2.262 | 2.411 | 0.938 | 0.925–0.952 | sí |
| DEL | 351 | dist_pelota | 245.236 | 284.831 | 0.861 | 0.823–0.908 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 297.052 | 281.572 | 1.055 | 1.012–1.098 | sí |
| ARQ | 123 | dist_pelota | 1079.476 | 1065.23 | 1.013 | 0.992–1.039 | sí |
| ARQ | 123 | vel | 1.653 | 1.76 | 0.939 | 0.883–0.993 | sí |
| ARQ | 123 | toques_min | 0.233 | 0.292 | 0.8 | 0.139–12.572 | sí |
| DEF | 542 | dist_ancla | 184.164 | 172.936 | 1.065 | 1.036–1.097 | sí |
| DEF | 542 | dist_pelota | 565.112 | 568.466 | 0.994 | 0.979–1.009 | sí |
| DEF | 542 | vel | 2.175 | 2.226 | 0.977 | 0.965–0.99 | sí |
| DEF | 542 | toques_min | 0.638 | 0.472 | 1.351 | 0.923–1.927 | sí |
| MCD | 73 | dist_ancla | 200.664 | 180.971 | 1.109 | 1.048–1.171 | sí |
| MCD | 73 | dist_pelota | 451.608 | 441.983 | 1.022 | 0.987–1.056 | sí |
| MCD | 73 | vel | 2.308 | 2.344 | 0.985 | 0.958–1.012 | sí |
| MCD | 73 | toques_min | 0.034 | 0.368 | 0.092 | 0.019–0.8 | sí |
| MED | 546 | dist_ancla | 191.863 | 178.685 | 1.074 | 1.045–1.108 | sí |
| MED | 546 | dist_pelota | 531.577 | 542.66 | 0.98 | 0.966–0.994 | sí |
| MED | 546 | vel | 2.238 | 2.327 | 0.962 | 0.947–0.976 | sí |
| MED | 546 | toques_min | 0.667 | 0.769 | 0.867 | 0.625–1.353 | sí |
| DEL | 316 | dist_ancla | 201.365 | 189.502 | 1.063 | 1.034–1.093 | sí |
| DEL | 316 | dist_pelota | 542.208 | 554.44 | 0.978 | 0.96–0.997 | sí |
| DEL | 316 | vel | 2.293 | 2.334 | 0.982 | 0.967–1.0 | sí |
| DEL | 316 | toques_min | 0.314 | 0.617 | 0.509 | 0.29–1.124 | sí |

