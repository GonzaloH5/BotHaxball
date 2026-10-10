# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_ajuste_corner_h2/ultimo.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.576 | 7.101 | 0.785 | 0.725–0.851 | sí |
| DEF | 552 | toques_min | 5.842 | 4.589 | 1.273 | 1.1–1.479 | sí |
| DEF | 552 | vel | 2.162 | 2.29 | 0.944 | 0.933–0.955 | sí |
| DEF | 552 | dist_pelota | 238.848 | 278.093 | 0.859 | 0.826–0.892 | sí |
| MCD | 92 | patadas_efectivas_min | 4.044 | 5.997 | 0.674 | 0.509–0.845 | sí |
| MCD | 92 | toques_min | 2.452 | 2.271 | 1.08 | 0.714–1.791 | sí |
| MCD | 92 | vel | 2.151 | 2.288 | 0.94 | 0.914–0.964 | sí |
| MCD | 92 | dist_pelota | 266.553 | 296.644 | 0.899 | 0.802–1.002 | sí |
| MED | 604 | patadas_efectivas_min | 5.275 | 7.033 | 0.75 | 0.689–0.817 | sí |
| MED | 604 | toques_min | 6.322 | 6.8 | 0.93 | 0.823–1.048 | sí |
| MED | 604 | vel | 2.132 | 2.319 | 0.92 | 0.908–0.932 | sí |
| MED | 604 | dist_pelota | 222.251 | 264.159 | 0.841 | 0.805–0.878 | sí |
| DEL | 351 | patadas_efectivas_min | 4.45 | 4.845 | 0.919 | 0.814–1.03 | sí |
| DEL | 351 | toques_min | 5.126 | 6.266 | 0.818 | 0.709–0.974 | sí |
| DEL | 351 | vel | 2.269 | 2.409 | 0.942 | 0.928–0.956 | sí |
| DEL | 351 | dist_pelota | 257.934 | 286.097 | 0.902 | 0.861–0.94 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 298.496 | 281.572 | 1.06 | 1.013–1.109 | sí |
| ARQ | 123 | dist_pelota | 1081.397 | 1065.23 | 1.015 | 0.995–1.04 | sí |
| ARQ | 123 | vel | 1.65 | 1.76 | 0.937 | 0.878–0.996 | sí |
| ARQ | 123 | toques_min | 0.138 | 0.292 | 0.474 | 0.038–0.811 | sí |
| DEF | 542 | dist_ancla | 184.742 | 172.884 | 1.069 | 1.041–1.096 | sí |
| DEF | 542 | dist_pelota | 566.741 | 568.515 | 0.997 | 0.982–1.012 | sí |
| DEF | 542 | vel | 2.193 | 2.225 | 0.986 | 0.971–0.999 | sí |
| DEF | 542 | toques_min | 0.493 | 0.493 | 0.999 | 0.694–1.465 | sí |
| MCD | 73 | dist_ancla | 191.763 | 180.971 | 1.06 | 1.004–1.122 | sí |
| MCD | 73 | dist_pelota | 445.927 | 441.983 | 1.009 | 0.969–1.053 | sí |
| MCD | 73 | vel | 2.303 | 2.344 | 0.983 | 0.96–1.009 | sí |
| MCD | 73 | toques_min | 0.419 | 0.368 | 1.138 | 0.085–12.814 | sí |
| MED | 546 | dist_ancla | 189.692 | 178.962 | 1.06 | 1.031–1.092 | sí |
| MED | 546 | dist_pelota | 534.481 | 543.085 | 0.984 | 0.972–0.997 | sí |
| MED | 546 | vel | 2.243 | 2.324 | 0.965 | 0.952–0.978 | sí |
| MED | 546 | toques_min | 0.638 | 0.733 | 0.871 | 0.59–1.347 | sí |
| DEL | 316 | dist_ancla | 206.225 | 189.542 | 1.088 | 1.056–1.121 | sí |
| DEL | 316 | dist_pelota | 539.475 | 553.278 | 0.975 | 0.955–0.997 | sí |
| DEL | 316 | vel | 2.232 | 2.331 | 0.957 | 0.938–0.978 | sí |
| DEL | 316 | toques_min | 0.443 | 0.615 | 0.72 | 0.389–1.73 | sí |

