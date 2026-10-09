# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_sinhist_ancho/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.579 | 7.183 | 0.777 | 0.711–0.838 | sí |
| DEF | 552 | toques_min | 5.181 | 4.555 | 1.137 | 0.975–1.311 | sí |
| DEF | 552 | vel | 2.104 | 2.287 | 0.92 | 0.908–0.933 | sí |
| DEF | 552 | dist_pelota | 235.62 | 278.003 | 0.848 | 0.813–0.878 | sí |
| MCD | 92 | patadas_efectivas_min | 4.367 | 6.005 | 0.727 | 0.566–0.921 | sí |
| MCD | 92 | toques_min | 3.269 | 2.265 | 1.443 | 0.893–2.442 | sí |
| MCD | 92 | vel | 2.053 | 2.281 | 0.9 | 0.867–0.933 | sí |
| MCD | 92 | dist_pelota | 242.382 | 296.427 | 0.818 | 0.713–0.92 | sí |
| MED | 604 | patadas_efectivas_min | 5.274 | 7.133 | 0.739 | 0.681–0.808 | sí |
| MED | 604 | toques_min | 5.906 | 6.797 | 0.869 | 0.763–0.968 | sí |
| MED | 604 | vel | 2.109 | 2.318 | 0.91 | 0.9–0.921 | sí |
| MED | 604 | dist_pelota | 222.985 | 264.289 | 0.844 | 0.814–0.876 | sí |
| DEL | 351 | patadas_efectivas_min | 3.857 | 4.804 | 0.803 | 0.7–0.908 | sí |
| DEL | 351 | toques_min | 5.259 | 6.136 | 0.857 | 0.745–1.004 | sí |
| DEL | 351 | vel | 2.213 | 2.417 | 0.916 | 0.901–0.93 | sí |
| DEL | 351 | dist_pelota | 247.385 | 282.443 | 0.876 | 0.842–0.918 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 289.31 | 281.738 | 1.027 | 0.987–1.068 | sí |
| ARQ | 123 | dist_pelota | 1081.231 | 1064.773 | 1.015 | 0.996–1.038 | sí |
| ARQ | 123 | vel | 1.526 | 1.76 | 0.867 | 0.812–0.919 | sí |
| ARQ | 123 | toques_min | 0.077 | 0.292 | 0.263 | 0.065–3.566 | sí |
| DEF | 542 | dist_ancla | 177.446 | 172.948 | 1.026 | 1.0–1.054 | sí |
| DEF | 542 | dist_pelota | 564.252 | 567.999 | 0.993 | 0.98–1.008 | sí |
| DEF | 542 | vel | 2.092 | 2.226 | 0.94 | 0.927–0.953 | sí |
| DEF | 542 | toques_min | 0.411 | 0.512 | 0.802 | 0.515–1.248 | sí |
| MCD | 73 | dist_ancla | 194.324 | 181.085 | 1.073 | 1.022–1.133 | sí |
| MCD | 73 | dist_pelota | 445.489 | 441.888 | 1.008 | 0.97–1.044 | sí |
| MCD | 73 | vel | 2.23 | 2.344 | 0.952 | 0.924–0.983 | sí |
| MCD | 73 | toques_min | 0.289 | 0.369 | 0.785 | 0.1–6.817 | sí |
| MED | 546 | dist_ancla | 182.74 | 179.084 | 1.02 | 0.995–1.047 | sí |
| MED | 546 | dist_pelota | 529.765 | 542.504 | 0.977 | 0.964–0.99 | sí |
| MED | 546 | vel | 2.179 | 2.322 | 0.938 | 0.924–0.951 | sí |
| MED | 546 | toques_min | 0.703 | 0.727 | 0.967 | 0.632–1.543 | sí |
| DEL | 316 | dist_ancla | 197.282 | 189.849 | 1.039 | 1.01–1.073 | sí |
| DEL | 316 | dist_pelota | 540.295 | 553.596 | 0.976 | 0.957–0.996 | sí |
| DEL | 316 | vel | 2.179 | 2.33 | 0.935 | 0.918–0.954 | sí |
| DEL | 316 | toques_min | 0.481 | 0.621 | 0.775 | 0.461–1.661 | sí |

