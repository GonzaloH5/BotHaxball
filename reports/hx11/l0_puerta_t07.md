# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_sinhist_full/model.pt`, temperatura 0.7, decisión cada 3 ticks, 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 6.95 | 7.046 | 0.986 | 0.913–1.049 | sí |
| DEF | 552 | toques_min | 2.591 | 4.474 | 0.579 | 0.5–0.67 | **no** |
| DEF | 552 | vel | 2.101 | 2.29 | 0.918 | 0.906–0.929 | sí |
| DEF | 552 | dist_pelota | 259.965 | 277.366 | 0.937 | 0.907–0.969 | sí |
| MCD | 92 | patadas_efectivas_min | 5.307 | 5.866 | 0.905 | 0.725–1.118 | sí |
| MCD | 92 | toques_min | 1.951 | 2.188 | 0.891 | 0.582–1.316 | sí |
| MCD | 92 | vel | 2.0 | 2.275 | 0.879 | 0.851–0.908 | sí |
| MCD | 92 | dist_pelota | 269.552 | 290.145 | 0.929 | 0.831–1.033 | sí |
| MED | 604 | patadas_efectivas_min | 6.594 | 7.052 | 0.935 | 0.874–1.003 | sí |
| MED | 604 | toques_min | 3.173 | 6.764 | 0.469 | 0.413–0.529 | **no** |
| MED | 604 | vel | 2.129 | 2.306 | 0.923 | 0.912–0.935 | sí |
| MED | 604 | dist_pelota | 248.739 | 257.945 | 0.964 | 0.932–1.0 | sí |
| DEL | 351 | patadas_efectivas_min | 4.884 | 4.775 | 1.023 | 0.903–1.163 | sí |
| DEL | 351 | toques_min | 2.526 | 6.359 | 0.397 | 0.332–0.466 | **no** |
| DEL | 351 | vel | 2.227 | 2.415 | 0.922 | 0.908–0.939 | sí |
| DEL | 351 | dist_pelota | 262.778 | 279.84 | 0.939 | 0.9–0.988 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 287.048 | 281.572 | 1.019 | 0.98–1.06 | sí |
| ARQ | 123 | dist_pelota | 1083.187 | 1065.23 | 1.017 | 0.998–1.039 | sí |
| ARQ | 123 | vel | 1.539 | 1.76 | 0.874 | 0.826–0.923 | sí |
| ARQ | 123 | toques_min | 0.125 | 0.292 | 0.429 | 0.12–5.0 | sí |
| DEF | 542 | dist_ancla | 179.414 | 173.054 | 1.037 | 1.007–1.066 | sí |
| DEF | 542 | dist_pelota | 565.021 | 567.811 | 0.995 | 0.982–1.009 | sí |
| DEF | 542 | vel | 2.067 | 2.227 | 0.928 | 0.915–0.942 | sí |
| DEF | 542 | toques_min | 0.221 | 0.447 | 0.495 | 0.32–0.732 | **no** |
| MCD | 73 | dist_ancla | 196.272 | 180.971 | 1.085 | 1.031–1.145 | sí |
| MCD | 73 | dist_pelota | 453.772 | 441.983 | 1.027 | 0.989–1.061 | sí |
| MCD | 73 | vel | 2.182 | 2.344 | 0.931 | 0.903–0.962 | sí |
| MCD | 73 | toques_min | 0.057 | 0.368 | 0.154 | 0.0–1.652 | sí |
| MED | 546 | dist_ancla | 184.345 | 178.789 | 1.031 | 1.005–1.062 | sí |
| MED | 546 | dist_pelota | 531.197 | 542.858 | 0.979 | 0.966–0.993 | sí |
| MED | 546 | vel | 2.165 | 2.324 | 0.932 | 0.918–0.945 | sí |
| MED | 546 | toques_min | 0.343 | 0.689 | 0.497 | 0.335–0.8 | sí |
| DEL | 316 | dist_ancla | 195.776 | 189.453 | 1.033 | 1.004–1.064 | sí |
| DEL | 316 | dist_pelota | 543.795 | 553.813 | 0.982 | 0.964–1.003 | sí |
| DEL | 316 | vel | 2.186 | 2.332 | 0.938 | 0.921–0.955 | sí |
| DEL | 316 | toques_min | 0.138 | 0.579 | 0.239 | 0.137–0.505 | **no** |

