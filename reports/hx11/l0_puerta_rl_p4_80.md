# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/rl_a3_p4/model_80.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.23 | 7.221 | 0.724 | 0.661–0.788 | sí |
| DEF | 552 | toques_min | 6.581 | 4.618 | 1.425 | 1.237–1.651 | sí |
| DEF | 552 | vel | 2.174 | 2.284 | 0.952 | 0.938–0.964 | sí |
| DEF | 552 | dist_pelota | 223.785 | 277.501 | 0.806 | 0.77–0.841 | sí |
| MCD | 92 | patadas_efectivas_min | 3.445 | 6.063 | 0.568 | 0.408–0.766 | sí |
| MCD | 92 | toques_min | 3.387 | 2.198 | 1.541 | 0.976–2.549 | sí |
| MCD | 92 | vel | 2.119 | 2.287 | 0.926 | 0.903–0.952 | sí |
| MCD | 92 | dist_pelota | 246.562 | 298.602 | 0.826 | 0.73–0.925 | sí |
| MED | 604 | patadas_efectivas_min | 4.543 | 7.178 | 0.633 | 0.578–0.693 | **no** |
| MED | 604 | toques_min | 6.76 | 6.923 | 0.976 | 0.863–1.108 | sí |
| MED | 604 | vel | 2.129 | 2.322 | 0.917 | 0.906–0.93 | sí |
| MED | 604 | dist_pelota | 217.945 | 265.936 | 0.82 | 0.787–0.856 | sí |
| DEL | 351 | patadas_efectivas_min | 4.201 | 4.749 | 0.885 | 0.764–1.018 | sí |
| DEL | 351 | toques_min | 5.313 | 6.303 | 0.843 | 0.709–1.01 | sí |
| DEL | 351 | vel | 2.255 | 2.411 | 0.935 | 0.922–0.949 | sí |
| DEL | 351 | dist_pelota | 243.558 | 281.153 | 0.866 | 0.833–0.908 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 289.205 | 281.572 | 1.027 | 0.986–1.07 | sí |
| ARQ | 123 | dist_pelota | 1072.067 | 1065.23 | 1.006 | 0.986–1.029 | sí |
| ARQ | 123 | vel | 1.744 | 1.76 | 0.991 | 0.935–1.054 | sí |
| ARQ | 123 | toques_min | 0.253 | 0.292 | 0.869 | 0.357–9.313 | sí |
| DEF | 542 | dist_ancla | 185.45 | 172.978 | 1.072 | 1.043–1.103 | sí |
| DEF | 542 | dist_pelota | 553.569 | 568.482 | 0.974 | 0.959–0.988 | sí |
| DEF | 542 | vel | 2.2 | 2.228 | 0.987 | 0.973–1.0 | sí |
| DEF | 542 | toques_min | 0.795 | 0.459 | 1.733 | 1.236–2.535 | sí |
| MCD | 73 | dist_ancla | 191.519 | 180.971 | 1.058 | 1.003–1.123 | sí |
| MCD | 73 | dist_pelota | 448.425 | 441.983 | 1.015 | 0.979–1.049 | sí |
| MCD | 73 | vel | 2.238 | 2.344 | 0.955 | 0.932–0.982 | sí |
| MCD | 73 | toques_min | 0.312 | 0.368 | 0.846 | 0.149–7.24 | sí |
| MED | 546 | dist_ancla | 187.48 | 179.042 | 1.047 | 1.021–1.077 | sí |
| MED | 546 | dist_pelota | 522.613 | 542.374 | 0.964 | 0.95–0.977 | sí |
| MED | 546 | vel | 2.243 | 2.323 | 0.966 | 0.952–0.978 | sí |
| MED | 546 | toques_min | 0.769 | 0.748 | 1.029 | 0.744–1.55 | sí |
| DEL | 316 | dist_ancla | 203.323 | 189.685 | 1.072 | 1.04–1.105 | sí |
| DEL | 316 | dist_pelota | 537.193 | 553.486 | 0.971 | 0.952–0.991 | sí |
| DEL | 316 | vel | 2.231 | 2.333 | 0.956 | 0.938–0.976 | sí |
| DEL | 316 | toques_min | 0.525 | 0.62 | 0.847 | 0.429–1.965 | sí |

