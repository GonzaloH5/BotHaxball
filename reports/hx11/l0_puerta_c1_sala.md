# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_ajuste_conduccion_c1/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.073 | 7.127 | 0.571 | 0.517–0.639 | **no** |
| DEF | 552 | toques_min | 8.689 | 4.595 | 1.891 | 1.657–2.165 | **no** |
| DEF | 552 | vel | 2.182 | 2.28 | 0.957 | 0.946–0.967 | sí |
| DEF | 552 | dist_pelota | 222.215 | 279.94 | 0.794 | 0.761–0.826 | sí |
| MCD | 92 | patadas_efectivas_min | 2.981 | 6.098 | 0.489 | 0.341–0.657 | **no** |
| MCD | 92 | toques_min | 4.585 | 2.423 | 1.892 | 1.206–3.109 | sí |
| MCD | 92 | vel | 2.141 | 2.301 | 0.931 | 0.903–0.961 | sí |
| MCD | 92 | dist_pelota | 243.144 | 298.337 | 0.815 | 0.721–0.908 | sí |
| MED | 604 | patadas_efectivas_min | 3.841 | 7.127 | 0.539 | 0.485–0.596 | **no** |
| MED | 604 | toques_min | 9.433 | 6.781 | 1.391 | 1.23–1.566 | sí |
| MED | 604 | vel | 2.174 | 2.315 | 0.939 | 0.929–0.95 | sí |
| MED | 604 | dist_pelota | 212.922 | 265.606 | 0.802 | 0.764–0.836 | sí |
| DEL | 351 | patadas_efectivas_min | 3.376 | 4.705 | 0.718 | 0.619–0.832 | sí |
| DEL | 351 | toques_min | 8.185 | 6.167 | 1.327 | 1.138–1.56 | sí |
| DEL | 351 | vel | 2.277 | 2.424 | 0.94 | 0.927–0.952 | sí |
| DEL | 351 | dist_pelota | 235.418 | 285.939 | 0.823 | 0.785–0.866 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 294.922 | 281.572 | 1.047 | 1.006–1.092 | sí |
| ARQ | 123 | dist_pelota | 1076.115 | 1065.23 | 1.01 | 0.991–1.035 | sí |
| ARQ | 123 | vel | 1.671 | 1.76 | 0.949 | 0.897–1.004 | sí |
| ARQ | 123 | toques_min | 0.263 | 0.292 | 0.903 | 0.309–10.731 | sí |
| DEF | 542 | dist_ancla | 187.805 | 172.993 | 1.086 | 1.054–1.117 | sí |
| DEF | 542 | dist_pelota | 564.245 | 568.521 | 0.992 | 0.978–1.008 | sí |
| DEF | 542 | vel | 2.173 | 2.226 | 0.976 | 0.962–0.991 | sí |
| DEF | 542 | toques_min | 1.002 | 0.502 | 1.994 | 1.386–3.028 | **no** |
| MCD | 73 | dist_ancla | 198.029 | 180.971 | 1.094 | 1.032–1.159 | sí |
| MCD | 73 | dist_pelota | 449.749 | 441.983 | 1.018 | 0.98–1.054 | sí |
| MCD | 73 | vel | 2.304 | 2.344 | 0.983 | 0.955–1.01 | sí |
| MCD | 73 | toques_min | 0.385 | 0.368 | 1.046 | 0.207–11.875 | sí |
| MED | 546 | dist_ancla | 191.863 | 178.641 | 1.074 | 1.046–1.108 | sí |
| MED | 546 | dist_pelota | 530.986 | 542.775 | 0.978 | 0.965–0.993 | sí |
| MED | 546 | vel | 2.227 | 2.327 | 0.957 | 0.943–0.971 | sí |
| MED | 546 | toques_min | 1.004 | 0.755 | 1.329 | 0.93–2.096 | sí |
| DEL | 316 | dist_ancla | 207.268 | 189.372 | 1.095 | 1.062–1.131 | sí |
| DEL | 316 | dist_pelota | 537.82 | 554.796 | 0.969 | 0.949–0.99 | sí |
| DEL | 316 | vel | 2.285 | 2.335 | 0.979 | 0.96–0.997 | sí |
| DEL | 316 | toques_min | 1.036 | 0.618 | 1.677 | 1.122–3.208 | sí |

