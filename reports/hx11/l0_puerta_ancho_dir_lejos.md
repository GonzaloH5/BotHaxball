# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/l0_sinhist_ancho/model_final.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 5.406 | 7.153 | 0.756 | 0.698–0.822 | sí |
| DEF | 552 | toques_min | 5.599 | 4.637 | 1.207 | 1.029–1.391 | sí |
| DEF | 552 | vel | 2.159 | 2.286 | 0.945 | 0.933–0.957 | sí |
| DEF | 552 | dist_pelota | 238.999 | 278.671 | 0.858 | 0.824–0.891 | sí |
| MCD | 92 | patadas_efectivas_min | 4.309 | 6.255 | 0.689 | 0.519–0.878 | sí |
| MCD | 92 | toques_min | 2.657 | 2.217 | 1.199 | 0.842–1.826 | sí |
| MCD | 92 | vel | 2.08 | 2.278 | 0.913 | 0.886–0.941 | sí |
| MCD | 92 | dist_pelota | 250.798 | 293.262 | 0.855 | 0.757–0.948 | sí |
| MED | 604 | patadas_efectivas_min | 5.381 | 7.089 | 0.759 | 0.696–0.821 | sí |
| MED | 604 | toques_min | 5.643 | 6.802 | 0.83 | 0.738–0.922 | sí |
| MED | 604 | vel | 2.144 | 2.321 | 0.924 | 0.912–0.935 | sí |
| MED | 604 | dist_pelota | 228.155 | 267.127 | 0.854 | 0.818–0.891 | sí |
| DEL | 351 | patadas_efectivas_min | 4.567 | 4.859 | 0.94 | 0.818–1.071 | sí |
| DEL | 351 | toques_min | 5.19 | 6.493 | 0.799 | 0.694–0.924 | sí |
| DEL | 351 | vel | 2.264 | 2.415 | 0.938 | 0.924–0.954 | sí |
| DEL | 351 | dist_pelota | 248.712 | 282.997 | 0.879 | 0.841–0.926 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 293.972 | 281.617 | 1.044 | 1.002–1.091 | sí |
| ARQ | 123 | dist_pelota | 1086.277 | 1065.121 | 1.02 | 1.0–1.044 | sí |
| ARQ | 123 | vel | 1.63 | 1.76 | 0.926 | 0.875–0.978 | sí |
| ARQ | 123 | toques_min | 0.222 | 0.292 | 0.76 | 0.093–11.252 | sí |
| DEF | 542 | dist_ancla | 184.852 | 173.063 | 1.068 | 1.039–1.098 | sí |
| DEF | 542 | dist_pelota | 560.204 | 568.425 | 0.986 | 0.971–1.001 | sí |
| DEF | 542 | vel | 2.186 | 2.227 | 0.982 | 0.968–0.995 | sí |
| DEF | 542 | toques_min | 0.454 | 0.502 | 0.905 | 0.597–1.347 | sí |
| MCD | 73 | dist_ancla | 198.973 | 181.025 | 1.099 | 1.036–1.167 | sí |
| MCD | 73 | dist_pelota | 455.735 | 441.916 | 1.031 | 0.996–1.067 | sí |
| MCD | 73 | vel | 2.288 | 2.343 | 0.976 | 0.953–1.001 | sí |
| MCD | 73 | toques_min | 0.357 | 0.369 | 0.969 | 0.177–11.208 | sí |
| MED | 546 | dist_ancla | 190.215 | 178.82 | 1.064 | 1.034–1.099 | sí |
| MED | 546 | dist_pelota | 530.217 | 543.148 | 0.976 | 0.963–0.991 | sí |
| MED | 546 | vel | 2.229 | 2.326 | 0.958 | 0.945–0.973 | sí |
| MED | 546 | toques_min | 0.795 | 0.743 | 1.069 | 0.712–1.663 | sí |
| DEL | 316 | dist_ancla | 203.115 | 189.391 | 1.072 | 1.038–1.111 | sí |
| DEL | 316 | dist_pelota | 535.956 | 554.662 | 0.966 | 0.947–0.987 | sí |
| DEL | 316 | vel | 2.267 | 2.335 | 0.971 | 0.952–0.989 | sí |
| DEL | 316 | toques_min | 0.441 | 0.618 | 0.714 | 0.41–1.543 | sí |

