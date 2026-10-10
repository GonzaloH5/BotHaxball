# Puerta P3 de L0 por puesto (partidos de prueba)

Con [`learn/hx11_puerta.py`](../../learn/hx11_puerta.py). Modelo `runs/hx11_l0/rl_a3_p4/model_40.pt`, temperatura 0.7, decisión cada 3 ticks, ruido persistente (renovación 0.33), patada persistente (renovación 0.33), 20 partidos de prueba. Razón bot/humano en las mismas ventanas, con IC 90%; **pasa** si el IC se superpone con [0.75; 1.33].

## Ventanas cerca de la pelota

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| DEF | 552 | patadas_efectivas_min | 4.053 | 7.29 | 0.556 | 0.5–0.611 | **no** |
| DEF | 552 | toques_min | 7.412 | 4.644 | 1.596 | 1.392–1.849 | **no** |
| DEF | 552 | vel | 2.164 | 2.286 | 0.947 | 0.934–0.959 | sí |
| DEF | 552 | dist_pelota | 221.28 | 277.861 | 0.796 | 0.762–0.826 | sí |
| MCD | 92 | patadas_efectivas_min | 3.173 | 5.932 | 0.535 | 0.364–0.704 | **no** |
| MCD | 92 | toques_min | 5.013 | 2.201 | 2.278 | 1.395–3.813 | **no** |
| MCD | 92 | vel | 2.1 | 2.296 | 0.915 | 0.89–0.942 | sí |
| MCD | 92 | dist_pelota | 232.158 | 299.298 | 0.776 | 0.678–0.868 | sí |
| MED | 604 | patadas_efectivas_min | 4.202 | 7.096 | 0.592 | 0.541–0.65 | **no** |
| MED | 604 | toques_min | 7.794 | 6.777 | 1.15 | 1.017–1.293 | sí |
| MED | 604 | vel | 2.138 | 2.32 | 0.921 | 0.91–0.932 | sí |
| MED | 604 | dist_pelota | 205.107 | 264.788 | 0.775 | 0.739–0.81 | sí |
| DEL | 351 | patadas_efectivas_min | 3.564 | 4.788 | 0.744 | 0.647–0.861 | sí |
| DEL | 351 | toques_min | 6.655 | 6.333 | 1.051 | 0.889–1.265 | sí |
| DEL | 351 | vel | 2.252 | 2.411 | 0.934 | 0.921–0.948 | sí |
| DEL | 351 | dist_pelota | 233.815 | 283.693 | 0.824 | 0.783–0.868 | sí |

## Ventanas al azar

| Línea | Ventanas | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|---|
| ARQ | 123 | dist_ancla | 294.542 | 281.572 | 1.046 | 1.004–1.087 | sí |
| ARQ | 123 | dist_pelota | 1070.022 | 1065.23 | 1.004 | 0.985–1.028 | sí |
| ARQ | 123 | vel | 1.713 | 1.76 | 0.974 | 0.917–1.03 | sí |
| ARQ | 123 | toques_min | 0.14 | 0.292 | 0.48 | 0.198–5.967 | sí |
| DEF | 542 | dist_ancla | 185.067 | 173.065 | 1.069 | 1.04–1.097 | sí |
| DEF | 542 | dist_pelota | 558.514 | 568.891 | 0.982 | 0.968–0.997 | sí |
| DEF | 542 | vel | 2.156 | 2.227 | 0.968 | 0.955–0.982 | sí |
| DEF | 542 | toques_min | 0.766 | 0.486 | 1.576 | 1.039–2.405 | sí |
| MCD | 73 | dist_ancla | 199.014 | 180.971 | 1.1 | 1.042–1.166 | sí |
| MCD | 73 | dist_pelota | 452.986 | 441.983 | 1.025 | 0.984–1.062 | sí |
| MCD | 73 | vel | 2.309 | 2.344 | 0.985 | 0.963–1.009 | sí |
| MCD | 73 | toques_min | 0.371 | 0.368 | 1.008 | 0.111–9.01 | sí |
| MED | 546 | dist_ancla | 187.924 | 178.856 | 1.051 | 1.024–1.083 | sí |
| MED | 546 | dist_pelota | 524.063 | 543.027 | 0.965 | 0.95–0.979 | sí |
| MED | 546 | vel | 2.256 | 2.326 | 0.97 | 0.957–0.983 | sí |
| MED | 546 | toques_min | 1.084 | 0.703 | 1.542 | 1.069–2.36 | sí |
| DEL | 316 | dist_ancla | 204.891 | 190.014 | 1.078 | 1.048–1.112 | sí |
| DEL | 316 | dist_pelota | 539.475 | 551.943 | 0.977 | 0.959–0.998 | sí |
| DEL | 316 | vel | 2.262 | 2.33 | 0.971 | 0.954–0.989 | sí |
| DEL | 316 | toques_min | 0.577 | 0.619 | 0.933 | 0.533–2.017 | sí |

