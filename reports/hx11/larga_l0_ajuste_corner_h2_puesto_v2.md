# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 321.151 | 316.202 | 1.016 | 0.92–1.121 | sí |
| ARQ | disp_x | 199.924 | 211.666 | 0.945 | 0.829–1.086 | sí |
| ARQ | disp_y | 197.592 | 182.725 | 1.081 | 0.989–1.177 | sí |
| ARQ | lado_arco | 0.988 | 0.985 | 1.003 | 0.989–1.016 | sí |
| ARQ | repliegue | -4.062 | 7.999 | -0.508 | -6.144–4.613 | sí |
| ARQ | va_a_la_pelota | 0.315 | 0.432 | 0.729 | -1.307–2.647 | sí |
| ARQ | cerca_pelota | 0.03 | 0.028 | 1.081 | 0.805–1.379 | sí |
| ARQ | dist_pelota | 1077.563 | 1049.582 | 1.027 | 0.975–1.083 | sí |
| ARQ | vel | 1.67 | 1.735 | 0.962 | 0.886–1.051 | sí |
| ARQ | cambios_tecla_s | 4.453 | 2.704 | 1.647 | 1.444–1.86 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 301.72 | 171.612 | 1.758 | 1.606–1.921 | **no** |
| DEF | disp_x | 149.391 | 111.85 | 1.336 | 1.224–1.46 | sí |
| DEF | disp_y | 137.025 | 114.577 | 1.196 | 1.074–1.337 | sí |
| DEF | lado_arco | 0.674 | 0.761 | 0.885 | 0.846–0.925 | sí |
| DEF | repliegue | -16.644 | -9.576 | 1.738 | -1.664–6.323 | sí |
| DEF | va_a_la_pelota | -0.011 | -0.256 | 0.041 | -2.231–1.255 | sí |
| DEF | cerca_pelota | 0.076 | 0.085 | 0.894 | 0.782–1.021 | sí |
| DEF | dist_pelota | 611.783 | 561.926 | 1.089 | 1.037–1.147 | sí |
| DEF | vel | 2.202 | 2.175 | 1.012 | 0.983–1.035 | sí |
| DEF | cambios_tecla_s | 4.597 | 3.668 | 1.253 | 1.174–1.334 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.025 | 0.0 | | | |
| MCD | zona | 198.212 | 181.463 | 1.092 | 0.907–1.309 | sí |
| MCD | disp_x | 136.488 | 114.43 | 1.193 | 1.012–1.417 | sí |
| MCD | disp_y | 135.82 | 129.862 | 1.046 | 0.921–1.172 | sí |
| MCD | lado_arco | 0.616 | 0.62 | 0.995 | 0.922–1.069 | sí |
| MCD | repliegue | 2.714 | 6.419 | 0.423 | -2.66–3.067 | sí |
| MCD | va_a_la_pelota | 0.25 | -0.117 | -2.138 | -7.601–6.14 | sí |
| MCD | cerca_pelota | 0.07 | 0.094 | 0.75 | 0.515–1.164 | sí |
| MCD | dist_pelota | 437.986 | 385.82 | 1.135 | 1.062–1.208 | sí |
| MCD | vel | 2.322 | 2.322 | 1.0 | 0.951–1.053 | sí |
| MCD | cambios_tecla_s | 4.507 | 4.11 | 1.097 | 0.891–1.439 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.0 | 0.0 | | | |
| MED | zona | 227.887 | 179.592 | 1.269 | 1.157–1.403 | sí |
| MED | disp_x | 135.467 | 116.174 | 1.166 | 1.059–1.282 | sí |
| MED | disp_y | 130.83 | 122.27 | 1.07 | 0.997–1.149 | sí |
| MED | lado_arco | 0.406 | 0.465 | 0.873 | 0.817–0.927 | sí |
| MED | repliegue | -13.39 | 1.306 | -10.256 | -10.638–14.621 | sí |
| MED | va_a_la_pelota | 0.117 | 0.172 | 0.679 | -1.283–3.548 | sí |
| MED | cerca_pelota | 0.091 | 0.095 | 0.961 | 0.841–1.077 | sí |
| MED | dist_pelota | 504.473 | 506.871 | 0.995 | 0.951–1.049 | sí |
| MED | vel | 2.235 | 2.252 | 0.992 | 0.971–1.012 | sí |
| MED | cambios_tecla_s | 4.725 | 3.435 | 1.375 | 1.292–1.468 | sí |
| MED | estorbo (s por saque, 143 saques) | 0.013 | 0.002 | | | |
| DEL | zona | 245.534 | 213.827 | 1.148 | 1.008–1.286 | sí |
| DEL | disp_x | 138.071 | 118.137 | 1.169 | 1.031–1.33 | sí |
| DEL | disp_y | 140.519 | 138.519 | 1.014 | 0.885–1.129 | sí |
| DEL | lado_arco | 0.237 | 0.26 | 0.911 | 0.824–1.03 | sí |
| DEL | repliegue | 4.525 | 7.76 | 0.583 | -1.962–3.298 | sí |
| DEL | va_a_la_pelota | -0.005 | 0.142 | -0.036 | -2.359–1.767 | sí |
| DEL | cerca_pelota | 0.073 | 0.075 | 0.978 | 0.821–1.138 | sí |
| DEL | dist_pelota | 584.658 | 572.986 | 1.02 | 0.951–1.107 | sí |
| DEL | vel | 2.103 | 2.197 | 0.957 | 0.916–0.997 | sí |
| DEL | cambios_tecla_s | 4.553 | 3.364 | 1.354 | 1.243–1.484 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.039 | 0.0 | | | |
