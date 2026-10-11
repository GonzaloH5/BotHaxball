# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 294.508 | 316.202 | 0.931 | 0.858–1.009 | sí |
| ARQ | disp_x | 190.844 | 211.666 | 0.902 | 0.811–1.009 | sí |
| ARQ | disp_y | 181.99 | 182.725 | 0.996 | 0.922–1.067 | sí |
| ARQ | lado_arco | 0.987 | 0.985 | 1.002 | 0.997–1.006 | sí |
| ARQ | repliegue | 3.311 | 7.999 | 0.414 | -2.574–4.891 | sí |
| ARQ | va_a_la_pelota | 0.335 | 0.432 | 0.774 | -0.912–3.088 | sí |
| ARQ | cerca_pelota | 0.028 | 0.028 | 1.02 | 0.809–1.253 | sí |
| ARQ | dist_pelota | 1038.711 | 1049.582 | 0.99 | 0.951–1.027 | sí |
| ARQ | vel | 1.713 | 1.735 | 0.987 | 0.918–1.066 | sí |
| ARQ | cambios_tecla_s | 4.577 | 2.704 | 1.693 | 1.502–1.911 | **no** |
| ARQ | congelado (s por min de juego abierto) | 0.0 | 1.282 | | | |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 210.473 | 171.612 | 1.226 | 1.132–1.331 | sí |
| DEF | disp_x | 119.857 | 111.85 | 1.072 | 0.985–1.181 | sí |
| DEF | disp_y | 119.208 | 114.577 | 1.04 | 0.953–1.128 | sí |
| DEF | lado_arco | 0.79 | 0.761 | 1.039 | 1.015–1.063 | sí |
| DEF | repliegue | -14.259 | -9.576 | 1.489 | -1.524–5.255 | sí |
| DEF | va_a_la_pelota | 0.015 | -0.344 | -0.043 | -1.7–0.818 | sí |
| DEF | cerca_pelota | 0.07 | 0.085 | 0.818 | 0.706–0.931 | sí |
| DEF | dist_pelota | 572.699 | 561.926 | 1.019 | 0.986–1.057 | sí |
| DEF | vel | 2.159 | 2.175 | 0.992 | 0.978–1.008 | sí |
| DEF | cambios_tecla_s | 4.694 | 3.668 | 1.28 | 1.209–1.351 | sí |
| DEF | congelado (s por min de juego abierto) | 0.1 | 0.059 | | | |
| DEF | estorbo (s por saque, 144 saques) | 0.003 | 0.0 | | | |
| MCD | zona | 200.268 | 181.463 | 1.104 | 0.967–1.253 | sí |
| MCD | disp_x | 121.892 | 114.43 | 1.065 | 0.932–1.248 | sí |
| MCD | disp_y | 151.608 | 129.862 | 1.167 | 1.03–1.301 | sí |
| MCD | lado_arco | 0.637 | 0.62 | 1.029 | 0.986–1.077 | sí |
| MCD | repliegue | -21.853 | 6.419 | -3.405 | -9.723–8.103 | sí |
| MCD | va_a_la_pelota | 0.166 | -0.11 | -1.507 | -8.82–9.542 | sí |
| MCD | cerca_pelota | 0.078 | 0.094 | 0.825 | 0.584–1.221 | sí |
| MCD | dist_pelota | 411.407 | 385.82 | 1.066 | 0.997–1.139 | sí |
| MCD | vel | 2.364 | 2.322 | 1.018 | 0.968–1.075 | sí |
| MCD | cambios_tecla_s | 4.686 | 4.11 | 1.14 | 0.909–1.525 | sí |
| MCD | congelado (s por min de juego abierto) | 0.0 | 0.0 | | | |
| MCD | estorbo (s por saque, 19 saques) | 0.01 | 0.0 | | | |
| MED | zona | 232.268 | 179.592 | 1.293 | 1.19–1.398 | sí |
| MED | disp_x | 134.121 | 116.174 | 1.154 | 1.063–1.253 | sí |
| MED | disp_y | 134.024 | 122.27 | 1.096 | 1.021–1.185 | sí |
| MED | lado_arco | 0.426 | 0.465 | 0.916 | 0.863–0.969 | sí |
| MED | repliegue | -10.964 | 1.306 | -8.397 | -8.278–10.54 | sí |
| MED | va_a_la_pelota | 0.24 | 0.126 | 1.901 | -4.44–7.264 | sí |
| MED | cerca_pelota | 0.103 | 0.095 | 1.083 | 0.962–1.213 | sí |
| MED | dist_pelota | 465.009 | 506.871 | 0.917 | 0.876–0.963 | sí |
| MED | vel | 2.292 | 2.252 | 1.017 | 1.002–1.033 | sí |
| MED | cambios_tecla_s | 4.817 | 3.435 | 1.402 | 1.323–1.484 | sí |
| MED | congelado (s por min de juego abierto) | 0.0 | 0.057 | | | |
| MED | estorbo (s por saque, 143 saques) | 0.003 | 0.002 | | | |
| DEL | zona | 230.876 | 213.827 | 1.08 | 0.982–1.184 | sí |
| DEL | disp_x | 122.013 | 118.137 | 1.033 | 0.947–1.126 | sí |
| DEL | disp_y | 137.324 | 138.519 | 0.991 | 0.889–1.082 | sí |
| DEL | lado_arco | 0.219 | 0.26 | 0.843 | 0.79–0.916 | sí |
| DEL | repliegue | 0.755 | 7.76 | 0.097 | -2.671–5.37 | sí |
| DEL | va_a_la_pelota | 0.035 | 0.234 | 0.149 | -1.604–1.398 | sí |
| DEL | cerca_pelota | 0.065 | 0.075 | 0.865 | 0.683–1.076 | sí |
| DEL | dist_pelota | 567.524 | 572.986 | 0.99 | 0.934–1.063 | sí |
| DEL | vel | 2.183 | 2.197 | 0.994 | 0.97–1.018 | sí |
| DEL | cambios_tecla_s | 4.599 | 3.364 | 1.367 | 1.267–1.48 | sí |
| DEL | congelado (s por min de juego abierto) | 0.0 | 0.187 | | | |
| DEL | estorbo (s por saque, 89 saques) | 0.0 | 0.0 | | | |
