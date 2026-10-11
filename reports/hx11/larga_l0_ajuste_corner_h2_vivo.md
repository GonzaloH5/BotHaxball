# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 295.568 | 316.202 | 0.935 | 0.849–1.022 | sí |
| ARQ | disp_x | 185.731 | 211.666 | 0.877 | 0.787–0.982 | sí |
| ARQ | disp_y | 190.307 | 182.725 | 1.041 | 0.955–1.129 | sí |
| ARQ | lado_arco | 0.99 | 0.985 | 1.006 | 0.998–1.015 | sí |
| ARQ | repliegue | 8.665 | 7.999 | 1.083 | -3.221–7.459 | sí |
| ARQ | va_a_la_pelota | 0.401 | 0.432 | 0.928 | -0.491–2.713 | sí |
| ARQ | cerca_pelota | 0.028 | 0.028 | 1.029 | 0.822–1.28 | sí |
| ARQ | dist_pelota | 1050.276 | 1049.582 | 1.001 | 0.959–1.038 | sí |
| ARQ | vel | 1.696 | 1.735 | 0.977 | 0.9–1.068 | sí |
| ARQ | cambios_tecla_s | 4.49 | 2.704 | 1.661 | 1.485–1.845 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 207.317 | 171.612 | 1.208 | 1.114–1.31 | sí |
| DEF | disp_x | 126.251 | 111.85 | 1.129 | 1.059–1.214 | sí |
| DEF | disp_y | 115.69 | 114.577 | 1.01 | 0.918–1.104 | sí |
| DEF | lado_arco | 0.788 | 0.761 | 1.036 | 1.014–1.061 | sí |
| DEF | repliegue | -11.845 | -9.576 | 1.237 | -1.128–3.871 | sí |
| DEF | va_a_la_pelota | -0.091 | -0.357 | 0.255 | -0.78–1.164 | sí |
| DEF | cerca_pelota | 0.07 | 0.085 | 0.824 | 0.734–0.926 | sí |
| DEF | dist_pelota | 583.04 | 561.926 | 1.038 | 0.998–1.087 | sí |
| DEF | vel | 2.146 | 2.175 | 0.987 | 0.969–1.004 | sí |
| DEF | cambios_tecla_s | 4.68 | 3.668 | 1.276 | 1.206–1.35 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.01 | 0.0 | | | |
| MCD | zona | 242.4 | 181.463 | 1.336 | 1.18–1.531 | sí |
| MCD | disp_x | 134.627 | 114.43 | 1.177 | 1.028–1.36 | sí |
| MCD | disp_y | 147.587 | 129.862 | 1.136 | 0.989–1.279 | sí |
| MCD | lado_arco | 0.545 | 0.62 | 0.88 | 0.814–0.944 | sí |
| MCD | repliegue | -13.144 | 6.419 | -2.048 | -7.205–5.493 | sí |
| MCD | va_a_la_pelota | 0.07 | -0.088 | -0.791 | -5.497–7.123 | sí |
| MCD | cerca_pelota | 0.098 | 0.094 | 1.045 | 0.765–1.461 | sí |
| MCD | dist_pelota | 413.574 | 385.82 | 1.072 | 0.986–1.159 | sí |
| MCD | vel | 2.34 | 2.322 | 1.008 | 0.961–1.059 | sí |
| MCD | cambios_tecla_s | 4.691 | 4.11 | 1.142 | 0.919–1.514 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.012 | 0.0 | | | |
| MED | zona | 221.429 | 179.592 | 1.233 | 1.123–1.372 | sí |
| MED | disp_x | 131.83 | 116.174 | 1.135 | 1.033–1.255 | sí |
| MED | disp_y | 126.225 | 122.27 | 1.032 | 0.965–1.106 | sí |
| MED | lado_arco | 0.424 | 0.465 | 0.912 | 0.863–0.96 | sí |
| MED | repliegue | -13.245 | 1.306 | -10.144 | -10.129–12.751 | sí |
| MED | va_a_la_pelota | 0.269 | 0.165 | 1.632 | -2.525–8.438 | sí |
| MED | cerca_pelota | 0.103 | 0.095 | 1.085 | 0.95–1.231 | sí |
| MED | dist_pelota | 479.325 | 506.871 | 0.946 | 0.906–0.99 | sí |
| MED | vel | 2.261 | 2.252 | 1.004 | 0.986–1.023 | sí |
| MED | cambios_tecla_s | 4.783 | 3.435 | 1.392 | 1.308–1.483 | sí |
| MED | estorbo (s por saque, 143 saques) | 0.008 | 0.002 | | | |
| DEL | zona | 232.972 | 213.827 | 1.09 | 0.985–1.214 | sí |
| DEL | disp_x | 133.125 | 118.137 | 1.127 | 1.025–1.238 | sí |
| DEL | disp_y | 144.813 | 138.519 | 1.045 | 0.938–1.162 | sí |
| DEL | lado_arco | 0.214 | 0.26 | 0.823 | 0.764–0.897 | sí |
| DEL | repliegue | -6.396 | 7.76 | -0.824 | -5.87–8.545 | sí |
| DEL | va_a_la_pelota | -0.146 | 0.237 | -0.615 | -3.223–0.349 | **no** |
| DEL | cerca_pelota | 0.069 | 0.075 | 0.923 | 0.748–1.123 | sí |
| DEL | dist_pelota | 579.381 | 572.986 | 1.011 | 0.943–1.091 | sí |
| DEL | vel | 2.133 | 2.197 | 0.971 | 0.947–0.996 | sí |
| DEL | cambios_tecla_s | 4.573 | 3.364 | 1.359 | 1.265–1.471 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.0 | 0.0 | | | |
