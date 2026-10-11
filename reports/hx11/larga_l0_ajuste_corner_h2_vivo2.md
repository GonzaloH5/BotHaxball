# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 309.861 | 316.202 | 0.98 | 0.896–1.068 | sí |
| ARQ | disp_x | 198.869 | 211.666 | 0.94 | 0.823–1.073 | sí |
| ARQ | disp_y | 190.641 | 182.725 | 1.043 | 0.966–1.121 | sí |
| ARQ | lado_arco | 0.984 | 0.985 | 0.999 | 0.997–1.001 | sí |
| ARQ | repliegue | 7.653 | 7.999 | 0.957 | -2.806–6.466 | sí |
| ARQ | va_a_la_pelota | 0.425 | 0.588 | 0.723 | 0.082–1.6 | sí |
| ARQ | cerca_pelota | 0.028 | 0.028 | 0.996 | 0.772–1.258 | sí |
| ARQ | dist_pelota | 1054.8 | 1049.582 | 1.005 | 0.966–1.043 | sí |
| ARQ | vel | 1.639 | 1.735 | 0.944 | 0.883–1.029 | sí |
| ARQ | cambios_tecla_s | 4.374 | 2.704 | 1.618 | 1.441–1.804 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 204.728 | 171.612 | 1.193 | 1.115–1.276 | sí |
| DEF | disp_x | 122.503 | 111.85 | 1.095 | 1.018–1.183 | sí |
| DEF | disp_y | 115.677 | 114.577 | 1.01 | 0.928–1.089 | sí |
| DEF | lado_arco | 0.794 | 0.761 | 1.043 | 1.018–1.069 | sí |
| DEF | repliegue | -13.32 | -9.576 | 1.391 | -1.638–4.6 | sí |
| DEF | va_a_la_pelota | -0.119 | -0.333 | 0.357 | -0.969–1.55 | sí |
| DEF | cerca_pelota | 0.069 | 0.085 | 0.81 | 0.716–0.916 | sí |
| DEF | dist_pelota | 574.138 | 561.926 | 1.022 | 0.986–1.056 | sí |
| DEF | vel | 2.14 | 2.175 | 0.984 | 0.97–0.999 | sí |
| DEF | cambios_tecla_s | 4.615 | 3.668 | 1.258 | 1.184–1.331 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.005 | 0.0 | | | |
| MCD | zona | 195.33 | 181.463 | 1.076 | 0.932–1.237 | sí |
| MCD | disp_x | 121.377 | 114.43 | 1.061 | 0.91–1.252 | sí |
| MCD | disp_y | 150.648 | 129.862 | 1.16 | 1.017–1.301 | sí |
| MCD | lado_arco | 0.636 | 0.62 | 1.026 | 0.981–1.076 | sí |
| MCD | repliegue | -9.711 | 6.419 | -1.513 | -6.047–5.444 | sí |
| MCD | va_a_la_pelota | -0.011 | -0.11 | 0.1 | -3.326–4.923 | sí |
| MCD | cerca_pelota | 0.073 | 0.094 | 0.778 | 0.53–1.184 | sí |
| MCD | dist_pelota | 409.523 | 385.82 | 1.061 | 0.984–1.14 | sí |
| MCD | vel | 2.365 | 2.322 | 1.019 | 0.97–1.074 | sí |
| MCD | cambios_tecla_s | 4.503 | 4.11 | 1.096 | 0.864–1.458 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.0 | 0.0 | | | |
| MED | zona | 232.098 | 179.592 | 1.292 | 1.19–1.395 | sí |
| MED | disp_x | 134.725 | 116.174 | 1.16 | 1.066–1.26 | sí |
| MED | disp_y | 134.996 | 122.27 | 1.104 | 1.03–1.196 | sí |
| MED | lado_arco | 0.428 | 0.465 | 0.92 | 0.867–0.972 | sí |
| MED | repliegue | -6.133 | 1.306 | -4.698 | -4.829–5.339 | sí |
| MED | va_a_la_pelota | 0.187 | 0.165 | 1.133 | -1.872–6.197 | sí |
| MED | cerca_pelota | 0.104 | 0.095 | 1.094 | 0.975–1.225 | sí |
| MED | dist_pelota | 462.534 | 506.871 | 0.913 | 0.873–0.959 | sí |
| MED | vel | 2.284 | 2.252 | 1.014 | 1.0–1.029 | sí |
| MED | cambios_tecla_s | 4.796 | 3.435 | 1.396 | 1.314–1.482 | sí |
| MED | estorbo (s por saque, 143 saques) | 0.007 | 0.002 | | | |
| DEL | zona | 236.33 | 213.827 | 1.105 | 0.992–1.224 | sí |
| DEL | disp_x | 126.316 | 118.137 | 1.069 | 0.977–1.176 | sí |
| DEL | disp_y | 143.723 | 138.519 | 1.038 | 0.938–1.148 | sí |
| DEL | lado_arco | 0.217 | 0.26 | 0.836 | 0.778–0.906 | sí |
| DEL | repliegue | 0.017 | 7.76 | 0.002 | -3.006–6.288 | sí |
| DEL | va_a_la_pelota | 0.043 | 0.265 | 0.163 | -1.606–1.083 | sí |
| DEL | cerca_pelota | 0.065 | 0.075 | 0.873 | 0.71–1.076 | sí |
| DEL | dist_pelota | 579.416 | 572.986 | 1.011 | 0.947–1.092 | sí |
| DEL | vel | 2.119 | 2.197 | 0.965 | 0.934–0.994 | sí |
| DEL | cambios_tecla_s | 4.552 | 3.364 | 1.353 | 1.25–1.477 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.035 | 0.0 | | | |
