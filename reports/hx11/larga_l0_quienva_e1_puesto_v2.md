# Prueba larga (fantasma 60 s entre humanos): l0_quienva_e1

Modelo `runs/hx11_l0/l0_quienva_e1/model.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 302.778 | 316.202 | 0.958 | 0.875–1.057 | sí |
| ARQ | disp_x | 172.882 | 211.666 | 0.817 | 0.729–0.919 | sí |
| ARQ | disp_y | 199.415 | 182.725 | 1.091 | 0.991–1.206 | sí |
| ARQ | lado_arco | 0.991 | 0.985 | 1.006 | 0.998–1.015 | sí |
| ARQ | repliegue | 11.28 | 7.999 | 1.41 | -5.548–8.025 | sí |
| ARQ | va_a_la_pelota | 0.506 | 0.432 | 1.171 | -2.074–4.438 | sí |
| ARQ | cerca_pelota | 0.026 | 0.028 | 0.935 | 0.698–1.188 | sí |
| ARQ | dist_pelota | 1064.173 | 1049.582 | 1.014 | 0.971–1.059 | sí |
| ARQ | vel | 1.682 | 1.735 | 0.969 | 0.897–1.049 | sí |
| ARQ | cambios_tecla_s | 4.479 | 2.704 | 1.656 | 1.463–1.867 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 301.314 | 171.612 | 1.756 | 1.614–1.908 | **no** |
| DEF | disp_x | 150.558 | 111.85 | 1.346 | 1.243–1.466 | sí |
| DEF | disp_y | 138.57 | 114.577 | 1.209 | 1.1–1.321 | sí |
| DEF | lado_arco | 0.686 | 0.761 | 0.901 | 0.861–0.938 | sí |
| DEF | repliegue | -26.578 | -9.576 | 2.775 | -7.788–12.667 | sí |
| DEF | va_a_la_pelota | 0.029 | -0.424 | -0.068 | -1.258–0.57 | **no** |
| DEF | cerca_pelota | 0.071 | 0.085 | 0.829 | 0.711–0.961 | sí |
| DEF | dist_pelota | 613.021 | 561.926 | 1.091 | 1.042–1.146 | sí |
| DEF | vel | 2.192 | 2.175 | 1.008 | 0.985–1.027 | sí |
| DEF | cambios_tecla_s | 4.572 | 3.668 | 1.246 | 1.173–1.323 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.011 | 0.0 | | | |
| MCD | zona | 198.399 | 181.463 | 1.093 | 0.964–1.241 | sí |
| MCD | disp_x | 123.217 | 114.43 | 1.077 | 0.917–1.246 | sí |
| MCD | disp_y | 147.87 | 129.862 | 1.139 | 0.999–1.283 | sí |
| MCD | lado_arco | 0.619 | 0.62 | 1.0 | 0.964–1.04 | sí |
| MCD | repliegue | 1.908 | 6.419 | 0.297 | -4.589–5.134 | sí |
| MCD | va_a_la_pelota | 0.248 | -0.088 | -2.82 | -9.595–13.682 | sí |
| MCD | cerca_pelota | 0.089 | 0.094 | 0.946 | 0.638–1.369 | sí |
| MCD | dist_pelota | 419.073 | 385.82 | 1.086 | 1.006–1.163 | sí |
| MCD | vel | 2.302 | 2.322 | 0.992 | 0.94–1.05 | sí |
| MCD | cambios_tecla_s | 4.754 | 4.11 | 1.157 | 0.949–1.516 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.034 | 0.0 | | | |
| MED | zona | 220.841 | 179.592 | 1.23 | 1.14–1.338 | sí |
| MED | disp_x | 127.173 | 116.174 | 1.095 | 1.01–1.18 | sí |
| MED | disp_y | 132.025 | 122.27 | 1.08 | 1.0–1.162 | sí |
| MED | lado_arco | 0.402 | 0.465 | 0.864 | 0.804–0.92 | sí |
| MED | repliegue | -10.558 | 1.306 | -8.087 | -7.642–10.216 | sí |
| MED | va_a_la_pelota | 0.048 | 0.257 | 0.187 | -0.715–1.17 | sí |
| MED | cerca_pelota | 0.084 | 0.095 | 0.889 | 0.772–1.011 | sí |
| MED | dist_pelota | 509.522 | 506.871 | 1.005 | 0.967–1.051 | sí |
| MED | vel | 2.238 | 2.252 | 0.994 | 0.976–1.012 | sí |
| MED | cambios_tecla_s | 4.756 | 3.435 | 1.384 | 1.301–1.477 | sí |
| MED | estorbo (s por saque, 143 saques) | 0.016 | 0.002 | | | |
| DEL | zona | 230.018 | 213.827 | 1.076 | 0.991–1.169 | sí |
| DEL | disp_x | 132.091 | 118.137 | 1.118 | 1.015–1.222 | sí |
| DEL | disp_y | 137.167 | 138.519 | 0.99 | 0.884–1.099 | sí |
| DEL | lado_arco | 0.217 | 0.26 | 0.832 | 0.763–0.917 | sí |
| DEL | repliegue | 1.986 | 7.76 | 0.256 | -3.372–4.568 | sí |
| DEL | va_a_la_pelota | 0.015 | 0.212 | 0.071 | -2.748–1.252 | sí |
| DEL | cerca_pelota | 0.073 | 0.075 | 0.979 | 0.85–1.135 | sí |
| DEL | dist_pelota | 571.434 | 572.986 | 0.997 | 0.936–1.075 | sí |
| DEL | vel | 2.149 | 2.197 | 0.978 | 0.953–1.004 | sí |
| DEL | cambios_tecla_s | 4.661 | 3.364 | 1.386 | 1.273–1.502 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.007 | 0.0 | | | |
