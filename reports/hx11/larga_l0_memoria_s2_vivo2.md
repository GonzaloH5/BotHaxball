# Prueba larga (fantasma 60 s entre humanos): l0_memoria_s2

Modelo `runs/hx11_l0/l0_memoria_s2/model.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 293.243 | 316.202 | 0.927 | 0.843–1.035 | sí |
| ARQ | disp_x | 190.137 | 211.666 | 0.898 | 0.789–1.033 | sí |
| ARQ | disp_y | 200.802 | 182.725 | 1.099 | 1.004–1.193 | sí |
| ARQ | lado_arco | 0.977 | 0.985 | 0.992 | 0.985–0.998 | sí |
| ARQ | repliegue | 17.126 | 7.999 | 2.141 | -5.472–10.289 | sí |
| ARQ | va_a_la_pelota | 0.241 | 0.432 | 0.557 | -1.347–2.136 | sí |
| ARQ | cerca_pelota | 0.034 | 0.028 | 1.241 | 0.952–1.603 | sí |
| ARQ | dist_pelota | 1034.361 | 1049.582 | 0.985 | 0.94–1.028 | sí |
| ARQ | vel | 1.706 | 1.735 | 0.983 | 0.9–1.084 | sí |
| ARQ | cambios_tecla_s | 4.472 | 2.704 | 1.654 | 1.467–1.852 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 211.767 | 171.612 | 1.234 | 1.133–1.343 | sí |
| DEF | disp_x | 117.567 | 111.85 | 1.051 | 0.971–1.134 | sí |
| DEF | disp_y | 121.322 | 114.577 | 1.059 | 0.955–1.17 | sí |
| DEF | lado_arco | 0.789 | 0.761 | 1.038 | 1.011–1.065 | sí |
| DEF | repliegue | -17.807 | -9.576 | 1.859 | -3.05–8.215 | sí |
| DEF | va_a_la_pelota | -0.037 | -0.338 | 0.108 | -0.727–0.819 | sí |
| DEF | cerca_pelota | 0.07 | 0.085 | 0.817 | 0.726–0.925 | sí |
| DEF | dist_pelota | 581.485 | 561.926 | 1.035 | 0.997–1.076 | sí |
| DEF | vel | 2.126 | 2.175 | 0.978 | 0.962–0.994 | sí |
| DEF | cambios_tecla_s | 4.593 | 3.668 | 1.252 | 1.182–1.33 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.005 | 0.0 | | | |
| MCD | zona | 207.094 | 181.463 | 1.141 | 1.012–1.294 | sí |
| MCD | disp_x | 124.615 | 114.43 | 1.089 | 0.942–1.265 | sí |
| MCD | disp_y | 144.689 | 129.862 | 1.114 | 1.0–1.222 | sí |
| MCD | lado_arco | 0.616 | 0.62 | 0.994 | 0.952–1.037 | sí |
| MCD | repliegue | -3.884 | 6.419 | -0.605 | -4.202–3.886 | sí |
| MCD | va_a_la_pelota | 0.217 | -0.088 | -2.468 | -12.682–12.154 | sí |
| MCD | cerca_pelota | 0.081 | 0.094 | 0.865 | 0.563–1.256 | sí |
| MCD | dist_pelota | 391.708 | 385.82 | 1.015 | 0.923–1.116 | sí |
| MCD | vel | 2.335 | 2.322 | 1.006 | 0.962–1.055 | sí |
| MCD | cambios_tecla_s | 4.762 | 4.11 | 1.159 | 0.921–1.534 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.015 | 0.0 | | | |
| MED | zona | 217.482 | 179.592 | 1.211 | 1.113–1.31 | sí |
| MED | disp_x | 123.666 | 116.174 | 1.064 | 0.978–1.147 | sí |
| MED | disp_y | 130.648 | 122.27 | 1.069 | 0.994–1.155 | sí |
| MED | lado_arco | 0.437 | 0.465 | 0.939 | 0.895–0.983 | sí |
| MED | repliegue | -11.792 | 1.306 | -9.032 | -9.154–11.245 | sí |
| MED | va_a_la_pelota | 0.247 | 0.151 | 1.639 | -4.669–8.179 | sí |
| MED | cerca_pelota | 0.104 | 0.095 | 1.09 | 0.964–1.217 | sí |
| MED | dist_pelota | 462.555 | 506.871 | 0.913 | 0.871–0.957 | sí |
| MED | vel | 2.281 | 2.252 | 1.013 | 0.999–1.028 | sí |
| MED | cambios_tecla_s | 4.895 | 3.435 | 1.425 | 1.34–1.518 | **no** |
| MED | estorbo (s por saque, 143 saques) | 0.006 | 0.002 | | | |
| DEL | zona | 231.417 | 213.827 | 1.082 | 0.96–1.208 | sí |
| DEL | disp_x | 123.69 | 118.137 | 1.047 | 0.969–1.13 | sí |
| DEL | disp_y | 139.384 | 138.519 | 1.006 | 0.902–1.13 | sí |
| DEL | lado_arco | 0.215 | 0.26 | 0.826 | 0.748–0.91 | sí |
| DEL | repliegue | 0.442 | 7.76 | 0.057 | -2.931–4.958 | sí |
| DEL | va_a_la_pelota | 0.004 | 0.301 | 0.012 | -0.957–0.824 | sí |
| DEL | cerca_pelota | 0.067 | 0.075 | 0.896 | 0.736–1.088 | sí |
| DEL | dist_pelota | 561.03 | 572.986 | 0.979 | 0.916–1.058 | sí |
| DEL | vel | 2.144 | 2.197 | 0.976 | 0.951–1.002 | sí |
| DEL | cambios_tecla_s | 4.527 | 3.364 | 1.346 | 1.243–1.463 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.019 | 0.0 | | | |
