# Prueba larga (fantasma 60 s entre humanos): l0_estado_s1

Modelo `runs/hx11_l0/l0_estado_s1/model.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 326.025 | 316.202 | 1.031 | 0.93–1.149 | sí |
| ARQ | disp_x | 198.369 | 211.666 | 0.937 | 0.782–1.147 | sí |
| ARQ | disp_y | 199.267 | 182.725 | 1.091 | 0.99–1.185 | sí |
| ARQ | lado_arco | 0.988 | 0.985 | 1.003 | 0.996–1.011 | sí |
| ARQ | repliegue | 6.648 | 7.999 | 0.831 | -2.283–4.221 | sí |
| ARQ | va_a_la_pelota | 0.068 | 0.432 | 0.157 | -1.913–2.402 | sí |
| ARQ | cerca_pelota | 0.029 | 0.028 | 1.03 | 0.775–1.322 | sí |
| ARQ | dist_pelota | 1098.904 | 1049.582 | 1.047 | 0.995–1.097 | sí |
| ARQ | vel | 1.64 | 1.735 | 0.945 | 0.875–1.028 | sí |
| ARQ | cambios_tecla_s | 4.403 | 2.704 | 1.629 | 1.419–1.858 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 292.471 | 171.612 | 1.704 | 1.557–1.856 | **no** |
| DEF | disp_x | 141.361 | 111.85 | 1.264 | 1.177–1.357 | sí |
| DEF | disp_y | 135.055 | 114.577 | 1.179 | 1.066–1.295 | sí |
| DEF | lado_arco | 0.687 | 0.761 | 0.902 | 0.868–0.937 | sí |
| DEF | repliegue | -23.673 | -9.576 | 2.472 | -4.067–10.796 | sí |
| DEF | va_a_la_pelota | 0.014 | -0.254 | -0.057 | -2.341–1.058 | sí |
| DEF | cerca_pelota | 0.074 | 0.085 | 0.87 | 0.754–1.009 | sí |
| DEF | dist_pelota | 606.113 | 561.926 | 1.079 | 1.029–1.138 | sí |
| DEF | vel | 2.199 | 2.175 | 1.011 | 0.986–1.033 | sí |
| DEF | cambios_tecla_s | 4.632 | 3.668 | 1.263 | 1.181–1.347 | sí |
| DEF | estorbo (s por saque, 144 saques) | 0.016 | 0.0 | | | |
| MCD | zona | 204.068 | 181.463 | 1.125 | 0.949–1.334 | sí |
| MCD | disp_x | 123.471 | 114.43 | 1.079 | 0.924–1.27 | sí |
| MCD | disp_y | 156.397 | 129.862 | 1.204 | 1.091–1.329 | sí |
| MCD | lado_arco | 0.644 | 0.62 | 1.04 | 0.984–1.105 | sí |
| MCD | repliegue | -1.793 | 6.419 | -0.279 | -3.664–3.724 | sí |
| MCD | va_a_la_pelota | 0.15 | -0.088 | -1.704 | -6.86–11.754 | sí |
| MCD | cerca_pelota | 0.079 | 0.094 | 0.841 | 0.659–1.095 | sí |
| MCD | dist_pelota | 420.098 | 385.82 | 1.089 | 1.017–1.158 | sí |
| MCD | vel | 2.32 | 2.322 | 0.999 | 0.951–1.05 | sí |
| MCD | cambios_tecla_s | 4.768 | 4.11 | 1.16 | 0.933–1.518 | sí |
| MCD | estorbo (s por saque, 19 saques) | 0.0 | 0.0 | | | |
| MED | zona | 219.19 | 179.592 | 1.22 | 1.142–1.307 | sí |
| MED | disp_x | 129.587 | 116.174 | 1.115 | 1.013–1.228 | sí |
| MED | disp_y | 127.955 | 122.27 | 1.046 | 0.979–1.123 | sí |
| MED | lado_arco | 0.408 | 0.465 | 0.878 | 0.821–0.932 | sí |
| MED | repliegue | -11.321 | 1.306 | -8.671 | -8.582–11.908 | sí |
| MED | va_a_la_pelota | 0.218 | 0.163 | 1.338 | -1.941–5.665 | sí |
| MED | cerca_pelota | 0.084 | 0.095 | 0.889 | 0.786–1.0 | sí |
| MED | dist_pelota | 504.745 | 506.871 | 0.996 | 0.958–1.04 | sí |
| MED | vel | 2.24 | 2.252 | 0.994 | 0.976–1.013 | sí |
| MED | cambios_tecla_s | 4.738 | 3.435 | 1.379 | 1.298–1.475 | sí |
| MED | estorbo (s por saque, 143 saques) | 0.014 | 0.002 | | | |
| DEL | zona | 226.458 | 213.827 | 1.059 | 0.973–1.154 | sí |
| DEL | disp_x | 127.663 | 118.137 | 1.081 | 0.992–1.183 | sí |
| DEL | disp_y | 135.033 | 138.519 | 0.975 | 0.882–1.061 | sí |
| DEL | lado_arco | 0.227 | 0.26 | 0.873 | 0.813–0.949 | sí |
| DEL | repliegue | -0.825 | 7.76 | -0.106 | -4.445–5.133 | sí |
| DEL | va_a_la_pelota | 0.023 | 0.187 | 0.124 | -1.427–1.713 | sí |
| DEL | cerca_pelota | 0.079 | 0.075 | 1.063 | 0.905–1.242 | sí |
| DEL | dist_pelota | 555.505 | 572.986 | 0.969 | 0.908–1.046 | sí |
| DEL | vel | 2.15 | 2.197 | 0.979 | 0.95–1.01 | sí |
| DEL | cambios_tecla_s | 4.709 | 3.364 | 1.4 | 1.289–1.521 | sí |
| DEL | estorbo (s por saque, 89 saques) | 0.023 | 0.0 | | | |
