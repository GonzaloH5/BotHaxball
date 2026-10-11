# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 313.571 | 316.202 | 0.992 | 0.91–1.077 | sí |
| ARQ | disp_x | 206.854 | 211.666 | 0.977 | 0.85–1.128 | sí |
| ARQ | disp_y | 192.194 | 182.725 | 1.052 | 0.967–1.133 | sí |
| ARQ | lado_arco | 0.987 | 0.985 | 1.002 | 0.998–1.007 | sí |
| ARQ | repliegue | 6.929 | 7.999 | 0.866 | -2.929–4.401 | sí |
| ARQ | va_a_la_pelota | 0.443 | 0.588 | 0.753 | 0.141–1.595 | sí |
| ARQ | cerca_pelota | 0.028 | 0.028 | 1.004 | 0.796–1.22 | sí |
| ARQ | dist_pelota | 1044.57 | 1049.582 | 0.995 | 0.957–1.033 | sí |
| ARQ | vel | 1.653 | 1.735 | 0.953 | 0.889–1.023 | sí |
| ARQ | cambios_tecla_s | 4.5 | 2.704 | 1.664 | 1.478–1.869 | **no** |
| ARQ | congelado (s por min de juego abierto) | 1.497 | 1.282 | | | |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 210.17 | 171.612 | 1.225 | 1.127–1.33 | sí |
| DEF | disp_x | 119.945 | 111.85 | 1.072 | 0.989–1.182 | sí |
| DEF | disp_y | 117.122 | 114.577 | 1.022 | 0.942–1.103 | sí |
| DEF | lado_arco | 0.789 | 0.761 | 1.037 | 1.014–1.062 | sí |
| DEF | repliegue | -12.424 | -9.576 | 1.297 | -1.016–4.143 | sí |
| DEF | va_a_la_pelota | 0.112 | -0.344 | -0.324 | -2.585–0.513 | **no** |
| DEF | cerca_pelota | 0.065 | 0.085 | 0.762 | 0.668–0.867 | sí |
| DEF | dist_pelota | 571.805 | 561.926 | 1.018 | 0.983–1.055 | sí |
| DEF | vel | 2.162 | 2.175 | 0.994 | 0.98–1.01 | sí |
| DEF | cambios_tecla_s | 4.61 | 3.668 | 1.257 | 1.186–1.326 | sí |
| DEF | congelado (s por min de juego abierto) | 0.1 | 0.059 | | | |
| DEF | estorbo (s por saque, 144 saques) | 0.003 | 0.0 | | | |
| MCD | zona | 206.348 | 181.463 | 1.137 | 0.986–1.299 | sí |
| MCD | disp_x | 123.988 | 114.43 | 1.084 | 0.975–1.234 | sí |
| MCD | disp_y | 155.965 | 129.862 | 1.201 | 1.077–1.324 | sí |
| MCD | lado_arco | 0.63 | 0.62 | 1.017 | 0.969–1.068 | sí |
| MCD | repliegue | -26.256 | 6.419 | -4.09 | -10.797–9.679 | sí |
| MCD | va_a_la_pelota | 0.06 | -0.11 | -0.54 | -7.239–6.745 | sí |
| MCD | cerca_pelota | 0.083 | 0.094 | 0.878 | 0.631–1.267 | sí |
| MCD | dist_pelota | 409.875 | 385.82 | 1.062 | 0.988–1.14 | sí |
| MCD | vel | 2.364 | 2.322 | 1.018 | 0.968–1.077 | sí |
| MCD | cambios_tecla_s | 4.686 | 4.11 | 1.14 | 0.911–1.51 | sí |
| MCD | congelado (s por min de juego abierto) | 0.0 | 0.0 | | | |
| MCD | estorbo (s por saque, 19 saques) | 0.01 | 0.0 | | | |
| MED | zona | 229.12 | 179.592 | 1.276 | 1.176–1.377 | sí |
| MED | disp_x | 131.566 | 116.174 | 1.132 | 1.045–1.228 | sí |
| MED | disp_y | 130.501 | 122.27 | 1.067 | 0.993–1.153 | sí |
| MED | lado_arco | 0.427 | 0.465 | 0.919 | 0.868–0.971 | sí |
| MED | repliegue | -11.149 | 1.306 | -8.539 | -8.589–10.571 | sí |
| MED | va_a_la_pelota | 0.237 | 0.145 | 1.64 | -3.674–7.544 | sí |
| MED | cerca_pelota | 0.101 | 0.095 | 1.059 | 0.942–1.187 | sí |
| MED | dist_pelota | 467.018 | 506.871 | 0.921 | 0.88–0.964 | sí |
| MED | vel | 2.29 | 2.252 | 1.017 | 1.002–1.033 | sí |
| MED | cambios_tecla_s | 4.832 | 3.435 | 1.407 | 1.325–1.49 | sí |
| MED | congelado (s por min de juego abierto) | 0.0 | 0.057 | | | |
| MED | estorbo (s por saque, 143 saques) | 0.003 | 0.002 | | | |
| DEL | zona | 230.3 | 213.827 | 1.077 | 0.978–1.182 | sí |
| DEL | disp_x | 122.26 | 118.137 | 1.035 | 0.949–1.132 | sí |
| DEL | disp_y | 136.806 | 138.519 | 0.988 | 0.885–1.08 | sí |
| DEL | lado_arco | 0.216 | 0.26 | 0.832 | 0.783–0.894 | sí |
| DEL | repliegue | -0.067 | 7.76 | -0.009 | -3.088–5.394 | sí |
| DEL | va_a_la_pelota | 0.078 | 0.234 | 0.335 | -1.042–1.617 | sí |
| DEL | cerca_pelota | 0.063 | 0.075 | 0.85 | 0.673–1.074 | sí |
| DEL | dist_pelota | 567.956 | 572.986 | 0.991 | 0.936–1.062 | sí |
| DEL | vel | 2.182 | 2.197 | 0.993 | 0.969–1.018 | sí |
| DEL | cambios_tecla_s | 4.631 | 3.364 | 1.377 | 1.273–1.489 | sí |
| DEL | congelado (s por min de juego abierto) | 0.0 | 0.187 | | | |
| DEL | estorbo (s por saque, 89 saques) | 0.0 | 0.0 | | | |
