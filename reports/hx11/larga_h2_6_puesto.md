# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 314.654 | 314.826 | 0.999 | 0.911–1.097 | sí |
| ARQ | disp_x | 179.4 | 209.0 | 0.858 | 0.748–1.01 | sí |
| ARQ | disp_y | 204.44 | 187.011 | 1.093 | 1.007–1.173 | sí |
| ARQ | lado_arco | 0.983 | 0.985 | 0.998 | 0.987–1.009 | sí |
| ARQ | repliegue | 6.039 | 5.444 | 1.109 | -4.229–6.111 | sí |
| ARQ | va_a_la_pelota | 0.38 | 0.694 | 0.547 | -1.039–1.665 | sí |
| ARQ | cerca_pelota | 0.027 | 0.028 | 0.991 | 0.728–1.284 | sí |
| ARQ | dist_pelota | 1081.323 | 1050.604 | 1.029 | 0.979–1.081 | sí |
| ARQ | vel | 1.679 | 1.749 | 0.96 | 0.894–1.028 | sí |
| ARQ | cambios_tecla_s | 4.465 | 2.678 | 1.667 | 1.472–1.884 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 302.927 | 170.605 | 1.776 | 1.621–1.942 | **no** |
| DEF | disp_x | 141.731 | 110.843 | 1.279 | 1.187–1.371 | sí |
| DEF | disp_y | 135.913 | 113.79 | 1.194 | 1.054–1.335 | sí |
| DEF | lado_arco | 0.66 | 0.753 | 0.877 | 0.839–0.915 | sí |
| DEF | repliegue | -22.405 | -13.036 | 1.719 | -1.204–4.819 | sí |
| DEF | va_a_la_pelota | -0.157 | -0.343 | 0.458 | -0.696–1.995 | sí |
| DEF | cerca_pelota | 0.076 | 0.085 | 0.894 | 0.781–1.027 | sí |
| DEF | dist_pelota | 616.97 | 567.569 | 1.087 | 1.028–1.149 | sí |
| DEF | vel | 2.213 | 2.176 | 1.017 | 0.995–1.036 | sí |
| DEF | cambios_tecla_s | 4.58 | 3.661 | 1.251 | 1.175–1.331 | sí |
| DEF | estorbo (s por saque, 143 saques) | 0.075 | 0.0 | | | |
| MCD | zona | 188.2 | 181.588 | 1.036 | 0.927–1.17 | sí |
| MCD | disp_x | 113.654 | 114.948 | 0.989 | 0.919–1.058 | sí |
| MCD | disp_y | 147.499 | 138.224 | 1.067 | 0.915–1.212 | sí |
| MCD | lado_arco | 0.642 | 0.627 | 1.025 | 0.985–1.065 | sí |
| MCD | repliegue | -3.681 | -6.323 | 0.582 | -0.983–4.692 | sí |
| MCD | va_a_la_pelota | 0.166 | -0.086 | -1.919 | -6.746–6.243 | sí |
| MCD | cerca_pelota | 0.075 | 0.087 | 0.858 | 0.646–1.169 | sí |
| MCD | dist_pelota | 435.739 | 392.689 | 1.11 | 1.035–1.173 | sí |
| MCD | vel | 2.36 | 2.336 | 1.01 | 0.968–1.058 | sí |
| MCD | cambios_tecla_s | 4.507 | 4.037 | 1.116 | 0.899–1.464 | sí |
| MCD | estorbo (s por saque, 18 saques) | 0.032 | 0.0 | | | |
| MED | zona | 221.469 | 178.597 | 1.24 | 1.151–1.337 | sí |
| MED | disp_x | 129.337 | 115.967 | 1.115 | 1.03–1.206 | sí |
| MED | disp_y | 135.347 | 121.121 | 1.117 | 1.044–1.202 | sí |
| MED | lado_arco | 0.404 | 0.469 | 0.86 | 0.8–0.915 | sí |
| MED | repliegue | -15.058 | -6.438 | 2.339 | -4.627–8.094 | sí |
| MED | va_a_la_pelota | 0.063 | -0.021 | -2.982 | -5.33–6.247 | sí |
| MED | cerca_pelota | 0.087 | 0.094 | 0.931 | 0.805–1.052 | sí |
| MED | dist_pelota | 502.603 | 511.009 | 0.984 | 0.944–1.033 | sí |
| MED | vel | 2.273 | 2.261 | 1.005 | 0.986–1.022 | sí |
| MED | cambios_tecla_s | 4.805 | 3.393 | 1.416 | 1.333–1.508 | **no** |
| MED | estorbo (s por saque, 140 saques) | 0.043 | 0.002 | | | |
| DEL | zona | 240.91 | 212.548 | 1.133 | 1.038–1.25 | sí |
| DEL | disp_x | 130.575 | 117.908 | 1.107 | 1.012–1.218 | sí |
| DEL | disp_y | 151.006 | 138.5 | 1.09 | 0.955–1.241 | sí |
| DEL | lado_arco | 0.209 | 0.256 | 0.818 | 0.725–0.907 | sí |
| DEL | repliegue | 4.329 | 8.996 | 0.481 | -3.463–4.061 | sí |
| DEL | va_a_la_pelota | 0.023 | 0.384 | 0.06 | -0.412–0.63 | **no** |
| DEL | cerca_pelota | 0.079 | 0.076 | 1.037 | 0.89–1.215 | sí |
| DEL | dist_pelota | 580.871 | 579.374 | 1.003 | 0.948–1.072 | sí |
| DEL | vel | 2.197 | 2.193 | 1.002 | 0.971–1.031 | sí |
| DEL | cambios_tecla_s | 4.72 | 3.407 | 1.386 | 1.26–1.507 | sí |
| DEL | estorbo (s por saque, 93 saques) | 0.029 | 0.0 | | | |
