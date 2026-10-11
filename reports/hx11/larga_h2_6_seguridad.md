# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 299.661 | 314.826 | 0.952 | 0.875–1.036 | sí |
| ARQ | disp_x | 195.094 | 209.0 | 0.933 | 0.821–1.083 | sí |
| ARQ | disp_y | 191.384 | 187.011 | 1.023 | 0.957–1.099 | sí |
| ARQ | lado_arco | 0.981 | 0.985 | 0.995 | 0.984–1.006 | sí |
| ARQ | repliegue | 4.24 | 5.444 | 0.779 | -2.644–5.622 | sí |
| ARQ | va_a_la_pelota | 0.246 | 0.694 | 0.355 | -0.507–1.023 | sí |
| ARQ | cerca_pelota | 0.03 | 0.028 | 1.083 | 0.919–1.279 | sí |
| ARQ | dist_pelota | 1055.152 | 1050.604 | 1.004 | 0.959–1.05 | sí |
| ARQ | vel | 1.666 | 1.749 | 0.952 | 0.868–1.036 | sí |
| ARQ | cambios_tecla_s | 4.432 | 2.678 | 1.655 | 1.46–1.878 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 179.11 | 170.605 | 1.05 | 0.976–1.121 | sí |
| DEF | disp_x | 123.883 | 110.843 | 1.118 | 1.033–1.207 | sí |
| DEF | disp_y | 117.35 | 113.79 | 1.031 | 0.954–1.108 | sí |
| DEF | lado_arco | 0.751 | 0.753 | 0.997 | 0.977–1.019 | sí |
| DEF | repliegue | -19.973 | -13.036 | 1.532 | -0.348–3.418 | sí |
| DEF | va_a_la_pelota | -0.182 | -0.342 | 0.534 | -0.219–1.511 | sí |
| DEF | cerca_pelota | 0.073 | 0.085 | 0.853 | 0.786–0.921 | sí |
| DEF | dist_pelota | 560.311 | 567.569 | 0.987 | 0.957–1.02 | sí |
| DEF | vel | 2.157 | 2.176 | 0.991 | 0.974–1.008 | sí |
| DEF | cambios_tecla_s | 4.594 | 3.661 | 1.255 | 1.189–1.328 | sí |
| DEF | estorbo (s por saque, 143 saques) | 0.009 | 0.0 | | | |
| MCD | zona | 191.759 | 181.588 | 1.056 | 0.941–1.184 | sí |
| MCD | disp_x | 129.192 | 114.948 | 1.124 | 1.004–1.252 | sí |
| MCD | disp_y | 145.068 | 138.224 | 1.05 | 0.888–1.199 | sí |
| MCD | lado_arco | 0.638 | 0.627 | 1.018 | 0.964–1.069 | sí |
| MCD | repliegue | -21.939 | -6.323 | 3.47 | -6.231–7.238 | sí |
| MCD | va_a_la_pelota | 0.251 | -0.086 | -2.902 | -11.448–7.201 | sí |
| MCD | cerca_pelota | 0.063 | 0.087 | 0.728 | 0.532–0.976 | sí |
| MCD | dist_pelota | 419.8 | 392.689 | 1.069 | 1.018–1.128 | sí |
| MCD | vel | 2.361 | 2.336 | 1.011 | 0.97–1.06 | sí |
| MCD | cambios_tecla_s | 4.35 | 4.037 | 1.077 | 0.855–1.449 | sí |
| MCD | estorbo (s por saque, 18 saques) | 0.019 | 0.0 | | | |
| MED | zona | 185.215 | 178.597 | 1.037 | 0.995–1.09 | sí |
| MED | disp_x | 130.796 | 115.967 | 1.128 | 1.05–1.201 | sí |
| MED | disp_y | 127.328 | 121.121 | 1.051 | 0.97–1.15 | sí |
| MED | lado_arco | 0.443 | 0.469 | 0.945 | 0.91–0.985 | sí |
| MED | repliegue | -15.827 | -6.438 | 2.458 | -3.883–8.212 | sí |
| MED | va_a_la_pelota | 0.079 | -0.007 | -11.348 | -6.932–6.105 | sí |
| MED | cerca_pelota | 0.092 | 0.094 | 0.986 | 0.878–1.102 | sí |
| MED | dist_pelota | 512.57 | 511.009 | 1.003 | 0.969–1.037 | sí |
| MED | vel | 2.265 | 2.261 | 1.001 | 0.988–1.015 | sí |
| MED | cambios_tecla_s | 4.743 | 3.393 | 1.398 | 1.318–1.494 | sí |
| MED | estorbo (s por saque, 140 saques) | 0.032 | 0.002 | | | |
| DEL | zona | 202.904 | 212.548 | 0.955 | 0.864–1.056 | sí |
| DEL | disp_x | 126.908 | 117.908 | 1.076 | 0.994–1.176 | sí |
| DEL | disp_y | 146.581 | 138.5 | 1.058 | 0.941–1.2 | sí |
| DEL | lado_arco | 0.241 | 0.256 | 0.943 | 0.857–1.037 | sí |
| DEL | repliegue | -15.951 | 8.996 | -1.773 | -14.489–10.329 | sí |
| DEL | va_a_la_pelota | 0.112 | 0.394 | 0.284 | -0.449–0.983 | sí |
| DEL | cerca_pelota | 0.075 | 0.076 | 0.987 | 0.864–1.113 | sí |
| DEL | dist_pelota | 566.185 | 579.374 | 0.977 | 0.929–1.038 | sí |
| DEL | vel | 2.204 | 2.193 | 1.005 | 0.976–1.033 | sí |
| DEL | cambios_tecla_s | 4.589 | 3.407 | 1.347 | 1.234–1.465 | sí |
| DEL | estorbo (s por saque, 93 saques) | 0.0 | 0.0 | | | |
