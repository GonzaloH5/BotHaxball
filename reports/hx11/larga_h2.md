# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 12 partidos de test, 527 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 289.617 | 287.031 | 1.009 | 0.947–1.077 | sí |
| ARQ | disp_x | 195.144 | 190.178 | 1.026 | 0.925–1.149 | sí |
| ARQ | disp_y | 174.522 | 165.956 | 1.052 | 0.975–1.131 | sí |
| ARQ | lado_arco | 0.973 | 0.964 | 1.009 | 1.0–1.018 | sí |
| ARQ | repliegue | 17.495 | 27.34 | 0.64 | -0.491–1.445 | sí |
| ARQ | va_a_la_pelota | -0.751 | -0.886 | 0.847 | 0.053–2.337 | sí |
| ARQ | cerca_pelota | 0.037 | 0.039 | 0.948 | 0.778–1.212 | sí |
| ARQ | dist_pelota | 1021.298 | 1017.945 | 1.003 | 0.975–1.03 | sí |
| ARQ | vel | 1.632 | 1.757 | 0.929 | 0.881–0.978 | sí |
| ARQ | cambios_tecla_s | 4.23 | 3.189 | 1.326 | 1.169–1.507 | sí |
| ARQ | estorbo (s por saque, 73 saques) | 0.052 | 0.0 | | | |
| DEF | zona | 178.798 | 173.819 | 1.029 | 0.988–1.078 | sí |
| DEF | disp_x | 122.966 | 107.003 | 1.149 | 1.091–1.222 | sí |
| DEF | disp_y | 119.206 | 112.91 | 1.056 | 1.001–1.11 | sí |
| DEF | lado_arco | 0.755 | 0.767 | 0.984 | 0.97–1.0 | sí |
| DEF | repliegue | 12.736 | 20.617 | 0.618 | 0.024–0.928 | sí |
| DEF | va_a_la_pelota | 0.077 | -0.093 | -0.836 | -6.895–5.867 | sí |
| DEF | cerca_pelota | 0.078 | 0.08 | 0.976 | 0.91–1.052 | sí |
| DEF | dist_pelota | 554.633 | 560.61 | 0.989 | 0.973–1.008 | sí |
| DEF | vel | 2.117 | 2.109 | 1.004 | 0.993–1.014 | sí |
| DEF | cambios_tecla_s | 4.837 | 3.942 | 1.227 | 1.174–1.277 | sí |
| DEF | estorbo (s por saque, 292 saques) | 0.086 | 0.021 | | | |
| MCD | zona | 215.523 | 197.568 | 1.091 | 0.976–1.234 | sí |
| MCD | disp_x | 132.615 | 120.933 | 1.097 | 0.961–1.237 | sí |
| MCD | disp_y | 145.575 | 131.179 | 1.11 | 0.966–1.288 | sí |
| MCD | lado_arco | 0.519 | 0.539 | 0.962 | 0.912–1.02 | sí |
| MCD | repliegue | 41.025 | 32.858 | 1.249 | 0.692–2.466 | sí |
| MCD | va_a_la_pelota | -0.316 | -0.042 | 7.604 | -9.571–15.559 | sí |
| MCD | cerca_pelota | 0.107 | 0.141 | 0.763 | 0.647–0.913 | sí |
| MCD | dist_pelota | 399.963 | 398.545 | 1.004 | 0.931–1.068 | sí |
| MCD | vel | 2.235 | 2.284 | 0.978 | 0.964–0.995 | sí |
| MCD | cambios_tecla_s | 5.108 | 4.096 | 1.247 | 1.058–1.44 | sí |
| MCD | estorbo (s por saque, 51 saques) | 0.064 | 0.0 | | | |
| MED | zona | 198.422 | 190.557 | 1.041 | 0.973–1.117 | sí |
| MED | disp_x | 133.432 | 116.324 | 1.147 | 1.09–1.221 | sí |
| MED | disp_y | 132.435 | 130.704 | 1.013 | 0.928–1.112 | sí |
| MED | lado_arco | 0.393 | 0.423 | 0.928 | 0.894–0.962 | sí |
| MED | repliegue | 14.036 | 12.885 | 1.089 | 0.142–2.304 | sí |
| MED | va_a_la_pelota | 0.003 | -0.068 | -0.043 | -4.147–5.993 | sí |
| MED | cerca_pelota | 0.084 | 0.092 | 0.906 | 0.842–0.975 | sí |
| MED | dist_pelota | 523.389 | 519.124 | 1.008 | 0.969–1.048 | sí |
| MED | vel | 2.169 | 2.202 | 0.985 | 0.971–0.998 | sí |
| MED | cambios_tecla_s | 4.827 | 3.609 | 1.337 | 1.292–1.383 | sí |
| MED | estorbo (s por saque, 280 saques) | 0.066 | 0.002 | | | |
| DEL | zona | 232.594 | 191.255 | 1.216 | 1.074–1.388 | sí |
| DEL | disp_x | 142.513 | 117.777 | 1.21 | 1.087–1.336 | sí |
| DEL | disp_y | 148.311 | 135.802 | 1.092 | 0.998–1.21 | sí |
| DEL | lado_arco | 0.253 | 0.256 | 0.987 | 0.902–1.095 | sí |
| DEL | repliegue | 2.627 | 0.243 | 10.829 | -2.369–5.516 | sí |
| DEL | va_a_la_pelota | -0.275 | 0.328 | -0.838 | -4.178–0.621 | **no** |
| DEL | cerca_pelota | 0.075 | 0.092 | 0.812 | 0.714–0.92 | sí |
| DEL | dist_pelota | 587.94 | 539.008 | 1.091 | 1.025–1.165 | sí |
| DEL | vel | 2.106 | 2.157 | 0.976 | 0.954–0.999 | sí |
| DEL | cambios_tecla_s | 4.784 | 3.494 | 1.369 | 1.284–1.453 | sí |
| DEL | estorbo (s por saque, 153 saques) | 0.058 | 0.0 | | | |
