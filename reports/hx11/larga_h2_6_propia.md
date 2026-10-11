# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 296.676 | 314.826 | 0.942 | 0.872–1.032 | sí |
| ARQ | disp_x | 189.775 | 209.0 | 0.908 | 0.809–1.047 | sí |
| ARQ | disp_y | 197.562 | 187.011 | 1.056 | 0.974–1.145 | sí |
| ARQ | lado_arco | 0.992 | 0.985 | 1.007 | 0.999–1.016 | sí |
| ARQ | repliegue | -0.758 | 5.444 | -0.139 | -3.485–3.383 | sí |
| ARQ | va_a_la_pelota | 0.396 | 0.694 | 0.57 | -0.204–1.216 | sí |
| ARQ | cerca_pelota | 0.034 | 0.028 | 1.209 | 0.906–1.573 | sí |
| ARQ | dist_pelota | 1039.721 | 1050.604 | 0.99 | 0.949–1.035 | sí |
| ARQ | vel | 1.681 | 1.749 | 0.961 | 0.894–1.029 | sí |
| ARQ | cambios_tecla_s | 4.549 | 2.678 | 1.698 | 1.514–1.894 | **no** |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 183.403 | 170.605 | 1.075 | 1.001–1.14 | sí |
| DEF | disp_x | 128.11 | 110.843 | 1.156 | 1.061–1.263 | sí |
| DEF | disp_y | 123.177 | 113.79 | 1.082 | 0.993–1.175 | sí |
| DEF | lado_arco | 0.748 | 0.753 | 0.993 | 0.972–1.018 | sí |
| DEF | repliegue | -16.327 | -13.036 | 1.253 | -0.133–2.736 | sí |
| DEF | va_a_la_pelota | -0.151 | -0.303 | 0.497 | -0.517–1.432 | sí |
| DEF | cerca_pelota | 0.073 | 0.085 | 0.853 | 0.781–0.922 | sí |
| DEF | dist_pelota | 559.331 | 567.569 | 0.985 | 0.957–1.018 | sí |
| DEF | vel | 2.169 | 2.176 | 0.997 | 0.979–1.012 | sí |
| DEF | cambios_tecla_s | 4.554 | 3.661 | 1.244 | 1.175–1.316 | sí |
| DEF | estorbo (s por saque, 143 saques) | 0.049 | 0.0 | | | |
| MCD | zona | 194.028 | 181.588 | 1.069 | 0.92–1.234 | sí |
| MCD | disp_x | 130.293 | 114.948 | 1.133 | 1.02–1.271 | sí |
| MCD | disp_y | 152.623 | 138.224 | 1.104 | 0.918–1.291 | sí |
| MCD | lado_arco | 0.635 | 0.627 | 1.012 | 0.977–1.052 | sí |
| MCD | repliegue | -16.059 | -6.323 | 2.54 | -3.447–6.015 | sí |
| MCD | va_a_la_pelota | 0.247 | -0.086 | -2.863 | -12.045–6.764 | sí |
| MCD | cerca_pelota | 0.068 | 0.087 | 0.782 | 0.552–1.051 | sí |
| MCD | dist_pelota | 446.418 | 392.689 | 1.137 | 1.08–1.212 | sí |
| MCD | vel | 2.327 | 2.336 | 0.996 | 0.95–1.045 | sí |
| MCD | cambios_tecla_s | 4.615 | 4.037 | 1.143 | 0.919–1.504 | sí |
| MCD | estorbo (s por saque, 18 saques) | 0.023 | 0.0 | | | |
| MED | zona | 184.946 | 178.597 | 1.036 | 0.983–1.094 | sí |
| MED | disp_x | 131.682 | 115.967 | 1.136 | 1.044–1.219 | sí |
| MED | disp_y | 125.424 | 121.121 | 1.036 | 0.971–1.11 | sí |
| MED | lado_arco | 0.439 | 0.469 | 0.937 | 0.899–0.982 | sí |
| MED | repliegue | -5.967 | -6.438 | 0.927 | -2.417–3.087 | sí |
| MED | va_a_la_pelota | -0.18 | -0.003 | 70.866 | -9.178–11.571 | sí |
| MED | cerca_pelota | 0.093 | 0.094 | 0.997 | 0.874–1.137 | sí |
| MED | dist_pelota | 503.581 | 511.009 | 0.985 | 0.951–1.022 | sí |
| MED | vel | 2.263 | 2.261 | 1.001 | 0.989–1.013 | sí |
| MED | cambios_tecla_s | 4.78 | 3.393 | 1.409 | 1.327–1.502 | sí |
| MED | estorbo (s por saque, 140 saques) | 0.03 | 0.002 | | | |
| DEL | zona | 204.846 | 212.548 | 0.964 | 0.877–1.066 | sí |
| DEL | disp_x | 128.696 | 117.908 | 1.091 | 1.012–1.184 | sí |
| DEL | disp_y | 145.213 | 138.5 | 1.048 | 0.917–1.201 | sí |
| DEL | lado_arco | 0.241 | 0.256 | 0.941 | 0.862–1.033 | sí |
| DEL | repliegue | -18.909 | 8.996 | -2.102 | -15.662–11.268 | sí |
| DEL | va_a_la_pelota | 0.346 | 0.397 | 0.871 | -0.033–2.493 | sí |
| DEL | cerca_pelota | 0.075 | 0.076 | 0.992 | 0.866–1.132 | sí |
| DEL | dist_pelota | 573.17 | 579.374 | 0.989 | 0.943–1.047 | sí |
| DEL | vel | 2.192 | 2.193 | 1.0 | 0.971–1.03 | sí |
| DEL | cambios_tecla_s | 4.659 | 3.407 | 1.368 | 1.242–1.497 | sí |
| DEL | estorbo (s por saque, 93 saques) | 0.042 | 0.0 | | | |
