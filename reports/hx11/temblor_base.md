# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 87 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 177.595 | 163.065 | 1.089 | 0.968–1.233 | sí |
| DEF | disp_x | 96.368 | 96.794 | 0.996 | 0.891–1.111 | sí |
| DEF | disp_y | 107.266 | 132.111 | 0.812 | 0.729–0.897 | sí |
| DEF | lado_arco | 0.863 | 0.827 | 1.044 | 0.991–1.114 | sí |
| DEF | repliegue | 12.547 | 13.831 | 0.907 | -1.364–3.701 | sí |
| DEF | va_a_la_pelota | -0.261 | -0.397 | 0.658 | -1.362–2.166 | sí |
| DEF | cerca_pelota | 0.073 | 0.085 | 0.866 | 0.657–1.133 | sí |
| DEF | dist_pelota | 490.091 | 502.136 | 0.976 | 0.92–1.034 | sí |
| DEF | vel | 2.182 | 2.176 | 1.003 | 0.975–1.04 | sí |
| DEF | cambios_tecla_s | 5.023 | 3.968 | 1.266 | 1.168–1.372 | sí |
| DEF | estorbo (s por saque, 51 saques) | 0.01 | 0.005 | | | |
| MED | zona | 209.28 | 177.886 | 1.176 | 1.088–1.29 | sí |
| MED | disp_x | 123.478 | 108.725 | 1.136 | 1.019–1.263 | sí |
| MED | disp_y | 119.515 | 131.479 | 0.909 | 0.825–1.005 | sí |
| MED | lado_arco | 0.334 | 0.347 | 0.963 | 0.873–1.084 | sí |
| MED | repliegue | 27.357 | 35.442 | 0.772 | 0.181–1.235 | sí |
| MED | va_a_la_pelota | -0.157 | 0.386 | -0.407 | -2.005–0.466 | **no** |
| MED | cerca_pelota | 0.107 | 0.097 | 1.1 | 0.916–1.331 | sí |
| MED | dist_pelota | 438.794 | 448.242 | 0.979 | 0.937–1.022 | sí |
| MED | vel | 2.275 | 2.265 | 1.004 | 0.982–1.028 | sí |
| MED | cambios_tecla_s | 5.047 | 3.763 | 1.341 | 1.22–1.499 | sí |
| MED | estorbo (s por saque, 52 saques) | 0.087 | 0.0 | | | |
| DEL | zona | 212.928 | 179.818 | 1.184 | 0.979–1.456 | sí |
| DEL | disp_x | 116.003 | 105.761 | 1.097 | 0.906–1.384 | sí |
| DEL | disp_y | 129.389 | 131.302 | 0.985 | 0.801–1.233 | sí |
| DEL | lado_arco | 0.11 | 0.141 | 0.779 | 0.554–0.97 | sí |
| DEL | repliegue | 0.295 | 22.338 | 0.013 | -4.019–1.531 | sí |
| DEL | va_a_la_pelota | -0.008 | 0.331 | -0.026 | -3.957–3.746 | sí |
| DEL | cerca_pelota | 0.051 | 0.087 | 0.588 | 0.476–0.715 | **no** |
| DEL | dist_pelota | 504.999 | 523.726 | 0.964 | 0.877–1.062 | sí |
| DEL | vel | 2.215 | 2.241 | 0.988 | 0.952–1.024 | sí |
| DEL | cambios_tecla_s | 4.738 | 3.505 | 1.352 | 1.204–1.511 | sí |
| DEL | estorbo (s por saque, 26 saques) | 0.0 | 0.0 | | | |
