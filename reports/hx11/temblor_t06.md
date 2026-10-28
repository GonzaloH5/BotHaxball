# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 87 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 184.638 | 163.065 | 1.132 | 1.012–1.268 | sí |
| DEF | disp_x | 110.612 | 96.794 | 1.143 | 1.014–1.29 | sí |
| DEF | disp_y | 114.856 | 132.111 | 0.869 | 0.779–0.969 | sí |
| DEF | lado_arco | 0.833 | 0.827 | 1.007 | 0.949–1.085 | sí |
| DEF | repliegue | -0.646 | 13.831 | -0.047 | -6.41–4.33 | sí |
| DEF | va_a_la_pelota | 0.125 | -0.232 | -0.54 | -7.689–5.314 | sí |
| DEF | cerca_pelota | 0.066 | 0.085 | 0.784 | 0.643–0.943 | sí |
| DEF | dist_pelota | 493.032 | 502.136 | 0.982 | 0.927–1.035 | sí |
| DEF | vel | 2.216 | 2.176 | 1.018 | 0.986–1.059 | sí |
| DEF | cambios_tecla_s | 4.73 | 3.968 | 1.192 | 1.096–1.294 | sí |
| DEF | estorbo (s por saque, 51 saques) | 0.04 | 0.005 | | | |
| MED | zona | 223.998 | 177.886 | 1.259 | 1.151–1.402 | sí |
| MED | disp_x | 134.121 | 108.725 | 1.234 | 1.094–1.369 | sí |
| MED | disp_y | 127.988 | 131.479 | 0.973 | 0.873–1.096 | sí |
| MED | lado_arco | 0.342 | 0.347 | 0.986 | 0.873–1.145 | sí |
| MED | repliegue | 19.857 | 35.442 | 0.56 | -0.383–0.909 | sí |
| MED | va_a_la_pelota | 0.403 | 0.288 | 1.398 | -0.552–5.195 | sí |
| MED | cerca_pelota | 0.114 | 0.097 | 1.166 | 0.973–1.416 | sí |
| MED | dist_pelota | 427.353 | 448.242 | 0.953 | 0.898–1.005 | sí |
| MED | vel | 2.323 | 2.265 | 1.026 | 1.003–1.05 | sí |
| MED | cambios_tecla_s | 4.698 | 3.763 | 1.248 | 1.121–1.402 | sí |
| MED | estorbo (s por saque, 52 saques) | 0.062 | 0.0 | | | |
| DEL | zona | 214.594 | 179.818 | 1.193 | 0.962–1.516 | sí |
| DEL | disp_x | 135.768 | 105.761 | 1.284 | 1.051–1.618 | sí |
| DEL | disp_y | 140.601 | 131.302 | 1.071 | 0.86–1.399 | sí |
| DEL | lado_arco | 0.121 | 0.141 | 0.863 | 0.616–1.121 | sí |
| DEL | repliegue | -0.275 | 22.338 | -0.012 | -3.703–2.156 | sí |
| DEL | va_a_la_pelota | -0.369 | 0.488 | -0.755 | -3.121–0.755 | sí |
| DEL | cerca_pelota | 0.049 | 0.087 | 0.562 | 0.423–0.739 | **no** |
| DEL | dist_pelota | 516.081 | 523.726 | 0.985 | 0.905–1.081 | sí |
| DEL | vel | 2.253 | 2.241 | 1.005 | 0.962–1.043 | sí |
| DEL | cambios_tecla_s | 4.424 | 3.505 | 1.262 | 1.119–1.421 | sí |
| DEL | estorbo (s por saque, 26 saques) | 0.0 | 0.0 | | | |
