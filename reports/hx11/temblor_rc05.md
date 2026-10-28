# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 88 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 210.731 | 177.096 | 1.19 | 1.065–1.338 | sí |
| DEF | disp_x | 115.382 | 99.18 | 1.163 | 1.035–1.294 | sí |
| DEF | disp_y | 113.511 | 128.293 | 0.885 | 0.734–1.067 | sí |
| DEF | lado_arco | 0.801 | 0.769 | 1.041 | 0.992–1.109 | sí |
| DEF | repliegue | -57.032 | -33.904 | 1.682 | 1.067–4.264 | sí |
| DEF | va_a_la_pelota | -0.204 | -0.317 | 0.643 | -2.283–2.888 | sí |
| DEF | cerca_pelota | 0.09 | 0.09 | 0.996 | 0.827–1.21 | sí |
| DEF | dist_pelota | 538.332 | 551.211 | 0.977 | 0.904–1.055 | sí |
| DEF | vel | 2.168 | 2.17 | 0.999 | 0.972–1.029 | sí |
| DEF | cambios_tecla_s | 4.619 | 3.821 | 1.209 | 1.112–1.324 | sí |
| DEF | estorbo (s por saque, 61 saques) | 0.014 | 0.0 | | | |
| MED | zona | 214.182 | 173.874 | 1.232 | 1.1–1.402 | sí |
| MED | disp_x | 118.146 | 112.413 | 1.051 | 0.938–1.188 | sí |
| MED | disp_y | 118.824 | 115.969 | 1.025 | 0.894–1.187 | sí |
| MED | lado_arco | 0.428 | 0.417 | 1.027 | 0.903–1.187 | sí |
| MED | repliegue | -36.655 | -29.577 | 1.239 | 0.097–3.011 | sí |
| MED | va_a_la_pelota | 0.267 | 0.241 | 1.106 | -2.126–5.488 | sí |
| MED | cerca_pelota | 0.119 | 0.119 | 1.006 | 0.814–1.301 | sí |
| MED | dist_pelota | 450.03 | 491.099 | 0.916 | 0.827–1.013 | sí |
| MED | vel | 2.289 | 2.274 | 1.007 | 0.978–1.038 | sí |
| MED | cambios_tecla_s | 4.59 | 3.395 | 1.352 | 1.2–1.535 | sí |
| MED | estorbo (s por saque, 59 saques) | 0.022 | 0.016 | | | |
| DEL | zona | 190.776 | 181.988 | 1.048 | 0.877–1.203 | sí |
| DEL | disp_x | 105.812 | 114.264 | 0.926 | 0.803–1.068 | sí |
| DEL | disp_y | 118.689 | 123.834 | 0.958 | 0.792–1.152 | sí |
| DEL | lado_arco | 0.139 | 0.147 | 0.945 | 0.821–1.116 | sí |
| DEL | repliegue | -19.561 | 4.312 | -4.536 | -6.928–9.391 | sí |
| DEL | va_a_la_pelota | 0.235 | 0.284 | 0.827 | -4.014–6.851 | sí |
| DEL | cerca_pelota | 0.05 | 0.078 | 0.645 | 0.478–0.837 | sí |
| DEL | dist_pelota | 544.516 | 541.166 | 1.006 | 0.897–1.162 | sí |
| DEL | vel | 2.193 | 2.218 | 0.989 | 0.916–1.051 | sí |
| DEL | cambios_tecla_s | 4.365 | 3.315 | 1.317 | 1.192–1.464 | sí |
| DEL | estorbo (s por saque, 34 saques) | 0.0 | 0.0 | | | |
