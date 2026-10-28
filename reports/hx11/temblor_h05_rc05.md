# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 88 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 696.289 | 177.096 | 3.932 | 2.985–5.282 | **no** |
| DEF | disp_x | 331.325 | 99.18 | 3.341 | 2.642–4.11 | **no** |
| DEF | disp_y | 263.392 | 128.293 | 2.053 | 1.583–2.796 | **no** |
| DEF | lado_arco | 0.657 | 0.769 | 0.854 | 0.742–0.975 | sí |
| DEF | repliegue | -43.91 | -33.904 | 1.295 | -0.155–4.128 | sí |
| DEF | va_a_la_pelota | -0.029 | -0.423 | 0.068 | -1.738–1.065 | sí |
| DEF | cerca_pelota | 0.071 | 0.09 | 0.786 | 0.59–1.084 | sí |
| DEF | dist_pelota | 926.316 | 551.211 | 1.681 | 1.351–2.083 | **no** |
| DEF | vel | 1.93 | 2.17 | 0.89 | 0.772–0.997 | sí |
| DEF | cambios_tecla_s | 0.864 | 3.821 | 0.226 | 0.188–0.264 | **no** |
| DEF | estorbo (s por saque, 61 saques) | 0.024 | 0.0 | | | |
| MED | zona | 685.105 | 173.874 | 3.94 | 3.026–5.15 | **no** |
| MED | disp_x | 390.119 | 112.413 | 3.47 | 2.628–4.588 | **no** |
| MED | disp_y | 297.439 | 115.969 | 2.565 | 2.078–3.208 | **no** |
| MED | lado_arco | 0.317 | 0.417 | 0.76 | 0.575–0.979 | sí |
| MED | repliegue | -59.732 | -29.577 | 2.02 | -2.584–11.807 | sí |
| MED | va_a_la_pelota | 0.368 | 0.093 | 3.958 | -7.002–14.134 | sí |
| MED | cerca_pelota | 0.094 | 0.119 | 0.791 | 0.567–1.045 | sí |
| MED | dist_pelota | 872.141 | 491.099 | 1.776 | 1.334–2.368 | **no** |
| MED | vel | 2.019 | 2.274 | 0.888 | 0.777–0.987 | sí |
| MED | cambios_tecla_s | 0.752 | 3.395 | 0.222 | 0.169–0.272 | **no** |
| MED | estorbo (s por saque, 59 saques) | 0.006 | 0.016 | | | |
| DEL | zona | 787.353 | 181.988 | 4.326 | 2.872–5.593 | **no** |
| DEL | disp_x | 458.616 | 114.264 | 4.014 | 2.967–4.864 | **no** |
| DEL | disp_y | 318.221 | 123.834 | 2.57 | 1.704–3.416 | **no** |
| DEL | lado_arco | 0.328 | 0.147 | 2.232 | 1.224–3.674 | sí |
| DEL | repliegue | -28.357 | 4.312 | -6.576 | -11.8–11.818 | sí |
| DEL | cerca_pelota | 0.03 | 0.078 | 0.384 | 0.197–0.596 | **no** |
| DEL | dist_pelota | 1120.209 | 541.166 | 2.07 | 1.532–2.588 | **no** |
| DEL | vel | 1.791 | 2.218 | 0.807 | 0.67–0.962 | sí |
| DEL | cambios_tecla_s | 0.745 | 3.315 | 0.225 | 0.169–0.292 | **no** |
| DEL | estorbo (s por saque, 34 saques) | 0.0 | 0.0 | | | |
