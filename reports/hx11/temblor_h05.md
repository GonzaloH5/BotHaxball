# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 87 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 750.214 | 163.065 | 4.601 | 3.553–5.774 | **no** |
| DEF | disp_x | 409.308 | 96.794 | 4.229 | 3.284–5.194 | **no** |
| DEF | disp_y | 283.006 | 132.111 | 2.142 | 1.746–2.548 | **no** |
| DEF | lado_arco | 0.758 | 0.827 | 0.917 | 0.826–1.02 | sí |
| DEF | repliegue | 22.781 | 13.831 | 1.647 | -1.979–6.086 | sí |
| DEF | va_a_la_pelota | -0.064 | -0.577 | 0.11 | -1.918–0.856 | sí |
| DEF | cerca_pelota | 0.041 | 0.085 | 0.479 | 0.338–0.674 | **no** |
| DEF | dist_pelota | 1104.537 | 502.136 | 2.2 | 1.723–2.709 | **no** |
| DEF | vel | 1.86 | 2.176 | 0.855 | 0.739–0.97 | sí |
| DEF | cambios_tecla_s | 0.911 | 3.968 | 0.23 | 0.192–0.27 | **no** |
| DEF | estorbo (s por saque, 51 saques) | 0.011 | 0.005 | | | |
| MED | zona | 879.104 | 177.886 | 4.942 | 3.717–6.382 | **no** |
| MED | disp_x | 503.977 | 108.725 | 4.635 | 3.601–5.888 | **no** |
| MED | disp_y | 325.18 | 131.479 | 2.473 | 1.993–3.054 | **no** |
| MED | lado_arco | 0.423 | 0.347 | 1.22 | 0.935–1.684 | sí |
| MED | repliegue | 4.442 | 35.442 | 0.125 | -1.497–0.769 | sí |
| MED | va_a_la_pelota | -0.17 | 0.234 | -0.725 | -7.969–4.57 | sí |
| MED | cerca_pelota | 0.074 | 0.097 | 0.755 | 0.564–0.942 | sí |
| MED | dist_pelota | 1117.246 | 448.242 | 2.493 | 1.955–3.104 | **no** |
| MED | vel | 1.847 | 2.265 | 0.816 | 0.689–0.925 | sí |
| MED | cambios_tecla_s | 0.806 | 3.763 | 0.214 | 0.168–0.266 | **no** |
| MED | estorbo (s por saque, 52 saques) | 0.137 | 0.0 | | | |
| DEL | zona | 705.447 | 179.818 | 3.923 | 2.776–5.183 | **no** |
| DEL | disp_x | 411.426 | 105.761 | 3.89 | 2.889–4.847 | **no** |
| DEL | disp_y | 325.916 | 131.302 | 2.482 | 1.781–3.375 | **no** |
| DEL | lado_arco | 0.18 | 0.141 | 1.279 | 0.625–2.403 | sí |
| DEL | repliegue | -19.695 | 22.338 | -0.882 | -7.279–3.295 | sí |
| DEL | va_a_la_pelota | -0.846 | 0.623 | -1.358 | -3.99–0.367 | **no** |
| DEL | cerca_pelota | 0.041 | 0.087 | 0.475 | 0.3–0.64 | **no** |
| DEL | dist_pelota | 893.505 | 523.726 | 1.706 | 1.399–2.044 | **no** |
| DEL | vel | 1.954 | 2.241 | 0.872 | 0.741–0.995 | sí |
| DEL | cambios_tecla_s | 0.843 | 3.505 | 0.24 | 0.194–0.293 | **no** |
| DEL | estorbo (s por saque, 26 saques) | 0.113 | 0.0 | | | |
