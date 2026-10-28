# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 4 partidos de test, 87 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| DEF | zona | 581.614 | 163.065 | 3.567 | 2.542–4.568 | **no** |
| DEF | disp_x | 279.645 | 96.794 | 2.889 | 2.196–3.616 | **no** |
| DEF | disp_y | 259.131 | 132.111 | 1.961 | 1.454–2.497 | **no** |
| DEF | lado_arco | 0.766 | 0.827 | 0.927 | 0.823–1.033 | sí |
| DEF | repliegue | 2.642 | 13.831 | 0.191 | -4.94–4.149 | sí |
| DEF | va_a_la_pelota | -0.803 | -0.526 | 1.525 | 0.803–4.044 | sí |
| DEF | cerca_pelota | 0.052 | 0.085 | 0.617 | 0.469–0.803 | sí |
| DEF | dist_pelota | 911.543 | 502.136 | 1.815 | 1.379–2.28 | **no** |
| DEF | vel | 2.03 | 2.176 | 0.933 | 0.836–1.034 | sí |
| DEF | cambios_tecla_s | 1.23 | 3.968 | 0.31 | 0.269–0.359 | **no** |
| DEF | estorbo (s por saque, 51 saques) | 0.009 | 0.005 | | | |
| MED | zona | 663.339 | 177.886 | 3.729 | 2.546–5.125 | **no** |
| MED | disp_x | 347.098 | 108.725 | 3.192 | 2.292–4.497 | **no** |
| MED | disp_y | 258.847 | 131.479 | 1.969 | 1.513–2.516 | **no** |
| MED | lado_arco | 0.49 | 0.347 | 1.412 | 1.1–1.851 | sí |
| MED | repliegue | 27.83 | 35.442 | 0.785 | -0.156–1.858 | sí |
| MED | va_a_la_pelota | -0.532 | 0.136 | -3.9 | -22.922–22.242 | sí |
| MED | cerca_pelota | 0.075 | 0.097 | 0.765 | 0.602–0.949 | sí |
| MED | dist_pelota | 899.647 | 448.242 | 2.007 | 1.489–2.625 | **no** |
| MED | vel | 2.065 | 2.265 | 0.912 | 0.792–1.015 | sí |
| MED | cambios_tecla_s | 1.2 | 3.763 | 0.319 | 0.259–0.381 | **no** |
| MED | estorbo (s por saque, 52 saques) | 0.065 | 0.0 | | | |
| DEL | zona | 657.836 | 179.818 | 3.658 | 2.506–4.9 | **no** |
| DEL | disp_x | 354.07 | 105.761 | 3.348 | 2.268–4.456 | **no** |
| DEL | disp_y | 303.947 | 131.302 | 2.315 | 1.633–3.228 | **no** |
| DEL | lado_arco | 0.182 | 0.141 | 1.292 | 0.595–2.436 | sí |
| DEL | repliegue | -7.473 | 22.338 | -0.335 | -4.323–1.994 | sí |
| DEL | va_a_la_pelota | -0.6 | 0.623 | -0.963 | -3.229–-0.062 | **no** |
| DEL | cerca_pelota | 0.051 | 0.087 | 0.587 | 0.392–0.808 | sí |
| DEL | dist_pelota | 839.183 | 523.726 | 1.602 | 1.312–1.91 | sí |
| DEL | vel | 1.872 | 2.241 | 0.836 | 0.701–0.959 | sí |
| DEL | cambios_tecla_s | 1.076 | 3.505 | 0.307 | 0.241–0.377 | **no** |
| DEL | estorbo (s por saque, 26 saques) | 0.0 | 0.0 | | | |
