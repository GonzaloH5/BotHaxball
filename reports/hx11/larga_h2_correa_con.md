# Prueba larga (fantasma 60 s entre humanos): l0_ajuste_corner_h2

Modelo `runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`, 6 partidos de test, 261 ventanas por lugar. Razón fantasma/humano con IC 90%; pasa si se superpone con [0.75; 1.33]. Estorbo: segundos por saque a < 35 px del que saca o < 45 px de la pelota, en pelotas paradas y saques iniciales de su equipo que saca otro.

| Línea | Métrica | Bot | Humano | Razón | IC 90% | ¿Pasa? |
|---|---|---|---|---|---|---|
| ARQ | zona | 305.769 | 316.202 | 0.967 | 0.877–1.07 | sí |
| ARQ | disp_x | 196.958 | 211.666 | 0.931 | 0.798–1.094 | sí |
| ARQ | disp_y | 186.643 | 182.725 | 1.021 | 0.931–1.107 | sí |
| ARQ | lado_arco | 0.988 | 0.985 | 1.003 | 0.997–1.011 | sí |
| ARQ | repliegue | 10.518 | 7.999 | 1.315 | -4.683–7.402 | sí |
| ARQ | va_a_la_pelota | 0.242 | 0.588 | 0.412 | -1.019–1.378 | sí |
| ARQ | cerca_pelota | 0.026 | 0.028 | 0.929 | 0.706–1.176 | sí |
| ARQ | dist_pelota | 1044.289 | 1049.582 | 0.995 | 0.956–1.035 | sí |
| ARQ | vel | 1.65 | 1.735 | 0.951 | 0.878–1.036 | sí |
| ARQ | cambios_tecla_s | 4.274 | 2.704 | 1.581 | 1.41–1.784 | **no** |
| ARQ | congelado (s por min de juego abierto) | 1.755 | 1.282 | | | |
| ARQ | estorbo (s por saque, 43 saques) | 0.0 | 0.0 | | | |
| DEF | zona | 207.228 | 171.612 | 1.208 | 1.125–1.299 | sí |
| DEF | disp_x | 113.993 | 111.85 | 1.019 | 0.953–1.092 | sí |
| DEF | disp_y | 110.306 | 114.577 | 0.963 | 0.902–1.025 | sí |
| DEF | lado_arco | 0.794 | 0.761 | 1.044 | 1.02–1.072 | sí |
| DEF | repliegue | -11.813 | -9.576 | 1.234 | -0.588–3.6 | sí |
| DEF | va_a_la_pelota | 0.235 | -0.287 | -0.819 | -3.904–0.263 | **no** |
| DEF | cerca_pelota | 0.067 | 0.085 | 0.785 | 0.69–0.889 | sí |
| DEF | dist_pelota | 573.113 | 561.926 | 1.02 | 0.986–1.055 | sí |
| DEF | vel | 2.155 | 2.175 | 0.991 | 0.977–1.006 | sí |
| DEF | cambios_tecla_s | 4.641 | 3.668 | 1.265 | 1.198–1.336 | sí |
| DEF | congelado (s por min de juego abierto) | 0.047 | 0.059 | | | |
| DEF | estorbo (s por saque, 144 saques) | 0.016 | 0.0 | | | |
| MCD | zona | 199.151 | 181.463 | 1.097 | 0.976–1.253 | sí |
| MCD | disp_x | 132.467 | 114.43 | 1.158 | 1.009–1.33 | sí |
| MCD | disp_y | 144.511 | 129.862 | 1.113 | 0.983–1.245 | sí |
| MCD | lado_arco | 0.627 | 0.62 | 1.012 | 0.969–1.065 | sí |
| MCD | repliegue | -11.431 | 6.419 | -1.781 | -4.868–5.914 | sí |
| MCD | va_a_la_pelota | 0.15 | -0.088 | -1.703 | -6.496–7.673 | sí |
| MCD | cerca_pelota | 0.075 | 0.094 | 0.8 | 0.547–1.2 | sí |
| MCD | dist_pelota | 423.386 | 385.82 | 1.097 | 1.033–1.168 | sí |
| MCD | vel | 2.348 | 2.322 | 1.011 | 0.968–1.066 | sí |
| MCD | cambios_tecla_s | 4.454 | 4.11 | 1.084 | 0.866–1.444 | sí |
| MCD | congelado (s por min de juego abierto) | 0.0 | 0.0 | | | |
| MCD | estorbo (s por saque, 19 saques) | 0.0 | 0.0 | | | |
| MED | zona | 221.75 | 179.592 | 1.235 | 1.12–1.351 | sí |
| MED | disp_x | 119.063 | 116.174 | 1.025 | 0.947–1.101 | sí |
| MED | disp_y | 127.689 | 122.27 | 1.044 | 0.978–1.122 | sí |
| MED | lado_arco | 0.424 | 0.465 | 0.911 | 0.863–0.959 | sí |
| MED | repliegue | -7.047 | 1.306 | -5.397 | -5.728–7.104 | sí |
| MED | va_a_la_pelota | -0.062 | 0.174 | -0.353 | -3.866–1.586 | sí |
| MED | cerca_pelota | 0.103 | 0.095 | 1.079 | 0.966–1.209 | sí |
| MED | dist_pelota | 461.897 | 506.871 | 0.911 | 0.874–0.954 | sí |
| MED | vel | 2.284 | 2.252 | 1.014 | 1.001–1.028 | sí |
| MED | cambios_tecla_s | 4.801 | 3.435 | 1.398 | 1.318–1.485 | sí |
| MED | congelado (s por min de juego abierto) | 0.0 | 0.057 | | | |
| MED | estorbo (s por saque, 143 saques) | 0.01 | 0.002 | | | |
| DEL | zona | 222.725 | 213.827 | 1.042 | 0.933–1.163 | sí |
| DEL | disp_x | 119.239 | 118.137 | 1.009 | 0.935–1.099 | sí |
| DEL | disp_y | 129.692 | 138.519 | 0.936 | 0.846–1.035 | sí |
| DEL | lado_arco | 0.219 | 0.26 | 0.842 | 0.766–0.925 | sí |
| DEL | repliegue | -10.738 | 7.76 | -1.384 | -7.702–8.801 | sí |
| DEL | va_a_la_pelota | 0.014 | 0.228 | 0.063 | -2.354–1.999 | sí |
| DEL | cerca_pelota | 0.074 | 0.075 | 0.987 | 0.816–1.193 | sí |
| DEL | dist_pelota | 557.62 | 572.986 | 0.973 | 0.923–1.035 | sí |
| DEL | vel | 2.174 | 2.197 | 0.99 | 0.966–1.012 | sí |
| DEL | cambios_tecla_s | 4.771 | 3.364 | 1.418 | 1.326–1.531 | sí |
| DEL | congelado (s por min de juego abierto) | 0.08 | 0.187 | | | |
| DEL | estorbo (s por saque, 89 saques) | 0.0 | 0.0 | | | |
