# Puerta de realismo P5: e1

- Sintéticos: 200 juegos de 90 s (C:/hxpod/runs/l0_quienva_e1/model.pt); referencia: 256 ventanas humanas de prueba; techo: 256 de entrenamiento contra prueba.
- **Dentro del techo: 18%** (criterio ≥ 80%). Colectivas a más de 2 techos: cambios_posesion_min, pases_min, pase_completo.
- **C2ST (clips de 10 s): AUC 0.988** (criterio ≤ 0,70). Rasgos que más delatan: cambios_tecla (0.364), jug_quietos (0.245), pelota_parada (0.173), juego_abierto (0.087), vel_jug_p90 (0.057).
- **No pasa.**

| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |
|---|---|---|---|---|---|---|---|---|
| colectiva | goles_10min | 0.933 | 0.911 | 1.119 | 0.137 | 0.263 | 0.5 | sí |
| colectiva | remates_min | 0.013 | 0.017 | 0.018 | 0.006 | 0.008 | 0.7 | sí |
| colectiva | cambios_posesion_min | 16.036 | 14.007 | 14.604 | 2.565 | 1.003 | 2.6 | **no** |
| colectiva | pases_min | 7.622 | 11.862 | 11.429 | 5.204 | 0.486 | 10.7 | **no** |
| colectiva | pase_completo | 0.312 | 0.465 | 0.453 | 0.156 | 0.015 | 10.1 | **no** |
| tactica | largo_equipo | 1060.121 | 868.915 | 882.864 | 190.999 | 19.090 | 10.0 | **no** |
| tactica | ancho_equipo | 992.615 | 966.302 | 1000.351 | 27.932 | 34.521 | 0.8 | sí |
| tactica | profundidad | 452.727 | 411.925 | 422.673 | 40.809 | 11.068 | 3.7 | **no** |
| tactica | vecino_mas_cercano | 251.138 | 239.295 | 244.670 | 11.830 | 6.027 | 2.0 | **no** |
| tactica | cortes_10min | 2.762 | 2.055 | 3.046 | 0.719 | 0.998 | 0.7 | sí |
| temporal | duracion_posesion_s | 2.610 | 3.272 | 3.246 | 0.664 | 0.107 | 6.2 | **no** |
| temporal | juego_abierto | 0.622 | 0.730 | 0.712 | 0.110 | 0.022 | 5.0 | **no** |
| temporal | espera_saque | 310.829 | 198.730 | 214.276 | 111.912 | 17.153 | 6.5 | **no** |
| temporal | laterales_10min | 26.328 | 19.498 | 19.654 | 6.868 | 0.903 | 7.6 | **no** |
| temporal | lateral_repetido | 0.233 | 0.015 | 0.017 | 0.218 | 0.005 | 45.4 | **no** |
| mecanica | cargadas_min | 0.367 | 0.228 | 0.238 | 0.139 | 0.048 | 2.9 | **no** |
| mecanica | contacto_s_jug_min | 0.302 | 0.586 | 0.615 | 0.284 | 0.033 | 8.6 | **no** |
| mecanica | racha_p90 | 25.666 | 57.203 | 63.090 | 31.550 | 6.133 | 5.1 | **no** |
| mecanica | vel_jugador | 1.872 | 2.151 | 2.149 | 0.279 | 0.006 | 44.2 | **no** |
| mecanica | vel_pelota | 3.410 | 4.451 | 4.494 | 1.040 | 0.053 | 19.7 | **no** |
| mecanica | cambios_tecla_s | 4.891 | 3.454 | 3.315 | 1.441 | 0.138 | 10.5 | **no** |
| mecanica | mas_cercano_abierto | 67.217 | 55.360 | 55.378 | 11.839 | 1.207 | 9.8 | **no** |
