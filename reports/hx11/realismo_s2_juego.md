# Puerta de realismo P5: s2_juego

- Sintéticos: 200 juegos de 90 s (C:/hxpod/runs/l0_memoria_s2/model.pt); referencia: 256 ventanas humanas de prueba; techo: 256 de entrenamiento contra prueba.
- **Dentro del techo: 9%** (criterio ≥ 80%). Colectivas a más de 2 techos: remates_min, pases_min, pase_completo.
- **C2ST (clips de 10 s): AUC 0.989** (criterio ≤ 0,70). Rasgos que más delatan: cambios_tecla (0.334), jug_quietos (0.213), pelota_parada (0.198), juego_abierto (0.113), vel_jug_p90 (0.051).
- **No pasa.**

| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |
|---|---|---|---|---|---|---|---|---|
| colectiva | goles_10min | 1.100 | 0.911 | 1.119 | 0.304 | 0.263 | 1.2 | **no** |
| colectiva | remates_min | 0.000 | 0.017 | 0.018 | 0.018 | 0.008 | 2.3 | **no** |
| colectiva | cambios_posesion_min | 15.705 | 14.007 | 14.604 | 1.826 | 1.003 | 1.8 | **no** |
| colectiva | pases_min | 7.537 | 11.862 | 11.429 | 4.465 | 0.486 | 9.2 | **no** |
| colectiva | pase_completo | 0.329 | 0.465 | 0.453 | 0.137 | 0.015 | 8.9 | **no** |
| tactica | largo_equipo | 1059.315 | 868.915 | 882.864 | 190.192 | 19.090 | 10.0 | **no** |
| tactica | ancho_equipo | 979.424 | 966.302 | 1000.351 | 18.699 | 34.521 | 0.5 | sí |
| tactica | profundidad | 453.364 | 411.925 | 422.673 | 41.446 | 11.068 | 3.7 | **no** |
| tactica | vecino_mas_cercano | 248.912 | 239.295 | 244.670 | 9.605 | 6.027 | 1.6 | **no** |
| tactica | cortes_10min | 2.644 | 2.055 | 3.046 | 0.684 | 0.998 | 0.7 | sí |
| temporal | duracion_posesion_s | 2.788 | 3.272 | 3.246 | 0.485 | 0.107 | 4.5 | **no** |
| temporal | juego_abierto | 0.642 | 0.730 | 0.712 | 0.090 | 0.022 | 4.1 | **no** |
| temporal | espera_saque | 334.028 | 198.730 | 214.276 | 135.100 | 17.153 | 7.9 | **no** |
| temporal | laterales_10min | 20.796 | 19.498 | 19.654 | 1.269 | 0.903 | 1.4 | **no** |
| temporal | lateral_repetido | 0.196 | 0.015 | 0.017 | 0.181 | 0.005 | 37.5 | **no** |
| mecanica | cargadas_min | 0.290 | 0.228 | 0.238 | 0.121 | 0.048 | 2.5 | **no** |
| mecanica | contacto_s_jug_min | 0.302 | 0.586 | 0.615 | 0.285 | 0.033 | 8.6 | **no** |
| mecanica | racha_p90 | 25.532 | 57.203 | 63.090 | 31.734 | 6.133 | 5.2 | **no** |
| mecanica | vel_jugador | 1.850 | 2.151 | 2.149 | 0.301 | 0.006 | 47.6 | **no** |
| mecanica | vel_pelota | 3.281 | 4.451 | 4.494 | 1.169 | 0.053 | 22.2 | **no** |
| mecanica | cambios_tecla_s | 4.855 | 3.454 | 3.315 | 1.404 | 0.138 | 10.2 | **no** |
| mecanica | mas_cercano_abierto | 60.940 | 55.360 | 55.378 | 5.578 | 1.207 | 4.6 | **no** |
