# Puerta de realismo P5: gail_p1

- Sintéticos: 200 juegos de 90 s (C:/hxpod/runs/gail_p1/model.pt); referencia: 256 ventanas humanas de prueba; techo: 256 de entrenamiento contra prueba.
- **Dentro del techo: 0%** (criterio ≥ 80%). Colectivas a más de 2 techos: cambios_posesion_min, pases_min, pase_completo.
- **C2ST (clips de 10 s): AUC 0.987** (criterio ≤ 0,70). Rasgos que más delatan: cambios_tecla (0.340), jug_quietos (0.198), pelota_parada (0.196), juego_abierto (0.109), vel_jug_media (0.077).
- **No pasa.**

| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |
|---|---|---|---|---|---|---|---|---|
| colectiva | goles_10min | 1.233 | 0.911 | 1.119 | 0.305 | 0.263 | 1.2 | **no** |
| colectiva | remates_min | 0.018 | 0.017 | 0.018 | 0.008 | 0.008 | 1.0 | **no** |
| colectiva | cambios_posesion_min | 16.121 | 14.007 | 14.604 | 2.357 | 1.003 | 2.3 | **no** |
| colectiva | pases_min | 7.300 | 11.862 | 11.429 | 4.588 | 0.486 | 9.4 | **no** |
| colectiva | pase_completo | 0.318 | 0.465 | 0.453 | 0.147 | 0.015 | 9.6 | **no** |
| tactica | largo_equipo | 1078.485 | 868.915 | 882.864 | 209.362 | 19.090 | 11.0 | **no** |
| tactica | ancho_equipo | 1005.137 | 966.302 | 1000.351 | 38.805 | 34.521 | 1.1 | **no** |
| tactica | profundidad | 457.797 | 411.925 | 422.673 | 46.373 | 11.068 | 4.2 | **no** |
| tactica | vecino_mas_cercano | 253.602 | 239.295 | 244.670 | 14.294 | 6.027 | 2.4 | **no** |
| tactica | cortes_10min | 5.097 | 2.055 | 3.046 | 3.026 | 0.998 | 3.0 | **no** |
| temporal | duracion_posesion_s | 2.595 | 3.272 | 3.246 | 0.678 | 0.107 | 6.3 | **no** |
| temporal | juego_abierto | 0.645 | 0.730 | 0.712 | 0.088 | 0.022 | 4.0 | **no** |
| temporal | espera_saque | 317.477 | 198.730 | 214.276 | 118.561 | 17.153 | 6.9 | **no** |
| temporal | laterales_10min | 21.929 | 19.498 | 19.654 | 2.559 | 0.903 | 2.8 | **no** |
| temporal | lateral_repetido | 0.183 | 0.015 | 0.017 | 0.168 | 0.005 | 34.9 | **no** |
| mecanica | cargadas_min | 0.223 | 0.228 | 0.238 | 0.104 | 0.048 | 2.2 | **no** |
| mecanica | contacto_s_jug_min | 0.286 | 0.586 | 0.615 | 0.300 | 0.033 | 9.1 | **no** |
| mecanica | racha_p90 | 24.658 | 57.203 | 63.090 | 32.558 | 6.133 | 5.3 | **no** |
| mecanica | vel_jugador | 1.977 | 2.151 | 2.149 | 0.173 | 0.006 | 27.4 | **no** |
| mecanica | vel_pelota | 3.449 | 4.451 | 4.494 | 1.001 | 0.053 | 19.0 | **no** |
| mecanica | cambios_tecla_s | 4.665 | 3.454 | 3.315 | 1.216 | 0.138 | 8.8 | **no** |
| mecanica | mas_cercano_abierto | 67.023 | 55.360 | 55.378 | 11.645 | 1.207 | 9.6 | **no** |
