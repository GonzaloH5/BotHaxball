# Puerta de realismo P5: e1_juego

- Sintéticos: 200 juegos de 90 s (C:/hxpod/runs/l0_quienva_e1/model.pt); referencia: 256 ventanas humanas de prueba; techo: 256 de entrenamiento contra prueba.
- **Dentro del techo: 18%** (criterio ≥ 80%). Colectivas a más de 2 techos: cambios_posesion_min, pases_min, pase_completo.
- **C2ST (clips de 10 s): AUC 0.987** (criterio ≤ 0,70). Rasgos que más delatan: cambios_tecla (0.391), jug_quietos (0.228), pelota_parada (0.160), juego_abierto (0.114), vel_jug_media (0.055).
- **No pasa.**

| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |
|---|---|---|---|---|---|---|---|---|
| colectiva | goles_10min | 1.000 | 0.911 | 1.119 | 0.138 | 0.263 | 0.5 | sí |
| colectiva | remates_min | 0.019 | 0.017 | 0.018 | 0.007 | 0.008 | 0.9 | sí |
| colectiva | cambios_posesion_min | 18.378 | 14.007 | 14.604 | 4.360 | 1.003 | 4.3 | **no** |
| colectiva | pases_min | 8.317 | 11.862 | 11.429 | 3.764 | 0.486 | 7.8 | **no** |
| colectiva | pase_completo | 0.312 | 0.465 | 0.453 | 0.153 | 0.015 | 9.9 | **no** |
| tactica | largo_equipo | 1071.216 | 868.915 | 882.864 | 202.094 | 19.090 | 10.6 | **no** |
| tactica | ancho_equipo | 970.854 | 966.302 | 1000.351 | 30.928 | 34.521 | 0.9 | sí |
| tactica | profundidad | 454.101 | 411.925 | 422.673 | 42.183 | 11.068 | 3.8 | **no** |
| tactica | vecino_mas_cercano | 250.449 | 239.295 | 244.670 | 11.142 | 6.027 | 1.8 | **no** |
| tactica | cortes_10min | 2.522 | 2.055 | 3.046 | 0.488 | 0.998 | 0.5 | sí |
| temporal | duracion_posesion_s | 2.470 | 3.272 | 3.246 | 0.803 | 0.107 | 7.5 | **no** |
| temporal | juego_abierto | 0.681 | 0.730 | 0.712 | 0.056 | 0.022 | 2.5 | **no** |
| temporal | espera_saque | 303.270 | 198.730 | 214.276 | 104.330 | 17.153 | 6.1 | **no** |
| temporal | laterales_10min | 21.829 | 19.498 | 19.654 | 2.726 | 0.903 | 3.0 | **no** |
| temporal | lateral_repetido | 0.170 | 0.015 | 0.017 | 0.155 | 0.005 | 32.3 | **no** |
| mecanica | cargadas_min | 0.263 | 0.228 | 0.238 | 0.080 | 0.048 | 1.7 | **no** |
| mecanica | contacto_s_jug_min | 0.341 | 0.586 | 0.615 | 0.245 | 0.033 | 7.4 | **no** |
| mecanica | racha_p90 | 26.576 | 57.203 | 63.090 | 30.640 | 6.133 | 5.0 | **no** |
| mecanica | vel_jugador | 1.872 | 2.151 | 2.149 | 0.278 | 0.006 | 44.1 | **no** |
| mecanica | vel_pelota | 3.294 | 4.451 | 4.494 | 1.156 | 0.053 | 21.9 | **no** |
| mecanica | cambios_tecla_s | 4.901 | 3.454 | 3.315 | 1.452 | 0.138 | 10.5 | **no** |
| mecanica | mas_cercano_abierto | 67.900 | 55.360 | 55.378 | 12.944 | 1.207 | 10.7 | **no** |
