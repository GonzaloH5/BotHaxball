# Puerta de realismo P5: e1_juego_r02

- Sintéticos: 200 juegos de 90 s (C:/hxpod/runs/l0_quienva_e1/model.pt); referencia: 256 ventanas humanas de prueba; techo: 256 de entrenamiento contra prueba.
- **Dentro del techo: 9%** (criterio ≥ 80%). Colectivas a más de 2 techos: cambios_posesion_min, pases_min, pase_completo.
- **C2ST (clips de 10 s): AUC 0.972** (criterio ≤ 0,70). Rasgos que más delatan: jug_quietos (0.286), pelota_parada (0.257), juego_abierto (0.179), cambios_tecla (0.108), vel_jug_p90 (0.060).
- **No pasa.**

| familia | métrica | sintético | prueba | entrenamiento | d | techo | veces | dentro |
|---|---|---|---|---|---|---|---|---|
| colectiva | goles_10min | 0.900 | 0.911 | 1.119 | 0.038 | 0.263 | 0.1 | sí |
| colectiva | remates_min | 0.013 | 0.017 | 0.018 | 0.006 | 0.008 | 0.7 | sí |
| colectiva | cambios_posesion_min | 16.736 | 14.007 | 14.604 | 2.964 | 1.003 | 3.0 | **no** |
| colectiva | pases_min | 7.338 | 11.862 | 11.429 | 4.579 | 0.486 | 9.4 | **no** |
| colectiva | pase_completo | 0.309 | 0.465 | 0.453 | 0.156 | 0.015 | 10.1 | **no** |
| tactica | largo_equipo | 1311.147 | 868.915 | 882.864 | 442.025 | 19.090 | 23.2 | **no** |
| tactica | ancho_equipo | 1082.549 | 966.302 | 1000.351 | 116.217 | 34.521 | 3.4 | **no** |
| tactica | profundidad | 507.649 | 411.925 | 422.673 | 96.424 | 11.068 | 8.7 | **no** |
| tactica | vecino_mas_cercano | 275.033 | 239.295 | 244.670 | 35.725 | 6.027 | 5.9 | **no** |
| tactica | cortes_10min | 3.353 | 2.055 | 3.046 | 1.303 | 0.998 | 1.3 | **no** |
| temporal | duracion_posesion_s | 2.720 | 3.272 | 3.246 | 0.597 | 0.107 | 5.6 | **no** |
| temporal | juego_abierto | 0.653 | 0.730 | 0.712 | 0.079 | 0.022 | 3.6 | **no** |
| temporal | espera_saque | 335.957 | 198.730 | 214.276 | 137.039 | 17.153 | 8.0 | **no** |
| temporal | laterales_10min | 21.829 | 19.498 | 19.654 | 2.369 | 0.903 | 2.6 | **no** |
| temporal | lateral_repetido | 0.191 | 0.015 | 0.017 | 0.176 | 0.005 | 36.6 | **no** |
| mecanica | cargadas_min | 0.294 | 0.228 | 0.238 | 0.065 | 0.048 | 1.4 | **no** |
| mecanica | contacto_s_jug_min | 0.320 | 0.586 | 0.615 | 0.266 | 0.033 | 8.1 | **no** |
| mecanica | racha_p90 | 26.996 | 57.203 | 63.090 | 30.220 | 6.133 | 4.9 | **no** |
| mecanica | vel_jugador | 1.829 | 2.151 | 2.149 | 0.321 | 0.006 | 50.9 | **no** |
| mecanica | vel_pelota | 3.229 | 4.451 | 4.494 | 1.221 | 0.053 | 23.2 | **no** |
| mecanica | cambios_tecla_s | 3.632 | 3.454 | 3.315 | 0.200 | 0.138 | 1.5 | **no** |
| mecanica | mas_cercano_abierto | 70.954 | 55.360 | 55.378 | 15.599 | 1.207 | 12.9 | **no** |
