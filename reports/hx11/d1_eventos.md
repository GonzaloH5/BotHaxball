# D1: eventos de HaxEleven

Etapa D1 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con las definiciones de [`tools/hx11_d1.py`](../../tools/hx11_d1.py). Son 394 partidos del estadio actual ("HaxEleven RSx11 [1800, 1200]") y 3849 minutos de juego abierto. Los datos están en [`d1_eventos.json`](d1_eventos.json).

## Controles contra lo que anota la sala

| Control | Sala | Detectado | Acuerdo |
|---|---|---|---|
| Goles (subidas del marcador) | 642 | 642 | 100.0% |
| Powershot anunciado, en una patada de juego abierto | 9091 | 8950 | 98.4% |
| % de la barra anunciado, en una pelota parada ejecutada | 6503 | 6491 | 99.8% |
| Faltas con autor en contacto >= 1 s | 233 | 220 | 94.4% |
| Offsides con el infractor tocando la pelota | 964 | 960 | 99.6% |
| Offsides con pasador | 964 | 961 | 99.7% |

## Frecuencia por minuto de juego abierto (los 22 jugadores)

| Evento | Por minuto | Total |
|---|---|---|
| pase | 28.186 | 108479 |
| recepcion | 14.862 | 57200 |
| intercepcion | 9.018 | 34707 |
| perdida | 7.289 | 28053 |
| bloqueo | 5.414 | 20836 |
| conduccion | 4.521 | 17400 |
| saque | 4.32 | 16628 |
| robo | 4.081 | 15708 |
| despeje | 3.304 | 12715 |
| usar | 2.284 | 8791 |
| recuperacion | 2.277 | 8762 |
| cuerpeo | 0.867 | 3335 |
| remate | 0.849 | 3268 |
| pared | 0.619 | 2382 |
| centro | 0.606 | 2334 |
| atajada | 0.284 | 1094 |
| offside | 0.25 | 964 |
| gol | 0.167 | 642 |
| falta | 0.061 | 233 |
| falta_recibida | 0.061 | 233 |

## Resultado de cada acción

| Acción | Completo | Rival | Salida | Gol | Otro | n |
|---|---|---|---|---|---|---|
| pase | 47.2% | 42.8% | 9.2% | 0.0% | 0.7% | 108479 |
| pase_largo | 6.9% | 61.9% | 29.8% | 0.0% | 1.5% | 9633 |
| centro | 25.5% | 63.5% | 10.7% | 0.0% | 0.4% | 2334 |
| remate | 0.3% | 73.7% | 10.2% | 15.9% | 0.0% | 3268 |
| despeje | 42.1% | 49.5% | 8.1% | 0.0% | 0.3% | 12715 |
| perdida | 0.0% | 87.2% | 12.8% | 0.0% | 0.0% | 28053 |

## Atributos

- **pase:** corto 98846, patada 101765, cuerpo 6714, largo 9633
- **recepcion:** de_primera 20771, controla 36429
- **saque_corner:** ejecutado 2865, pierde_3s 2423, gol_15s 178, conserva_3s 442
- **saque_lateral:** ejecutado 10061, pierde_3s 6702, gol_15s 130, conserva_3s 3359, no_ejecutado 54
- **saque_saque_arco:** ejecutado 1896, conserva_3s 703, gol_15s 18, pierde_3s 1193
- **remate:** al_arco 2489, afuera 779
- **saque_tiro_libre:** ejecutado 1746, conserva_3s 517, gol_15s 41, pierde_3s 1229
- **gol:** a_favor 520, con_asistencia 331, en_contra 122
- **falta:** penal 4
- **saque_penal:** ejecutado 6, pierde_3s 4, gol_15s 3, conserva_3s 2

## Powershot: resultado según la carga

| Carga | Acciones | Gol | Completo | Rival | Salida |
|---|---|---|---|---|---|
| 0-24% | 1630 | 0.7% | 34.4% | 59.3% | 5.2% |
| 100% | 268 | 4.5% | 10.8% | 64.2% | 19.8% |
| 25-49% | 6525 | 1.1% | 31.0% | 59.5% | 7.6% |
| 50-74% | 513 | 5.1% | 14.2% | 65.1% | 15.4% |
| 75-99% | 14 | 0.0% | 14.3% | 57.1% | 28.6% |

## Espera hasta sacar (ticks: p10, p50, p90)

- corner: [157, 274, 463]
- lateral: [88, 163, 290]
- saque_arco: [173, 272, 423]
- tiro_libre: [136, 256, 404]
- penal: [132, 186, 338]
