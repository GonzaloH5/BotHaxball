# D2: puestos y formación en HaxEleven

Etapa D2 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md), con el método de [`tools/hx11_d2.py`](../../tools/hx11_d2.py). Son 394 partidos (788 equipos) del estadio actual. Los datos están en [`d2_puestos.json`](d2_puestos.json).

**Falta la validación:** el arquero sale de `!gk` y es exacto, pero los roles de campo y la formación no tienen verdad todavía. Hace falta la muestra del usuario (`python -m tools.hx11_d2 muestra`) o las alineaciones de Discord.

## Formación del partido sin pelota

| Formación | Equipos | % |
|---|---|---|
| 4-1-4-1 | 190 | 24.1% |
| 4-3-3 | 157 | 19.9% |
| 3-4-3 | 156 | 19.8% |
| 4-2-3-1 | 153 | 19.4% |
| 4-4-2 | 55 | 7.0% |
| 3-5-2 | 40 | 5.1% |
| 5-4-1 | 14 | 1.8% |
| 4-5-1 | 10 | 1.3% |
| 5-3-2 | 7 | 0.9% |
| otra | 6 | 0.8% |

## Formación del partido con pelota

| Formación | Equipos | % |
|---|---|---|
| 4-1-4-1 | 201 | 25.5% |
| 3-4-3 | 175 | 22.2% |
| 4-3-3 | 146 | 18.5% |
| 4-2-3-1 | 144 | 18.3% |
| 4-4-2 | 46 | 5.8% |
| 3-5-2 | 42 | 5.3% |
| 5-4-1 | 12 | 1.5% |
| 5-3-2 | 9 | 1.1% |
| otra | 7 | 0.9% |
| 4-5-1 | 6 | 0.8% |

## Con pelota | sin pelota (los 15 pares más comunes)

| Par | Equipos |
|---|---|
| 4-1-4-1 | 4-1-4-1 | 172 |
| 3-4-3 | 3-4-3 | 140 |
| 4-2-3-1 | 4-2-3-1 | 132 |
| 4-3-3 | 4-3-3 | 125 |
| 4-4-2 | 4-4-2 | 39 |
| 3-5-2 | 3-5-2 | 33 |
| 3-4-3 | 4-3-3 | 14 |
| 5-4-1 | 5-4-1 | 10 |
| 4-1-4-1 | 4-3-3 | 9 |
| 3-4-3 | 4-1-4-1 | 7 |
| 3-4-3 | 4-2-3-1 | 7 |
| 4-3-3 | 4-4-2 | 6 |
| 4-1-4-1 | 4-2-3-1 | 6 |
| 4-3-3 | 4-2-3-1 | 6 |
| otra | otra | 6 |

## Ventanas de 3 minutos sin pelota

| Formación | Equipos | % |
|---|---|---|
| 4-1-4-1 | 893 | 22.2% |
| 3-4-3 | 765 | 19.0% |
| 4-3-3 | 739 | 18.4% |
| 4-2-3-1 | 685 | 17.0% |
| 4-4-2 | 279 | 6.9% |
| 3-5-2 | 194 | 4.8% |
| otra | 147 | 3.7% |
| incompleta | 91 | 2.3% |
| 4-5-1 | 90 | 2.2% |
| 5-4-1 | 83 | 2.1% |
| 5-3-2 | 60 | 1.5% |

## Estabilidad

- Cambios de formación sin pelota entre ventanas seguidas, por equipo y partido: media 1.6, p90 3.
- Parte de las muestras en que un jugador está en su rol dominante (jugadores con >= 100 s de juego): p10 0.342, p50 0.647, p90 0.877.
- Costo contra la plantilla canónica (z por eje; "otra" si pasa de 0.35): p10 0.103, p50 0.175, p90 0.26.
