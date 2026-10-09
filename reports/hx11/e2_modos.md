# E2: modos y versiones del script por partido

Auditoría del 2026-10-08 (§2.3, E2), con [`tools/hx11_modos.py`](../../tools/hx11_modos.py) sobre 394 partidos aptos. Los criterios están en el docstring y se fijaron antes de mirar el resultado. Datos por partido en `data/hx11/modos.json`.

## Modo de cada partido

| Modo | Prendido | Apagado | Indeterminado | Anunciado contra inferido |
|---|---|---|---|---|
| faltas | 185 | 8 | 201 | 2 contradicciones |
| offside | 313 | 4 | 77 | 2 contradicciones |
| powershot | 394 | 0 | 0 | 0 contradicciones |
| faltas_l0 | 185 | 31 | 178 | 0 contradicciones |

- **Faltas cobradas sobre oportunidades** (contacto >= 110 ticks) en los partidos con faltas prendidas: 0.928. **Oportunidades sin cobro** (ninguna falta hasta 20 ticks después del contacto) en esos partidos: 21.5%.
- **`faltas_l0`** (lo que usa la imitación): prendido si hubo cobros o anuncio; apagado si se anunció o si P(apagado) >= 0.85, con P(apagado) = 1 / (1 + q^k) para k oportunidades sin ningún cobro, q la tasa sin cobro de arriba y previa 0,5; desconocido si no.
- **Contactos que llegan a 100–114 ticks** (justo debajo de la regla, firma de un exploit si el RL los concentra): 288 en todo el banco.
- **Partidos con un cambio de modo anunciado a mitad de partido:** 9.

## Por mes

| Mes | Partidos | Faltas prendidas / apagadas | Offside prendido / apagado | Powershot prendido / apagado | Con `!gk` | Anuncia modos |
|---|---|---|---|---|---|---|
| 2026-01 | 8 | 3 / 0 | 4 / 0 | 8 / 0 | 4 | 0 |
| 2026-02 | 7 | 1 / 1 | 3 / 0 | 7 / 0 | 6 | 0 |
| 2026-03 | 1 | 0 / 0 | 1 / 0 | 1 / 0 | 0 | 0 |
| 2026-04 | 6 | 1 / 1 | 3 / 0 | 6 / 0 | 3 | 0 |
| 2026-05 | 12 | 4 / 1 | 8 / 0 | 12 / 0 | 8 | 0 |
| 2026-07 | 9 | 2 / 0 | 7 / 0 | 9 / 0 | 7 | 1 |
| 2026-08 | 62 | 31 / 1 | 59 / 0 | 62 / 0 | 44 | 11 |
| 2026-09 | 267 | 127 / 4 | 206 / 4 | 267 / 0 | 162 | 36 |
| 2026-10 | 22 | 16 / 0 | 22 / 0 | 22 / 0 | 16 | 2 |

## Marcadores de versión del script (anuncios presentes)

| Marcadores | Partidos |
|---|---|
| barra + gk + powershot_pct | 192 |
| barra + powershot_pct | 123 |
| anuncia_modos + barra + gk + powershot_pct | 36 |
| barra + gambeta + gk + powershot_pct | 18 |
| barra + gambeta + powershot_pct | 11 |
| anuncia_modos + barra + powershot_pct | 8 |
| anuncia_modos + barra + gambeta + gk + powershot_pct | 4 |
| anuncia_modos + barra + gambeta + powershot_pct | 2 |

## Por parte

| Parte | Faltas | Offside | Powershot |
|---|---|---|---|
| dev | {'indeterminado': 22, 'prendido': 16, 'apagado': 2} | {'prendido': 33, 'indeterminado': 6, 'apagado': 1} | {'prendido': 40} |
| test | {'indeterminado': 30, 'prendido': 22} | {'prendido': 42, 'indeterminado': 10} | {'prendido': 52} |
| test_grupos | {'indeterminado': 11, 'prendido': 1} | {'prendido': 7, 'indeterminado': 5} | {'prendido': 12} |
| train | {'indeterminado': 138, 'prendido': 146, 'apagado': 6} | {'indeterminado': 56, 'prendido': 231, 'apagado': 3} | {'prendido': 290} |
