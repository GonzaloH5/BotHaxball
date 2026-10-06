# Conformidad del simulador en SANGUCHITO RS X4

2026-10-06. Pod QuickPod (1× 3060), código de la rama `reinicio`. Datos crudos en `reports/sanguchito/`.

## Qué se agregó

- **Mapa:** `stadiums/sanguchito_rs_x4.hbs`, exportado de la grabación `SanguREC-29-9-2026-18h12m` (variante en juego).
- **Script de la sala:** `contract.MAPS["sanguchito_rs_x4"]`, con la bandera `FIX_SANGU` en el kernel. Al usar ese mapa se suma sola, junto con `FIX_LATERAL`.
- **Origen de las reglas:** se midieron en las 7 grabaciones (121 saques). Ver el comentario en `env/rs4z/contract.py`.
  - Masa de jugadores 0,5 constante.
  - Rombo en el saque inicial.
  - Lateral en (x de salida, ±690,84):
    - barrera c1 en ±557,76; los rivales que la cruzaron pasan a ±542,76;
    - entra con |y| < 678,325 después de una patada;
    - una corrida de más de 270 px o una entrada sin patada se lo dan al rival.
  - Córner y saque de arco con la pelota fija (invMass 0).
    - Disco de 361,71 en el punto del córner, que solo choca con los defensores.
    - Área de 840/353,11 con empuje a ±825 en el saque de arco.
    - Solo los libera una patada del ejecutor hacia la cancha. La velocidad es v = S·(pelota − pateador después)/|pelota − pateador antes|, con S = 10,35124 en córner y 13,96832 en saque de arco.
    - Curva: (−vx/35, ∓0,05) en córner y (0, −0,02505·vy) en saque de arco. Se mantiene 5 ticks, después decae ×0,97 y vale 0 a los 149.
    - Sin patada válida se libera a los 599 ticks.
- **Supuesto sin verificar:** el plazo del lateral (599 ticks). Ningún lateral grabado pasó de 301.

## Resultados

**Física en juego abierto**: `tools.rs4z_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*'`

| | n | p50 | p90 | p99 |
|---|---|---|---|---|
| 1 tick, pelota | 83 622 | 2e-5 | 1e-4 | 1,4e-4 px |
| 60 ticks, pelota | 790 | 0,002 | 0,012 | 0,045 px |
| 60 ticks, jugadores | 790 | 0,0014 | 0,0026 | 0,014 px |

- La hipótesis de masa 0,3 da un error de 14,6 px en el p90 a 60 ticks. Queda descartada: la masa es 0,5 constante.
- Reloj del saque inicial: congelado en 14 265 de 14 265 ticks.

**Saques, con los jugadores forzados a su estado grabado**: `tools.rs4z_restart_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*'`

| tipo | n | detectados | tipo, ejecutor y punto | liberación | pelota a +59 ticks, p90 / máx |
|---|---|---|---|---|---|
| lateral | 69 | 69 | 69/69, punto ≤ 0,0004 px | 64/64 en el mismo tick, más 5 cedidos al rival que coinciden | 0,007 / 0,030 px |
| córner | 20 | 20 | 20/20, punto exacto | 20/20 | 0,016 / 0,040 px |
| saque de arco | 12 | 12 | 12/12, punto exacto | 12/12 | 0,048 / 0,052 px |

- La liberación de córner y saque de arco sale con desfase +1. Se debe a la convención de frame de los eventos de patada, no a un error de dinámica: la trayectoria coincide.
- Rombo del saque inicial: error máximo de 7e-5 px en todos los reposicionamientos en juego.

**Tests:** 52/52, incluidos los 8 nuevos de `tests/rs4z/test_rs4z_sangu.py`.

## Hallazgos sobre los datos

- Las grabaciones mezclan variantes de estadio:
  - 2 tramos de Classic, 1 de Real Soccer Revolution y 1 de Training RSR en `data/rs4_jsonl`;
  - en 2 grabaciones de Sanguchito, la tanda de penales final usa otra variante, con 53 segmentos y otras posiciones.
- Las dos herramientas de conformidad ahora usan solo los tramos jugados en el estadio del mapa (`stadium_frames`).
- **El dataset de imitación tiene que aplicar el mismo filtro.**
- RS ONE no cambió:
  - el kernel previo al reinicio (7b21f76) y el actual dan resultados idénticos en física y córners sobre las mismas grabaciones;
  - la única diferencia es un sorteo de último toque desconocido, que ahora sale con semilla fija.
