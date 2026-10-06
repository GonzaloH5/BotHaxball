# Conformidad del simulador con el dataset X4 (498 grabaciones de Sanguchito)

2026-10-06, sesión autónoma. Antes, el script de SANGUCHITO RS X4 se había validado con 7 grabaciones (`reports/sanguchito_conformance.md`). El dataset nuevo trae 498.

## Saques: `tools.rs4z_restart_conformance --map sanguchito_rs_x4 --pattern 'SanguREC*'`

Los jugadores se fuerzan a su estado grabado en cada tick, así que el error que se mide es solo del árbitro y de la pelota. Resultado final, con las dos correcciones de abajo (`restart_conformance_sanguchito.json`):

| Tipo | n | Detectados | Tipo / ejecutor correctos | Punto (p90) | Fin coincidente (liberación o cesión) | Desfase de la liberación | Pelota a +59 ticks: p90 / máx |
|---|---|---|---|---|---|---|---|
| Lateral | 6.259 | 6.258 | 6.258 / 6.258 | 0,0002 px | **6.211 / 6.229 (99,7%)** | 0 en 6.215 | 0,008 px / 195 px (1 caso) |
| Córner | 1.480 | 1.480 | 1.479 / 1.479 | 0 | **1.463 / 1.463 (100%)** (antes de corregir el kernel: 1.449) | +1 en 1.455, 0 en 8 | 0,008 px / 0,39 px |
| Saque de arco | 1.155 | 1.155 | 1.151 / 1.155 | 0 | **1.146 / 1.146 (100%)** | +1 en 1.143, 0 en 3 | 0,097 px / 91 px |

- Quedan 18 laterales (0,3%) en los que el fin no coincide: no se investigaron todavía.

- La liberación de córner y saque de arco sale con desfase +1: es la convención del evento de patada (se graba un frame antes), no un error de dinámica.
- La cesión del lateral sale con desfase 0.
- El rombo del saque inicial coincide en todos los reposicionamientos en juego: p90 7e-5 px.
- Los pocos valores máximos grandes vienen de tramos con la tanda de penales o de una pelota teletransportada por el saque siguiente.

## Correcciones de esta sesión

### 1. Herramienta: laterales cedidos al rival

- **Síntoma:** al principio, 548 de 6.229 laterales salían como "no coinciden". En todos, la sala le daba el lateral al rival y el simulador "no hacía nada".
- **Causa:** la herramienta dejaba de simular un tick antes del saque siguiente. En un lateral cedido, el saque siguiente es justamente la cesión.
- **Verificación:** siguiendo tick a tick un caso (`SanguREC-1-10-2026-20h09m`, lateral del frame 19316), el simulador cede en el paso que produce el registro 19485 y la sala coloca la pelota del rival en 19486. Es la convención de ±1 de los demás saques.
- **Cesiones:** de los 629 laterales cedidos, el 69% es por corrida. La pelota está a 263–270 px del punto en el tick anterior, lo que coincide con el umbral de 270 del contrato. Casi todo el resto es entrada sin patada: la pelota cruza |y| = 678,325.
- Con la herramienta corregida, coinciden 6.211 de 6.229 laterales (99,7%).

### 2. Kernel: el córner de Sanguchito exige patada hacia la cancha también en y, juzgada al patear

- **Síntoma:** ~2,5% de los córners se liberaban en el simulador entre 28 y 256 ticks antes que en la sala.
- **Medición:** se tomaron todas las patadas del equipo ejecutor durante córners y saques de arco liberados por una patada (`data/x4_ticks`), con la posición del pateador en el registro del evento de patada:

| Regla | Patadas que liberan, bien clasificadas | Patadas ignoradas, bien clasificadas |
|---|---|---|
| x hacia la cancha (contrato anterior), posición después del tick | 2.727 / 2.746 | 322 / 346 |
| x hacia la cancha y, en el córner, también y; posición al patear (antes del tick) | **2.746 / 2.746** | **346 / 346** |

- En el saque de arco la componente y no importa: 566 de 1.222 liberaciones van hacia afuera en y.
- **Cambio:** `env/rs4z/kernel.py` (`FIX_SANGU`). El kernel guarda pelota − pateador antes del tick (`kinfo[:, 3:5]`) y la regla usa eso.
- **Tests:**
  - `tests/rs4z/test_rs4z_sangu.py::test_corner_kick_outward_in_y_does_not_release`;
  - `test_corner_direction_is_judged_before_the_tick`;
  - regresión con la grabación real `tests/rs4z/test_x4_real_conformance.py`: un córner con cuatro patadas ignoradas se libera en el mismo tick que en la sala.

## Demora de colocación de saques

Ver `reports/x4/dataset_audit.md`:
- Sanguchito coloca sin demora, igual que el contrato.
- La distribución de RS ONE coincide con los rangos uniformes del contrato.
- Los tests de la mecánica están en `tests/rs4z/test_rs4z_script_rs.py`: córner demorado con disco en el punto, y Sanguchito sin demora.

## Pendiente

- **Física en juego abierto** con las 498 grabaciones. Ya está verificada con 7 (p90 a 60 ticks: 0,012 px). La física es la misma de RS ONE, y la conformidad de saques sobre 498 grabaciones da 0,008–0,1 px a 59 ticks.
- **2K23 (B1):** el script sigue siendo un supuesto (igual a RS ONE desplazado 70 px), salvo la demora del lateral. Se corrió la conformidad sobre las 2 grabaciones de prueba con bots (13 saques; `restart_conformance_2k23_bots.json`). El script de la sala no depende de quién juega, y **no coincide**:
  - córner y saque de arco: la trayectoria tras la patada se desvía 46 y 26 px a +59 ticks (impulso o curva distintos de RS ONE);
  - lateral: el punto real difiere 2–4 px del de salida (valores con un decimal, p. ej. −79,4 o 368,2) y el simulador libera 2–27 ticks antes.
  - Con 13 saques no se puede ajustar la regla. Hace falta grabar partidos humanos en 2K23 (o la sala de prueba con los bots quietos y humanos sacando) y repetir `tools.rs4z_script_probe` y la conformidad, como se hizo con Sanguchito.
  - Mientras tanto, el RL usa 2K23 solo en el 15% de los partidos, y conviene no tomar sus saques como fieles.
