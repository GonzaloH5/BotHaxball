# Traspaso: HaxEleven D0–D3 (estado al 2026-10-08)

Documento para que otro agente siga desde acá sin la conversación previa. El plan general es
[`PLAN_HAXELEVEN.md`](PLAN_HAXELEVEN.md); la memoria del proyecto está en
`C:\Users\gonza\.claude\projects\C--Users-gonza-Documents-HaxballRL\memory\haxeleven-x11.md`.

## Objetivo y reglas que no cambian

**Objetivo:** un bot que entra como cliente a un partido humano de HaxEleven 11v11 y juega el puesto que elige el usuario con `--pos`. Lo elige solo el usuario: nunca el técnico, ni el chat, ni la inferencia a partir de los compañeros.

**Etapas:**
- **Hechas y commiteadas:** S0–S6 (simulador fiel). Último commit: `ef0d147`.
- **Pedido vigente del usuario (2026-10-08, tercera sesión):**
  - retomar con la auditoría [`AUDITORIA_PLAN_HAXELEVEN_2026-10-08.md`](AUDITORIA_PLAN_HAXELEVEN_2026-10-08.md) como criterio permanente, evaluada críticamente;
  - completar D2, D3 y las marcas;
  - preparar los datos y avanzar hacia una primera L0 funcional, con pruebas chicas y evaluación frecuente;
  - pedir aprobación para cambios grandes de arquitectura, gasto de GPU o riesgos bloqueantes.
- **La revisión crítica de la auditoría** está en [`REVISION_AUDITORIA_HAXELEVEN.md`](REVISION_AUDITORIA_HAXELEVEN.md). Sus cambios P1–P10 ya están en el plan.

**Reglas:**
- **Todo corre local.** No usar el pod para HaxEleven salvo que el usuario lo pida. La PC tiene poca RAM libre (1,5–3 GB): usar como máximo 3 procesos.
- **Nombres, chat e ids de Discord solo en `data/`,** que no se versiona. Nunca en `reports/` ni en commits.
- **No ejecutar ni decodificar el script ofuscado de la sala.**
- **Al commitear, solo los archivos HaxEleven.** Codex trabaja en paralelo en el repo: `learn/x4_ppo.py`, `tools/x4_html_replay.py`, `reports/x4/*`, `deploy/rs4z/*`, `learn/x4_coop_*` y `docs/COOPERACION_*` no son de esta línea. Mensajes de commit en español y con la línea de coautoría del sistema.
- **Python:** `.venv/Scripts/python.exe`. La de sistema no tiene numba.
- **Grafo de código:** después de modificar código, correr `graphify update .`, según el CLAUDE.md del usuario.

## Qué pidió el usuario en esta etapa (2026-10-08) y quedó en el plan

En HaxBall todos tienen la misma velocidad y físico, y no hay regate. La ventaja sale de ganar la posición, usar al rival o apoyarse en un compañero. Hay que no perder referencias ni marcas: con una línea de 4 rígida, un 9 humano se ubica para que un pase deje al defensor detrás.

Ya está en el plan (D1, D2, D3, L1, L2 y §6 B):
- pared, usar y cuerpeo;
- quién marca a quién con el HMM de Franks 2015;
- margen sobre la espalda, peligro OBSO y "marca perdida";
- rivales explotadores en L1;
- la restricción de puesto va sobre la zona, no sobre el punto del ancla.

## Archivos de esta etapa (sin commitear todavía)

**Modificados:**

| Archivo | Cambio |
|---|---|
| `bridge/hx11_ticks.js` | El header trae `players: [{id, name}]`; nuevo evento `join` con el campo `player` (no `name`). |
| `tools/hx11_npz.py` | `convert(lines, out_base \| None, max_players=96, return_arrays=False)`. |
| `tools/hx11_script.py` | `Match(base=None, arrays=None, name=None)` acepta arreglos en memoria. |
| `docs/PLAN_HAXELEVEN.md` | Agregados de referencias y marcas en D1–D3, L1, L2 y §6. |

**Nuevos:**

| Archivo | Qué hace |
|---|---|
| `tools/hx11_d0.py` | Caché compacto por partido. Subcomandos `uno`, `lote` y `resumen`. |
| `tools/hx11_d1.py` | Eventos. Subcomandos `uno`, `lote` y `muestra`. |
| `tools/hx11_d2.py` | Roles y formación. Subcomandos `uno`, `lote` y `muestra`. |
| `tools/hx11_marcas.py` | Referencias y marcas. Subcomandos `uno` y `lote`. |
| `tools/hx11_d3.py` | Control del espacio, VAEP y valor de estado. Subcomando `todo`. |
| `tests/test_hx11_data.py` | 9 tests; pasan todos con `.venv/Scripts/python.exe -m pytest -q tests/test_hx11_data.py`. |
| `reports/hx11/d0_cache.{md,json}` | **Desactualizado:** es de la corrida anterior al arreglo del arquero. Regenerarlo. |

### D0 (`tools/hx11_d0.py`)

- **Entrada:** `node bridge/hx11_ticks.js <hbr2> --max-minutes 0 --stdout`, pasado a `build()`.
- **Salida:** `data/hx11/d0/<nombre>.npz` y `.json`.
  - 22 lugares fijos: 0–10 rojo (ataca hacia +x), 11–21 azul.
  - Cuantización int16: posiciones ×10, velocidades ×1000.
  - `phase` con los modos del kernel: 0 abierto, 1 salida pendiente, 2 pelota parada, 3 congelamiento, 4 gol, 5 saque inicial. Además `piece_kind`, `piece_team` y `spot`.
  - Arquero: `gk_pid` (declarado), `gk_inferred` y `gk_src` (1 declarado, 2 inferido).
  - El `.json` trae decisiones de la sala, patadas, números anunciados (powershot %, barra %) y la cobertura del arquero.
- **`--max-players`:** 48 por defecto (ahorra RAM). El lote reintenta con 128 si un partido tiene más.
- **Arreglo de esta sesión:**
  - **Problema:** el `!gk` suele escribirse con el partido detenido, y antes se descartaba. 311 de 399 partidos quedaban sin arquero.
  - **Ahora:** el anuncio se asocia al primer tick posterior (`searchsorted`). Si no hay anuncio, el arquero se infiere: el jugador más cercano a su arco, el más frecuente en bloques de 1 min (`GK_BLOCK = 3600`). Con bloques de 10 s coincidía 89% con el declarado; con 1 min, 100% en el partido de prueba.
  - **Ojo:** el arquero declarado puede jugar lejos del arco (mediana de 915 px en un partido).
- **Lote:** `--jobs 3` tarda unos 11 s por partido, ~72 min para 399. Saltea los partidos que ya tienen `.npz`. Al final escribe `data/hx11/d0/manifest.json`, con split, sesión, grupos y cobertura del arquero por partido.

### D1 (`tools/hx11_d1.py`)

- **Definiciones** en el docstring:
  - toque a ≤ 0,6 px o patada; tramos por jugador;
  - acción final (pase, centro, remate, despeje, pérdida) con su resultado;
  - recepción, intercepción, bloqueo, atajada, robo, recuperación, conducción;
  - gol con asistencia; falta con autor por contacto; offside con infractor y pasador; saques;
  - pared, usar y cuerpeo.
- **Salida:**
  - `data/hx11/d1/<nombre>.json`, con `control` (comparación contra lo que anota la sala) y `events`;
  - `data/hx11/d1/<nombre>.npz`, con `owner` (último toque por tick, −1 con la pelota muerta) y `seg`.
- **Validado en un partido** (versión vieja de D0): goles 6/6, powershot 29/30, barra 20/20, faltas 4/4, offsides 5/5 por toque.
- **`lote`:** lee el manifiesto de D0 y escribe `reports/hx11/d1_eventos.{md,json}`.
- **`muestra --per-kind 4`:** escribe `data/hx11/d1/muestra_revision.md` con nombres (sacados de `data/hx11/meta_*.jsonl`) y tiempo de la repetición, para que el usuario lo revise a ojo. No versionar.

### D2 (`tools/hx11_d2.py`)

- **Muestreo:** juego abierto, un tick de cada 6.
- **Coordenadas del equipo,** relativas al centroide (Bialkowski 2014).
- **Roles:** EM con 10 gaussianas y Hungarian en numba (`hungarian`, `assign_frames`).
- **Formación:** por partido y por ventana de 3 min, con y sin pelota, contra 9 plantillas canónicas. Los roles se nombran línea-lado (DEF-I, MCD-CI, MED-C, DEL-D, …) y el arquero es ARQ.
- **Anclas:** la posición media de cada rol, con y sin pelota.
- **Rol dominante:** los 10 más presentes reciben roles distintos por Hungarian.
- **Arquero:** el declarado; si no, el inferido; si no, el más cercano al arco.
- **Salida:** `data/hx11/d2/<nombre>.{npz,json}`. `lote` escribe `reports/hx11/d2_puestos.{md,json}` y `muestra` escribe `data/hx11/d2/muestra_revision.md`, con nombres.
- **Prueba en un partido:** rojo 4-3-3 sin pelota y 4-2-3-1 con pelota; azul 4-2-3-1. La estabilidad de rol por jugador da 0,3–0,88.

### Marcas (`tools/hx11_marcas.py`)

- **Necesita D0, D1 y D2;** usa `control` y `VMAX` de D3.
- **Quién marca a quién:** HMM de Franks por defensor (permanencia 0,97), γ ajustado por mínimos cuadrados.
  - **Hallazgo:** γ ≈ (0,9–0,96 atacante, ~0 pelota, ~0,06 arco). El modelo colapsa a "el atacante más cercano", así que **no** sirve para decir si la marca es personal o zonal.
  - **Pendiente:** una medida mejor, por ejemplo cuánto sigue el defensor el desplazamiento de su referencia en ventanas de 3 s.
- **Margen = tiempo del atacante − tiempo del defensor** (corregido; antes estaba invertido). Mide la carrera al punto (min(x del defensor, x del atacante) − 200, y del atacante), en coordenadas del defensor; negativo es espalda ganada.
- **Peligro:** control del espacio × xG, con una logística de distancia y ángulo ajustada con los remates de train.
- **Marca perdida:** el margen cruza −10 ticks con el atacante habilitado y peligro ≥ 0,02. El resultado se mide en 5 s.
- **Prueba en un partido:** 14 marcas perdidas, 9 terminan en remate (64%), contra 9% de remate en muestras al azar de la posesión rival. La señal es fuerte.
- **`lote`:** escribe `reports/hx11/d2_marcas.{md,json}` y lee reacción y sigma de `reports/hx11/d3_valor.json` (si no existe, usa 15 y 10). Por eso conviene correr D3 antes que marcas.

### D3 (`tools/hx11_d3.py`)

- **Control del espacio:** Spearman con vmax = 2,86 px/tick (de la física), patada típica de 10 px/tick y frenado 0,99.
  - Reacción y sigma se ajustan por verosimilitud de los pases en una grilla (reacción 5–45, sigma 4–25).
  - Usa 40 partidos de train y 20 de dev, con 3 procesos.
  - En un partido elegía la esquina chica de la grilla; por eso se amplió.
- **VAEP:** MLP de torch para P(marcar en 10 s) y P(recibir en 10 s) por acción, con la acción y las dos anteriores. Se agrega por línea del puesto con D2.
- **Valor de estado:** MLP sobre rasgos que también existen en el simulador. Φ = P(a favor) − P(en contra), para el shaping de L2.
- **Datos:** split de S0; test incluye test_grupos. Hace falta `data/hx11/d0/manifest.json`.
- **Salida:** `data/hx11/d3/` (`vaep.pt`, `estado.pt`, valores por acción) y `reports/hx11/d3_valor.{md,json}`.
- **Estado:** todavía no corrió con datos completos.

## Estado al 2026-10-08, tercera sesión (con la auditoría integrada)

### Hecho y verificado

| Bloque | Resultado | Evidencia |
|---|---|---|
| Higiene del banco | 5 de 399 grabaciones no son 11v11 (una 1 contra 1 hacía fallar D2). `apto` en el manifiesto; **394 aptos** (290 de entrenamiento, 40 de desarrollo, 64 de prueba). D1, D2, D3 y marcas usan solo los aptos (`apt_names`). | `python -m tools.hx11_d0 plantel` |
| D1 | Corrida otra vez: un anuncio de powershot se pegaba a dos patadas (101,6% → 98,4%). Goles 642/642. | `reports/hx11/d1_eventos.md` |
| D2 | 394 partidos. Familia 4-3-3 63%, 3-4-3 20%, 4-4-2 7%, 3-5-2 5% (40 equipos). | `reports/hx11/d2_puestos.md` |
| Validación interna de D2 | Rol dominante igual entre mitades 83,4%. Por tick: 63% en su rol, 11% en permuta y 25% en otro. Ancla de 3 min identificable en el 76,5%. **Decisión P2:** L0 se condiciona con el ancla medida del jugador por mitad y la línea, no con el rol por tick. | `tools/hx11_d2_valida.py`, `reports/hx11/d2_validacion.md` |
| E2 modos | Powershot prendido en 394. Offside 313 prendido, 4 apagado, 77 indeterminados. Faltas (`faltas_l0`) 185 prendidas, 31 apagadas, 178 desconocidas. Con faltas prendidas, la sala cobra el 78,5% de los contactos de ≥ 110 ticks; el contacto antes de un cobro es p50 115. | `tools/hx11_modos.py`, `data/hx11/modos.json`, `reports/hx11/e2_modos.md` |
| D3 + E3 | Modelos A y B. AUC: VAEP 0,905 / 0,906; estado 0,907 / 0,870. ECE ≤ 0,004 (A sobreestima el decil alto). **Fiabilidad del VAEP por jugador entre partidos:** r 0,20 (A) y 0,28 (B), contra 0,50 del pase completo. Φ A–B Spearman 0,90. Control del espacio con σ en el borde (4) y sobreconfiado. | `reports/hx11/d3_valor.md` §4 |
| Marcas | **Seguimiento** (βa por equipo y partido): p10 0,21, p50 0,28, p90 0,35, marca mayormente zonal. Con la referencia que se aleja, la sigue el mismo 25%, la toma otro 34%, queda libre 42%. La **"marca perdida" NO se valida a escala**: 167.506 eventos, remate o gol 6,2% contra 4,7% al azar (× 1,3). El 64% de un partido (n = 14) no se repitió. Hay que rediseñarla (nivel 2) antes de usarla para evaluar. | `reports/hx11/d2_marcas.md` |
| L0: datos | `learn/hx11_data.py`: muestras (partido, t, lugar, retardo d ∈ [8, 36]) con la convención de X4. Condición P2 (ancla y línea), modos P1, coordenadas del equipo y rotación de la acción del azul. Test del espejo (y → −y) contra un partido reflejado. | `tests/test_hx11_data.py` (11/11) |
| L0: política y entrenador | `learn/hx11_policy.py` (conjuntos, como X4) y `learn/hx11_bc.py` (NLL total, en cambios de tecla, de patear, por línea; espejo como aumento). | prueba de humo |
| Banco de lazo cerrado (B5) | `learn/hx11_closed_loop.py`. **Control:** con las teclas grabadas, el jugador se desvía p50 0,1 px y p90 2,9 px a 5 s; 4 de 32 ventanas se cortan antes por una parada del simulador. | `runs/hx11_l0/lazo_humano_humo.json` |
| Plan | Revisión de la auditoría con N1–N7, P1–P10 y tres niveles de experimentos; plan actualizado con las puertas de §8. | `docs/REVISION_AUDITORIA_HAXELEVEN.md`, `docs/PLAN_HAXELEVEN.md` |

### L0-mini: lo que mostró el lazo cerrado (2026-10-08)

**Hallazgos, todos [R] y medidos en partidos de desarrollo con las mismas ventanas:**

1. **Hay que decidir cada 3 ticks y sostener la acción, como X4** (`frame_skip=3` en `env/rs4z`, la evaluación y `deploy/bot.js`).
   - Decidiendo en cada tick, el bot cambiaba de tecla 45 veces por segundo, contra 2,7 de los humanos.
   - La historia propia de la observación está a 3, 6… ticks, así que la decisión por tick queda sin coherencia.
   - Va en `FRAME_SKIP` de `learn/hx11_closed_loop.py`.
2. **La historia propia de teclas causa confusión causal (copycat; de Haan 2019).** Brazos con la misma configuración (30 partidos, 6000 pasos, semilla 1), 640 ventanas desde el jugador más cercano a la pelota:

   | Brazo | Patadas/min | Toques/min | Dist. al ancla | Desvío 5 s | Cambios/s |
   |---|---|---|---|---|---|
   | Con historia | 0,65 | 1,81 | 252 | 302 | 3,8 |
   | Sin historia | 2,37 | 1,55 | 219 | 238 | 14,7 |
   | Humano | 5,1 | 5,0 | 217 | — | 3,8 |

   - **Sin historia:** la ubicación queda humana y las patadas suben 3,7 veces (tasas totales: 1,61 contra 0,43 por minuto), pero el bot vibra.
   - **Arreglos de inferencia que no sirven:** la histéresis empeora la ubicación; la temperatura 0,5 ayuda poco.
3. **La escala ayuda.** L0-mini (60 partidos, 15.000 pasos, con historia) patea 2,21/min, contra 0,65 del brazo de 30 partidos y 6000 pasos.
4. **El problema central es que nadie va a la pelota.**
   - **En el partido reactivo** (los 22 con el modelo):

     | Métrica | Modelo | Humano |
     |---|---|---|
     | Velocidad (px/tick), con historia | 1,77 | 2,26 |
     | Velocidad (px/tick), sin historia | 0,82 | 2,26 |
     | Jugador más cercano a la pelota, p50 (px) | 260–296 | 57 |
     | Patadas por jugador y minuto | 0,02–0,05 | 0,72 |

   - En partidos que arrancan con saque, a veces nadie saca.
   - **Conclusión: una población de L0 no sirve como L1** (confirma la auditoría §2.6 en HaxEleven).
   - **Para el bot solo entre 21 humanos,** la brecha es tocar la pelota: de 3,4 a 4,7 veces menos que el humano.
5. **Brazos en curso** (30 partidos, 6000 pasos, semilla 1):
   - `drophist`: historia borrada en el 50% de las muestras de entrenamiento y entera en la inferencia;
   - `v2drop`: lo mismo más los rasgos de "voy yo" (`--extra`, `hx11_data.engage`: ranking de cercanía a la pelota en el equipo, ventaja sobre el compañero y el rival más cercanos, también hacia la pelota proyectada). Con esos rasgos, en los datos, el más cercano de su equipo patea 4 veces más.
   - **Comparación:**

     ```
     python -m learn.hx11_l0_compare runs/hx11_l0/conhist runs/hx11_l0/sinhist runs/hx11_l0/drophist runs/hx11_l0/v2drop runs/hx11_l0/mini
     ```

     escribe `reports/hx11/l0_brazos.md`.

### Comparación de brazos (`reports/hx11/l0_brazos.md`)

En los dos lazos, con las mismas ventanas:
- **Ningún brazo en CPU cierra la brecha:**
  - **no reactivo:** toques 1,5–2,0/min contra 6,3 del humano; patadas 0,6–2,5 contra 5,7;
  - **reactivo:** cambios de posesión 0,4–1,4/min contra 10,8.
- **Sin historia:** el mejor en patadas (2,48) y en ubicación (220 contra 215), pero vibra.
- **Dropout de historia:** no ayuda; en la inferencia vuelve a mandar la historia.
- **Rasgos de "voy yo":** casi no cambian nada.
- **Más datos (mini):** mejoran las patadas.
- **Corrección de medida (2026-10-08):** "patadas/min" y "toques/min" salían de promediar la tasa de cada ventana. Las ventanas cortadas por una salida (una patada en medio segundo = 120/min) la inflaban, y por eso el mismo modelo dio 2,5 y 7,8. Ahora el resumen usa total sobre tiempo total (`summarize`).
  - **Con esa medida, en 640 ventanas cerca de la pelota:**

    | Modelo | Patadas/min | Toques/min |
    |---|---|---|
    | Humano | 4,7 | 4,6 |
    | Con historia (30 partidos) | 0,43 | 1,38 |
    | Sin historia (30 partidos) | 1,61 | 1,00 |
    | Mini (60 partidos) | 1,13 | 1,26 |

  - Las conclusiones cualitativas no cambian. Las tablas de `reports/hx11/l0_brazos*.md` anteriores a la corrección tienen la media vieja.
- **Brazo en curso:** sin historia más sobremuestreo cerca de la pelota (`--cerca-frac 0.5`; natural 7,7% → 50% del lote). Se compara solo en `reports/hx11/l0_brazos_cerca.md`.

**Más brazos y pruebas en el banco no reactivo** (640 ventanas cerca de la pelota). La métrica de patadas pasa a ser la **efectiva**: en el simulador, cuando la física ejecuta la patada; en los humanos, la de D0. Contar flancos de la tecla castigaba a quien la sostiene.

| Brazo | Patadas efectivas/min | Toques/min | Velocidad | Dist. al ancla | Cambios/s |
|---|---|---|---|---|---|
| Humano | 6,0 | 4,6 | 2,33 | 217 | 3,7 |
| Sin historia (30 partidos) | 2,03 | 1,07 | 1,73 | 216 | 14,8 |
| Con historia (30 partidos) | 0,96 | 1,46 | 2,18 | 249 | 3,7 |
| Sin historia + cabeza de duración (`--hold`) | 0,63 | 1,39 | 2,21 | 255 | 2,1 |

- **Cabeza de duración:** arregla la vibración y la velocidad, pero sostener a ciegas le quita reacción cerca de la pelota. Acortar la duración en la inferencia (× 0,5 o × 0,33) no lo recupera. No sirve como arreglo.
- **Sobremuestreo cerca de la pelota** (`--cerca-frac 0.5`): no mejora los toques.
- **El retardo no es la causa:** sin historia con d = 8 patea 1,71/min y con d = 30, 1,89/min.

**Para la PC local, después del servidor:** la brecha de tocar la pelota es de la política. Las opciones con evidencia que quedan son la escala (corridas remotas) y cambiar el objetivo, por ejemplo L0.5 (ventaja por acción) o un RL dirigido a tocar la pelota, siempre con las puertas.

**Respuesta del usuario (2026-10-08): no hay pod con GPU.**
- Hubo un servidor CPU por 3 horas: `ssh -p 41001 a156cb9b-57aa-4ee4-9383-24eefdeb0bc8@84.242.173.26`. Es un Threadripper con cuota de **2 CPUs** y 95 GB de RAM.
- **Dos corridas con los 290 partidos,** en `/workspace/hx`:
  - `runs/full_conhist`: con historia, 35.000 pasos o 117 min;
  - `runs/full_sinhist_hold`: sin historia más la cabeza de duración, 30.000 pasos o 125 min. Se puede evaluar con o sin la duración.
- **Copias:** cada 15 min se bajan a `runs/hx11_l0/full_*`.
- **Después:** se sigue en CPU local.
- Opciones que se habían planteado (ya no aplican):
- **A) Pod (3060):**
  - 290 partidos de entrenamiento (5 veces L0-mini), la mejor configuración de los brazos y un modelo más grande;
  - curva de escala E6 (25, 50 y 100%);
  - unas horas de GPU por corrida.
  - Hace falta `pod_sync` o copiar solo `learn/hx11_*`, `tools/hx11_d0.py` y `data/hx11/{d0,d1,d2}`, sin pisar lo de Codex.
- **B) Solo CPU local:** como máximo unos 120 partidos por la RAM (~12 MB por partido); horas por corrida.
- **C) Antes de escalar, más brazos chicos en CPU:**
  - acciones en bloque contra la vibración;
  - pérdida con más peso en los cambios y las patadas.

### Cómputo desde la noche del 2026-10-08: la PC del hermano del usuario (el servidor de 3 horas quedó abandonado)

**El servidor de 3 horas quedó abandonado** por pedido del usuario.
- `runs/hx11_l0/full_sinhist_hold`: sin historia más cabeza de duración, 290 partidos. Su mejor checkpoint es el del paso 10.000.
- `runs/hx11_l0/full_conhist`: con historia; murió en el paso 5000.

**Primer punto de escala** (`reports/hx11/l0_escala_parcial.md`, 640 ventanas cerca de la pelota, sin la cabeza de duración):

| Modelo | Patadas efectivas/min | Toques/min |
|---|---|---|
| 290 partidos, 10.000 pasos | 3,07 | 0,83 |
| 30 partidos, 6000 pasos | 1,88 | 1,11 |
| Humano | 6,74 | 6,00 |

- La escala sube las patadas sin empeorar la ubicación (221 contra 217).
- El partido reactivo sigue fallando.

**PC prestada: `hxpod@100.99.31.43`, por Tailscale.** Es el "pod" gratis, prestado por el hermano del usuario.
- **Máquina:** Windows 10, i5-11400F (6 núcleos y 12 hilos), 24 GB de RAM, RX 580 (no se usa: torch no la soporta en Windows y no se le puede bajar la prioridad), ~19 GB libres en C:.
- **Cuenta:** `hxpod` es un usuario estándar, oculto en la pantalla de inicio. Su clave está en `C:\ProgramData\ssh\hxpod-claves`, con una regla `Match User` en `sshd_config`.
- **Scripts de preparación:** `tools/pod_windows/preparar_hxpod.ps1` y `arreglar_hxpod.ps1`, que corrió el dueño como administrador.
- **Sesión SSH:** no tiene perfil (`USERPROFILE` es `C:\WINDOWS`), así que todo va a `C:\hxpod` (`entorno.bat`). PowerShell corre en modo restringido y WMI está denegado; se usa Python con ctypes.
- **Software:** Python 3.12 portable vía `uv`, con torch 2.14.1 CPU, numpy y numba en `C:\hxpodenv` (`instalar.bat`). El código va en `C:\hxpodepo` y los datos de los 330 partidos de entrenamiento y desarrollo en `C:\hxpodepo\data`.
- **Límites acordados con el usuario** (`tools/pod_windows/lanzar.py`, job object de Windows):
  - hilos lógicos 8–11;
  - tope duro del 33% de CPU;
  - prioridad Idle;
  - 8 GB de RAM con prioridad de memoria baja (el tope ya cortó una vez nuestra carga sin afectar la PC);
  - no lanza si `C:\hxpod` pasa de 6 GB o si al disco le quedan menos de 8 GB libres;
  - el proceso se separa de la sesión SSH y no abre ventanas.
- **Velocidad:** unos 9 pasos/s de BC con lotes de 1024. Con la dieta de RAM de `hx11_data`, un partido ocupa ~15 MB; 290 partidos son ~5 GB.
- **Estado y detención:** `C:\hxpodenv\Scripts\python.exe C:\hxpod\estado.py [--detener NOMBRE]` mide y detiene todo el árbol de procesos, porque el `python.exe` del venv es un lanzador.
- **Si apaga la PC:**
  - `learn/hx11_bc.py` guarda un checkpoint completo y atómico (pesos, optimizador, paso e historial) cada ≤ 5 min y en cada evaluación, en solo `model.pt` y `ultimo.pt`;
  - `--reanudar` sigue desde ahí;
  - `python -m tools.pod_windows.vigilar --nombre <n> -- <args de hx11_bc>` corre en la PC del usuario: cada 10 min baja los resultados a `runs/hx11_l0/<n>/` y relanza si se cortó.
- **Corrida en curso:** `l0_sinhist_full`: sin historia, 290 partidos, 80.000 pasos, unas 2,5–3 h. El vigilante escribe en `runs/hx11_l0/vigilante_l0_sinhist_full.log`.

### En curso al escribir esto

- **L0-mini en CPU:** 60 partidos de entrenamiento y 15 de desarrollo, 15.000 pasos con lotes de 1024. Salida en `runs/hx11_l0/mini/` y log en `runs/hx11_l0/mini.log`.
- **Después:** evaluarlo en el banco de lazo cerrado:

  ```
  .venv/Scripts/python.exe -m learn.hx11_closed_loop --model runs/hx11_l0/mini/model.pt --matches 10 --starts 320 --out runs/hx11_l0/mini/lazo.json
  ```

  y compararlo con `--policy humano`.

### Lo que sigue, en orden

1. **Leer L0-mini en los dos lazos.** Atención a la pasividad: patadas y toques por minuto contra el humano en las mismas ventanas, distancia a la pelota y al ancla.
2. **Según el resultado:** ajustar la observación o el modelo con pruebas chicas, y decidir con el usuario el **L0 completo en el pod** (GPU, requiere aprobación).
3. **E4:** tabla de potencia, barata.
4. **Techos humano contra humano** para la puerta de realismo de L1 (B7). Se pueden calcular ya con D1–D3.
5. **Muestras para el usuario.** Contienen nombres: quedan en `data/` y no se versionan.
   - `python -m tools.hx11_d1 muestra --per-kind 4`;
   - `python -m tools.hx11_d2 muestra --matches 10`;
   - una planilla liviana de E1 (vocabulario de `--pos` → ancla y unos 10 equipos-partido).
6. **Mejoras de análisis de nivel 2,** solo cuando una evaluación las necesite:
   - recalibrar el control del espacio (grilla más chica de σ, calibración en desarrollo) (P10);
   - rediseñar la marca perdida;
   - árbitro en sombra en otros meses (E2b).
7. **Cerrar el bloque:**
   - memoria;
   - `graphify update .`;
   - commit solo de los archivos HaxEleven. Hay que revisar antes que ningún reporte tenga nombres de jugadores.

### Archivos nuevos o modificados en esta sesión (sin commitear)

- **Nuevos:**
  - `tools/hx11_d2_valida.py`, `tools/hx11_modos.py`;
  - `learn/hx11_data.py`, `learn/hx11_policy.py`, `learn/hx11_bc.py`, `learn/hx11_closed_loop.py`;
  - `docs/REVISION_AUDITORIA_HAXELEVEN.md`;
  - `reports/hx11/d2_validacion.*`, `e2_modos.*`.
- **Modificados:**
  - `tools/hx11_d0.py` (`roster`, `apt_names`, subcomando `plantel`);
  - `tools/hx11_d1.py` (powershot, aptos);
  - `tools/hx11_d2.py` (aptos);
  - `tools/hx11_d3.py` (aptos, modelos B, calibración, fiabilidad);
  - `tools/hx11_marcas.py` (aptos);
  - `tests/test_hx11_data.py`;
  - `docs/PLAN_HAXELEVEN.md`.
- **Reportes regenerados:**
  - `d0_cache.*`, `d1_eventos.*`, `d2_puestos.*`, `d2_marcas.*`, `d3_valor.*`.
