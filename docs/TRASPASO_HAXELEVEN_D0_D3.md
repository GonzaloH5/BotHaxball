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
- **Software:** Python 3.12 portable vía `uv`, con torch 2.14.1 CPU, numpy y numba en `C:\hxpodenv` (`instalar.bat`). El código va en `C:\hxpod
epo` y los datos de los 330 partidos de entrenamiento y desarrollo en `C:\hxpod
epo\data`.
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

### Resultado de escala en prueba (2026-10-09; `reports/hx11/l0_escala_prueba.md`)

- **Corrida:** `l0_sinhist_full` (sin historia, 290 partidos, 80.000 pasos) terminó en la PC prestada en ~2,2 h a 9,8 pasos/s. NLL de desarrollo 1,406. El modelo está en `runs/hx11_l0/l0_sinhist_full/model.pt`.
- **Partidos de prueba, no reactivo** (640 ventanas cerca de la pelota):

  | Modelo | Patadas efectivas/min | Toques/min | Velocidad | Dist. al ancla |
  |---|---|---|---|---|
  | **Este modelo** | 6,42 | 2,15 | 1,95 | 218 |
  | 290 partidos, 10k pasos | 3,20 | 0,88 | — | — |
  | 30 partidos, 6k pasos | 2,19 | 1,30 | — | — |
  | Humano | 7,16 | 4,63 | 2,34 | 196 |

- **Reactivo** (los 22 con el modelo):

  | Métrica | Modelo | Humano |
  |---|---|---|
  | Cambios de posesión por minuto | 9,0 | 10,8 |
  | Patadas por jugador y minuto | 0,46 | 0,72 |
  | Toques por jugador y minuto | 0,11 | 0,40 |
  | Jugador más cercano a la pelota | 114 px | 79 px |

  **Ya se juega al fútbol.** P13 queda matizado.
- **Inferencia:**
  - **Temperatura 0,7:** patadas 6,49 contra 6,64 de los humanos, toques 2,56 contra 4,46, velocidad 2,14 y 11,4 cambios/s.
  - **Decidir cada 6 ticks:** patadas 4,77, toques 2,88 y 6,8 cambios/s.
  - Quedan dos brechas: los toques y la vibración (el humano tiene 4 cambios/s).
- **En curso:** `l0_sinhist_ancho` (la misma receta con hidden 512 y entidades 256, 1,5 M parámetros) en la PC prestada, de noche. El vigilante escribe en `runs/hx11_l0/vigilante_l0_sinhist_ancho.log`.

### Puerta P3 por puesto en prueba (2026-10-09; `reports/hx11/l0_puerta_t07.md`, `learn/hx11_puerta.py`)

Modelo `l0_sinhist_full` con temperatura 0,7, 20 partidos de prueba y 1600 ventanas cerca de la pelota más 1600 al azar. Criterio fijado antes de mirar: el IC 90% de la razón bot/humano se superpone con [0,75; 1,33].

| Métrica | Razón bot/humano | ¿Pasa? |
|---|---|---|
| Patadas efectivas | 0,9–1,02 | sí, en todas las líneas |
| Velocidad | 0,88–0,94 | sí |
| Distancia a la pelota | 0,93–1,03 | sí |
| Distancia al ancla | 1,02–1,09 | sí |
| Arquero | — | sí |
| Toques, cerca de la pelota | DEF 0,58, MED 0,47, DEL 0,40 | **no** |
| Toques, al azar | DEF 0,50, DEL 0,24 | **no** |

- **Lectura:** patea como un humano pero **no conduce ni retiene** la pelota, y eso importa para el powershot (~90 ticks de contacto) y el cuerpeo.
- **Hipótesis:** la vibración de teclas (13 cambios/s contra 4). Decidir cada 6 ticks subió los toques al 61%.

### Bot de sala: piezas listas (2026-10-09)

- **`export/to_onnx_hx11.py`:** ONNX + meta, con el vocabulario de `--pos`. Verificado contra torch: 1,2e-6 en probabilidades. Exportado `deploy/hx11/l0.onnx`.
- **`learn/hx11_fase.py`:** fase de juego **causal** para la sala. Usa el estado del juego, las colocaciones de la pelota por el script (el mismo evento de D0), las patadas y los goles; el equipo que saca sale de la regla del último toque. Contra D0, en 10 partidos de prueba:
  - acuerdo por tick 97,3% (juego abierto 99,99%, pelota parada 98,2%, saque inicial 100%, gol 99,3%);
  - tipo de pelota parada 100%;
  - equipo que saca 90,4%;
  - la "salida pendiente" de D0, que es hacia atrás, queda como juego abierto.
- **`learn/hx11_cerebro.py` (cerebro):**
  - ventana viva con el formato de D0, la fase causal y la observación de `build` con el ancla y la línea de `--pos`;
  - `set_pos` para cambiar de puesto en caliente.
  - **Paridad con el entrenamiento** (`learn/hx11_cerebro_paridad.py`, partido de prueba, 202 decisiones de juego abierto): fase igual al 100% y **diferencia de probabilidades 0,0**.
- **`deploy/hx11/` (cliente node):**
  - `frame.js`: arma el tick desde la sala o una repetición; colocaciones, patadas, goles, `!gk` y offside.
  - `cerebro_proc.js`: proceso del cerebro.
  - `bot.js`: cliente en vivo con `--join` y `--pos`. Se une con una clave nueva y no elige equipo. Retardo = eco medido (por defecto 22, de S6). Comandos de consola: `pos X`, `pausa`, `sigue`, `salir`.
  - **Prueba de punta a punta** (`test_replay.js`, una grabación de prueba a velocidad real): una decisión cada 3,0 ticks; cerebro p50 5 ms y p95 53 ms; dirección igual a la del humano de ese lugar en el 54%.
- **Falta probarlo en una sala real.** Es el piloto C0 con el usuario:

  ```
  node deploy/hx11/bot.js --join <link> --pos LI --name HX-Bot
  ```

  - El admin mueve al bot a un equipo.
  - Partidos grabados.
  - De 2 a 3 puestos.
  - Métricas de proceso contra el banco (E4: con 6–10 partidos solo se ven fallas groseras).

### Partido de 22 bots en la sala del usuario (2026-10-09, madrugada)

- **Montaje:** `deploy/hx11/partido.js`, un cliente por rol (rojo 4-3-3, azul 4-4-2) con el cerebro en modo de varios bots. El ingreso tiene que ir de a uno, con unos 10 s entre bots, porque el servidor maestro limita por IP ("Error de conexión Master"). Entraron los 22 y el usuario le dio admin a un bot, que repartió los equipos.
- **Lo que vio el usuario:** pocas jugadas razonables, mucho ruido, extrapolación enorme en todas las fichas, muchos bots quietos y la sala con lag (picos de ping del host).
- **Grabación del host:** `data/hx11/salas/partido_bots_20261009_0149.hbr2` (no se versiona). Diagnóstico con `tools/hx11_rec_bots.py`, guardado en `reports/hx11/partido_bots_20261009.json`, contra 8 partidos humanos de prueba:
  - **quietos:** del 26% al 55% del juego abierto, contra 2% en los humanos;
  - **velocidad media:** 1,03 contra 2,0;
  - **distancia a la zona:** 300 a 660 contra unos 180;
  - **teclas congeladas:** rachas de más de 1 s sin ningún cambio de tecla en el 20 al 70% del tiempo, de hasta 14 s. Un bot (A10) no se movió nunca.
- **Causa medida:** el proceso de Node con los 22 clientes usaba 1,2 núcleos, con el hilo principal saturado. Cada cliente de node-haxball simula el partido y lo vuelve a simular con cada tecla atrasada que llega. Las teclas salían a ráfagas. Además, los bots cambian de tecla unas 10 veces por segundo contra unas 3,8 de un humano, y el host reenvía cada cambio a todos.
- **Arreglo:**
  - `partido.js` reparte los bots en `--procesos` (6 por defecto), cada uno con su observador y su cerebro;
  - el proceso padre ordena los ingresos de a uno;
  - cada 10 s se informa por proceso: ticks por segundo del observador (tiene que dar 60), decisiones por segundo (20), retraso p95 del bucle y eco.
  - **Histéresis de la dirección** en el cerebro (`--hyst`), para bajar los cambios de tecla.
- **Tercer y cuarto intento (madrugada del 2026-10-09):**
  - Con un cerebro por proceso, los bots andaban bien: ~60 ticks/s por proceso y 20 decisiones por segundo. Pero la PC del usuario quedó al 100% de CPU (cerebros 4,3 núcleos, node 3,7, el navegador del host 1,75) y su ping subió a 1500.
  - Con **un solo cerebro** en el proceso principal (≈ 1,1 núcleos para los 22; prueba sin sala `deploy/hx11/test_partido_replay.js`: los 22 con teclas, respuesta p50 8 ms) anduvo de nuestro lado: observador 59–61 ticks/s, 20 decisiones por segundo, eco p50 8 ticks y los dos arqueros confirmados. La CPU quedó en 70% (node 5,4 núcleos, el host 4,5).
  - **Arqueros:** el `!gk` al cambiar de equipo no alcanzaba. Ahora se repite cada 8 s hasta ver el anuncio "[✅] … es el Arquero del Red/Blue".
  - **El ping del host volvió a dispararse.** El usuario decidió **abandonar por ahora el partido de 22 bots** y probar con **21 humanos y un bot**, que es el objetivo real. Un host en la misma PC que 22 clientes de node-haxball no aguanta.
- **Lo del modelo, aparte de la sala:** el lazo reactivo del simulador con 22 bots (`learn/hx11_selfplay.py`, ahora con métricas de posicionamiento; 16 juegos de 60 s en prueba, temperatura 0,7) da, contra humanos en las mismas ventanas:

  | Métrica | Bot / humano |
  |---|---|
  | Toques | 0,21 |
  | Patadas | 0,57 |
  | Jugador más cercano a la pelota | 150 px contra 58 |
  | Bloque lejos de la jugada | 1,6 |
  | Juego abierto | 0,71 |
  | Goles | 6 veces más |

  - Distancia a la zona: 0,93. La zona la cuidan.
  - En juego abierto, la mediana del más cercano queda en 85–100 px contra 58 desde los primeros 5 s, así que no es una degradación lenta.
  - **Conclusión:** con 22 imitadores nadie sale a disputar la pelota (cada uno "espera" al que la disputaría en los datos). Es un fallo de coordinación y de estados fuera de distribución. La inferencia no lo arregla. Es material para L0.5 o L1 (por ejemplo, ajuste corto contra estadísticas humanas) y para la puerta de realismo P5.
- **Retención de pelota** (lazo no reactivo, 10 partidos de prueba, 640 ventanas cerca de la pelota, temperatura 0,7). Proporción respecto del humano:

  | Variante | Patadas | Toques | Cambios/s |
  |---|---|---|---|
  | Cada 3 ticks | 98% | 57% | 11,4 |
  | Cada 4 | 86% | 63% | 8,6 |
  | Cada 6 | 71% | 75% | 5,9 |
  | Adaptativo (6 a ≤ 30 px) | 70% | 76% | 10,2 |
  | Adaptativo a ≤ 60 px | 71% | 63% | 9,7 |
  | Dirección sostenida y patada libre | 85% | 52% | 10,5 |
  | Histéresis 0,5 | 70% | 40% | 2,1 |
  | Histéresis 0,3 (más fuerte: se mantiene si p ≥ 0,3 × máx) | 62% | 38% | 1,8 |
  | Histéresis 0,8 (la más suave) | 80% | 50% | 2,7 |
  | **Red ancha** (`l0_sinhist_ancho`, paso 42k de 80k, NLL de desarrollo 1,384 contra 1,406), cada 3 | 101% | 56% | 11,2 |

  | **Red ancha, paso 60k** (NLL de desarrollo 1,372), cada 3 | 99% | **71%** | 10,9 |
  | **Red ancha final, paso 80k** (NLL 1,368), cada 3 | 99% | 64% | 10,8 |

  - Entre puntos de control los toques varían de 56% a 71% con el mismo banco, así que el 71% del paso 60k era en parte ruido de una sola evaluación.
  - Con el modelo final, la red ancha mejora la retención unos 7 puntos sobre la angosta (64% contra 57%). No alcanza la puerta.

  - A los 42k la red ancha no cambiaba la retención: 56% contra 57% de la angosta.
  - **A los 60k los toques suben a 71%**, con las patadas en 99%. Es la primera variante que deja los dos cerca de la puerta [0,75; 1,33].
  - Es una sola evaluación, así que hay que confirmarla con el modelo final y con la puerta por puesto (`learn/hx11_puerta.py`).
- **Corte de la PC prestada (2026-10-09):**
  - Dejó de responder entre las 03:55 y las 04:11.
  - Entre los pasos 64k y 66k el reloj de la corrida salta 11,3 h con el mismo pid, así que la PC estuvo **suspendida**; al despertar siguió sola.
  - Además, el Tailscale de la PC del usuario quedó trabado ("starting", sin IP) hasta que el usuario lo reinició, a las 16:50.
  - El vigilante había muerto con la sesión de Claude Code (último registro 04:57). **Lección:** el vigilante no sobrevive al cierre de la sesión, y una PC prestada que se suspende pausa la corrida sin perder nada.
- **`l0_sinhist_ancho` terminado:** 80.000 pasos y NLL de desarrollo **1,3676**, contra 1,4064 de la angosta. Ya está bajado a `runs/hx11_l0/l0_sinhist_ancho/` (`model_final.pt`).

  **La histéresis quedó descartada para la sala.** En el lazo no reactivo, 0,8 parecía buena: dividía por 4 los cambios de tecla y las patadas seguían dentro de la puerta. Pero ese lazo arranca en juego abierto y con humanos alrededor.
  - **Segundo partido en la sala con 0,8** (grabación `partido_bots_20261009_0212.hbr2`): 20 de 22 bots sin dirección el 100% del tiempo y el 54% del tiempo en saque inicial.
    - En el saque todos están quietos y la histéresis sostenía el "quieto".
    - Se reprodujo en el lazo reactivo desde saques (`learn/hx11_selfplay.py --solo-saques --hyst 0.8`): velocidad 0,54 del humano y más cercano a la pelota a 237 px contra 92. Sin histéresis, 1,07 y 108 px.
  - **Sin sostener el "quieto"** (arreglo en `sample`): corren 1,48 veces la velocidad humana, se abren 1,34 veces la separación humana y el más cercano queda a 172 px.
  - **Conclusión:** la histéresis queda en 0 por defecto. Para el temblor y la carga del host hay que buscar otra cosa, como decidir cada 4–6 ticks o una cabeza de duración entrenada.

  Ninguna variante deja a la vez patadas y toques en [0,75; 1,33]. Patear y retener compiten. Lo que queda es estructural: la red ancha en curso, una cabeza de duración o entrenar para decisiones más espaciadas.

### Validación de puestos (E1 liviano)

- **Parte 1, hecha:** el usuario aprobó el dibujo de `--pos` (`reports/hx11/pos_vocab.html`).
  - **Observación suya:** el mediapunta y otros tienen libertad para moverse por todo el ancho.
  - **Extensión anotada en el plan (§4):** condicionar también por la "libertad" medida del jugador.
- **Parte 2, opcional:** `data/hx11/d2/muestra_revision.md`, con nombres. Solo si el usuario conoce esos grupos.

### Puerta P3 con la red ancha final (2026-10-09; `reports/hx11/l0_puerta_ancho_t07.md`)

Mismo banco que la puerta anterior: temperatura 0,7, 20 partidos de prueba, 1600 ventanas cerca de la pelota y 1600 al azar, el mismo criterio. Toques por minuto, razón bot/humano con su IC 90%:

| Línea | Angosta, cerca | **Ancha, cerca** | Angosta, al azar | **Ancha, al azar** | Puesto con la ancha |
|---|---|---|---|---|---|
| ARQ | — | — | 0,43 (pasa) | 0,53 (pasa) | **pasa** |
| DEF | 0,58 (0,50–0,67) no | **0,65 (0,56–0,75) pasa** | 0,50 no | 0,54 (pasa) | **pasa** (justo) |
| MCD | 0,89 pasa | 0,83 pasa | 0,15 (pasa) | 0,25 (pasa) | **pasa** |
| MED | 0,47 no | 0,51 (0,45–0,57) no | 0,50 pasa | 0,46 pasa | no |
| DEL | 0,40 no | 0,58 (0,50–0,67) no | 0,24 no | 0,49 pasa | no |

- El resto de las métricas pasa en todas las líneas: patadas 0,9–1,2, velocidad 0,86–0,95, distancia a la pelota 0,88–1,03 y al ancla 1,02–1,07.
- **Lectura:** la red ancha pasa la puerta en ARQ, DEF y MCD. MED y DEL siguen cortos de toques cerca de la pelota (los humanos de esas líneas conducen más: 6,2–6,8 toques/min contra ~3,5 del bot).
- **Para el piloto con 21 humanos:** empezar por puestos defensivos (ARQ, LI, LD, DFC, MCD). La ancha es el modelo por defecto de `bot.js` y `partido.js` (`runs/hx11_l0/l0_sinhist_ancho/model_final.pt`). La paridad del cerebro con este modelo da diferencia 0,0.

### Retención: la causa era el sorteo de la tecla de patear (2026-10-09, noche)

- **Diagnóstico** (`learn/hx11_closed_loop.py`, métricas nuevas de control, control largo y patadas cargadas): todo el déficit estaba en las posesiones largas. El control en rachas de 90 ticks o más estaba al 33–42% del humano, mientras el control corto estaba a la par.
- **Mecanismo:** el bot patea un 33% más por segundo de control. La patada sale al *apretar* la tecla, y el sorteo independiente cada 3 ticks partía una apretada en varias.
  - En los datos humanos, la probabilidad de patear por segundo de control es casi constante (~0,8/s), así que no faltaba un "reloj de posesión".
  - El ruido persistente de la dirección (`StickyNoise`, Gumbel-max) baja la vibración a nivel humano, pero no mejora los toques: la vibración no era la causa.
- **Arreglo de inferencia:** tecla de patear persistente (`--renovar-patada`, en `sample`, el lazo, el selfplay, la puerta y el cerebro de la sala). Elegido en partidos de **desarrollo** (12 partidos, 640 ventanas):

  | Renovación | Patadas | Toques | Control largo |
  |---|---|---|---|
  | 1 | 0,98 | 0,57 | 0,33 |
  | 0,7 | 0,95 | 0,63 | 0,46 |
  | 0,5 | 0,91 | 0,68 | 0,44 |
  | 0,33 | 0,79 | 0,78 | 0,72 |

- **Puerta P3 en prueba** (`reports/hx11/l0_puerta_ancho_patada033.md`): **pasan los 5 puestos.**
  - Patadas 0,73–0,80 (el IC toca 0,75).
  - Toques cerca de la pelota: DEF 1,14, MCD 1,44, MED 0,87 y DEL 0,86.
  - Velocidad 0,87–0,95 y ubicación 1,02–1,07.
  - Corrió con el árbitro antes de los dos últimos cambios del lateral; en las ventanas no reactivas cerca de la pelota eso casi no pesa.
- **Acciones en bloque** (`--chunk`, `Ensemble` de ACT): el ajuste `l0_ancho_bloque` (20.000 pasos desde la ancha) terminó en la PC prestada, pero **todavía no se evaluó**.

### Simulador: reglas de pelota parada corregidas con las grabaciones (2026-10-09, noche)

El usuario las vio en los partidos simulados (visor `learn/hx11_replay_html.py`, ahora con las líneas reales del estadio y la pelota parada en juego):

- **Saque de arco:** la sala saca a los rivales del área grande a |x| = 1035, con su misma y, un tick después de la colocación (328 de 328 en el banco). Faltaba en `env/hx11` (`_gk_clear_box`).
- **Lateral** (anuncios "COMO NO VAS A SABER SACAR UN LATERAL! 💀" y 6603 laterales del banco):
  - **Llevar la pelota por la línea es legal** (122 px de mediana). Hay planos solo para la pelota en |y| = 1238,14, que la frenan en 1230.
  - **Patear con la pelota ya adentro** (metida empujando) es mal sacado: 36 de 37 casos.
  - **Meterla, volver a sacarla y patear desde afuera** es legal: 24 de 25 casos, adentro hasta 66 ticks.
  - **Metida y dejada adentro sin patear:** mal sacado pasados `LATERAL_INSIDE_TICKS` = 90 (regla del usuario; los humanos casi no lo hacen).
  - **Patada que no mete la pelota** en 45 ticks: mal sacado contra el **último que la tocó** (la sala repite a los ~51 ticks). Si la toca un rival afuera, vuelve a sacar el mismo equipo (6 de 8).
  - Las primeras versiones de ese día estaban mal: el supuesto de 2 s desde la colocación, y "cualquier empujón es mal sacado", que trababa a los bots en un ida y vuelta.
- **Pruebas:** 15 en `tests/test_hx11_env.py`, una por regla. El árbitro en sombra contra las 31 grabaciones corre con las reglas finales (`runs/hx11_l0/s4_arbitro_v2.json`).
- **Pelotas paradas de los bots en el lazo reactivo** (`learn/hx11_pelota_parada.py`): con las reglas correctas, **mal sacado 48% contra 2% humano**. Además, a veces ningún bot va a sacar (laterales de 27 s, un córner de 47 s).

### Ajuste corto del lateral (L0.5 dirigido, aprobado por el usuario el 2026-10-09)

- `learn/hx11_ajuste_lateral.py`: **autoimitación filtrada** (Oh et al. 2018).
  1. Episodios de lateral en el simulador desde laterales reales de entrenamiento, con los 22 bots con la configuración de sala (temperatura 0,7, patada 0,33).
  2. Se guardan las decisiones del equipo que saca, cerca de la pelota, en los laterales bien sacados.
  3. Ajuste con `learn.hx11_bc --init <ancha> --extra-data <colecta.npz> --extra-frac f`.
- **Prueba chica:** 75% bien, 22% mal y 3% por tiempo.
- **Medición:** tasa de "bien" sobre 400 laterales de desarrollo (`base_dev`) antes y después, y la puerta P3 completa antes de usar el modelo.

- **Resultado del ajuste r1** (`runs/hx11_l0/l0_ajuste_lateral_r1`; 2000 pasos, 20% del lote con 105.416 decisiones de laterales bien sacados; NLL humana 1,370 contra 1,368): **empeoró**, de 75% a 70% bien sacados en los mismos 400 laterales de desarrollo. **No se usa.**
  - Con 75% de éxito, filtrar por episodio casi no da contraste.
  - Además, entrenar sobre acciones sorteadas a temperatura 0,7 y volver a aplicar 0,7 endurece al bot.
- **La causa real era la patada persistente.** Todos los fallos eran "metida empujando sin patear a tiempo": el que saca se guardaba la patada.
  - Con sorteo independiente: 91% bien.
  - **Con la persistente solo en juego abierto** (`piece_rows`: independiente en pelota parada y saque inicial; también en el cerebro de la sala, con la fase causal): **90% bien, 10% mal** (los humanos rondan el 97%).
  - La retención en desarrollo **mejora**: toques 0,91, control largo 0,90, patadas 0,80.
  - Con 22 bots, el lateral sigue mal (61%): el compañero más cercano está a 140–830 px al colocar. Es el fallo de coordinación, no la patada.
- **Simulador final contra la sala** (árbitro en sombra v4, `runs/hx11_l0/s4_arbitro_v4.json`): colocaciones 1509 aciertos, 43 de más y 30 no vistas (original 1490, 53 y 49). Precisión 96,0% y recall 96,8%, **mejor que el original en todo**. El lateral, refinado con los toques: si lo vuelve a tocar el que sacó o un compañero, cuenta como sacado (370 + 35 casos del banco).

### Primer partido real con humanos y córners (2026-10-09, noche)

- **Piloto C0:** bot `GonzaloH4` (avatar 5, bandera uy), puesto **MCD**, en una sala ajena ("[H11] BOV vs ???").
  - Configuración: red ancha final, patada persistente solo en juego abierto, `--sin-saques corner,tiro_libre,saque_arco`.
  - Entró bien, con eco de 8 ticks. A las 22:16 **lo baneó un admin ("ST")**; motivo desconocido.
  - El usuario notó mucho temblor: el bot cambia de tecla ~11 veces por segundo contra 3,8 del humano, y los clientes lo extrapolan. Puede haber sido la causa del ban.
  - **No se lo volvió a meter:** sería saltarse el ban. Luego el usuario pidió no meterlo más.
  - **Pendiente:** la rec, si el usuario la tiene, y bajar el temblor (dirección persistente `--renovar 0.33`, a validar con la patada persistente).
- **Bot de sala:** `--flag/--lat/--lon/--avatar` en `bot.js`. Con la PC del usuario ocupada (mediciones, el vigilante bajando modelos), el cerebro llegó a p95 de 1,2 s; libre, 8–9 ms. **Durante un partido no se corre nada pesado en esa PC.**
- **Córners de los bots** (`learn/hx11_corner.py`, 254 córners de desarrollo, en la PC prestada):

  | Variante | Desde atrás | Italiano | Al área | 1.er toque atacante | Remate | Gol | Sacados |
  |---|---|---|---|---|---|---|---|
  | Base | 38% | 50% | 18% | 7% | 2% | 2% | 94% |
  | Ajuste h1 (humanos que sacan córners, 10% del lote, 2000 pasos) | 48% | 45% | 28% | 13% | 8% | 2% | 95% |
  | Base + regla (solo patea desde atrás) | 99% | 0% | 27% | 12% | 5% | 3,5% | **68%** |
  | h1 + regla | 98% | 0,6% | 41% | 31% | 8% | 4,5% | **70%** |
  | Humanos | 95% | 3,5% | 92% | 36% | 40% | 4,3% | 100% |

  - **Potencia:** el bot patea con la barra baja en el 10% de los córners (masa p10 1,15 contra 3,30 humana; la pelota sale a 5 px/tick contra 16). El humano espera la barra alta. La barra **sí** está en el simulador y en la observación (la masa de la pelota).
  - **Lectura:** la regla arregla el "hacia dónde", pero el bot no aprendió a **rodear la pelota**: 30% sin sacar.
  - **En curso:** ajuste fuerte h2 (30% del lote, 6000 pasos, `runs/hx11_l0/l0_ajuste_corner_h2`). Después: córners con y sin regla, laterales, retención (`C:/hxpod/runs/ajuste_corner/eval_h2_driver.py`) y la puerta P3 en la PC del usuario (la PC prestada no tiene los partidos de prueba).
- **Comba:** toda comba queda entera 6 ticks y después ×0,981 por tick. Coeficientes ajustados; error del powershot −38%. Falta la comba de patadas sin carga en juego abierto.

### Ajuste fuerte del córner (h2) y reglas de patada (2026-10-09, noche)

- **h2:** 6000 pasos con el 30% del lote de decisiones de humanos que sacan córners (`runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt`). Es el **modelo por defecto del bot**.
  - Puerta P3 con la configuración de sala: **pasan los 5 puestos** (MCD: patadas 0,67, toques 1,08, ubicación 1,06). `reports/hx11/l0_puerta_h2_sala.md`.
  - Laterales en desarrollo: 91,5% bien. Retención: 0,79 / 0,93 / 0,84.
- **Regla de patada en pelotas paradas con barra** (`learn/hx11_corner.py --regla --regla-barra`; en el cerebro, `--regla-saques`, apagada por defecto): solo se patea desde atrás de la pelota respecto de adónde va, y con la barra alta (masa ≥ 2,8, o pasados 10 s).
  - **Córner, h2 + reglas:** 84% sacados, italiano 0%, llega al área 41%, sale por otro lado 30%, remate 10%, velocidad p50 19,1 (humanos: 100%, 3,5%, 92%, —, 40%, 20,0).
  - **Tiro libre, h2 + reglas:** sale de la cancha 28% (base 37%, humanos 6%), avance 610 px (base 166, humanos 882), potencia igual a la humana.
  - Siguen por debajo del humano en puntería, así que con humanos esas pelotas paradas las sacan ellos (`--sin-saques`).
- **Temblor:** dirección persistente solo lejos de la pelota (`--renovar 0.33`, por defecto en el bot de sala). Cambios de tecla al azar 10,1 → 4,7 por segundo (humano 3,0); cerca de la pelota el juego no cambia.

### Conducción y powershot, faltas y offsides (2026-10-09/10, madrugada)

- **Diagnóstico** (lazo no reactivo con h2; métricas nuevas en `learn/hx11_closed_loop.py`):
  - El bot controla la pelota como el humano (≤ 4 px: 1,07), pero **corta el contacto**: contacto seguido ≥ 90 ticks 0,57, rachas que llegan al primer nivel de carga 0,51, patadas cargadas 0,43.
- **Ajuste c1** (desde h2, 4000 pasos, 10% córner + 20% decisiones de humanos llevando la pelota pegada, `learn/hx11_corner.py humanos --tipo conduccion`): **se pasa**.
  - Contacto largo 1,30, rachas de carga 1,15, toques 1,41.
  - Patadas 0,58 y patadas cargadas solo 0,56.
  - Falla la puerta (MCD: patadas 0,49, toques 1,89). **Descartado.**
  - Promedios de pesos h2/c1 al 35% y 60%: las patadas caen y las cargadas no suben (0,28 / 0,37).
  - **Lectura:** copiar la conducción enseña a retener, no cuándo patear cargado. Es material para un ajuste que mire el resultado (RL/L2).
- **Acciones en bloque (ACT) evaluadas y descartadas** (`l0_ancho_bloque`, retención en desarrollo con la configuración de sala):
  - sin ensamble: patadas 0,67 y toques 0,93 (h2: 0,77 y 0,93);
  - con ensamble de 8: toques 0,72 y contacto largo 0,26;
  - el temblor no baja (~7 cambios/s).
  La dirección persistente lejos de la pelota resolvió mejor lo mismo.
- **Faltas y offsides** (lazo reactivo de 22 bots, por 10 min de juego abierto): **0 faltas**; **5,5 offsides** contra ~1,7 humanos (estimado con la proporción de la sala, 138 offsides y 49 faltas en 31 partidos). Pesa para un bot delantero; el MCD casi no queda en offside.

### Coordinación de 22 bots: dónde está la brecha, regla del lateral y "quién va" (e1) (2026-10-10, madrugada)

El usuario eligió la opción A (quién va + refuerzo). Mediciones en el lazo reactivo de 22 bots (`learn.hx11_selfplay`, 24 juegos de 90 s en desarrollo, temperatura 0,7, `--renovar 0.33 --renovar-patada 0.33`, retardo fijo de 15 ticks como la sala), contra los humanos en las mismas ventanas. Scripts de diagnóstico en el scratchpad de la sesión (cercano por fase, pelotas paradas por tipo, laterales repetidos, contacto).

- **El retardo no explica la brecha:** con d = 8, 15 o 36 el contacto con la pelota y el más cercano casi no cambian.
- **"Quién va" en juego abierto era moderado:** el más cercano a la pelota estaba 1,23–1,38 veces más lejos que entre humanos. La brecha grande estaba en las pelotas paradas (1,8 veces).
- **Laterales con 22 bots:** el **49% se repetía** en el mismo punto (humanos 4%) y había 41 laterales cada 10 min (humanos 19).
  - Casi todos eran **patadas paralelas a la línea**: el que saca llega caminando por la línea y patea apenas toca la pelota.
  - Física del simulador, verificada: desde afuera con ≥ 5° la pelota entra; desde adentro o paralela corre por la línea, y solo vale si la toca un compañero (480 ticks).
  - Humanos: ángulo hacia adentro p10 6°, p50 27° (3.124 laterales).
- **Regla del lateral** (`kick_rule_ok` en `learn/hx11_policy.py`): patear solo desde afuera de la pelota, con ≥ 6° hacia la cancha. Las posiciones se llevan al momento de la decisión con la velocidad (`seen_now`); sin esto se bloqueaban patadas buenas.
  - **Con 22 bots:** repetidos 49% → **21%**, laterales cada 10 min 41 → 27, juego abierto 62% → 66%. Lo que queda: la mete empujando sin patear.
  - **Con compañeros humanos** (colector desde laterales reales, dos semillas): baja de 90% a 84% bien. El bot bloqueado empuja la pelota adentro, y la patada por la línea a un compañero humano cercano sí servía.
  - Por eso va **aparte y apagada**: `--regla-lateral` en el cerebro (no usar con humanos) y en `learn.hx11_selfplay`. `--regla-saques` sigue siendo solo córner, tiro libre y saque de arco. Pruebas: `tests/test_hx11_regla.py`.
- **e1, la señal de "quién va"** (`runs/hx11_l0/l0_quienva_e1/model.pt`):
  - Ajuste desde h2 con los rasgos `engage` (`--extra`, observación v2: puesto en la cercanía a la pelota y ventaja sobre el compañero y el rival más cercanos), con las 5 entradas nuevas en cero (`--init` ahora rellena columnas).
  - 20.000 pasos con los 290 partidos y 30% de córner v2 (`humanos_corner_v2.npz`), en la PC del hermano.
  - NLL de desarrollo 1,3716 → 1,3705: casi igual, como con 30 partidos.
  - En el lazo de 22 bots, **mejora en las dos semillas** (h2 → e1):
    - más cercano en juego abierto: 1,27 → 1,16 y 1,38 → 1,29 veces el humano;
    - offsides: 3,8 → 2,0 y 1,8 → 0,9 cada 10 min (humanos ~1,7–2,3);
    - juego abierto: 66% → 67% y 58% → 62%;
    - toques: +8% y +14%.
  - **Puerta P3 de un bot entre humanos** (`reports/hx11/l0_puerta_e1_sala.md`): pasa todo menos el **MCD cerca de la pelota**. Toca 2,02 veces lo humano (IC 1,37–2,93; h2: 1,08). Con humanos alrededor, el MCD de e1 va de más a la pelota.
  - **Decisión:** el bot de sala sigue con **h2**; e1 es mejor para la población de 22 bots (lazo reactivo, refuerzo).
- **Lo que más falta ahora es llevar la pelota:**
  - Contactos por minuto iguales al humano (1,0), pero el tiempo con alguien en contacto es 0,4 del humano y las rachas largas (p90) 0,3: 24 contra 82 ticks.
  - Las rachas de ≥ 6 ticks terminan en patada en el 34% (humanos 50%; largo p50 12 contra 28 ticks), se escapan en el 52% (44%) y con un rival encima en el 14% (6%).
  - Es el mismo hueco que la retención y el powershot: copiar la conducción no lo arregló (c1). Es lo que tiene que atacar el refuerzo (A3).

### Refuerzo A3, piloto corto (aprobado por el usuario el 2026-10-10)

- **`learn/hx11_rl.py`:**
  - Un protagonista por partido (jugador de campo al azar) arranca de h2 y juega contra 21 bots e1, con las reglas de patada en ellos.
  - Muestreo igual que en la sala: temperatura 0,7, ruido persistente y `--sin-saques` del bot.
  - PPO con penalización KL(π‖π_ref) exacta sobre las 18 acciones, a la misma temperatura; π_ref es h2 congelada y el coeficiente se adapta hacia `--kl-objetivo`.
  - El crítico arranca del tronco de h2. Premios configurables con `--premios k=v,...`.
  - En la PC del hermano corre con `tools.pod_windows.vigilar --modulo learn.hx11_rl`: unos 20 s por iteración de 4096 decisiones.
- **Control:** retención entre humanos (`learn.hx11_closed_loop --cerca`, la misma configuración de sala que da para h2: patadas 0,77, contacto largo 0,57, rachas de carga 0,51, patadas cargadas 0,43) y después la puerta P3.
- **p1** (premio por contacto hasta 2,5 s, pase +0,05, pérdida −0,15): en la iteración 70 retiene más (contacto largo 0,73, rachas 0,76) pero **patea mucho menos (0,48)** y carga menos (0,34). Es el mismo resultado que c1. **Cortado.**
  - **Por qué:** en el banco, solo el 38% de las posesiones humanas termina en un compañero y el 47% en un rival, pero igual la pelota avanza (101 y 182 px). Con una pérdida cara, patear "no conviene", y eso castiga el juego humano normal.
- **p2** (terreno ganado por posesión, con signo, 0,0005/px; pase +0,05; pérdida −0,05; patada cargada +0,1; sin contacto): en la iteración 40 patea más (0,83), pero **lleva mucho menos** la pelota (contacto largo 0,34, rachas 0,22, cargadas 0,30). Se la saca de encima hacia adelante. **Cortado.**
- **p3** (p2 con patada cargada +0,5):
  - en la iteración 40 queda como h2 (cargadas 0,44, patadas 0,80, contacto largo 0,46);
  - en la 65 empeora: cargadas 0,34, rachas de carga 0,38, patadas 0,84.
  - La PC del hermano dejó de responder a las 04:46 del 2026-10-10 (y el Tailscale local quedó en "NoState"), así que la corrida quedó cortada ahí.
  - La corrida en realidad terminó las 200 iteraciones (la PC solo perdió la conexión). Los controles 40–200 dan patadas cargadas entre 0,31 y 0,49, sin tendencia, y llevan la pelota menos que h2.
- **Lectura de p1–p3:** con tres juegos de premios, ninguno sube las patadas cargadas entre humanos; cada premio solo cambia retener por patear. Se entrena entre bots (e1) y se evalúa entre humanos: lo que conviene contra estos compañeros y rivales bot no es lo que mejora contra humanos.
- **p4, entre humanos grabados** (`--entorno humanos`: ventanas de 8 s de partidos de entrenamiento como el jugador de campo más cercano a la pelota; los otros 21 repiten sus teclas; premios de p3). Unos 12 s por iteración.
  - **Retención entre humanos** (desarrollo):
    - control 40: cargadas 0,59, contacto largo 0,93, rachas 0,93, patadas 0,67;
    - control 80: cargadas 0,53, contacto largo 0,64, rachas 0,63, patadas 0,77;
    - controles 120–200: patean cada vez menos (0,63) y las cargadas vuelven a 0,46.
  - **Puerta P3 del control 80** (`reports/hx11/l0_puerta_rl_p4_80.md`): pasa 35 de 36, pero **falla el MED cerca de la pelota en patadas** (0,63, IC 0,58–0,69).
  - **Prueba reactiva** (`hx11_protagonista_bots`: protagonista entre 21 bots e1 que sí reaccionan, unos 33 min de juego abierto por modelo): contacto h2 0,66 s/min, p4-40 0,39, p4-80 0,47; patadas cargadas, casi ninguna en los tres.
  - **Lectura:** **p4 mejora porque los humanos grabados no reaccionan** al bot. Aprendió que aguantar la pelota es seguro, y frente a rivales que reaccionan no se sostiene. Con una muestra chica (10 min) había parecido lo contrario: era ruido.
- **Conclusión del piloto A3:** ninguno de los dos entornos sirve para refuerzo. Los bots no son realistas y los humanos grabados no reaccionan. **El bot de sala sigue con h2.** Para refuerzo haría falta un entorno reactivo y realista: una población que pase la puerta de realismo, o partidos reales.

- **Puerta P3 del control 40 de p4** (`reports/hx11/l0_puerta_rl_p4_40.md`): falla 5 chequeos: patadas de DEF, MCD y MED, y toques de DEF y MCD (de más).

### Seguir mejorando la imitación (pedido del usuario, 2026-10-10)

- **Partido real del bot (grabación del usuario, `HaxEleven-9-10-2026-22h26m.hbr2`):** GonzaloH4 jugó 135 s de MCD (83 de juego abierto) y lo echaron "por dos decisiones erradas". El análisis (scratchpad `analizar_bot_real.py` y `jugadas_bot.py`) muestra cuáles:
  1. **1:47:** la pelota venía hacia su campo pegada a la banda. La alcanzó de frente y la pateó **para atrás por la línea**, y fue lateral del rival en x = −1378, sin nadie que lo apurara (rival más cercano a ~80 px).
  2. **2:06:** la llevó 1,5 s en campo propio y la pateó **directo a un rival** (recuperada en 0,4 s). El contraataque llegó cerca del área y el bot no volvió.
  - Además: 6,5 cambios de tecla por segundo (humanos de esa sala 3,6), distancia mediana a la pelota 420 px contra 581, y toques 1,27 por minuto contra 0,66.
  - La causa es la misma que la de los laterales: **patea apenas llega a la pelota, sin acomodarse**.
- **Métricas de calidad de patada** en el lazo cerrado (`kick_outcomes` en `learn/hx11_closed_loop.py`): patadas hacia atrás en campo propio, regaladas (la toca después un rival), afuera y a un compañero, del bot contra el humano en las mismas ventanas.
  - Las ventanas humanas son, por construcción, de juego abierto, así que "afuera" humano es 0: lo del bot es exceso.
  - **h2:** atrás en campo propio 2,04 veces el humano; afuera 0,69 por minuto; a un compañero 0,34.
- **Regla de patada en juego abierto** (`open_kick_ok` en `learn/hx11_policy.py`; `--regla-juego` en el lazo cerrado y la puerta). No patea:
  - **pegado a la banda hacia afuera:** en el banco, el 3% de las patadas humanas, y el 56% de esas sale;
  - **hacia atrás en campo propio sin un compañero en esa dirección** (35°, 700 px): los humanos patean hacia atrás en el 6% de sus patadas, casi siempre como pase (42% a un compañero).
  - **h2 con la regla:** atrás 2,04 → 1,12; afuera 0,69 → 0,40; pero patadas 0,77 → 0,68. A un compañero, sin cambio (0,32). Puerta P3 con la regla: **36/36** (`reports/hx11/l0_puerta_h2_reglajuego.md`).
  - Con esto quedó **encendida por defecto en el bot de sala** (`--regla-juego` en el cerebro; `--sin-regla-juego` en `deploy/hx11/bot.js` la apaga). Prueba sin sala con la grabación del partido real: decide cada 3 ticks, 5–7 ms.
  - **Patadas atrás separadas en pase y error** (lazo cerrado, con unos 310 min de juego cerca de la pelota por variante, h2; por minuto):

    | | atrás que van a un rival o afuera | afuera | patadas efectivas | a un compañero |
    |---|---|---|---|---|
    | humano | 0,049 | — | 6,50 | 1,50 |
    | sin regla | 0,228 | 0,79 | 5,31 | 0,47 |
    | primera regla (−0,5, 35°, 700 px) | 0,146 | 0,64 | 4,92 | 0,48 |
    | **v3 (por defecto): cualquier componente hacia atrás; compañero a 25°, 500 px, sin rival a 45 px de la línea ni a 70 del compañero** | **0,087** | **0,55** | 4,81 | 0,47 |

    La regla saca errores pero no enseña a elegir: las patadas que llegan a un compañero siguen en un tercio de las del humano.
- **Puerta de realismo P5** (`learn/hx11_realismo.py`):
  - **Métricas:** 22, en cuatro familias (colectivas, tácticas, temporales, mecánicas). Distancia W1 por ventana de 90 s: sintético contra humano de prueba, con techo = humano de entrenamiento contra prueba.
  - **C2ST:** red chica sobre 20 rasgos de clips de 10 s, con validación cruzada por partido; da el AUC y qué rasgos delatan al bot.
  - **Criterio del plan:** ≥ 80% dentro del techo, ninguna colectiva a más de 2 techos, AUC ≤ 0,70.
  - **e1, 200 juegos de 90 s** (`reports/hx11/realismo_e1.md`): **18% dentro del techo, C2ST AUC 0,988. No pasa.**
    - Dentro: goles, remates, ancho y cortes por falta u offside.
    - Lo que más delata al bot, según el C2ST: cambios de tecla (4,9/s contra 3,5), jugadores quietos, tiempo de pelota parada (juego abierto 62% contra 73%), velocidad de los jugadores (0,87).
    - Colectivas a más de 2 techos: pases (7,6 contra 11,9 por min), pase completo (31% contra 47%) y cambios de posesión.
    - También fuera: largo del equipo (1060 contra 870 px), espera de los saques (311 contra 199 ticks), laterales (26 contra 19,5 cada 10 min, 23% repetidos), contacto (0,5) y el más cercano (67 contra 55 px).
    - Siguiente, en curso en la PC del hermano: la misma población con la regla de juego abierto y con la dirección más persistente (`--renovar 0.2`).

### Lo que sigue, en orden

0. **(2026-10-10) Refuerzo A3:** ver la sección de arriba. Lo de abajo es la lista anterior.

1. **Puerta P3 con la patada mixta** (`reports/hx11/l0_puerta_ancho_patada033_mixta.md`). Si pasa, es la configuración por defecto del bot de sala.
2. **Evaluar `l0_ancho_bloque`** (acciones en bloque, con y sin ensamble) en el mismo banco, sobre desarrollo.
3. **Piloto C0 con 21 humanos y el bot**, con `--renovar-patada 0.33`. Grabar y analizar con `tools/hx11_rec_bots.py`.
4. **Córner** (pedido del usuario: "lo mismo que el lateral, cómo funcionan en la sala"). Lo medido hasta ahora:
   - Colocación, barrera de 750 px, disco del punto y jugadores sin masa: coinciden con la sala (99,6–100%). La pelota no se puede mover antes de la patada.
   - **"CORNER ITALIANO? 🍕"** (2 casos) es la patada que sale hacia afuera por la línea de fondo (pega en la red de afuera y no llega al área). La sala lo cobra como salida normal, saque de arco para el rival; el simulador ya lo hace.
   - **"¡SE ACABA EL TIEMPO PARA SACAR EL CORNER!"** es solo un aviso. En ~10.600 pelotas paradas del banco la sala nunca cedió por tiempo: córners pateados hasta 1366 ticks después. `PIECE_TIMEOUT` pasó de 600 (cesión supuesta) a 1800 (solo red de seguridad).
   - **Comba del córner:** el modelo del simulador tiene ~20% de error relativo en la gravedad (p50 0,0035 contra 0,0152 de la sala). Falta ajustarlo con los 18.000 ticks de córner (pendiente de S2).
   - **Falta medir a los bots en córners**, con un colector como el del lateral: tiempo hasta sacar, tasa de córner italiano, si llega al área, primer toque atacante o defensor, tiros y goles.
5. **Lateral después de la patada, refinado con el árbitro en sombra (v2 tenía 33 cobros de más):**
   - si un compañero la toca antes de entrar, el lateral cuenta como sacado (plazo de seguridad de 480 ticks);
   - si la toca un rival afuera, el lateral vuelve al que sacaba (`RI_LAT_THROWER`, `RI_LAT_TOUCHED`).
   Árbitro en sombra v3: `runs/hx11_l0/s4_arbitro_v3.json`.
6. **"¿Quién va?" en pelotas paradas y en la disputa de la pelota** (el fallo de coordinación de 22 imitadores), y los rivales que vuelven a entrar al área en el saque de arco: ajuste dirigido como el del lateral.
5. **¿Más datos ayudan?** La prueba de la mitad de los partidos. El usuario la postergó.
6. **Pendientes de análisis:** techos humano contra humano para la puerta de realismo de L1 (B7), y las mejoras de nivel 2 (P10, marca perdida, E2b).

### Archivos de la sesión del 2026-10-08 (ya commiteados en a806f15 y 4c86209)

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
