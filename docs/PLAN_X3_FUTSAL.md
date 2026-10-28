# Plan: bot competitivo de Futsal x3

## Objetivo

Crear un bot independiente para jugar Futsal x3 en el estadio `Futsal x3 by Bazinga from HaxMaps`, usando las grabaciones de `replays_real/stadiums/x3futsal` para aprender el comportamiento humano y, después, mejorar mediante entrenamiento competitivo en simulación. El bot se desplegará como modalidad separada: sus datos, política, modelos, configuración y pruebas no deben alterar el pipeline ni los modelos de X4/RS4-Z.

El objetivo final no es solo imitar las grabaciones: es jugar partidos completos y ser competitivo contra una referencia humana de esa modalidad, con las reglas y física representadas y la cadencia de control del despliegue host-local.

## Inventario inicial

- Hay **146 archivos `.hbr2`** en `replays_real/stadiums/x3futsal`.
- El estadio proporcionado está en `stadiums/futsal3bybazinga.hbs`; declara 76 segmentos, 130 vértices, 8 discos fijos y 2 arcos.
- Sus parámetros exportados incluyen radio de pelota 6,25, masa inversa 1,5, aceleración de jugador 0,11 y fuerza de patada 5.
- Las grabaciones pueden cambiar de estadio durante una partida. Solo se usarán los ticks cuyo estadio activo corresponda al mapa objetivo; los tramos de otros estadios no entrarán al dataset ni a la evaluación.
- Estos datos son un inventario inicial, no una certificación: todavía hay que medir duplicados, duración, integridad, número de jugadores y distribución de estadios por grabación.

## F0. Auditoría realizada (2026-10-27)

Herramienta: `python -m tools.x3futsal_audit --workers 2`. Reutiliza el conversor existente sin modificar su comportamiento y guarda la caché, el índice, el informe y el split en `data/x3futsal_jsonl/` (datos locales ignorados por Git).

- De 146 archivos, **144 son únicos** y **2 son duplicados exactos por SHA-256**. Los 144 únicos terminaron con el callback `fin del replay`; tres no contienen ningún tramo del mapa objetivo.
- **141 replays contienen el estadio objetivo**. El filtro compara una identidad semántica normalizada del estadio que resuelve traits y física; no depende del nombre ni de que el JSON de replay conserve la misma representación del `.hbs`.
- Se conservaron **6.592.342 ticks** (1.831,21 min, unas 30,5 h) del mapa y se excluyeron **129.876 ticks** (36,08 min) de otros estadios. Los límites de los spans son `[inicio, fin)` y los cambios intermedios quedan fuera.
- **6.350.843 ticks (96,34%) son 3v3**. El resto tiene planteles variables; no se asumirá que todos los frames sean 3v3 hasta decidir su tratamiento en F2.
- En los spans objetivo hay 1.791 goles, 83.900 patadas y 367 eventos `game_start`. Como una grabación puede contener varias partidas, la unidad de split es el archivo `.hbr2` completo para que sus ticks y partidos no se separen.
- Split inicial reproducible, semilla `20261027`: **112 train / 14 dev / 15 test** (141 replays), distribuido por archivo. F2 lo reemplaza por un split agrupado por sala para evitar fuga entre replays relacionados.
- Informe detallado por replay, tramos, IDs, estados, planteles, teclas y eventos: `data/x3futsal_jsonl/audit.json`. Índice de conversión: `index.json`. Asignaciones: `splits.json`.

## Principios de implementación

1. Identificar los tramos por la identidad del estadio exportado, no solo por el nombre ni por el número de segmentos. Los cambios intermedios deben excluirse con límites de frame precisos.
2. Separar entrenamiento, desarrollo y prueba por grabación/sesión; nunca repartir ticks de un mismo partido entre splits.
3. Medir la distribución real de planteles, teclas y cadencia de control antes de fijar contratos de observación o supuestos de 3 contra 3.
4. Reutilizar el parser de replay, el motor físico, utilidades y runtime cuando su semántica aplique, pero no reutilizar a ciegas el contrato RS4-Z/X4 de ocho jugadores ni sus reglas de sala.
5. Cada etapa tiene pruebas y criterios de avance. No se inicia RL ni se despliega un modelo hasta que simulación, datos y baseline de imitación estén validados.

## Etapas

### F0. Auditar grabaciones y fijar el split

- Convertir las 146 grabaciones con `bridge/replay_to_jsonl.js` y registrar SHA-256, duración, estado de conversión y cambios de estadio.
- Crear un índice específico de X3, reutilizando el patrón de `tools/rs4_jsonl_cache.py` sin cambiar el comportamiento de las colecciones existentes.
- Comparar cada estadio de cada tramo con `stadiums/futsal3bybazinga.hbs` mediante una identidad canónica estable del contenido. Verificar también que la conversión conserva los frames de `stadium_change`.
- Generar spans `[inicio, fin)` del mapa objetivo y contar ticks/partidos incluidos y excluidos. Informar separadamente replays duplicados, rotos, incompletos y grabaciones sin tramos válidos.
- Medir por span la cantidad de jugadores por equipo, IDs, estados de juego/saque, goles, eventos y teclas. Definir qué hacer con planteles incompletos o excedentes según su frecuencia observada; no rellenar ni descartar por suposición.
- Separar partidas completas por sesión en entrenamiento/desarrollo/prueba, con proporciones y semilla reproducibles.

**Gate:** cumplido en la auditoría inicial; el filtro compara el estado normalizado del estadio y los splits mantienen completo cada replay.

### F1. Validar física y reglas de Futsal x3

- Confirmar qué elementos son reglas estándar de HaxBall y cuáles, si los hay, provienen de scripts/eventos de la sala.
- Cargar el estadio real en `sim/` y verificar geometría, colisiones, arcos, discos y parámetros de jugador/pelota contra los `.hbr2`.
- Extender o crear un contrato de entorno X3 que modele el número de jugadores real, los saques, goles, reloj y reinicios observados. Mantener el contrato aislado de RS4-Z.
- Comparar trayectorias simuladas con los tramos de replay, excluyendo transiciones en que el script cambie propiedades de discos, y cuantificar errores antes de entrenar.
- Añadir pruebas de regresión para la identidad del mapa, spans y mecánicas que efectivamente aparezcan en las grabaciones.

**Cierre F1 (2026-10-27):**

- Se verificó el ciclo de partido observado: `state == 1` es juego activo; un gol lleva a `state == 2`; `positions_reset` precede al siguiente `state == 0`; `time_up` cierra en `state == 3`. Solo los tramos activos y con marcador/plantel estables se usan para la comparación de un tick.
- La muestra determinista de 12 replays comparó **418.888 transiciones 3v3**. El máximo p90 por replay fue **0,00010 px para la pelota** y **0,0001414 px para jugadores**; el máximo p99 fue **0,0001415 px** y **0,0001542 px**, respectivamente. Hubo 7 transiciones con error de pelota mayor a 1 px (todas en un replay; máximo **1,9038 px**) y ninguna con error de jugador mayor a 0,5 px. Los casos grandes son ráfagas cortas alrededor de contactos/kicks; algunos no tienen evento `kick` cercano, así que su causa exacta queda como limitación conocida, no como física sistemáticamente distinta.
- La muestra adicional de 4 replays con más `disc_props` aportó **164.740 transiciones** y mantuvo los mismos p90. En los 1.300 eventos `disc_props` auditados, todos afectan discos de jugador: hay cambios de radio (15/20/25/50 px) y de `invMass` (0,5/0,7/0,75); no se observaron cambios de `bCoef` ni `damping`. El análisis del stream en su orden real detectó **58.704 ticks activos** con física no estándar en 4 replays. La herramienta ahora excluye esos ticks y las transiciones adyacentes; una de esas grabaciones tiene `invMass` alterado durante todo el tramo activo, por lo que no aporta transiciones estándar.
- También se excluyen ticks con jugadores exactamente superpuestos y los dos ticks vecinos: esa configuración degenerada aparece en estados de transición y el motor no puede reproducir de forma determinista la separación del servidor. En la muestra de 12 se apartaron **151 transiciones** por este criterio.
- Cobertura: **96,34% de los ticks objetivo son 3v3**. Para el primer baseline de X3 se fijó el contrato de plantel en tres jugadores por equipo; los demás planteles se conservan en la auditoría, pero no se mezclan con esta validación ni se entrenarán hasta tener soporte explícito.

**Gate F1: completado para la física estándar del estadio.** Las excepciones detectadas están cuantificadas y filtradas de la comparación; las ráfagas de error de pelota permanecen documentadas como riesgo para la etapa competitiva. F2 reutiliza esos filtros y reemplaza el split por replay con uno agrupado por sala. Los informes locales están en `data/x3futsal_jsonl/physics.json`, `physics_eventful.json` y `physics_property_overrides.json`.

### F2. Dataset y baseline de imitación humana

- Separar por nombre de sala normalizado (NFKC, espacios recortados y `casefold`); las grabaciones sin sala forman grupos individuales. Optimizar el reparto determinísticamente para balancear ticks 3v3 sin dividir grupos.
- Mantener una muestra por transición válida `t → t+1`; la etiqueta usa el input registrado en `t`, convención verificada con `Replayer.step`. Elegir solo spans del estadio objetivo, `state 0` (saque/reinicio) o `state 1` (juego activo), 3v3 estable, marcador estable y física estándar; excluir eventos de cambio/reinicio y vecinos de posiciones degeneradas.
- Guardar shards compactos versionados por replay con discos, acciones, referencia de frame, marcador/reloj y metadatos en `data/x3futsal_bc_kickoff/`.
- Contrato inicial: observación del jugador, pelota y contexto de marcador/reloj, más conjuntos separados de compañeros/rivales y la acción anterior del jugador con máscara de continuidad; espejar el eje X para que cada equipo ataque hacia `+x`. Acción de 18 clases: 9 direcciones y botón de patada. El runtime deberá conservar la última acción propia.
- Entrenar BC con modelo de conjuntos compartido y únicamente en train; escoger checkpoints con NLL de dev. Evaluar precisión conjunta/dirección/patada, cambios de acción y repetir-input como baseline; reservar test.
- El bot corre en la sala hosteada por el usuario: no añadir ping de HaxBall a la evaluación; medir aparte solo la cadencia/costo de inferencia del runtime.

**Avance F2 (2026-10-27):**

- Split agrupado por sala generado con semilla `20261027`: **53 grupos train, 7 dev y 7 test** (141 grabaciones elegibles). Los grupos permanecen separados y los ticks 3v3 se distribuyen **80,0001% / 10% / 10%**. La asignación está en `data/x3futsal_jsonl/splits.json`.
- Contrato y herramientas de extracción implementados en `learn/x3futsal_data.py`, `learn/x3futsal_policy.py` y `learn/x3futsal_bc.py`; tests cubren acción reflejada, alineación temporal y simetría de observación.
- El nodo Tailscale responde y tiene PyTorch CPU; los entrenamientos futuros deberán pasar por `tools/pod_windows/lanzar.py`, con afinidad limitada a cuatro hilos lógicos (aprox. dos núcleos físicos), cuota dura de **30% CPU** y `--threads 2`. No lanzar trabajo fuera de ese límite.
- Dataset completo y validado con `--refresh-manifest`: 141/141 grabaciones procesadas, 137 shards con transiciones y **5.340.741 transiciones** (4.265.838 train, 537.351 dev, 537.552 test). Cuatro grabaciones no aportaron muestras tras los filtros estrictos. El split test permanece fuera de la selección y evaluación del modelo.
- Tras detectar que los saques/reinicios (`state 0`) estaban fuera del entrenamiento, el extractor v2 generó `data/x3futsal_bc_kickoff/` sin tocar la versión histórica: **138 shards**, 6.009.660 transiciones (4.788.902 train, 611.412 dev, 609.346 test). El aumento de 668.919 muestras corresponde a transiciones válidas en state 0; el test se mantiene separado y no se usa en entrenamiento/selección.
- El smoke test remoto de 20 pasos confirmó carga del dataset, entrenamiento y checkpoints bajo el launcher; en 1.024 muestras dev obtuvo NLL 2,286 y accuracy 19,2%, frente a 18,4% de mayoría y 90,4% de repetir-input. Es solo una comprobación del pipeline, demasiado corta para juzgar la política.
- El primer baseline, sin historial de acciones, completó 20.000 pasos en el pod. Su mejor checkpoint de dev (paso 19.000; evaluación fija de 20.000 muestras) obtuvo NLL 1,372, accuracy 50,3%, kick recall 21,0% y accuracy 19,6% en acciones cambiadas; mayoría obtuvo 18,9% y repetir-input **89,2%**. No supera la baseline temporal y no se promueve como política.
- Para modelar la persistencia, la política v2 incorpora la acción previa del jugador y una máscara para los límites de segmentos. El smoke test completó 20 pasos; el entrenamiento no ponderado se detuvo en el paso 5.000 al colapsar a repetir la acción previa (accuracy 89,2%, pero solo 0,1% en acciones cambiadas), sin mejorar la baseline temporal.
- Se probó un objetivo que pondera ×8 las muestras donde cambia la acción, manteniendo NLL/accuracy de dev sin ponderar para selección. La corrida de 20.000 pasos completó dentro del límite de CPU; el mejor checkpoint (paso 14.000) obtuvo NLL 0,700, accuracy 83,8%, kick recall 35,3% y accuracy 11,6% en acciones cambiadas. En una comparación justa sobre el mismo subconjunto con historial, repetir-input obtuvo 89,2%; el mejor margen de cambio igualó esa cifra solo cuando volvió a repetir todas las acciones (0% de aciertos en cambios). Checkpoints y métricas se conservaron localmente en `runs/x3futsal_bc/`; ningún modelo se libera y el gate F2 sigue abierto. El split test permanece reservado.

**Gate:** supera las baselines de predicción en desarrollo, no muestra errores de paridad y produce episodios cerrados jugables sin fallos de observación/acción. El NLL/accuracy de dev no sustituye la evaluación competitiva de F3.

### F3. Evaluación competitiva en lazo cerrado

- Crear una evaluación reproducible con lados intercambiados y semillas; como el bot corre en la sala hosteada por el usuario, no sumar ping de HaxBall.
- Comparar BC contra una política scripted sencilla, versiones anteriores y un pool fijo que crezca con los checkpoints.
- Registrar victorias/empates/derrotas, goles/diferencia de goles, pérdidas y proxies de pases, tiros y movimiento/formación relevantes al futsal.
- Mantener el split de prueba reservado mientras se diagnostica/selecciona política; usarlo una sola vez cuando el protocolo quede fijado.
- Calibrar el gate cuantitativo con la fuerza y variabilidad medidas del BC y del pool; no escoger umbrales después de ver un resultado favorable.

**Implementación y piloto (2026-10-28):**

- `learn/x3futsal_eval.py` ejecuta partidos 3v3 cerrados con el motor validado y acciones cada 3 frames. Compara el BC v2, el BC inicial, una política scripted y repetir-input en pares, con semillas fijas y lados intercambiados.
- El piloto inicial probó retardos artificiales de 0/1/2 decisiones, pero no representan el despliegue: el bot corre en la sala hosteada por el usuario y no recibe ping de HaxBall. La versión actual del evaluador elimina esos retardos y reporta el control host-local; conserva partidos, agregados por política/lado, goles, patadas, tiros aproximados, pases inferidos por toque posterior, pérdidas, toques y métricas de movimiento.
- Un timeout de 300 ticks evita saques atascados; sus ocurrencias quedan registradas como `safetyKickoffs` y no se confunden con un saque exitoso.
- Smoke y piloto se ejecutaron en el nodo remoto con `tools/pod_windows/lanzar.py`, tope de **30% CPU**, cuatro hilos lógicos permitidos y `--threads 2`. El smoke cargó ambos checkpoints y produjo reportes completos. El piloto corto fue una comprobación funcional, no una estimación estadística de fuerza.
- Piloto fijo `seed=20261028`: **72 partidos** de 30 segundos, dos por lado y latencia, en seis emparejamientos y tres escenarios. El BC v2 empató **4/4 sin goles** contra el BC inicial y repetir-input en todos los retardos. Contra scripted perdió **3 de 4** (GD −1,25) a 0 ms y **4 de 4** (GD −1,00 / −1,50) a 50/100 ms. La scripted ganó los **8/8** partidos contra BC en los escenarios con retardo; muestra pequeña, no comparable con humanos.
- Se registraron saques de seguridad con frecuencia (4 eventos en los 4 partidos frente al BC inicial/repeat; 6, 6 y 8 eventos en los 4 partidos frente a scripted a 0/50/100 ms). Es una señal de atasco que requiere investigación; no se considera un comportamiento competitivo aceptable. Todos los empates frente a BC/repeat fueron 0-0.
- Reporte reproducible en `reports/x3futsal_eval_pilot_20261028.json`; smoke en `reports/x3futsal_eval_smoke.json`. Pruebas dirigidas locales: **9 pasaron y 2 se omitieron** por dependencias no disponibles en ese entorno; el piloto sí cargó y ejecutó ambos checkpoints en el nodo con PyTorch CPU.
- Dos smokes con la misma semilla y configuración produjeron JSON idénticos; las agregaciones del piloto reportan cada política a ambos lados y también separan rojo/azul.
- El split test sigue reservado: este piloto no leyó muestras de test ni se usó para seleccionar/promover checkpoints.
- La revisión del replay mostró que `state 0` contiene entradas y posiciones válidas de los seis jugadores antes de comenzar juego activo. El primer extractor BC omitía estas muestras; el contrato v2 amplía el filtro a estados 0/1 y conserva el split por sala. El dataset está reconstruido y el candidato v2 se entrena aparte, sin reemplazar los checkpoints históricos.
- El reentrenamiento kickoff-aware v2 completó 20.000 pasos bajo el límite remoto de 30% de CPU. En dev, su mejor NLL fue 0,700 (paso 4.000), con accuracy 84,5%, kick recall 20,5% y 10,3% en acciones cambiadas; el checkpoint final obtuvo NLL 0,726, accuracy 82,6%, kick recall 36,4% y 12,6% en acciones cambiadas. El checkpoint `best` y el final se conservaron por separado en `runs/x3futsal_bc/kickoff_v2/`.
- El piloto host-local reproducible `seed=20261029` ejecutó **240 partidos de 60 segundos**, 8 por lado en cada emparejamiento, sin latencia HaxBall modelada. `kickoff_v2/best` empató 16/16 contra cada política BC/repeat, pero perdió 16/16 contra scripted con diferencia de goles −3,00 por partido y 3,50 saques de seguridad por partido. `kickoff_v2/last` fue algo menos pasivo: contra scripted obtuvo 4 empates y 12 derrotas, diferencia −1,25 y 1,50 saques de seguridad por partido; aun así no marcó ningún gol. Todos los modelos BC produjeron cero patadas en los partidos contra scripted en este protocolo, mientras scripted sí completó ataques y marcó.
- El diagnóstico de las trazas confirma que el problema ya no es únicamente el timeout: el `best` toma solo 78 decisiones de movimiento por equipo en un minuto contra scripted y no realiza ninguna patada; el `last` se mueve más, pero sigue sin patear. Por tanto, añadir más pasos al mismo objetivo BC no es suficiente. El siguiente experimento debe medir explícitamente las acciones de saque y la paridad de `action_to_sim`/alcance de patada antes de iniciar RL o liberar un modelo.
- La traza de acciones mostró que ambos checkpoints emitían `stay` para los seis jugadores desde el primer tick de saque. Se añadió una asistencia determinista de saque en `ModelPolicy`: durante `sim.kickoff` usa la misma aproximación validada de `scripted`, y vuelve al modelo BC apenas la pelota entra en juego. El smoke remoto confirmó cero `safetyKickoffs`.
- El piloto completo con asistencia (`seed=20261030`, 240 partidos de 60 segundos, 8 por lado y emparejamiento) eliminó los saques atascados, pero no alcanzó el gate competitivo. `kickoff_v2/last` perdió 16/16 contra scripted (GD −2,31, 0,81 patadas propias por partido); `kickoff_v2/best` empató 8 y perdió 8 (GD −1,50, cero patadas propias). Como referencia, `changed_v1` empató 8 y perdió 8 (GD −0,50) y el baseline inicial perdió 16/16 (GD −1,50). La asistencia corrige la entrada al episodio, no la calidad del juego activo.
- La auditoría específica de inactividad encontró carreras de `stay` + velocidad menor a 0,08 de hasta 17.723 frames en un jugador y 532 carreras de al menos 600 frames en los shards. En varios casos los jugadores estaban fuera del campo (`x=±605` con límite aproximado `±550`), confirmando AFK/slots abandonados. El dataset v3 activo excluye transiciones contaminadas por posiciones fuera del campo o por 600 frames consecutivos de inactividad; se redujo `stay` de 10,52% a 8,80% en train, pero la tasa de patadas quedó prácticamente igual (3,68% a 3,69%).
- El candidato v3 con ponderación explícita de patadas completó 20.000 pasos bajo el límite remoto. Su mejor dev fue NLL 0,672, accuracy 85,1%, kick recall 81,9% y accuracy de acciones cambiadas 9,2%. En el piloto host-local (`seed=20261031`, 336 partidos de 60 segundos, 8 por lado) no superó el comportamiento cerrado: `active_kickweight_v1/best` perdió 16/16 contra scripted (GD −2,00, 0,50 patadas por partido) y `last` perdió 16/16 (GD −2,50, 1,00 patadas por partido). La mejora offline de patadas no se tradujo en posesión ni goles.

**Gate: NO CUMPLIDO.** El runner, el entrenamiento kickoff-aware y el protocolo host-local funcionan, pero ni el BC kickoff-aware ni el candidato filtrado de AFK superan a scripted. El filtrado de AFK mejora la distribución offline, pero no resuelve el error de lazo cerrado. No iniciar F4 ni liberar el modelo. Mantener el split test reservado hasta obtener una política jugable.

### F4. Aprendizaje competitivo desde BC

- Iniciar RL desde un checkpoint BC, conservando el BC como referencia y mecanismo de protección contra olvidar el comportamiento útil.
- Empezar con recompensas directas de gol y resultado. Añadir shaping solo con hipótesis, ablación y criterio medible; no premiar pases por defecto.
- Entrenar partidos con variedad de lados, planteles y estados iniciales, bajo el control host-local. Incluir arranques desde estados de replay si mejora la cobertura de situaciones, sin filtrar información de prueba.
- Mantener una liga/pool pequeño de BC y snapshots para evitar sobreajuste al self-play más reciente.
- Guardar evaluaciones periódicas y detener/revertir checkpoints que empeoren la fuerza competitiva o presenten conductas degeneradas (pasividad, no ejecutar saques, perseguir la pelota sin estructura).

**Gate:** mejora estadísticamente respaldada contra BC y el pool, mantiene cobertura de situaciones y supera controles de seguridad del comportamiento antes de aumentar el presupuesto.

### F5. Exportación y prueba en sala

- Exportar modelo y metadatos con versión explícita del contrato X3, mapa, normalización, temperatura y cadencia de control.
- Añadir una entrada/ruta de despliegue dedicada a X3; no cambiar el modelo predeterminado ni las opciones X4.
- Probar primero en sala privada y con permiso, grabando observaciones, acciones, latencia y resultado para comparar runtime con evaluación.
- Incorporar las nuevas grabaciones solo después de procesarlas y mantener un conjunto de prueba temporal separado para detectar regresiones.

**Gate:** paridad de observación y acción con el modelo evaluado, estabilidad de runtime y resultado competitivo confirmado en partidas controladas.

## Entregables previstos

- Herramientas e índice de conversión/filtrado de replays X3.
- Caché y splits por sesión, con estadísticas de cobertura y calidad.
- Entorno de simulación y contrato de observación/acción específicos o una extensión explícita y probada de componentes compartidos.
- BC reproducible, pipeline de evaluación competitivo y entrenamiento RL desde BC.
- Exportación ONNX y despliegue aislado para Futsal x3.

## Siguiente trabajo

F3 queda detenido en el gate: el BC v2 pierde contra scripted y los saques de seguridad son frecuentes. La extracción BC actual conserva únicamente `state == 1` (juego activo), mientras que el runner también exige decisiones durante el saque; por eso el saque es una situación fuera de la distribución de entrenamiento y una explicación plausible de los timeouts, aún por verificar. Contrastar esa hipótesis con las acciones y transiciones de saque de los replays, definir una política/dataset de reinicio con pruebas y repetir el piloto. No iniciar F4 ni consultar el split test hasta cerrar el protocolo y la selección.
