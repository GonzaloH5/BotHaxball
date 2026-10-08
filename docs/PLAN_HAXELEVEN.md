# Plan HaxEleven X11: un bot que ocupa un puesto en un partido humano

Borrador del 2026-10-07, pendiente de aprobación. Ese mismo día se revisó con las respuestas del usuario: reglas de falta y offside, penales, laterales, cómo se organizan los partidos, sala y script (§10). Es solo plan y diseño: no se corrió nada, no se usó el pod y no se procesaron las grabaciones. Sigue la regla de `docs/PLAN.md`: cada decisión cita trabajos publicados, y lo que es deducción nuestra va marcado **[inferencia]**.

## 0. Objetivo

**Qué queremos (usuario, 2026-10-07):**
- Un bot que entra a un partido humano de HaxEleven 11v11.
- Al agregarlo en la sala se le indica un puesto, y desde ahí se maneja solo.
- Ocupa ese puesto de forma correcta, sea cual sea la formación del equipo. Las más usadas son 4-3-3, 4-4-2, 3-4-3 y 3-5-2, pero el bot no debe depender de ninguna.
- Su nivel es competente: no es una carga para el equipo.

**Las formaciones no son rígidas.** Juegan personas no profesionales, y HaxBall permite libertades que el fútbol real no tiene:
- cuerpear para que llegue un apoyo;
- sacar rápido un lateral aprovechando la barrera que genera;
- otras por el estilo.

El bot tiene que entender la función del puesto, no una casilla fija.

**Cómo se juegan los partidos (usuario):**
- Son casuales, no de liga.
- Los 22 jugadores entran a la sala en grupos armados por Discord.
- Cada grupo tiene un **técnico**, que arma la formación por canal de voz y administra todo.

Consecuencias:
- **El puesto del bot lo elige el usuario al agregarlo, y nada más.** No lo define el técnico, ni la voz, ni lo que hagan los compañeros.
- Las indicaciones que el técnico da por voz no le llegan al bot: juega su puesto con lo que ve en la cancha.
- Las siglas de los archivos son grupos, no equipos de liga.

### Qué cambia respecto de X4

| | X4 (Sanguchito) | X11 (HaxEleven) |
|---|---|---|
| Con quién juega el bot | con otros bots: el equipo entero es del modelo | con 21 humanos: el bot es uno, o unos pocos |
| Problema de fondo | coordinación desde cero (self-play) | **ad hoc teamwork:** colaborar con compañeros desconocidos sin acuerdo previo (Stone 2010, AAAI; Mirsky 2022, 2202.10450) |
| Tiempo con la pelota | ~1/8 | ~1/22: casi todo el valor está en ubicarse sin la pelota **[inferencia]** |
| Riesgo principal | el pase no emerge | que el bot aprenda a coordinar consigo mismo o con la BC, y no con humanos (Carroll 2019, 1910.05789) |
| Mecánicas | patada, X | además: powershot con carga porcentual y curva, faltas por contacto sostenido, offside, barra de fuerza en la pelota parada; no hay slide |
| Costo del simulador | 9 jugadores y pelota | 23 discos móviles: medido en S1, 2,7 veces más lento por partido que 4v4 y el mismo costo por jugador |

### Consecuencias de diseño

1. **La vara no es "ganarle a la BC en self-play".** La vara es el **valor de reemplazo**: el mismo equipo con el bot en el puesto k, contra el mismo equipo con un humano (o su modelo) en ese puesto.
   - Lección de X4: ganarle a la BC el 85% no trajo juego colectivo, porque los rivales eran de la misma familia (`docs/AUDITORIA_POD_2026-10-07.md`).
2. **Los compañeros de entrenamiento son modelos de humanos, no copias del bot.**
   - Carroll 2019: entrenar con un modelo humano por imitación rinde más con humanos reales que el self-play.
   - Strouse 2021 (FCP, 2110.08176) y Yu 2023 (HSP, 2302.01605): lo decisivo es la **diversidad** de compañeros, incluidos compañeros "sesgados".
3. **La imitación es la base, no solo el ancla.** En X11, buena parte de la coordinación ya la aportan los 21 humanos. El bot necesita estar donde estaría un humano en ese puesto y decidir bien con la pelota.
   - Es justo lo que modela la imitación multiagente con roles (Le 2017, 1703.03121, y el "ghosting" de Le et al., MIT Sloan 2017).
   - El RL solo refina, con ancla KL a la imitación (HR-PPO 2403.19648; piKL 2112.07544).

## 1. Punto de partida

### El banco de grabaciones

Fuente: `replays_real/stadiums/haxelevenx11`. Solo se contaron los nombres de archivo; las grabaciones no se abrieron.

- **753 archivos** `.hbr2`, 157 MB.
- **43 son copias** "(1)". Las 6 que se compararon son idénticas byte a byte, así que quedan unas **710 únicas**.
- **80 llevan las siglas de los dos grupos que jugaron** (`AV_KJ_HaxEleven-…`, unas 19 siglas: AV, CMC, ZFC, BOV, SX, BOL…). Son partidos casuales entre grupos de Discord, cada uno con su técnico.
- **630 son genéricas** (`HaxEleven-<fecha>`), más algunas `CustomHaxEleven-…`: posible variante de mapa o de script.
- **Las fechas de los nombres mezclan formatos** ("9-14-2026" y "14-9-2026"). La fecha y la versión del script se toman de la grabación, no del nombre.
- Faltan medir la duración, el plantel por tramo (11v11, cambios, AFK) y los estadios por tramo (S0).

### El script de la sala: no sirve como fuente de reglas

- El creador no puede pasar las reglas exactas.
- El script que tiene el usuario (`HaxEleven-FaCunDoRF.txt`) es una sola línea de 1,7 MB, protegida:
  - `eval` con chequeo de integridad ("the code has been tampered");
  - todos los textos codificados;
  - flujo de control aplanado;
  - constantes ofuscadas.
- No hay nada legible sobre las reglas: no aparecen `setDiscProperties`, `onPlayerBallKick` ni ningún nombre de propiedad, y el único nombre visible es `HBInit`.
- Decodificarlo exigiría ejecutar su código, y el proyecto no ejecuta scripts de sala.
- **Las reglas salen de las grabaciones (S2) y de pruebas con bots cliente en la sala (S5).** Lo que dijo el usuario sobre faltas, offside, penales y laterales entra como hipótesis a verificar, no como verdad.

### Qué se reutiliza

Es infraestructura verificada contra el juego real (`README.md`):

- **Motor de física** (`sim/physics.py`).
  - Error p90 de 0,009–0,012 px a 60 ticks en mapas RS.
  - Admite n_red × n_blue jugadores.
  - Ya modela un powershot de otro script (`ball-physics.js`): carga binaria a los 96 ticks, invMass de la pelota 2,3, y curva como gravedad perpendicular proporcional a la velocidad lateral del jugador. **Es una plantilla, no la regla de HaxEleven.**
- **Lectura de grabaciones con el motor original** (`bridge/replay_to_jsonl.js`, `replay_meta.js`): los estados grabados ya incluyen todo lo que el script hizo con `setDiscProperties`.
- **Método de conformidad** de RS4-Z-2 y Sanguchito:
  - física a 1 y 60 ticks;
  - saques con los jugadores forzados (`tools/rs4z_conformance.py`, `rs4z_restart_conformance.py`);
  - "contrato" del script con la evidencia de cada regla (`env/rs4z/contract.py`).
- **Aprendizaje X4** (`learn/`):
  - imitación con el retardo de sala;
  - MAPPO con ancla KL y shaping de potencial con el EPV;
  - métricas de parecido humano (W1);
  - métricas contra el copycat (de Haan 2019, 1905.11979).
- **Despliegue** (`deploy/bot.js`, `join_bots.js`): se agrega `--pos`.

### Qué no se reutiliza

- **El kernel `env/rs4z`:** está especializado en 4v4 y en el script de RS. El árbitro de HaxEleven es un kernel nuevo (`env/hx11`) con la misma estructura.
- **Las macros de pase:** el RL las abandonó (de 295 inicios por actualización a 11; `AUDITORIA` §"Abandono de la macro"). No van por defecto.

## 2. Simulador fiel (S0–S6)

**Principio: lo que la física no explica, lo hizo el script.**
- Se simula un tick con física pura de HaxBall, con las entradas grabadas, y se compara con el tick grabado siguiente.
- El residuo aísla cada intervención del script: velocidad, posición, invMass, gravedad, color, cMask/cGroup o radio de un disco.
- Cada familia de intervenciones se vuelve una regla del contrato **HX11-1**, con su evidencia: N eventos y % de acuerdo.
- Así se armaron RS4-Z-2 y Sanguchito: 101/101 saques, y 2.746 patadas que liberan clasificadas sin errores.

### S0. Inventario (CPU liviano)
- Deduplicar por SHA-256.
- Correr `replay_meta` en todas, para obtener:
  - sala y estadio por tramo;
  - planteles por tick (11v11, cambios, AFK);
  - duración, goles y pausas;
  - nombres y avatares: ¿declaran número o puesto?;
  - chat de los humanos: ¿declaran puestos (sirve para las etiquetas de D2) o hay mensajes del árbitro? Hay que verificar si la grabación guarda los anuncios del script.
- Agrupar en sesiones los partidos seguidos de los mismos jugadores.
- Splits por partido y por bloque de fechas, sin fuga entre tiempos del mismo partido.
- Reservar grupos enteros para medir la generalización: cada técnico tiene su estilo.
- Salida: `reports/hx11/inventario.json`.

**Hecho el 2026-10-07** ([reporte](../reports/hx11/inventario.md)):
- **Archivos:** 711 únicos. Cada uno es un partido entero de unos 12,7 min, no un tiempo.
- **Datos útiles:** 142 h de 11v11 exacto en mapas x11.
- **Mapa:** tres versiones en el tiempo:
  - [1875, 1064], de junio a agosto;
  - tamaño aleatorio, en agosto;
  - [1800, 1200], la actual.
- **Anuncios:** quedan grabados y anuncian el % del powershot, la potencia de la barra y el arquero (`!gk`). Las faltas y los offsides no se anuncian.
- **La sala anuncia:**
  - offside desde mitad de cancha;
  - falta a los **2 s**, que no coincide con los 6 s que describió el usuario;
  - modos que se pueden apagar: faltas, offside, powershot y gambeta.

### S1. Estadio y física base
- Exportar el estadio desde la grabación, como se hizo con Sanguchito, e importarlo con `tools.import_stadium` como `stadiums/haxeleven_x11.hbs`.
- Registrar:
  - medidas de la cancha, arcos y áreas (importan para faltas y offside);
  - física del jugador: radio, aceleración, kickStrength, kickback;
  - física de la pelota;
  - segmentos y discos con cMask, que son candidatos a barreras.
- Catalogar las variantes de estadio de los tramos: entrenamiento, penales, `Custom`.
- **Conformidad de juego abierto** en ventanas sin residuo de script: p90 a 60 ticks ≤ 0,02 px, del orden de RS.
- **Rendimiento:** medir ticks/s del motor con 22 jugadores.
  - Si no alcanza, optimizar las colisiones, por ejemplo descartando pares lejanos con una grilla.
  - Siempre con la prueba de equivalencia exacta contra el motor actual.

**Hecho el 2026-10-07 con el estadio actual, [1800, 1200]** ([reporte](../reports/hx11/s1_fisica.md)):
- **Estadio:** `stadiums/haxeleven_x11.hbs`.
- **Física de juego abierto exacta en 8 grabaciones:**
  - 1 tick: 74.523 transiciones con error máximo de 0,001 px, que es el redondeo;
  - 60 ticks: p90 de 0,014 px la pelota y 0,003 px los jugadores.
- **Cómo:** el simulador aplica tick a tick las propiedades que deja el script (el 19% de las transiciones), con el mismo orden de discos que el juego.
- **Barra de fuerza:** es un disco que rebota entre otros dos (discos 28–30); se modela en S3.
- **Velocidad:** el motor con 22 jugadores cuesta lo mismo por jugador que 4v4 (9,1 M ticks de jugador/s en esta PC). No hace falta optimizarlo.

### S2. Arqueología del script

Nueva herramienta `tools/hx11_residuals.py`: por tick y por disco, registra todo cambio que la física no explica y lo clasifica. Para cada mecánica se mide lo siguiente.

**a) Powershot con carga porcentual** (según el usuario: 10% verde, … 50% amarillo/naranja, … 100% rojo; también se usa para pases largos y tiene curva).
- **Indicador:** qué cambia de color (pelota o jugador), y la tabla color → %.
- **"Pegado":** umbral de distancia entre el borde del jugador y la pelota. El otro script usaba ≤ 0,1 px y muestreaba cada 30 ticks.
- **Carga:** cuánto sube por paso y cada cuántos ticks.
- **Reinicio de la carga:** qué la reinicia (separarse, toque rival, patada, saque), y si se conserva después de separarse. En el otro script se conservaba.
- **Efecto:** invMass de la pelota o multiplicador de velocidad en función del %, y kickback.
- **Curva:**
  - gravedad al patear: dirección y magnitud, en función de la velocidad lateral, la tecla o el %;
  - duración y decaimiento.
- **Validación:** % predicho igual al grabado en ≥ 99% de los ticks con contacto; velocidad de salida con error p90 ≤ 1%; trayectoria con curva a +60 ticks con p90 ≤ 2 px.

**b) Faltas por contacto sostenido.**

Regla según el usuario (hipótesis a verificar):
- **Poseedor:** es el último que tocó la pelota.
- **Falta:** un rival que se queda pegado al poseedor durante **6 s**, sin tocar la pelota en ese tiempo.
- No importan la velocidad ni el ángulo: también cuenta chocarlo de atrás en carrera.
- No hay tarjetas.
- Hay penales.

Detección:
- El tiro libre se reconoce por la pelota recolocada fuera de un punto de saque, la barrera y la posible quietud de los jugadores.

Qué falta medir:
- **Contacto:** distancia entre centros ≤ r₁ + r₂ + ε. Hay que medir ε.
- **Temporizador:** si se reinicia al separarse aunque sea un tick, o si se acumula.
- **Umbral exacto:** 6 s son 360 ticks, salvo que el script muestree cada N ticks. El otro script muestreaba cada 30.
- **Posesión:** si se mantiene con la pelota lejos del poseedor, mientras nadie más la toque.
- **Varios rivales pegados a la vez:** si sus tiempos se suman.
- **Lugar del cobro:** en el punto de la falta o donde está la pelota.
- **Penal:** desde qué límite del área se cobra.

Validación:
- F1 ≥ 0,99 en las llamadas, con el momento a ±3 ticks.
- Se cuentan también los falsos positivos: contactos que la regla cobraría y la sala no cobró.

**c) Offside.**

Regla según el usuario (hipótesis a verificar): la del fútbol.
- Un jugador está en offside si está por delante del anteúltimo rival en el momento del pase.
- En los laterales no hay offside.

Detección:
- Tiro libre a favor del equipo que defiende, en la posición del adelantado.

Qué falta medir:
- **"Pase":** cualquier patada o toque de un compañero, o solo la patada.
- **Medición:** desde el centro del disco o desde el borde, y con qué tolerancia en px. En el fútbol, estar a la par es estar habilitado.
- **Campo propio:** si hay exención, como en el fútbol.
- **Delante de la pelota:** si se exige estar por delante de la pelota, como en el fútbol.
- **Córner y saque de arco:** si también eximen. En el fútbol eximen; el usuario solo mencionó el lateral.
- **Cuándo se cobra:** al patear o cuando el adelantado toca la pelota. Es decir, si hay offside "pasivo".

Validación:
- Acuerdo ≥ 99% en llamadas y en no-llamadas, sobre todos los pases con un receptor cerca de la línea.

**d) Córners, tiros libres y saques de arco con barra de fuerza** (según el usuario: la barra va de verde a rojo, ida y vuelta, y define la fuerza de salida).
- **Representación:** cómo se muestra la barra (disco, color, posición) y si el jugador la ve.
- **Barra:**
  - período y forma de la oscilación (¿triangular?);
  - fase al iniciar el saque: fija o aleatoria;
  - curva fase → velocidad de salida.
- **Ejecución:**
  - si la pelota queda fija y el script le asigna velocidad, como en los córners de Sanguchito;
  - la dirección;
  - barrera para el rival (radio, duración);
  - plazo y qué pasa al vencer;
  - quién puede ejecutar.
- **Validación:**
  - clasificación de tipo, punto y equipo al 100%, el estándar de Sanguchito;
  - velocidad de salida por fase con error p90 ≤ 2%.

**e) Laterales.**
- **Mecánica:** no usan la barra (usuario). Hay que confirmar si son una patada normal o si el script fija la velocidad.
- **Barrera:** cuándo aparece, cuánto dura y si desaparece al sacar. Esto define cuánto vale "sacar rápido aprovechando la barrera".
- **Offside:** no hay en los laterales (usuario).

**f) Penales.**
- Cuándo se cobran: el límite del área.
- Cómo se ejecutan:
  - posiciones del ejecutor, del arquero y del resto;
  - si usan la barra;
  - plazo;
  - qué puede hacer el arquero.

**g) Resto del árbitro.**
- Saque inicial, entretiempo y duración.
- Reloj: ¿se congela en los saques?
- Gol, cambios de jugadores y reglas del arquero.
- No hay tarjetas.
- Confirmar que no hay slide: no debe haber residuos de ese tipo.

**h) Lo inesperado.** Cualquier residuo en los jugadores (sprint, stamina, frenado tras una falta) se cataloga antes de seguir. Algunas salas de 11v11 tienen mecánicas de ese tipo **[hipótesis por verificar]**.

**Versiones del script:** el banco abarca de enero a octubre de 2026. Se compara cada regla por fecha; si cambió, hay un contrato por versión.

### S3. Contrato HX11-1 y kernel `env/hx11`
- Cada regla queda con sus parámetros, su fuente (N eventos, archivo) y su estado: verificada, aproximada o supuesta.
- El kernel es numba y parametrizado, para poder barrer variantes de las reglas no resueltas.
- Agrega estado público para el actor: % de carga, fase de la barra, temporizadores de falta, línea de offside y estado de la barrera.

### S4. Conformidad completa (`tools/hx11_conformance.py`)

| Prueba | Criterio propuesto |
|---|---|
| Física en juego abierto | p90 a 60 ticks ≤ 0,02 px |
| Powershot (carga, salida, curva) | ver S2a |
| Faltas | F1 ≥ 0,99; momento ±3 ticks |
| Offside | acuerdo ≥ 99% (llamadas y no-llamadas) |
| Pelota parada | tipo, punto y equipo al 100%; salida p90 ≤ 2% |
| Partido "re-arbitrado" | desde cada saque, con las entradas grabadas, el siguiente evento del árbitro (tipo, equipo, lugar) coincide en ≥ 99% |

Los umbrales son propuestas, a fijar por el usuario antes de medir.

### S5. Pruebas con bots cliente en la sala real, solo para lo que las grabaciones no resuelvan
- La sala donde va a jugar el bot solo admite clientes, y hay permiso. Alcanza: no hace falta alojar el script.
- Un admin o un técnico arranca el partido.
- Bots con guion (`join_bots.js`) hacen jugadas controladas y se graban:
  - **falta:** pegarse al poseedor 330, 350, 360, 370 y 390 ticks, y separarse un tick a mitad de camino, para ver si el temporizador se reinicia;
  - **offside:** receptor a −10, −2, 0, +2 y +10 px del anteúltimo defensor; en campo propio; desde un córner;
  - **powershot:** mantener el contacto exactamente N ticks y patear;
  - **barra:** patear saques en fases controladas.
- Cada prueba se pre-registra y se repite. La grabación de la sala es el dato.
- Claude no ejecuta el script de la sala (regla vigente).

### S6. Latencia
- Medir en la sala donde va a jugar el bot, como cliente, con `latency_probe.js` y `--trace`. La ubicación del host y los 22 jugadores pueden cambiarla.
- En RS fue de 9–12 ticks.
- Se entrena con la distribución medida (convención de retardo de `PLAN.md` E1).

**Salida de la etapa S:** se avanza cuando S4 pasa, con las reglas abiertas documentadas. No se entrena sobre un árbitro no verificado: es la lección de 2K23 (B1).

## 3. Datos y etiquetas (D0–D3)

**D0. Caché por tick**, como `tools/x4_ticks.py`:
- posiciones, velocidades y entradas de los 22 jugadores;
- estado del script de S3;
- marcador y reloj, solo para el crítico.

**D1. Eventos**, validados con una muestra que revisa el usuario, como los ejemplos de `x4_mechanics`:
- pase, recepción, remate, powershot (con %), pase largo, centro y despeje;
- conducción, intercepción y robo;
- falta cometida y recibida, offside y saques.

**D2. Puestos y formación.**
- **Coordenadas relativas al equipo:** se normalizan por la dirección de ataque y se miden el centroide y la superficie del equipo (Frencken 2011, *Eur. J. Sport Sci.*).
- **Asignación de rol por tick:** plantillas de rol con EM y Hungarian, por partición de mínima entropía (Bialkowski 2014, ICDM).
- **Cambios de rol y de formación:** SoccerCPD (Kim 2022, KDD, 2206.10926) separa los cambios tácticos de los intercambios momentáneos. Contra expertos, acierta el 86,5% de las posiciones: hay que esperar ruido de ese orden.
- **Clasificación de formación por situación:** 4-3-3, 4-4-2, 3-4-3, 3-5-2 u "otra" (Müller-Budack 2019, 1910.00412, "Does 4-4-2 exist?").
- **Forma con y sin pelota por separado:** un 3-5-2 defiende como 5-3-2 (Shaw y Glickman 2019, Barça Sports Analytics Summit).
- **Etiquetas de verdad para validar el etiquetador.** No hay alineaciones de liga, porque los partidos son casuales. Opciones, en orden:
  - si los técnicos publican la formación o la alineación en Discord (imagen o texto), se usa;
  - si los nombres o avatares declaran el puesto, se usan;
  - si no, el usuario etiqueta una muestra de unos 20 partidos: formación de cada grupo y puesto de cada jugador.

**D3. Valor.**
- **Valor de posesión de 11v11** (EPV: Fernández, Bornn y Cervone 2021, 2011.09426; VAEP: Decroos 2019, 1802.07127).
- **Control del espacio** y oportunidad sin pelota (pitch control y OBSO: Spearman 2018, MIT Sloan; generación de espacio: Fernández y Bornn 2018, MIT Sloan).
- **Usos:**
  - medir el valor agregado por jugador y por puesto;
  - shaping de potencial, de suma cero, como en X4 (Ng 1999).

## 4. Diseño del bot

### Puesto como condición, no como casilla
- **El puesto se representa con un ancla continua,** aprendida en D2: la posición media del rol en coordenadas del equipo, con y sin pelota.
- **Se le suma un token de línea:** arquero, defensa, medio o ataque.
- **Así, puestos parecidos de formaciones distintas quedan vecinos.** Por ejemplo, "LI de 4-4-2" y "CRI de 3-5-2".
- **La formación no es una entrada.** El puesto es fijo: lo elige el usuario.
  - Lo que cambia de una formación a otra es cómo se juega ese puesto.
  - Eso el bot lo ve en la ubicación de compañeros y rivales, que forman parte de su observación.
  - **[inferencia]** Por eso no debería depender de la formación; se prueba en la evaluación (§6).

**Vocabulario de `--pos`:**
- ARQ;
- DFC (central), LI y LD (laterales), CRI y CRD (carrileros);
- MCD (volante central), MC, MCO (enganche), MI y MD;
- EI y ED (extremos), DC (centrodelantero) y SD (segundo delantero).

### Política
- **Una sola red para todos los puestos,** condicionada por el ancla.
- **Entidades como conjuntos:** 10 compañeros y 11 rivales, con encoder compartido y atención o pooling.
  - Smit et al. 2023 (JAAMAS) llevan el 11v11 a 1 GPU y 1 CPU con atención lineal en la cantidad de agentes.
  - Base: Deep Sets, 1703.06114.
- **Observación del actor:**
  - propio, pelota y entidades;
  - % de carga y curva activa;
  - líneas de offside de los dos equipos;
  - estado del saque: tipo, equipo, fase de la barra, barrera y plazo;
  - temporizadores de falta: los míos sobre el poseedor y los del rival sobre mí;
  - geometría y acciones propias pendientes (latencia).
- **Sin marcador ni reloj en el actor** (preferencia del usuario); van solo al crítico.
- **Salida: las 18 teclas, como en X4.**
  - Cargar el powershot es pegarse y patear.
  - La barra es patear en el momento justo.
  - Las dos cosas se pueden hacer con teclas y vienen de los humanos.
  - Sin macros por defecto (lección de X4).
- **Cabezas auxiliares supervisadas:** dónde estaría un humano de mi puesto a +1 s y a +2 s (ghosting).
  - **[inferencia]** Mejoran la representación del juego sin pelota.
  - Dan un "punto objetivo" legible, útil para depurar y como respaldo.

### Referencia programada, independiente de los datos
- **Ubicación al estilo RoboCup:**
  - la formación por triangulación de Delaunay, que lleva la posición de la pelota a la posición de cada rol (Akiyama y Noda 2008, RoboCup 2007, LNCS), ajustada con los datos humanos de D2;
  - posicionamiento por situación (SBSP, Reis, Lau y Oliveira 2001);
  - roles flexibles (Stone y Veloso 1999, *Artificial Intelligence*).
- **Con la pelota:** reglas simples.
- **Usos:**
  - línea base de "puesto correcto";
  - compañero y rival del pool de evaluación;
  - plan de respaldo en sala.

### Entrada a la sala

El bot entra como cliente, y el usuario le elige el puesto:

```
node deploy/hx11/join_bots.js --join <link> --pos LI
```

- **El puesto es el que pasa el usuario con `--pos`, y no cambia solo.**
  - El bot no lo deduce de sus compañeros.
  - No recibe órdenes del técnico ni de otros jugadores.
- **Si el usuario quiere cambiarlo durante el partido,** lo hace él desde la consola del bot (por ejemplo, `pos MC`). No hay comandos de chat para otros jugadores.
- **Saques:** el bot cede los saques que un humano de su puesto no suele tomar (se aprende quién saca cada cosa por puesto, D1). Así no le "roba" el córner al ejecutor habitual.

## 5. Entrenamiento (L0–L2)

### L0. Imitación condicionada por el puesto, con latencia de sala
- **Datos:** cada humano de cada partido aporta una trayectoria, 22 por partido. Son muchas más trayectorias de jugador que en X4 (8 por partido), a medir en S0.
- **Pérdida:** NLL de las teclas, más las cabezas auxiliares de posición.
- **Métricas:** separadas en cambios de tecla (copycat) y por contexto: abierto, saque propio o rival, con o sin pelota.
- **Variante pre-registrada, no por defecto:** pesar las demostraciones por valor agregado (D3), o filtrar a los mejores jugadores del banco.
  - Motivo: en una sala pública el humano promedio no es la vara de "competente".

### L1. Población de compañeros y rivales humanos sintéticos
- **La base es la BC jugando los otros 21 puestos** (Carroll 2019).
- **Diversidad** (FCP 2110.08176; HSP 2302.01605):
  - checkpoints de la BC en distintas etapas;
  - temperaturas distintas;
  - estados iniciales de equipos y formaciones reales;
  - compañeros sesgados (más individualistas, más defensivos, más directos), con un ajuste corto con recompensas sesgadas, como en HSP;
  - el equipo programado de §4.

### L2. RL "de puesto"
Se hace solo si L0 no alcanza el criterio de competencia (§6 B).
- **Quién aprende:** 1 a 3 puestos por equipo, con roles distintos; el resto son de L1.
- **Recompensa:** gol ±1 de equipo, más el shaping de potencial con el EPV de 11v11, de suma cero.
- **Ancla KL a la imitación condicionada por el puesto** (HR-PPO; piKL). λ se elige con un barrido corto pre-registrado.
  - El ancla es lo que mantiene el puesto.
  - Si en la evaluación el bot se sale del puesto, se activa el brazo B: una restricción lagrangiana sobre la distancia al ancla **[inferencia, pre-registrado]**.
- **Como en X4:**
  - 40% de arranques desde estados humanos;
  - latencia aleatoria con la distribución de S6;
  - crítico central con todo, más marcador y reloj.
- **Por qué no self-play:** el objetivo es complementar a humanos, no a sí mismo (Carroll 2019; Strouse 2021).

**Cómputo:** queda para decidir.
- Las etapas S, D y la evaluación en lazo abierto (§6 A) son CPU livianas.
- L0 y L2 necesitan GPU. L2 además corre 21 redes de compañeros por partido: hay que hacer la inferencia en lote en la GPU.

## 6. Evaluación: ¿ocupa bien el puesto? ¿es competente? ¿es una carga?

Todo se mide con la latencia de sala y con criterios fijados antes de medir.

### A. Lazo abierto, sobre las grabaciones de prueba (sin simulador)
- **Error de ghosting:** posición del bot a +1, +2 y +5 s contra el humano de ese puesto, por puesto (Le 2017).
  - Tiene que mejorar dos líneas base: la posición de plantilla y la referencia programada.
- **Intercambio de puesto:** en un estado real se le da al bot otro puesto. Su trayectoria tiene que ir a la zona del puesto nuevo: el etiquetador de D2 tiene que coincidir con el puesto pedido en ≥ 90% (propuesta). Prueba que el condicionamiento funciona.
- **Decisiones con la pelota:** acuerdo con el humano en pasar, rematar, cargar powershot o conducir, y NLL en los cambios de tecla.

### B. Lazo cerrado en el simulador, con los compañeros de L1: el partido de reemplazo
- **Valor de reemplazo:** en el puesto k, el bot contra la BC, con los mismos estados y semillas.
  - Se mide la diferencia de goles y de EPV por 90 min, con IC bootstrap.
  - **"No es carga":** el límite inferior del IC al 90% es ≥ −δ en cada puesto y en cada formación. δ lo fija el usuario.
- **Fidelidad al puesto,** con el mismo etiquetador que a los humanos:
  - % de tiempo en el puesto pedido;
  - mapa de calor (W1) contra los humanos de ese puesto;
  - distancia a los compañeros de línea;
  - offsides cometidos, en atacantes, y provocados, en defensores;
  - faltas cometidas y recibidas;
  - uso y resultado del powershot;
  - ejecución de saques.
  - Cada métrica tiene que quedar dentro del rango humano prueba-vs-entrenamiento, el mismo criterio que en X4.
- **Independencia de la formación:**
  - se reserva una formación entera (por ejemplo, todo el 3-5-2 fuera de entrenamiento) y se evalúa ahí;
  - se prueba con las cuatro formaciones y con formas irregulares.
- **Robustez:** con ningún estilo de compañero de L1 el valor de reemplazo cae debajo de −δ.

### C. Sala real
- Partidos humanos organizados por el usuario, con el bot en al menos 3 puestos distintos.
- Con grabación: la grabación entra como dato y como prueba de conformidad.
- **Las mismas métricas,** comparadas con los humanos de ese puesto en el banco.
- **Encuesta corta a los compañeros** (1–5): ¿estuvo en su puesto?, ¿decidió bien?, ¿fue una carga? Carroll y Strouse usan valoraciones humanas además del puntaje.

**Niveles:**
- **"No es carga" (mínimo):** pasa B, y en C sus métricas de puesto quedan dentro del rango humano.
- **"Competente" (objetivo):** mediana humana de su puesto en valor agregado por 90 min, en sala.

## 7. Fundamentos de fútbol 11 y de HaxEleven

Cada fundamento se mide primero en los humanos, por puesto (D1–D3). Así queda la referencia y la métrica de evaluación.
- **No se premian:** Liu 2022 (2105.12196) encontró que el shaping de estilo rinde peor, y la biblioteca de X4 advierte que una métrica no es un objetivo.
- **El único shaping es el potencial del EPV.**
- Se crea `docs/conocimiento/haxeleven_x11/`, una biblioteca como la de RS X4, con grado de evidencia y con el meta medido en el banco: formaciones usadas, frecuencia de cada mecánica.

### Nuevos por tener 11 por lado

| Fundamento | Cómo se mide | Fuente | Entra como |
|---|---|---|---|
| Líneas y distancia entre líneas (bloque compacto) | largo y ancho del equipo; profundidad del bot respecto de su línea | Frencken 2011; Bialkowski 2014 | evaluación |
| Línea defensiva y offside (subir en bloque, trampa) | x del defensor respecto de su línea; offsides provocados | regla medida en S2c | observación y evaluación |
| Desmarque de ruptura en habilitación | carrera respecto de la línea al momento del pase; offsides cometidos | Spearman 2018 (OBSO) | evaluación |
| Amplitud y carriles (no repetir carril y línea) | ocupación de 5 carriles × 3 líneas | Fernández y Bornn 2018 | evaluación |
| Cobertura y permuta | quién cubre la zona que deja el que sale a presionar | Stone y Veloso 1999 (roles flexibles) | evaluación |
| Presión: quién salta, quién cierra líneas | distancia al poseedor y a las líneas de pase, por puesto | Spearman 2017 (probabilidad de pase, MIT Sloan) | evaluación |
| Transiciones | tiempo hasta recuperar la forma tras perder o ganar la pelota | Shaw y Glickman 2019 | evaluación |
| Forma con y sin pelota | cambio de plantilla por fase (3-5-2 → 5-3-2) | Shaw y Glickman 2019 | etiqueta (D2) |
| Cambio de frente y pase largo | éxito del pase largo según el % de carga | D1 | evaluación |
| Arquero: ángulo, salida y juego con el pie | posición respecto de pelota y arco; saques de arco con la barra | D1–D2 | evaluación |
| Pelota parada preparada | receptor del córner y ubicación de los atacantes | TacticAI (Wang 2024, 2310.10553) | evaluación; posible modelo aparte |

### Propios de HaxBall y HaxEleven (descritos por el usuario)

| Fundamento | Por qué importa | Cómo se mide |
|---|---|---|
| **Cuerpear** para esperar apoyo | proteger la pelota además provoca la falta del rival que se queda pegado 6 s | tramos de protección (definición de `x4_mechanics`) y resultado: apoyo, falta a favor, pérdida |
| **Entrar y salir** al marcar | 6 s pegado al poseedor sin tocar la pelota es tiro libre en contra, también de atrás en carrera (penal en el área) | temporizador de contacto propio al despegarse; faltas cometidas por puesto |
| **Powershot:** cuánto cargar | cargar exige quedarse pegado: lleva tiempo y expone a la presión | % al patear según la presión y el resultado (gol, pase largo completo, pérdida) |
| **Curva** | rodear la barrera o al arquero | uso según la situación (tiro libre, remate) |
| **Saque rápido con la barrera** | la barrera aleja al rival unos instantes | tiempo hasta sacar contra la continuidad, en los humanos |
| **Barra de fuerza:** cuándo patear | la fase define la fuerza: pase corto o largo | fase elegida según el tipo de saque y el destino |
| **Sin slide** | solo se roba con el cuerpo o con la patada | robos por tipo |
| **Rifle y usar** (X4) | a verificar si existen en 11v11 | `x4_mechanics` adaptado |

## 8. Etapas y criterios de avance

| Etapa | Produce | Criterio para seguir | Cómputo |
|---|---|---|---|
| S0 | inventario, partidos, splits | archivos únicos, tramos 11v11 identificados | CPU liviano |
| S1 | `.hbs`, física base, ticks/s con 22 jugadores | física p90 ≤ 0,02 px | CPU liviano |
| S2–S3 | catálogo de residuos, contrato HX11-1, `env/hx11` | todas las familias explicadas o marcadas | CPU liviano |
| S4 | conformidad completa | tabla de S4 | CPU medio |
| S5–S6 | pruebas en sala, latencia | según lo que S4 deje abierto | sala, con el usuario |
| D0–D3 | caché, eventos, roles, EPV | etiquetador validado con la muestra del usuario | CPU medio |
| L0 | imitación por puesto | §6 A | GPU |
| B | partido de reemplazo | §6 B | CPU y GPU |
| L1–L2 | solo si B no pasa con L0 | §6 B | GPU |
| C | sala real | §6 C | sala |

No se avanza por presupuesto agotado sin una decisión explícita del usuario.

## 9. Riesgos

- **Compañeros sintéticos más débiles que los humanos.** El valor de reemplazo en el simulador sobreestima, y el bot puede explotar sus defectos (Carroll 2019). Mitigaciones: el ancla KL, la diversidad de L1 y la prueba C, que es la única que certifica.
- **Errores del etiquetador de roles.** SoccerCPD llega al 86,5% contra expertos. Hay que validar con alineaciones reales antes de condicionar con él.
- **Reglas del script que cambian con la fecha,** o variantes `Custom` mezcladas. Mitigación: contrato por versión (S2).
- **El motor con 22 jugadores puede ser lento** para el RL. Por eso L2 es condicional y la imitación es la base.
- **Ruido en las grabaciones** de salas públicas: AFK, partidos desparejos, tramos sin 11v11. Se filtran en S0.
- **Partidos casuales de nivel desigual.** "Competente" se mide contra la mediana del puesto en este banco, no contra la de un grupo fuerte.
- **Indicaciones por voz.** El técnico habla por voz y el bot no lo escucha.
  - El puesto no depende de eso: lo fija el usuario.
  - Pero si un grupo cambia de plan en el partido (por ejemplo, pasa a presionar arriba), el bot lo nota recién cuando sus compañeros se mueven distinto, y puede reaccionar tarde **[inferencia]**. Se mide en C.
- **El script es una caja negra.** Una regla que no aparece en las grabaciones ni en las pruebas de S5 queda como supuesto en el contrato, y se marca así.

## 10. Respuestas del usuario y decisiones abiertas

### Respondido el 2026-10-07

| Tema | Respuesta | Dónde entra |
|---|---|---|
| Reglas del creador | no puede pasarlas; el script está ofuscado y no se puede leer | §1, S2, S5 |
| Faltas | poseedor = último que tocó; falta si un rival se le pega 6 s sin tocar la pelota, con cualquier velocidad y ángulo, también de atrás en carrera | S2b |
| Offside | regla del fútbol: por delante del anteúltimo rival al momento del pase; no hay en laterales | S2c |
| Penales, tarjetas | hay penales; no hay tarjetas | S2b, S2f |
| Laterales | no usan la barra | S2e |
| Partidos | casuales, entre grupos de Discord con un técnico que arma la formación por voz | §0, §1, D2, §4 |
| Puesto del bot | lo elige el usuario al agregarlo, sin más; no lo define el técnico ni se deduce de los compañeros | §0, §4 |
| Sala del bot | solo como cliente, con permiso | §4, S5, S6 |

### Abierto

1. ¿Los técnicos publican la formación o la alineación en Discord? Si no, ¿el usuario puede etiquetar unos 20 partidos? (D2)
2. ¿El arquero tiene alguna mecánica especial?
3. ¿Qué son las `CustomHaxEleven`?
4. ¿Un admin puede arrancar partidos para las pruebas de S5 en la sala del bot?
5. Los umbrales de S4, el δ de "no es carga" y el tamaño de la prueba en sala.
6. Dónde corre cada etapa (local o pod) cuando se apruebe el plan.
