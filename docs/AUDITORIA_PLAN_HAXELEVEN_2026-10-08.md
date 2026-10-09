# Auditoría del plan HaxEleven: ¿puede este entrenamiento producir un jugador competente frente a humanos?

Fecha: 8 de octubre de 2026. Audita [`PLAN_HAXELEVEN.md`](PLAN_HAXELEVEN.md) tal como está (borrador del 7 y 8 de octubre).

- **No se modificó el plan ni el código.** Solo se leyeron reportes y datos, y se corrieron análisis de lectura en un directorio temporal.
- **Método:** se buscó activamente por qué el entrenamiento podría fracasar aunque esté bien programado. Se buscaron trabajos que contradigan decisiones del plan, no que las confirmen.

**Convención de evidencia (se usa en todo el documento):**
- **[L]** resultado publicado, con la cita revisada en esta auditoría;
- **[R]** evidencia de este repositorio, con el archivo o el número;
- **[H]** hipótesis no validada (incluye inferencias mías).

**Advertencia general [H]:** no hay ningún trabajo publicado sobre un agente que juegue con 21 humanos en un deporte en tiempo real. La literatura más cercana es Overcooked y Hanabi (2 jugadores), conducción autónoma (muchos agentes, sin rival), Google Research Football (todo el equipo es del modelo) y MOBA 5v5. Cualquier resultado que se traslade a HaxEleven es una extrapolación.

---

## 1. Resumen ejecutivo

**Veredicto:** no autorizaría hoy una campaña larga de entrenamiento con la arquitectura actual.
- No es porque el diseño general esté mal.
- Es porque **todavía no existe ninguna forma validada de saber si funcionó.** El plan mide el éxito con herramientas que el propio entrenamiento usa: el mismo etiquetador de puestos, el mismo EPV y los mismos compañeros sintéticos. Ninguna de esas tres piezas está validada contra humanos.

**Lo sólido:**
- el simulador: física exacta y pelota parada casi exacta;
- los eventos de D1, con controles contra la sala;
- las particiones por sesión, con grupos reservados;
- el marco de "valor de reemplazo";
- la latencia medida y usada como observación;
- la decisión de no premiar estilo.

**Lo débil o sin demostrar, en orden de gravedad:**
1. **L1 no tiene una prueba de validez.** La evidencia de X4 indica que una población de imitaciones juega un fútbol distinto al humano (§2.6).
2. **No hay evaluación con potencia estadística.** Con 1,67 goles por partido, medir "no es carga" por goles exige miles de partidos en el simulador y es imposible en la sala (§9.2).
3. **El etiquetador de puestos no tiene verdad de referencia.** Además, la estabilidad de rol que mide es baja (mediana 0,72) y detecta formaciones que no coinciden con las que describió el usuario (§2.4).
4. **El EPV sería a la vez shaping y juez.** Eso es circular.
5. **La puerta en lazo abierto (§6 A) no predice el lazo cerrado.** En conducción la correlación entre las dos evaluaciones llegó a ser negativa [L: Dauner 2023].
6. **El banco es heterogéneo en modos y versiones del script.** Las reglas se validaron en 30 partidos que no lo representan (§2.3).
7. **L2 tiene un problema de crédito que el plan no aborda:** el bot es 1 de 22, y su efecto sobre goles y EPV es pequeño frente al ruido.

**Qué falta demostrar antes de una campaña larga** (detalle en §11 F):
- etiquetador validado;
- valor validado e independiente del entrenamiento;
- tabla de potencia;
- L0 evaluado en tres lazos;
- puerta de realismo de L1;
- un piloto temprano en sala;
- un piloto corto de RL con ablaciones y auditoría de exploits.

---

## 2. Hechos nuevos que cambian la lectura del plan

Medidos en esta auditoría sobre `data/hx11/` y `reports/`. Los scripts quedaron fuera del repositorio.

### 2.1 Hay menos datos utilizables de lo que dice el plan [R]

- El plan y el pedido hablan de **142 h**. En el estadio actual, [1800, 1200], hay **88,4 h (399 partidos)**, y el entrenamiento tiene **61,2 h de 11v11 (293 partidos)** (`reports/hx11/d0_cache.md`).
- Las otras ~64 h son de mapas viejos: [1875, 1064] con aceleración al patear de 0,085, y tamaños aleatorios. D0 no las procesó.
- Por jugador y puesto: 61 h × 22 ≈ 1.350 horas-jugador, unas **120 por rol de campo** si los 10 roles se reparten parejo.

### 2.2 El juego tiene pocos goles y mucho caos [R]

| Medida (399 partidos) | Valor |
|---|---|
| Goles por partido | 1,67 (desvío 1,54); 24% de los partidos sin goles |
| Diferencia de gol por partido | desvío **1,36**; 33% de empates |
| Pase completo | **47,2%** (42,8% va al rival) |
| Remate que termina en gol | 16,1% |
| Ventanas de 10 s de juego abierto con gol | ≈ 3% |
| **Acciones con pelota por jugador y partido** | **p10 9, p50 18, p90 31** |
| Penales ejecutados | **6** en 399 partidos |
| Córners y tiros libres ejecutados | 2.896 y 1.769 |

**Consecuencias:**
- La señal de goles es escasísima para un jugador entre 22.
- El potencial EPV estará casi siempre cerca de 0.
- Cada partido aporta solo unas 18 decisiones con pelota del bot.
- Para penales no hay datos.

### 2.3 Las reglas se validaron en una muestra que no representa al banco [R]

- **Muestra de S2/S4:** 30 partidos, del 27 de septiembre al 5 de octubre, con **45 faltas (1,5 por partido)**.
- **El resto del banco:** (233 − 45) / 369 = **0,51 por partido**, unas 3 veces menos.
  - El 61% de los partidos (245 de 399) no tiene ninguna falta.
  - Offsides: 4,5 por partido en la muestra contra 2,4 en el banco; entre el 21% y el 26% de los partidos no tiene ninguno.
- **El umbral de la falta sí es estable donde hay faltas:** el contacto al cobrar tiene p50 de 115 y p90 de 130 ticks en agosto, septiembre y las fechas ambiguas. No aparece ningún bloque con una regla de 6 s.
- **Lectura [H]:** lo que cambia entre partidos es **si el modo está activo o qué frecuencia tiene**, no la regla. 149 partidos no tienen `!gk`, así que hay al menos dos versiones del script.
- **Problema:** los modos por partido no están reconstruidos (pendiente 7 de S2). La imitación aprendería una mezcla de conductas: sostener el contacto es gratis donde las faltas están apagadas y caro donde no.

### 2.4 Puestos: el etiquetador es inestable y no tiene verdad de referencia [R]

Sobre los 47 partidos que D2 alcanzó a procesar antes de fallar:
- **Estabilidad de rol por jugador de campo:** p10 0,40; p25 0,54; **p50 0,72**; p90 0,91. Un cuarto de los jugadores pasa más del 46% del tiempo en otro rol que el suyo.
- **Formaciones detectadas:**
  - con pelota: 4-1-4-1 (36 de 94), 4-3-3 (21), 4-2-3-1 (14), 3-4-3 (13), 4-4-2 (5), **3-5-2 (3)**;
  - sin pelota: 2 formaciones distintas por equipo y partido en ventanas de 3 min (p50).
- **Contraste con el usuario:** dijo que las más usadas son 4-3-3, 4-4-2, 3-4-3 y 3-5-2.
  - O el etiquetador confunde formas cercanas (4-1-4-1 contra 4-3-3), o la realidad es otra. Sin verdad de referencia no se sabe.
  - **El 3-5-2 casi no aparece:** la prueba del plan que reserva "todo el 3-5-2" (§6 B) no tendría datos.
- **HMM de marcas (Franks):** colapsa a "el atacante más cercano" (traspaso). La medida nueva de seguimiento da βa ≈ 0,28: marca mayormente zonal, en un solo partido.

### 2.5 En HaxEleven, X4 ya mostró que la imitación en lazo abierto no anticipa el lazo cerrado [R]

`docs/REPORTE_2026-10-06_noche.md` (Sanguchito 4v4):
- En lazo abierto la probabilidad de patada estaba calibrada: 0,028 contra 0,026 de los humanos.
- En lazo cerrado, la imitación chica pateaba 0,55 veces por minuto contra 3,2 de los humanos.

### 2.6 Una población de imitaciones juega otro fútbol [R]

Self-play de la imitación final de X4 contra los humanos (`reports/x4/pass_chain_bc_final.json` frente a `pass_chain_human.json`):

| | Imitación (32 partidos) | Humanos |
|---|---|---|
| Pases por minuto | 5,7 | 12,3 |
| Precisión de pase | 48,5% | 67,3% |
| Desmarques útiles | **3,1%** | **10,8%** |
| Progresivos por minuto | 0,87 | 1,91 |

- La imitación logra apoyos (apoyo_frac 0,42 contra 0,38) pero **no logra que esos apoyos sirvan**.
- Es exactamente el tipo de "fútbol artificial con estadísticas individuales plausibles" que preocupa en L1.
- En 11v11 hay 21 imitaciones interactuando, no 7. **[H]** Lo esperable es peor, no mejor.

### 2.7 El RL de X4 no movió el juego colectivo [R]

`docs/AUDITORIA_POD_2026-10-07.md`:
- **Resultado:** 85% de victorias contra la imitación sin aprobar el criterio colectivo.
- **Miniatura en CPU:** con KL de 0,05–0,12, el RL **se alejó de la pelota más que la propia imitación**. El ancla KL por acción no impidió la deriva de posición.
- **Ruido de evaluación:** un mismo checkpoint midió 0,776 y 0,687 en dos evaluaciones de 16 partidos.

---

## 3. Auditoría de la arquitectura por etapa

### 3.1 D0–D3: datos, eventos, puestos y valor

| | D0 caché | D1 eventos | D2 puestos | D3 valor |
|---|---|---|---|---|
| **Qué produce** | estados y entradas por tick | pases, recepciones, faltas, pared, usar, cuerpeo… | rol por tick, formación, anclas, marcas | VAEP, valor de estado Φ, control del espacio, marca perdida |
| **Estado** | hecho; arquero conocido 100% | hecho; goles 668/668, barra 99,6%, offside 99,6% | falló a los 47 partidos; arreglo sin correr | no corrió |
| **Validación externa** | contra la sala | contra los anuncios de la sala | **ninguna** | **ninguna** |
| **Supuestos no demostrados** | — | definiciones de "toque" (0,6 px); goles en contra 20% | que los roles por plantilla sean los puestos que el usuario va a pedir; que la estabilidad baja sea movilidad real y no ruido | que el valor esté calibrado; que el valor en estados del simulador (con la imitación) sea el mismo que en estados humanos |
| **Error que se propaga** | poco | eventos mal atribuidos pasan a métricas | **condiciona L0, define el ancla de L2 y mide la fidelidad en la evaluación: el mismo sesgo en tres lugares** | **shaping de L2 y juez de §6: el mismo sesgo en dos lugares** |

**¿Es el método adecuado?**
- **D0 y D1 sí.**
- **D2 es razonable** (Bialkowski 2014 es el estándar en tracking). Pero necesita verdad de referencia antes de usarse para condicionar y para medir. SoccerCPD llega al 86,5% contra expertos en fútbol profesional; en partidos casuales de HaxBall no hay ningún dato.
- **D3 es razonable como herramienta de análisis.** No puede ser el único juez.

**Cómo demostrar que funcionan:**
- **D2:** acuerdo contra las etiquetas del usuario en ~20 partidos, por puesto y a nivel de tramo, y no por tick (§11 D, E1).
- **D3:**
  - calibración en prueba (curva de confiabilidad, Brier contra la tasa base);
  - un segundo modelo de valor independiente (otra familia, otros rasgos);
  - **fiabilidad por mitades** del valor agregado por jugador: si un jugador aporta en el partido A, ¿aporta en el B? Sin esa fiabilidad, la métrica es ruido y no mide habilidad.

### 3.2 L0: imitación condicionada por el puesto

- **Qué aprende:** p(teclas | estado demorado, puesto, acciones propias pendientes). Es decir, lo que haría en promedio un humano de ese puesto viendo ese estado con ese retardo.
- **Capacidades que debería desarrollar:**
  - ubicación por puesto;
  - uso de mecánicas comunes: powershot, barra, cuerpeo;
  - estilo humano.
- **Qué no puede garantizar:**
  - recuperarse de estados que los humanos no visitan (deriva);
  - superar a los humanos promedio;
  - decidir mejor que el promedio;
  - adaptarse a una latencia distinta de la humana.
- **Supuestos:**
  - que el estado demorado con D ∈ [8, 36] sea una entrada causalmente correcta para acciones humanas tomadas con otro retardo, desconocido. En X4 no se pudo estimar (sin codo) [R].
  - que el condicionamiento por puesto se aprenda a pesar del ruido de las etiquetas [H].
- **Errores que se transmiten:**
  - la imitación es el **ancla de L2** y el **compañero de L1**: sus defectos aparecen en las dos etapas;
  - **si es pasiva, L1 es pasiva, y el RL aprende a jugar contra y con pasivos** [R: X4 §2.5].
- **¿Es el método adecuado?** Sí como base: es lo que hacen AlphaStar, VPT, HR-PPO y TiKick. **No como producto final:** ver §4.

### 3.3 L1: población de compañeros y rivales

- **Qué aprende:** nada. Es el entorno. Su calidad define todo lo que se mide en el simulador.
- **Supuesto central, no demostrado:** que 21 imitaciones autónomas produzcan partidos cuyo comportamiento colectivo se parezca al humano. La evidencia de X4 dice lo contrario [R §2.6].
- **Riesgo de transmisión:** máximo.
  - Si L1 es irreal, el valor de reemplazo, la decisión de hacer L2 y lo que aprende L2 quedan sesgados.
  - Y no se ve, porque todas las métricas del simulador son relativas a L1.
- **Detalle en §5.**

### 3.4 L2: RL de puesto

- **Qué aprende:** a mejorar el retorno (goles más el shaping EPV), cerca de la imitación, contra y con L1.
- **Puede desarrollar** decisiones mejores que el promedio en estados críticos y estrategias adaptadas a la latencia del bot.
  - Por ejemplo, defender con más margen: la imitación no puede aprenderlo porque imita márgenes humanos.
- **No puede garantizar:**
  - transferencia a humanos (lo aprendido es una mejor respuesta a L1);
  - fidelidad al puesto (el KL por acción no la protege [R §2.7]);
  - que no explote el árbitro ni a L1.
- **Detalle en §6.**

### 3.5 La puerta en lazo abierto (§6 A) no sirve como puerta

- **El error de ghosting (posición a +1, +2 y +5 s) es una medida de pronóstico.**
  - En nuPlan, los planificadores aprendidos ganaron en lazo abierto y perdieron en lazo cerrado. La mejor nota en lazo abierto la sacó un modelo que ignoraba a los demás agentes. Las dos evaluaciones estaban "fundamentalmente desalineadas" [L: Dauner 2023].
  - En WOSAC, el método con menor minADE no fue el más realista [L: Montali 2023].
- **[H]** Un modelo que minimiza el ghosting tiende a la posición media, porque es la que minimiza el error cuadrático. Puede quedar bien ubicado en promedio y ser pasivo.
- **Recomendación:** el lazo abierto queda como diagnóstico. La puerta de L0 es el **lazo cerrado no reactivo** (los otros 21 reproducidos de la grabación) más el **lazo cerrado reactivo** (§4.4).

---

## 4. Auditoría profunda de la imitación (L0)

### 4.1 ¿Interpretará situaciones o reproducirá patrones?

**Reproducirá la correlación estado → tecla.**
- **La "interpretación" está implícita.** Si el humano se ubica entre su referencia y el arco, la red puede aprender a hacerlo como función del estado, sin representar "por qué".
- **Ejemplo:** un lateral que se cierra porque el central salió a presionar. La imitación aprende la correlación si la situación está bien cubierta en los datos.
- **Fuera de esa cobertura no hay garantía.** La literatura de imitación lo reporta como confusión causal: el modelo usa rasgos correlacionados en vez de la causa [L: de Haan 2019; Codevilla 2019].
- **No hay forma de saber cuál de las dos cosas aprendió sin una intervención:** mover en el estado la causa (el central que sale) manteniendo lo demás, y ver si la respuesta cambia en la dirección humana. Ver las pruebas contrafácticas en §7.

### 4.2 ¿Generalizará a estados distintos? ¿Qué pasa cuando sus acciones cambian el partido?

- **Deriva:** cada error lo lleva a estados menos vistos, donde se equivoca más. El error crece más rápido que lineal con el horizonte [L: Ross 2011; resumido en Kumar 2022 como "suboptimalidad por estados fuera del dataset"].
- **En X11 el efecto puede ser menor que en X4** [H]: el bot es 1 de 22 y los humanos "corrigen" el partido. Por eso el riesgo principal no es la deriva del partido sino la del propio bot respecto de su zona y de la pelota.
- **Cómo evitarla** (cada una con su prueba; §11 D):
  1. **Medirla primero** en lazo cerrado no reactivo: 21 humanos reproducidos y el bot en un puesto, de 3 a 10 s desde miles de estados reales. Sin la medida no hay base para elegir un remedio.
  2. **Aumento con perturbaciones:** desplazar al jugador en la observación y supervisar la acción que vuelve a la trayectoria humana (ChauffeurNet, Bansal 2018). Barato y sin simulador.
  3. **Mejora offline sobre datos reales** (AW-BC/IQL, §4.6). Corrige decisiones sin depender de L1.
  4. **RL con ancla** (L2), solo si 1–3 no alcanzan. Lu 2022 muestra que imitación más RL supera a cada uno solo en los escenarios difíciles [L].
- **DAgger no es posible:** no hay un humano para consultar. Las variantes con simulador (DART) necesitan un experto ejecutable.

### 4.3 ¿Alcanzan 61 h (no 142 h)?

**Depende de la capacidad.**

| Capacidad | Ejemplos en entrenamiento (aprox.) | Juicio |
|---|---|---|
| Ubicación sin pelota por rol | ~120 h-jugador por rol, ×60 ticks/s | probablemente suficiente para el promedio [H] |
| Decisiones con pelota | ~18 por jugador y partido → ~116.000 en 293 partidos (unos 11.600 por rol) | justo para lo común; escaso para situaciones raras [H] |
| Córners y tiros libres | 2.896 / 1.769 en total | escaso por puesto (ejecutor, receptor) |
| Penales | **6** | **sin datos:** guion o fuera del alcance |
| Situaciones críticas (2v1 en contra, espalda ganada) | se miden en D3 | desconocido |

**Comparación con la literatura:**
- El ghosting en fútbol se hizo con decenas o cientos de partidos de tracking [L: Le 2017].
- AlphaStar usó 971.000 grabaciones y VPT 70.000 horas.
- Maia, millones de partidas por nivel.
- **Este banco es chico para imitación profunda y mediano para tracking deportivo.**

**Cómo saberlo con datos (E6):**
- curva de escala: entrenar con 25%, 50% y 100% del entrenamiento, y medir en lazo cerrado no reactivo, no en NLL;
- si sigue mejorando al 100%, conviene sumar los mapas viejos con el mapa como condición (precedente de X4 con RS ONE [R]).

**Cobertura por puesto, situación y habilidad:**
- tabla de conteo por rol × fase (abierto, saque propio o rival, con o sin pelota) × zona × nivel del jugador;
- el nivel del jugador se estima a partir de los partidos en que aparece (1.948 nombres; muchos repiten).

### 4.4 ¿La NLL de teclas alcanza?

**No como criterio.**
- La NLL total está dominada por "repetir la tecla". X4 ya separó los cambios de tecla [R].
- Pero aun una NLL perfecta en los cambios no asegura buena conducta en lazo cerrado [L: Dauner 2023; R §2.5].

**Propuesta de evaluación en tres lazos** (la misma para todas las variantes):
1. **Abierto** (diagnóstico): NLL en cambios de tecla, por contexto y por puesto.
2. **Cerrado no reactivo:** el bot controla un jugador y los otros 21 se reproducen de la grabación, durante 3–10 s, desde ≥ 5.000 estados de prueba. Se mide:
   - distancia a la trayectoria humana y a la zona del puesto;
   - desvío de la pelota respecto del jugador;
   - acciones con pelota.
   - Es la prueba "non-reactive closed-loop" de nuPlan.
3. **Cerrado reactivo:** partidos con L1, para métricas colectivas, con la advertencia de §5.

### 4.5 ¿Conviene sumar objetivos auxiliares o temporales?

Cada uno es un brazo pre-registrado, no un agregado por defecto:

| Agregado | Hipótesis | Evidencia | Métrica que lo justifica |
|---|---|---|---|
| **Historia** (secuencia corta o recurrencia) | adaptarse a compañeros y detectar a un AFK requiere memoria | Knott 2021: un compañero AFK solo se detecta con memoria [L]. Riesgo de "copycat" con historia [L: Wen 2020] | pruebas de estilo de compañero (§7, capacidad 11) y NLL en cambios de tecla |
| **Ghosting +1/+2 s** (ya en el plan) | mejora la representación sin pelota | Le 2017 [L] (abierto) | lazo cerrado no reactivo: deriva respecto de la zona |
| **Próximo evento** (receptor del pase, quién presiona) | fuerza a representar la intención del rival y del compañero | macro-intenciones mejoran el realismo de trayectorias multiagente [L: Zhan 2019] | intercepciones por oportunidad y anticipación (§7) |
| **Acciones en bloque** (k decisiones) | reduce la vacilación y el error compuesto | ACT, Zhao 2023 [L, manipulación] | cambios de tecla/s y deriva en lazo cerrado |

**[H]** Ninguno está demostrado en este dominio. Se agrega solo el que mejora su métrica en lazo cerrado, con varias semillas: la varianza entre corridas de imitación es alta [L: Codevilla 2019].

### 4.6 ¿Cómo evitar el techo de los humanos mediocres?

**El banco es casual:** pase completo 47%. Imitar el promedio produce, en el mejor caso, un jugador promedio; en lazo cerrado, uno peor (deriva).

| Opción | Evidencia | Riesgo | Prueba |
|---|---|---|---|
| **Condicionar por habilidad** (estimada por jugador con muchos partidos) y desplegar en el nivel alto | Maia: modelos por nivel predicen mejor a ese nivel; Maia-2 lo hace en una red [L] | extrapolar donde hay pocos datos de nivel alto | lazo cerrado: ¿el nivel alto mejora sin perder parecido? |
| **Imitación pesada por ventaja** (AW-BC, CRR, IQL) con el valor de D3 | TiKick, en 11v11 de GRF: de 83,4% a 94,4% de victorias contra la IA incluida al sumar los pesos; **CQL dio 0%** [L]. Kumar 2022: el RL offline supera a la imitación con datos ruidosos y estados críticos [L] | depende del valor; el offline ingenuo puede fallar (CQL) | igual que arriba, más el conjunto de escenarios críticos |
| **Filtrar por resultado del partido o por valor agregado de un partido** (lo que propone el plan) | en entornos estocásticos, condicionar por el retorno confunde suerte con habilidad [L: Paster 2022] | **alto:** con un desvío de 1,36 en la diferencia de gol, un partido no dice nada del jugador | no recomendado así: usar la ventaja por acción o la habilidad promediada sobre muchos partidos |

**Recomendación:** reemplazar la "variante pre-registrada" del plan (pesar por valor agregado o filtrar a los mejores) por dos brazos separados:
- habilidad estimada por jugador;
- ventaja por acción.

Cada uno se compara contra la imitación base en lazo cerrado.

---

## 5. Auditoría de los 21 agentes sintéticos (L1)

### 5.1 Diferencias esperables entre humanos reales e imitaciones autónomas

| Diferencia | Por qué aparece | Evidencia |
|---|---|---|
| **Pasividad o deriva** | error compuesto; la imitación nunca llega a situaciones de patada | [R: X4, 0,55 contra 3,2 patadas por minuto antes de escalar datos] |
| **Coordinación "hueca"** | apoyos que no sirven: cada imitación reproduce su parte sin la intención compartida | [R: desmarques útiles 3,1% contra 10,8%] |
| **Sin intención sostenida** | sin memoria ni plan, cada decisión se re-muestrea | [L: Zhan 2019 necesitó macro-intenciones para trayectorias coordinadas] |
| **Menos variedad que los humanos** | la imitación aprende la moda; la temperatura agrega ruido, no estilos | [L: GAMMA, Liang 2024: modelar el estilo como latente mejora la cooperación con humanos reales] |
| **Sin reacción a señales humanas** | voz, chat, conocerse del grupo | [L: Gao 2023: en MOBA con humanos, el canal de comunicación fue clave] |
| **Sin comportamiento fuera de distribución** | humanos AFK, enojados, que abandonan el puesto, novatos | [L: Knott 2021, pruebas de robustez de compañero] |

### 5.2 ¿Se amplifican los errores entre imitaciones?

**[H] fuerte, apoyada por [R]:** sí.
- Con una imitación entre 21 humanos, los humanos corrigen. Con 21 imitaciones, el error de una crea un estado raro para las otras 20.
- El fútbol resultante puede tener estadísticas individuales humanas y un comportamiento colectivo artificial: menos pases útiles, bloques sin compacidad, presión descoordinada.
- En X4, con 8 imitaciones, ya pasó [R §2.6].

### 5.3 ¿Puede el bot explotar los defectos de L1?

**Sí. Es el resultado esperado de una mejor respuesta a una población fija.**
- **[L]** Carroll 2019 entrena con modelos de humanos precisamente porque el self-play explota convenciones propias. Pero una mejor respuesta a la imitación explota los defectos de la imitación.
- **[L]** Strouse 2021 encontró que FCP, sin datos humanos y con diversidad de compañeros, superó con humanos reales al agente entrenado con la imitación humana (BCP), en rendimiento y en preferencia.
- **Ejemplos plausibles [H]:**
  - si los compañeros de L1 son pasivos, el bot aprende a ser individualista, porque pasar no paga;
  - si los delanteros rivales de L1 no buscan la espalda, el bot aprende a no cubrirla.
- **Este es el riesgo que los "rivales explotadores" del plan buscan cubrir.** Pero un explotador entrenado en el mismo simulador también puede explotar defectos del simulador, no del bot.

### 5.4 ¿Alcanza la diversidad propuesta? ¿Qué compañeros faltan?

**Lo propuesto:** checkpoints, temperaturas, estados reales, compañeros sesgados por recompensa (HSP), equipo programado y explotadores.

**Lo que falta:**
1. **Diversidad anclada en datos.** Un espacio de estilos aprendido de los humanos reales: embebido por jugador o latente por tramo, como GAMMA o los modelos individuales de Maia. Así se muestrean estilos que existen en la población real y con su frecuencia. Los sesgos de HSP los inventa el diseñador [H].
2. **Compañeros "malos" realistas:** AFK, que no vuelven, que abandonan el puesto, novatos. Knott 2021 [L].
3. **Formaciones irregulares:** 10v11 e inferioridad. El 11v11 completo cubre p10 de 0,85 de los ticks por partido [R].
4. **Compañeros que cambian de plan a mitad del partido**, como el técnico que ordena presionar arriba.

**Cómo medir la diversidad:**
- cobertura del espacio de estilos humanos: proyectar cada jugador humano y cada agente de L1 en los mismos rasgos por puesto (distancia a la pelota, altura, pases, presión, cuerpeo);
- medir qué fracción de humanos tiene un agente de L1 cerca, y al revés.

### 5.5 Evaluación independiente de L1 (puerta de realismo)

**No se avanza al RL ni se mide el valor de reemplazo hasta pasarla.** Todo se compara con la **variabilidad entre partidos humanos**: el mismo criterio "dentro del rango humano prueba-vs-entrenamiento" de X4.

**Diseño:**
- **Partidos sintéticos:** N ≥ 200 partidos completos de L1, desde saques iniciales y desde estados reales de prueba.
- **Referencia:** los 106 partidos de prueba y prueba de grupos, separados por grupo.
- **Calibración:** humanos de entrenamiento contra humanos de prueba dan el "techo" de cada distancia.

| Familia | Métricas |
|---|---|
| Colectivas | goles, remates y xG por partido; posesión; pases y su largo; pase completo; pases por pérdida; intercepciones, robos y recuperaciones por minuto; saques por tipo |
| Tácticas | largo y ancho del bloque, distancia entre líneas, altura de la línea, offsides cometidos y provocados, carriles ocupados, marcas perdidas por 10 min, margen de espalda |
| Temporales | duración de la posesión, tiempo hasta recuperar la forma tras perder la pelota, espera de los saques, ritmo de cambios de tecla |
| Situacionales | respuesta a 2v1, 3v2 y contraataques desde estados reales; tasa de remate tras marca perdida; continuidad tras recibir bajo presión |
| Mecánicas HaxEleven | powershot por nivel y resultado, faltas y cuerpeo, pared y usar por minuto, fase de la barra por tipo de saque |

**Pruebas:**
1. **Distribuciones:** cada métrica, distancia W1 normalizada contra la variación humano-humano. **Criterio propuesto:** ≥ 80% de las métricas dentro del techo y ninguna de las colectivas a más de 2 veces el techo.
2. **Clasificador de dos muestras (C2ST):** una red entrenada para distinguir clips de 10 s reales y sintéticos, con la posición de los 22 y la pelota.
   - AUC cerca de 0,5 es indistinguible. La importancia de rasgos dice **dónde** difiere.
   - **Criterio propuesto:** AUC ≤ 0,70, con humanos-contra-humanos de otro grupo como referencia, que debería dar cerca de 0,5.
   - **[H]** Es la prueba más dura y la más informativa.
3. **Desarrollos desde estados reales (estilo WOSAC):** desde un estado de prueba, 32 continuaciones de 5–10 s con L1. Se mide la verosimilitud de la continuación real bajo la distribución de las continuaciones: pelota, forma del equipo, próximo evento [L: Montali 2023].
4. **Juicio humano a ciegas:** 3 a 5 conocedores de HaxEleven ven clips cortos sin identificación y eligen cuál es real (como Zhan 2019 con analistas, y el Navigation Turing Test [L]).
5. **Validez de la respuesta humana** (requiere el piloto en sala C0):
   - con el bot en la sala, ¿los humanos le pasan, lo presionan y lo cubren como lo hace L1?
   - se compara la misma situación en el simulador y en la sala.
   - **Es la única prueba de que L1 se parece a los humanos *en la interacción con el bot*,** que es lo que importa.

**Si L1 no pasa:**
- **Arreglos con evidencia** [L: Igl 2022 Symphony, con discriminador y búsqueda; Montali 2023: ganó el único método entrenado en lazo cerrado]:
  - ajuste en lazo cerrado de la población con un discriminador (realismo como recompensa);
  - o mezcla con estructura programada (formación por Delaunay) para sostener la forma.
- **No se reemplaza la prueba por más diversidad:** la diversidad no arregla el realismo.

---

## 6. Auditoría del RL (L2)

### 6.1 ¿Es apropiado PPO?

- Para el bot, con 21 compañeros congelados, el problema es **RL de un solo agente en un entorno estocástico y parcialmente observable**. PPO con crítico asimétrico es una opción estándar y robusta [L: MAPPO, Yu 2022; AlphaStar y VPT usan variantes on-policy].
- **No encontré evidencia de que otro optimizador sea claramente superior en este escenario.**
  - Off-policy (SAC discreto, V-trace) reutilizaría muestras caras: cada tick exige 21 inferencias. Waymo usó BC-SAC [L: Lu 2022].
- **El cuello de botella no es el optimizador sino la señal** (§6.4). **Recomendación:** mantener PPO; medir antes la relación señal/ruido del gradiente (E14).

### 6.2 Crítico central

- **Adecuado** [L: crítico asimétrico]: ve todo, incluidas las acciones pendientes de los otros.
- **Pero no resuelve la atribución:** el valor del estado lo mueven los 22.
- **Alternativa con respaldo:** ventaja contrafáctica (COMA, Foerster 2018) o recompensa de diferencia con potencial (Devlin, Yliniemi, Kudenko y Tumer 2014 [L]). Comparar el Φ siguiente con la acción del bot contra el Φ siguiente con la acción de la imitación en el mismo estado.
- **Costo:** un desarrollo extra corto por muestra. Se justifica solo si E14 muestra que la señal/ruido es inútil sin ella.

### 6.3 Ancla KL: ¿protege el puesto o limita superar a los humanos?

- **¿Protege el puesto? Mal.**
  - El KL es por acción y por estado. Pequeñas desviaciones por decisión se acumulan en un desplazamiento grande.
  - En la miniatura de X4, con KL de 0,05–0,12, **el RL se alejó de la pelota más que la imitación** [R].
  - **Para proteger el puesto hay que restringir la distribución de estados, no las acciones:** una restricción sobre los rasgos de ubicación por puesto y fase (el brazo B del plan), o un discriminador de ocupación como AMP o GAIL.
- **¿Limita superar a los humanos? Poco, si λ es chico.**
  - El óptimo regularizado es π ∝ π_BC · exp(Q/λ). Reponderar la imitación permite superar al humano promedio.
  - AlphaStar llegó a Gran Maestro con ancla a su imitación y piKL juega fuerte y humano [L].
  - Lo que el ancla impide son **estrategias** que la imitación casi nunca muestra: secuencias largas de acciones improbables.
- **Fidelidad al puesto ≠ imitación exacta:**
  - **La fidelidad se mide** en la distribución de rasgos de ubicación (altura, carril, distancia a su línea y a su referencia) condicionada por la fase, contra los humanos de ese puesto.
  - **La imitación exacta se mide** en el KL por acción.
  - **Se pueden separar:** un bot puede tener KL alto y ubicación humana (decide distinto con la pelota), o KL bajo y ubicación derivada.
  - **El plan debería medir y restringir lo primero.**

### 6.4 Recompensa: goles más el shaping EPV

- **Goles:** con 1,67 por partido, el efecto de 1 jugador entre 22 sobre la diferencia de gol queda enterrado en un desvío de 1,36 por partido [R].
  - En la miniatura de X4 "casi no hay goles en los rollouts" [R].
  - **Los goles solos no alcanzan.**
- **Shaping EPV:**
  - **Teoría:** el shaping por potencial no cambia los equilibrios de Nash del juego estocástico, pero **sí cambia la exploración y a qué equilibrio se llega** [L: Devlin y Kudenko 2011].
  - **Práctica [H]:** con horizonte finito, KL y GAE, los errores del EPV orientan el aprendizaje.
  - **Revisar** que el potencial en el estado terminal o truncado esté bien tratado. Un corte por tiempo con Φ ≠ 0 rompe la invariancia. Hay que confirmarlo en el código heredado de X4: **no verifiqué que esté mal**.
- **¿Mide el valor táctico?**
  - El valor de D3 está entrenado en estados humanos y con continuaciones humanas.
  - En estados del simulador con L1, o ante tácticas nuevas del bot, es una extrapolación.
  - Con la tasa base de ~3% de goles cada 10 s, Φ cerca de 0 tiene diferencias chicas y ruidosas.
- **¿Puede explotarlo?** En el sentido estricto del shaping por potencial, los ciclos suman cero. Pero:
  - (a) con γ < 1 hay un sesgo por quedarse en estados de Φ alto o bajo;
  - (b) el bot puede **buscar estados que el modelo de valor sobrevalora** (zonas sin datos).
  - Se audita comparando el valor del modelo con el resultado real de esos estados en desarrollos largos.
- **Juez:** si el EPV es el shaping, **no puede ser la métrica principal de "aporta al resultado"** (ley de Goodhart). Hace falta un segundo modelo de valor independiente, y goles con potencia suficiente.

### 6.5 1 a 3 puestos aprendiendo y el resto congelado

- **Con 2 o 3 aprendices de parámetros compartidos en el mismo equipo**, pueden crear convenciones entre ellos: el modo de falla del self-play [L: Carroll 2019]. En la sala nunca va a tener otro aprendiz al lado.
- **Recomendación:** **un aprendiz por equipo**, en un puesto sorteado. El brazo con 3 aprendices solo se usa si E15 muestra que no cambia la conducta.
- **Congelar al resto es FCP** [L], y está bien si la población es válida y diversa (§5).

### 6.6 ¿Está justificado el 40% de arranques humanos?

- **No hay evidencia específica del 40%:** viene de X4.
- **Lo que sí tiene evidencia** es concentrar el entrenamiento en los escenarios difíciles. Lu 2022 encontró que entrenar con el 10% más difícil da la política más robusta [L].
- **Brazo pre-registrado:** 0%, 40% y 80% de arranques reales, y "estados críticos" (los que preceden a remates, marcas perdidas, contraataques y saques). Se elige por el conjunto de escenarios críticos, no por recompensa.

### 6.7 ¿Cuándo puede el RL superar a L0 y cuándo empeorarlo?

**Puede mejorarlo si se cumple todo esto:**
- (a) L0 falla en estados identificables, por deriva, latencia o decisiones críticas;
- (b) la señal en esos estados es informativa y atribuible al bot;
- (c) L1 es lo bastante realista para que la mejora transfiera;
- (d) el ancla preserva la compatibilidad.

**Puede empeorarlo si:**
- explota el árbitro (§7, faltas subcobradas);
- explota a L1 (§5.3);
- deriva del puesto (§6.3);
- sobreajusta al EPV (§6.4);
- crea convenciones entre aprendices (§6.5).

**Hoy no hay evidencia de ninguna de las cuatro condiciones.**

### 6.8 Ablaciones necesarias para justificar cada componente

| Componente | Ablación | Métrica principal | Criterio |
|---|---|---|---|
| Ancla KL | λ ∈ {0,02; 0,06; 0,2} | fidelidad de ubicación (W1 por puesto) y escenarios críticos | elegir el menor λ que mantiene la ubicación dentro del rango humano |
| Restricción de zona | con o sin | mismo | adoptarla solo si el KL no alcanza |
| Shaping EPV | con o sin; EPV A contra EPV B | goles (con potencia) y valor del modelo independiente | adoptar solo si mejora la métrica independiente |
| Arranques reales | 0, 40, 80% y críticos | escenarios críticos | — |
| Aprendices por equipo | 1 o 3 | convenciones: pases entre aprendices frente a pases a la imitación | 1 por defecto |
| Contrafáctico | con o sin | señal/ruido del gradiente y mejora por muestra | solo si E14 lo pide |
| Aleatorizar reglas | con o sin | sensibilidad de la conducta a las variantes | obligatorio en L2 |

---

## 7. Auditoría del simulador y los datos: bloqueante o tolerable

| Problema | Evidencia | Afecta a | Dictamen | Por qué |
|---|---|---|---|---|
| **Faltas: recall 71%** (la sala cobra y el simulador no) | S4 [R] | L2, sobre todo en defensores | **bloqueante para L2** sin aleatorizar; tolerable para L0 | el error va en la dirección peligrosa: en el simulador el contacto sale más barato y el RL lo aprende |
| Offside 93,5 / 90% | S4 [R] | L2: delanteros y línea defensiva | tolerable con aleatorización y auditoría | errores en los dos sentidos; el falso positivo premia la trampa del offside en el simulador |
| **Modos y versiones por partido sin reconstruir** | §2.3 [R] | L0 (conducta mezclada), D3, realismo de L1 | **bloqueante leve para la imitación final;** no para un primer piloto | la imitación aprende a sostener el contacto donde las faltas estaban apagadas |
| **Validación de reglas sobre 30 partidos de un solo período** | §2.3 [R] | todo lo que usa el árbitro | bloqueante para dar por cerrado S4 | 3 veces más faltas por partido que el banco |
| Pelota parada supuesta (plazo 600, lateral mal sacado, comba con 11–23% de error, `!lado`/`!centro`) | S3–S4 [R] | L2 en saques; evaluación de saques | tolerable si los saques no son objetivo del RL o se aleatorizan | los saques son 20% de los ticks; el RL podría aprender a demorar cerca del plazo supuesto |
| Penal (2–6 casos) | S2, D1 [R] | todo lo de penales | tolerable: penal con guion | sin datos no hay imitación posible |
| Orden de discos en choques múltiples | S3 [R] | detalle de forcejeos | tolerable | cambia el detalle, no la distribución (a verificar) |
| **Latencia: retardo constante sorteado contra ráfagas reales** | S6 [R]: 40–110 fps, bimodal | L0 y L2 en sala | tolerable **si** se prueba con trazas reales | las ráfagas son retardo variable dentro del episodio, no solo otro retardo; se pueden reproducir las trazas de S6 en el simulador |
| Retardo humano desconocido | X4 sin codo [R] | qué aprende la imitación | **riesgo de diseño** [H] | la imitación con D = 25 aprende a "adivinar" qué haría un humano que vio más reciente |
| **Etiquetas de puesto sin verdad** | §2.4 [R] | L0 (condición), L2 (ancla y restricción), evaluación de fidelidad | **bloqueante** | el mismo sesgo entra al entrenamiento y al juez |
| Sesgo del banco (grupos, salas públicas, nivel casual) | S0 [R] | todo | tolerable si se mide | el bot jugará en la sala del usuario; comparar su nivel con el banco |
| Mapas viejos excluidos | §2.1 [R] | cantidad de datos | tolerable | decidir con la curva de escala (E6) |

**Explotación del simulador:** para cada regla con supuestos o recall < 95%, se mide **la distribución humana del rasgo crítico y se vigila en el RL**. Por ejemplo:
- ticks de contacto con el poseedor (cuánta masa de la distribución cae entre 100 y 115);
- tiempo hasta sacar (cuánto cerca de 600);
- distancia a la línea de offside al recibir.

Una concentración en el borde de la regla es la firma de un exploit (E16).

---

## 8. Auditoría del aprendizaje táctico

**Principio:** que la red *pueda representar* algo (porque la observación lo contiene) no prueba que *lo use*. Para cada capacidad hacen falta dos pruebas:
- **conductual:** ¿el resultado se parece al humano?
- **contrafáctica:** si se cambia en el estado solo la causa, ¿cambia la respuesta en la dirección correcta?

| Capacidad | Mecanismo que la desarrollaría | Datos que la respaldan | Prueba conductual | Prueba contrafáctica |
|---|---|---|---|---|
| 1. Posicionamiento por puesto | imitación condicionada y ghosting | D2 (sin verdad todavía) | W1 de ubicación por puesto y fase en lazo cerrado; intercambio de puesto en lazo cerrado | mover la pelota o el bloque y comparar el desplazamiento del bot con la media condicional humana |
| 2. Coberturas, relevos y permutas | imitación implícita | **no hay detector** de cobertura en D1/D2 | escenarios reales donde un compañero sale de su zona: ¿quién la ocupa en N s? | desplazar al compañero en el estado inicial |
| 3. Anticipación y lectura | imitación con retardo (obliga a predecir) y RL | intercepciones 9/min [R] | intercepciones por oportunidad (líneas de pase cerca del bot); tiempo de reacción tras la patada | desviar la dirección de un pase en curso |
| 4. Desmarques y rupturas | imitación (débil en X4: 3,1% contra 10,8% [R]) y RL con OBSO | D3 (espaldas, OBSO), offsides 2,4 por partido | carreras a la espalda en el momento del pase; offsides cometidos por carrera; OBSO generado | mover la línea defensiva rival |
| 5. Defensa de espacios y marcas | imitación | marca perdida (64% termina en remate en un partido [R]); zonal βa ≈ 0,28 | marcas perdidas por 10 min contra humanos del puesto; margen de espalda | ubicar un 9 rival a la espalda |
| 6. Presión coordinada | imitación; la coordina el grupo humano | distancia al poseedor por puesto (D3) | con ≥ 2 compañeros presionando, ¿cierra su línea de pase? | sacar a un compañero de la presión |
| 7. Líneas de pase y apoyos | imitación | apoyos en X4 parecidos al humano [R] | disponible como opción (línea libre y distancia) cuando un compañero tiene la pelota | cerrar con un rival la línea actual |
| 8. Recepción y continuidad | imitación y RL | recepciones: 20.845 de primera, 36.624 controlando [R] | conservación tras recibir bajo presión; pase completo después | sumar presión en la recepción |
| 9. Transiciones | imitación y RL | pérdidas y recuperaciones en D1 | tiempo hasta volver a la zona tras perder la pelota | — |
| 10. Formaciones desconocidas | encoder de conjuntos y puesto | formaciones detectadas sesgadas (§2.4); 3-5-2 casi ausente | grupos reservados (LPM, VB); compañeros programados en formaciones raras | cambiar la formación de los compañeros programados |
| 11. Compañeros de distintos estilos | diversidad de L1 y **memoria** (que la red no tiene hoy) | estilos por jugador (a estimar) | pruebas tipo Knott: compañero AFK, pasivo, individualista, que no vuelve | — |
| 12. Mecánicas de HaxEleven | imitación (no se premian) | powershot 9.214; cuerpeo 3.362; pared 2.391; usar 8.834 [R] | uso y resultado por puesto contra humanos; fase de la barra por saque; saque rápido con barrera | — |

**Lectura:**
- Las capacidades 2, 4 y 11 son las más débiles.
  - La 2 no tiene ni detector.
  - La 4 ya falló en X4.
  - La 11 necesita memoria, que la arquitectura no tiene.
- La 5 es la que el usuario señaló como más importante ("no perder referencias"). Tiene el detector más prometedor (marca perdida), pero **validado en un solo partido**.

---

## 9. Evaluación de competencia humana

### 9.1 ¿Alcanza el valor de reemplazo?

**El concepto es correcto** para trabajo en equipo ad hoc. Tiene tres problemas:
1. **En el simulador es relativo a L1.** Si L1 no es realista, el valor de reemplazo puede tener sesgo en cualquier dirección.
2. **La línea base "imitación en el puesto" no es "humano en el puesto".** La imitación en lazo cerrado es más débil que un humano [R §2.5–2.6]. Superarla no implica nivel humano.
3. **En la sala, con goles, no tiene potencia** (§9.2).

### 9.2 Potencia estadística

**Fórmula:** tamaño por brazo para no-inferioridad, n ≈ 2(z₁₋α + z₁₋β)² σ² / δ².

**Con los datos del banco:**
- σ = 1,36 goles de diferencia por partido;
- α = 0,05 a una cola y potencia 0,8.

| Margen δ (goles por partido) | Partidos por brazo |
|---|---|
| 0,1 | ≈ 2.300 |
| 0,2 | ≈ 570 |
| 0,3 | ≈ 250 |
| 0,5 | ≈ 90 |

**Consecuencias:**
- **Simulador:** 2.300 partidos por celda es factible en el pod. Pero con 15 puestos × 4 formaciones × estilos son cientos de celdas.
  - Hay que medir si las semillas comunes reducen la varianza. **[H]** La dinámica es caótica y probablemente reducen poco.
  - Y medir la reducción de varianza de un resultado basado en valor (E4).
- **Sala:** "aporta al resultado" por goles **no es medible** con decenas de partidos.
  - Hay que usar métricas de proceso.
  - Ejemplo: en 20 partidos el bot tiene ~360 acciones con pelota, y una tasa como el pase completo se estima con ±4,3 puntos (IC 90%).

### 9.3 Protocolo de cinco niveles

**Controles comunes:**
- **positivo:** el equipo programado de §4, que debería ser detectable como bot;
- **negativo:** humanos contra humanos;
- **condicionamiento:** la imitación con el puesto equivocado.

Todas las pruebas se pre-registran y se usan varias semillas de cada modelo.

| Nivel | Lazo abierto | Simulador | Sala real | Criterio de aprobación (propuesto) |
|---|---|---|---|---|
| **1. Parece humano** | NLL en cambios de tecla (solo diagnóstico) | C2ST sobre clips del jugador (bot contra humano del mismo puesto, desde los mismos estados, lazo no reactivo) | jueces a ciegas sobre clips anonimizados de grabaciones (Navigation Turing Test [L]) | AUC del C2ST ≤ 0,65; jueces con IC 90% de acierto que incluye ≤ 60% |
| **2. Ocupa su puesto** | intercambio de puesto | % del tiempo en la zona del puesto, W1 de ubicación por fase, distancia a su línea, intercambio en lazo cerrado; en 4 formaciones más formas irregulares | las mismas métricas en las grabaciones de sala; el etiquetador **validado** (E1) | cada métrica dentro del rango humano del puesto (p10–p90 de jugadores-partido) |
| **3. Decide bien tácticamente** | — | conjunto de escenarios críticos desde estados reales, con respuestas aceptables definidas por (a) la distribución humana condicional y (b) el resultado en desarrollos: P(remate en contra en 5 s), etc.; Knott 2021 [L] | expertos a ciegas puntúan decisiones (bot contra humano del puesto, mismo tipo de situación) | tasa de aprobación ≥ mediana humana del puesto en esos mismos escenarios; IC 90% por bootstrap de escenarios |
| **4. Aporta al resultado** | — | valor de reemplazo contra la imitación **y** contra un "humano sustituto" (la mejor imitación validada), con goles en la potencia de §9.2 más el valor independiente, por estilo de compañero | diseño cruzado dentro de cada sesión: partidos alternados con el bot y con un humano en el mismo puesto; métricas de proceso en su zona (marcas perdidas, OBSO concedido, pérdidas); modelo jerárquico por grupo | no-inferioridad: límite inferior del IC 90% ≥ −δ (δ lo fija el usuario) |
| **5. Iguala o supera a humanos competentes** | — | contra imitaciones condicionadas en habilidad alta | percentil del bot en la distribución humana del puesto, para cada métrica de proceso; comparación por pares a ciegas ("¿quién jugó mejor el puesto?") con modelo Bradley–Terry | percentil ≥ 50 contra humanos de habilidad ≥ mediana; ≥ 50% en las comparaciones por pares, con IC |

**Separaciones que este protocolo permite:**
- 1 sin 2: parece humano pero no está en su puesto (ubicación fuera del rango).
- 2 sin 3: está en su puesto pero decide mal (falla los escenarios).
- 3 sin 4: decide bien pero no mueve el resultado (posible: el valor de un jugador es chico). El nivel 4 es de no-inferioridad.
- 4 sin 5: no es carga pero no llega al nivel competente.

**Regla:** ninguna métrica usada en el entrenamiento (EPV de shaping, ghosting, KL) es métrica principal de evaluación sin un par independiente.

**Encuesta a compañeros:** secundaria y sesgada si saben que es un bot. Puede usarse a ciegas solo con consentimiento informado previo de los participantes de la sesión.

---

## 10. Investigación: evidencia a favor y en contra de cada decisión

| Decisión del plan | A favor | En contra o con matiz | Veredicto |
|---|---|---|---|
| Compañeros = modelos de humanos (Carroll) | Carroll 2019: con humanos reales, el agente entrenado con el modelo humano supera al de self-play [L] | Strouse 2021: FCP sin datos humanos superó a BCP con humanos [L]. Knott 2021: el promedio de recompensa con modelos humanos oculta fallas [L] | usar modelos humanos **más** diversidad anclada en datos; validar L1 (§5.5) |
| Imitación como base, RL solo refina | AlphaStar, VPT y TiKick [L]. Lu 2022: imitación y RL superan a cada uno solo en lo difícil [L] | Cornelisse et al. 2026: en conducción, RL primero con poca imitación como regularizador da agentes compatibles con humanos [L, otro dominio, con recompensa densa] | mantener la base, pero **esperar** que L0 solo no alcance [R §2.5] y planificar L0.5 |
| Ancla KL | HR-PPO (λ 0,02–0,08): parecido humano sin perder rendimiento [L]. piKL, AlphaStar [L] | Protege acciones, no ubicación [R §2.7]. AH2AC2 reporta que λ tiene que ser grande en entrenamientos largos [L] | mantener, más una restricción sobre la ubicación |
| Shaping por potencial EPV | Ng 1999; Devlin y Kudenko 2011: preserva los equilibrios [L] | cambia la exploración y el equilibrio alcanzado [L]. Juez circular [H] | usar con un modelo independiente para evaluar |
| No premiar estilo | Liu 2022: los términos de pase y territorio rindieron peor que el esquema estándar [L] | el esquema estándar sí usaba shaping denso (velocidad a la pelota y de la pelota al arco) [L] | se sostiene |
| Ghosting y lazo abierto como puerta | Le 2017 [L] (lazo abierto, un jugador) | Dauner 2023: correlación negativa entre lazo abierto y cerrado [L]. WOSAC: minADE no ordena bien [L] | **lazo abierto solo como diagnóstico** |
| Pesar demostraciones por valor agregado | Kumar 2022; TiKick [L] | Paster 2022: condicionar por el retorno falla en entornos estocásticos [L] | ventaja por acción o habilidad promediada, no resultado por partido |
| Rivales explotadores (AlphaStar) | liga de AlphaStar [L] | con un simulador con supuestos, el explotador también explota al simulador [H] | primero como sonda de evaluación |
| Sin canal de comunicación | restricción del usuario | Gao 2023: en MOBA con humanos, la comunicación fue clave [L] | aceptado; medir el costo en la sala |
| Partidos de reemplazo con imitación en el puesto | marco ad hoc [L: Stone 2010] | la imitación es más débil que un humano [R] | sumar un "humano sustituto" validado |

**Citas del plan revisadas:**
- **Liu 2022 (2105.12196):** "el shaping de estilo rinde peor" se sostiene con un matiz. Los términos de coordinación "rindieron bien pero peor en general" que el esquema estándar.
- **HR-PPO, piKL, Carroll, Strouse, Le 2017, Kim 2022 (SoccerCPD), Müller-Budack 2019:** existen y se usan en el sentido correcto.
- **Strouse 2021:** su resultado principal **matiza** al plan. Superó al agente entrenado con la imitación humana.

**Qué está demostrado, qué es del repositorio y qué es hipótesis:**
- **[L]** La imitación deriva en lazo cerrado. El lazo abierto no predice el cerrado. Imitación con RL supera a cada uno en escenarios difíciles. El RL offline supera a la imitación con datos ruidosos. El shaping por potencial preserva equilibrios. Condicionar por el retorno falla en entornos estocásticos. Los modelos humanos promedio no detectan fallas de robustez.
- **[R]** La imitación de X4 fue pasiva y colectivamente hueca en lazo cerrado. El RL de X4 no mejoró lo colectivo y se alejó de la pelota con el ancla. El banco tiene 61 h de entrenamiento en el mapa actual, pocos goles, heterogeneidad de modos y etiquetas de puesto inestables.
- **[H]** Que en 11v11 la deriva del partido sea menor porque los humanos corrigen. Que los errores entre 21 imitaciones se amplifiquen más que en 4v4. Que el condicionamiento por puesto sobreviva al ruido de las etiquetas. Que el RL pueda atribuirse mejoras con 1 entre 22.

---

## 11. Resultado

### A. Diagnóstico

**Sólido (mantener sin cambios):**
- **simulador S0–S6:** física exacta, pelota parada ≥ 95%, latencia medida con eco propio;
- **D0 y D1,** con controles contra la sala;
- **particiones** por sesión y grupos reservados;
- **actor** sin marcador ni reloj, por preferencia del usuario;
- no premiar estilo;
- el marco de valor de reemplazo;
- la referencia programada como control.

**Débil (corregir antes de entrenar en serio):**
- la puerta en lazo abierto;
- la variante "pesar por valor agregado";
- la protección del puesto por KL;
- 2–3 aprendices por equipo;
- el 40% sin justificación;
- la ausencia de memoria para adaptarse a compañeros;
- la diversidad de L1 inventada en vez de aprendida.

**Puede impedir el objetivo (bloqueante):**
1. L1 sin puerta de realismo;
2. una evaluación sin potencia y con juez circular (EPV y etiquetador);
3. un etiquetador sin verdad de referencia;
4. el árbitro subcobrando faltas, para L2;
5. el banco con modos y versiones sin reconstruir.

### B. Riesgos prioritarios

Escala 1–3:
- **Impacto:** 3 si invalida el resultado.
- **Probabilidad:** 3 si la evidencia ya lo muestra.
- **Detección:** 3 si es difícil de detectar sin una prueba dedicada.

| # | Riesgo | Impacto | Prob. | Detección | Evidencia |
|---|---|---|---|---|---|
| 1 | L1 juega un fútbol artificial; todo lo medido en el simulador queda sesgado | 3 | 3 | 3 | [R §2.6] |
| 2 | Juez circular: EPV y etiquetador entrenan y evalúan | 3 | 3 | 3 | diseño del plan |
| 3 | Sin potencia: se declara éxito o fracaso por ruido | 3 | 3 | 2 | [R: ruido de X4], §9.2 |
| 4 | L0 pasiva o con deriva en lazo cerrado; L1 hereda la pasividad | 3 | 2 | 1 | [R §2.5] |
| 5 | El RL explota a L1 (individualismo, no cubrir) | 3 | 2 | 3 | [L: Carroll, Strouse] |
| 6 | El RL explota faltas subcobradas o supuestos de saque | 2 | 2 | 2 | [R S4] |
| 7 | Etiquetas de puesto ruidosas: condición débil, fidelidad mal medida | 3 | 2 | 3 | [R §2.4] |
| 8 | Señal de RL inútil (1 entre 22, pocos goles) | 2 | 3 | 1 | [R §2.2] |
| 9 | Deriva del puesto pese al KL | 2 | 2 | 1 | [R §2.7] |
| 10 | Banco heterogéneo en modos: conducta mezclada | 2 | 2 | 2 | [R §2.3] |
| 11 | Latencia en ráfagas no modelada | 2 | 2 | 1 | [R S6] |
| 12 | Sin memoria: no se adapta a compañeros AFK o con otros estilos | 2 | 2 | 2 | [L: Knott] |
| 13 | Sin comunicación: tarde ante cambios de plan del técnico | 1 | 3 | 2 | [L: Gao 2023] |

**Orden de ataque:** 1, 2, 3 y 7 primero. Son baratos de investigar y sin ellos ningún otro resultado se puede interpretar. Después 4 y 10. El 5, 6, 8 y 9 solo importan si hay L2.

### C. Mejoras

Cada una con su razón, su hipótesis y cómo se mide.

1. **Puerta de medición G0 antes de cualquier entrenamiento.**
   - **Razón:** riesgos 2, 3 y 7.
   - **Hipótesis:** el etiquetador acierta ≥ 85% por tramo, y el valor está calibrado y es fiable por mitades.
   - **Medición:** E1–E4.
2. **Condicionar por el puesto del tramo, no por el rol del tick.**
   - **Razón:** el rol por tick sale de la posición actual, así que filtra la respuesta en la condición [H]. Lo que el usuario fija con `--pos` es un puesto por tramo.
   - **Medición:** intercambio de puesto en lazo cerrado (E8).
3. **Evaluar L0 en tres lazos;** el lazo abierto deja de ser puerta.
   - **Razón:** [L: Dauner 2023; R §2.5].
   - **Medición:** E6.
4. **L0.5: mejora offline sobre datos humanos reales** (ventaja por acción o habilidad), antes del RL en línea.
   - **Razón:** [L: Kumar 2022; TiKick; Maia]. No depende de L1.
   - **Hipótesis:** mejora los escenarios críticos sin perder parecido.
   - **Medición:** E9.
5. **Puerta de realismo de L1,** con diversidad aprendida de los datos.
   - **Razón:** riesgo 1.
   - **Medición:** E10 y E11.
6. **Piloto temprano en sala (C0) con la mejor L0 o L0.5.**
   - **Razón:** es la única fuente de verdad sobre cómo responden los humanos al bot. Valida L1 (§5.5, prueba 5) y mide el costo de la latencia real. Su costo es organizativo, no de cómputo.
   - **Medición:** E12.
7. **En L2:**
   - un aprendiz por equipo;
   - aleatorizar las reglas no resueltas;
   - restringir la ubicación y no solo el KL;
   - evaluar con un valor independiente;
   - auditar exploits.
   - **Razón:** riesgos 5, 6 y 9.
   - **Medición:** E14–E16.
8. **Memoria en la política** (historia corta), como brazo.
   - **Razón:** riesgo 12.
   - **Hipótesis:** mejora las pruebas de estilo de compañero sin empeorar la NLL en cambios de tecla (copycat).
   - **Medición:** E7.

**No recomiendo agregar todavía** (sin evidencia de que hagan falta):
- liga de explotadores para entrenar: primero como sonda;
- ajustes de HSP con recompensas sesgadas: primero la diversidad aprendida;
- ventaja contrafáctica: solo si E14 lo pide;
- macros: la lección de X4.

### D. Experimentos

**Costos:** estimaciones de orden de magnitud **[H]**, basadas en el D0 medido (57 min para 399 partidos con 3 procesos), el kernel (500.000 ticks de partido por segundo en la PC) y la imitación de X4 (~37.000 muestras/s en CPU).

| ID | Hipótesis | Método | Métrica | Criterio de éxito (propuesto) | Costo |
|---|---|---|---|---|---|
| **E1** | El etiquetador identifica puestos | el usuario etiqueta ~20 partidos (formación por grupo y puesto por jugador y tramo), o se usan alineaciones de Discord | acuerdo por tramo y por puesto; matriz de confusión; ¿4-1-4-1 es 4-3-3? | ≥ 85% por tramo (referencia SoccerCPD 86,5%) | usuario: ~5–8 h; CPU: minutos |
| **E2** | Las reglas y los modos son estables por bloque | inferir los modos por partido (cobros, anuncios); árbitro en sombra sobre 30 partidos de agosto y de comienzos de septiembre | recall y precisión por bloque; partidos con faltas u offside apagados | igual que S4 en cada bloque, o contrato por versión | CPU ~1 h local |
| **E3** | El valor es válido | D3 más un segundo modelo independiente; calibración en prueba; fiabilidad por mitades del valor por jugador | ECE, Brier contra la tasa base; correlación A-B por jugador; acuerdo entre modelos | calibrado; fiabilidad con IC > 0 | CPU unas horas |
| **E4** | Hay un protocolo con potencia | varianza por puesto de las métricas de proceso en el banco; efecto de las semillas comunes en el simulador | partidos necesarios por nivel y celda | existe un protocolo de sala factible para los niveles 1–3 | CPU: minutos; pod: ~1 h |
| **E5** | El retardo de entrenamiento no daña | sonda de verosimilitud del retardo (como X4); lazo cerrado con trazas reales de S6 | NLL en función de d; deriva en función de la traza | sin degradación con trazas reales frente al retardo fijo | CPU, horas |
| **E6** | La imitación base y su escala | imitación al 25, 50 y 100%; tres lazos; 3 semillas | deriva (no reactivo), métricas colectivas (reactivo), NLL en cambios | curva que se aplana; si no, sumar mapas viejos con el mapa como condición | GPU: ~3 × 3 × (2–6 h) |
| **E7** | Historia y auxiliares ayudan | brazos de §4.5, de a uno contra la base | la métrica de cada brazo en la tabla | mejora con IC en lazo cerrado y en ≥ 2 de 3 semillas | GPU: 2–6 h por brazo y semilla |
| **E8** | El condicionamiento funciona | puesto por tramo contra rol por tick contra ancla continua; intercambio en lazo cerrado | % del tiempo en la zona pedida tras el intercambio; tiempo en llegar | ≥ 90% en zona (propuesta del plan) en lazo cerrado | incluido en E6 |
| **E9** | La mejora offline supera a la imitación | AW-BC o IQL con el valor de E3; condicionado por habilidad | escenarios críticos y parecido humano | mejora en críticos sin salir del rango humano | GPU: ~2 × E6 |
| **E10** | L1 es realista | §5.5, pruebas 1–4 | W1 contra el techo, AUC del C2ST, verosimilitud de desarrollos, jueces | ≥ 80% de métricas dentro del techo; AUC ≤ 0,70 | pod: 200 partidos ≈ 1–3 h; jueces: 1–2 h de usuarios |
| **E11** | La diversidad cubre a los humanos | proyección por estilos | cobertura de humanos y de agentes | ≥ 80% de los humanos con un agente cerca | CPU |
| **E12** | El bot no es carga en sala (piloto) | 6–10 partidos, 2–3 puestos, con consentimiento; grabación | fallas graves; métricas de proceso; respuesta humana al bot | nada fuera del p10–p90 humano en fallas graves | usuario: 2 sesiones |
| **E13** | Los humanos responden al bot como L1 | situaciones emparejadas sala-simulador con el mismo bot | pases al bot, presión sobre el bot, coberturas | dentro del rango de §5.5 | CPU, con los datos de E12 |
| **E14** | El RL tiene señal | señal/ruido del gradiente y varianza explicada de la ventaja del bot | escala de ruido del gradiente; ΔΦ atribuible al bot por contrafáctico | lotes factibles; si no, contrafáctico o L2 descartado | pod ~1 h |
| **E15** | Ablaciones de L2 | tabla de §6.8, piloto corto | §6.8 | §6.8 | pod: decenas de horas GPU |
| **E16** | Sin exploits | distribución de los rasgos en el borde de cada regla, RL contra humanos; aleatorización de reglas | masa cerca de 113 ticks de contacto, de 600 de espera y de la línea de offside | dentro del rango humano | incluido en E15 |

### E. Arquitectura recomendada

**Se mantiene L0–L2.** Se agregan puertas de medición, una etapa offline y un piloto temprano:

```
S0–S6 (hecho)
 └─ G0  Medición: verdad de puestos (E1) · modos y versiones (E2) · valor independiente (E3) · potencia (E4) · retardo (E5)
     └─ L0   Imitación por puesto del tramo; brazos de a uno; evaluación en tres lazos (E6–E8)
         └─ L0.5 Mejora offline sobre datos reales: ventaja por acción / habilidad (E9)
             └─ L1   Población con diversidad aprendida; PUERTA DE REALISMO (E10–E11)
                 └─ C0   Piloto en sala con L0/L0.5 (E12) → valida L1 en interacción (E13)
                     └─ L2   RL dirigido, opcional: 1 aprendiz por equipo, ubicación restringida,
                             reglas aleatorizadas, auditoría de exploits, valor independiente (E14–E16)
                         └─ C   Evaluación de cinco niveles (§9.3)
```

**Diferencias con el plan:**
1. La evaluación y la validación van **antes** del entrenamiento.
2. L1 tiene una **puerta propia**.
3. Se agrega **L0.5** offline, que usa interacciones humanas reales y no depende de L1.
4. **La sala entra temprano,** como fuente de verdad, no solo al final.
5. L2 pasa a ser **dirigido a fallas medidas**, no un refinamiento general.

### F. Decisión final

**No autorizaría una campaña larga de entrenamiento con la arquitectura actual.**

No porque el diseño sea incorrecto en lo general, sino porque hoy ningún resultado de esa campaña sería interpretable. **Falta demostrar, en este orden:**
1. **Que el puesto se mide bien:** etiquetador ≥ 85% contra la verdad del usuario (E1).
2. **Que el valor se mide bien e independiente del entrenamiento:** calibración y fiabilidad por jugador (E3).
3. **Cuántos partidos hacen falta** para cada nivel de §9.3, en el simulador y en la sala (E4).
4. **Qué modos y versiones** tiene cada partido de entrenamiento, y que el árbitro vale fuera de los 30 partidos de S4 (E2).
5. **Que L0 o L0.5 se sostiene en lazo cerrado no reactivo** (deriva y puesto dentro del rango humano) (E6–E9).
6. **Que L1 pasa la puerta de realismo** (E10), y que los humanos de la sala responden al bot como L1 (E12–E13).
7. Solo entonces, **un piloto corto de L2** con ablaciones, que muestre que hay señal (E14), que no hay exploits (E16) y que mejora sobre L0/L0.5 con una métrica independiente y con potencia (E15).

Recién con 1–7 tendría sentido una campaña larga, y solo en la dirección que el piloto haya mostrado útil.

---

## Fuentes

**Revisadas en esta auditoría:**
- Knott et al. 2021, *Evaluating the Robustness of Collaborative Agents* — https://arxiv.org/abs/2101.05507
- Kumar, Hong, Singh y Levine 2022, *When Should We Prefer Offline RL Over Behavioral Cloning?* — https://arxiv.org/abs/2204.05618
- Lu et al. 2022, *Imitation Is Not Enough* (BC-SAC, Waymo) — https://arxiv.org/abs/2212.11419
- Dauner et al. 2023, *Parting with Misconceptions about Learning-based Vehicle Motion Planning* — https://arxiv.org/abs/2306.07962
- Montali et al. 2023, *The Waymo Open Sim Agents Challenge* — https://arxiv.org/abs/2305.12032
- Liu et al. 2022, *From Motor Control to Team Play in Simulated Humanoid Football* — https://arxiv.org/abs/2105.12196
- Huang et al. 2021, *TiKick* — https://arxiv.org/abs/2110.04507
- Paster, McIlraith y Ba 2022, *You Can't Count on Luck* — https://arxiv.org/abs/2205.15967
- Cornelisse et al. 2026, *Human-like autonomy emerges from self-play and a pinch of human data* — https://arxiv.org/abs/2606.19370
- Cornelisse y Vinitsky 2024, HR-PPO — https://arxiv.org/abs/2403.19648
- Ad-Hoc Human-AI Coordination Challenge (AH2AC2) 2025 — https://arxiv.org/abs/2506.21490
- Liang et al. 2024, GAMMA — https://arxiv.org/abs/2411.13934
- Hu et al. 2022, *Human-AI Coordination via Human-Regularized Search and Learning* — https://arxiv.org/abs/2210.05125
- Codevilla et al. 2019, *Exploring the Limitations of Behavior Cloning for Autonomous Driving* — https://arxiv.org/abs/1904.08980
- McIlroy-Young et al. 2020, Maia — https://arxiv.org/abs/2006.01855
- Devlin et al. 2021, *Navigation Turing Test* — https://arxiv.org/abs/2105.09637
- Igl et al. 2022, *Symphony* — https://arxiv.org/abs/2205.03195
- Zhan et al. 2019, *Generating Multi-Agent Trajectories using Programmatic Weak Supervision* — https://arxiv.org/abs/1803.07612
- Strouse et al. 2021, *Collaborating with Humans without Human Data* (FCP) — https://arxiv.org/abs/2110.08176
- Devlin y Kudenko 2011, *Theoretical Considerations of Potential-Based Reward Shaping for Multi-Agent Systems* — https://eprints.whiterose.ac.uk/75111/
- Gao et al. 2023, *Towards Effective and Interpretable Human-Agent Collaboration in MOBA Games* — https://arxiv.org/abs/2304.11632
- Van Roy et al. 2020, *Valuing On-the-Ball Actions in Soccer: A Critical Comparison of xT and VAEP* — https://lirias.kuleuven.be/retrieve/558069

**Citadas sin revisar el texto en esta auditoría** (conocidas; verificar antes de citarlas en el plan): Ross, Gordon y Bagnell 2011 (DAgger); de Haan et al. 2019 (confusión causal); Wen et al. 2020 (copycat); Bansal et al. 2018 (ChauffeurNet); Zhao et al. 2023 (ACT); Foerster et al. 2018 (COMA); Devlin, Yliniemi, Kudenko y Tumer 2014 (recompensas de diferencia con potencial); Yu et al. 2022 (MAPPO).
