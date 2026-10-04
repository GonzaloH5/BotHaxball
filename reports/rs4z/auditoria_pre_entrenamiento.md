# Auditoría previa al entrenamiento RS4-Z (RS4 4v4 desde cero)

Fecha: 2026-10-03, actualizada el 2026-10-04 con el humo de liga, exploiters y compuertas. Rama `rs4z`. Plan aprobado: `~/.claude/plans/fallamos-catastroficamente-en-muchas-agile-whisper.md`.

**Estado: infraestructura lista para pruebas largas; el entrenamiento largo NO se lanzó y requiere tu aprobación.**
La sección 10 lista lo que todavía queda abierto.

## 1. Resumen

| Bloque | Qué se hizo | Evidencia principal |
|---|---|---|
| Simulador | Kernel numba nuevo con árbitro integrado; paridad exacta con el árbitro anterior (modo v1) y mecanismos reales del script (v2) | 1,39 M transiciones reales: error de 1 tick p99 1,4e-4 px (redondeo de las grabaciones) |
| Correcciones | F1 masa, F2 reloj, F3 fin de partido, F4 reloj del actor, F5 barreras, F7 franja muerta, F8 decodificador, F9 ejercicios fuera del árbitro, F10 escala 600, F11 rewards por defecto, F22 disco del córner | tabla de la sección 2 |
| Scripted | RS-Pro (nuevo, numba) con niveles L0–L5 y estilos continuos | escalera monotónica; L5 le gana a L0 92%; estructura en bandas humanas |
| Curriculum | 7 etapas con compuertas pre-registradas; 26 tareas | `train/rs4z/stages.py`, `reports/rs4z/gates.json` |
| Rewards | goles ±1 + resultado ±0,3 + PBRS (xT humano, acceso, distancia en 1v0) con retiro | identidad telescópica y suma cero verificadas por test; tramposos sin ganancia |
| Entrenador | PPO + crítico centralizado (MAPPO), inicio aleatorio, liga PFSP, compuertas | 83 tests RS4-Z; humos en Pod (secciones 9 y 9.1) |
| Despliegue | observación v2 portada a Node; export ONNX sin entradas privilegiadas | paridad obs 0,0; logits Node 6,7e-8 |

## 2. Problemas encontrados y correcciones

| # | Problema | Evidencia | Corrección (verificada) |
|---|---|---|---|
| F1 | Tras cada reposicionamiento la sala juega con invMass 0,5 (mapa) hasta la primera patada de un saque; el sim usaba 0,3 siempre | 72 archivos: 431 cambios a 0,3, 99% a 0–3 ticks de una patada; 370 en laterales y 59 al liberar córner/saque de arco; 34% de las transiciones en juego ocurren con 0,5 | Árbitro v2: 0,5 tras reposicionar, 0,3 con la patada del lateral o la liberación de la pieza. A 60 ticks en esos tramos: p90 de la pelota 0,02 px con el modelo nuevo contra 14,7 px con 0,3 fijo |
| F2 | El reloj corría durante la espera del saque inicial | 322 447 de 322 452 ticks de espera con reloj congelado | Reloj congelado en el saque inicial |
| F3 | El fin de partido se bootstrapeaba aunque el crítico ve el reloj | `haxball_env.py:1673` | Fin de partido = terminal real; truncación sólo en cortes artificiales |
| F4 | El actor veía un reloj (`tfrac`) y el export fijaba 7200 contra 10800 entrenado | `export/to_onnx.py:151` | Obs v2 sin reloj; metadatos del export desde el checkpoint |
| F5 | La barrera del saque de arco teletransportaba rivales hasta 175 px | `rs_one_referee.py:204` | Se usan los segmentos c0 reales del mapa vía cGroup, como el script |
| F6 | (Reportado como error) El disco de 33 px bloquea al ejecutor | Disco 3 del mapa: cMask rojo+azul; se retira 180 ticks después de colocar la pelota | **No era error**: es la regla real; se reproduce con el disco real del motor |
| F7 | Franja de la esquina donde nunca se cobraba la salida | `rs_one_referee.py:122` | Regla sin huecos + test |
| F8 | El decodificador público etiquetaba como córner el 6,4% de los laterales | `public_signals.py:92` | Node/Python clasifican por el punto exacto del contrato |
| F9 | Los ejercicios sintéticos de saques no pasaban por el árbitro | `rs4_v3._place` | Motor de ejercicios nuevo: todo saque vía `start_restart` |
| F10 | `field_half_h = 600` (la línea es 670) en obs, scripted y colocaciones | `sim/stadium.py:257` | Código nuevo con medidas del contrato; obs a escala real |
| F11 | `RewardConfig` por defecto no era neutro | `env/rewards.py` | Registro nuevo con todo en 0 |
| F22 | El córner real excluye a los defensores con un disco de radio 445 en (±1150, ±740); el sim aproximaba 100 px alrededor del punto | 510 de 510 córners; en 81 893 ticks de defensores: p1 de distancia al centro 460,0 (mínimo 455) | Disco real del script en el motor (sólo choca con el equipo defensor) |
| — | Barreras de laterales y saque de arco | El script agrega c1/c0 al cGroup de los rivales | Se reproducen igual (segmentos reales, de ambos lados) |
| — | Paso de 3 ticks y latencia | El bot cliente tiene 6–11 ticks de retraso | Latencia por jugador en el kernel + últimas 3 decisiones en la obs |
| — | Variantes reales del mapa | kickStrength 5,75 en 9/61; radio 8 en 5/61; la geometría 76ae sólo difiere en el plano de la pelota | Aleatorización por partido desde S5 (expuesta en la obs) |
| — | Saques trabados gratis (atajo de A1) | liberación de 900 ticks | Plazo de entrenamiento 600: el saque pasa al rival (sin multa ajustable); evaluación sin plazo |

Desviaciones aceptadas (sin efecto relevante en el aprendizaje): la sala cobra las salidas 1–9 ticks tarde (laterales) y ~20–40 en córners y saques de arco; el 3% de los laterales va al otro equipo; la curva del córner es un ajuste (residuo p90 0,04); el orden de colisiones rojo→azul (mitigado por color aleatorio); asimetrías del mapa real (pared de jugadores 0,1/0; arco c0 70/59,26) inactivas en juego. Formato real: partidos de ~605–681 s, sin gol de oro.

## 3. Contrato RS4-Z-2 y conformidad

- `env/rs4z/contract.py` (constantes y mecanismos), `env/rs4z/kernel.py` (física + árbitro, una llamada paralela por decisión).
- **Paridad v1** (`tests/rs4z/test_rs4z_referee_parity.py`): 768 000 ticks-partido con resincronización; error máximo 2,3e-13 px y cero discrepancias de eventos (goles, salidas, laterales, córners, saques de arco). En trayectoria libre los eventos coinciden durante >2000 ticks hasta que el redondeo numpy/numba (1e-17) crece por el caos de las colisiones.
- **Conformidad v2** (`tools/rs4z_conformance.py`, `reports/rs4z/conformance.json`): 1 392 809 transiciones reales, p99 de 1 tick 1,4e-4 px en ambas fases de masa.
- **Reglas v2** (`tests/rs4z/test_rs4z_rules.py`, 15 tests): masa por fase, reloj, disco del córner sólo para defensores, área del saque de arco por segmentos (sin teletransporte), disco del punto 180 ticks, barrera lateral de ambos lados sólo para rivales, franja de la esquina, plazos de entrenamiento, gol no terminal y fin de partido terminal, jugadores inactivos, latencia exacta.

## 4. Observación, acciones, latencia y despliegue

- Obs v2 (`env/rs4z/obs_v2.py`): propia + 7 entidades con máscara; escala real (1150, 670); x espejado para el azul; últimas 3 decisiones; latencia; variante del mapa; señales públicas de saque y saque inicial; fase de masa. **Sin reloj, marcador ni último toque real.** Rango acotado (test sobre 1500 decisiones) e invariante al cambio de color (test).
- Crítico (sólo entrenamiento): reloj, marcador, diferencia, último toque, plazos internos, fase de masa, latencia rival.
- 18 acciones, 3 ticks por decisión (20 Hz), latencia por jugador 0–12 ticks.
- Node: `deploy/rs4z/obs_v2.js`; paridad con Python 0,0 en 2500 casos (incluye córners, saques de arco, saque inicial, latencias y planteles incompletos). ONNX del actor: paridad PyTorch–onnxruntime 1e-7 y en Node 6,7e-8.
- **Pendiente** antes de jugar con humanos: el modo rs4z del bot de sala (`deploy/bot.js`) que arme este estado desde la sala (historial de acciones, retraso medido, fase de masa). El contrato y la función ya están portados y probados.

## 5. Scripted RS-Pro

`bots/rspro/` (numba). Rival, compañero y generador de situaciones; nunca objetivo de imitación ni de recompensa.

- **Percepción humana**: estado con retraso de reacción (L5: 8 ticks; L0: 39), ruido y anticipación; su propio cuerpo en tiempo real.
- **Cinemática exacta**: tiempo de llegada con la física real (verificado ±1 tick), predicción de la pelota (error <1e-9 antes del contacto), intercepción cortada en la línea de gol.
- **Cerebro de equipo**: fases con histéresis; roles (portador/presionante, apoyo corto, amplio, profundo, seguridad, cobertura, marca, último hombre) por asignación de costo mínimo con costo de cambio; los compañeros no controlados (aprendices o humanos) ocupan el rol más cercano y los bots completan el resto; nunca arquero fijo por slot.
- **Portador**: valor esperado (éxito × valor de posesión − fracaso × valor para el rival) entre tiro (puntos del arco válidos para su error angular), pase al pie/adelantado/en profundidad, autopase, conducción, despeje; costo de preparación (acomodarse detrás de la pelota mientras llega la presión); compromiso hasta ejecutar salvo opción claramente mejor; compensación de la velocidad previa de la pelota en la patada; ninguna trayectoria hacia el arco propio.
- **Defensa**: presión pegada del lado del arco con quite seguro cuando hay cobertura detrás; **último hombre que contiene** (L1+: sobre la línea pelota→arco a 70–110 px, retrocede con el poseedor y sólo cierra en zona de tiro o con la pelota suelta); **relevo** (si el presionante quedó pasado, presiona el mejor ubicado del lado del arco); atajada en la línea de gol, seguridad que cubre el arco con la pelota cerca, despejes que nunca apuntan al arco propio.
- **Ejecución**: navegación de órbita alrededor de la pelota (no la toca al rodearla) y frenado por velocidad deseada.
- **Saques**: ejecutor por costo, espera de apoyos, impulsos reales del córner (×1,98) y del saque de arco (×2,71).

Validación (`reports/rs4z/rspro_ladder.json`, 256 partidos de 3 min por par, ambos colores, estilos al azar; versión final con contención, relevo, salida sólo con ventaja clara y zona de tiro sin tirarse, §9.2; entre paréntesis la versión original):

| Par | Puntos del nivel superior (IC95) |
|---|---|
| L1–L0 | 0,54 (0,54) |
| L2–L1 | 0,67 (0,67) |
| L3–L2 | 0,62 (0,65) |
| L4–L3 | 0,55, IC 0,51–0,58 (0,57) |
| L5–L4 | 0,53, IC 0,49–0,56 (0,58) |
| L5–L3 / L2 / L1 / L0 | 0,62 / 0,70 / 0,84 / 0,91 (0,65 / 0,77 / 0,90 / 0,92) |

La parte alta se comprimió: con la defensa que contiene hay menos goles y más empates, así que los puntos se acercan a 0,5 aunque haya diferencia de nivel. El ritmo de goles de L5 contra L5 es 0,074 por minuto, igual al humano (0,071): el RS4 real también se juega con muy pocos goles.

Estructura L5 contra L5 frente a humanos (69 grabaciones, mismas definiciones): dispersión 150 (humano 166), profundidad 323 (362), anchura 257 (260), distancias a la pelota 113/224/301/416 (96/203/283/392), aglomeración ~7% (4%), flotación junto al arco 0%; pases por pérdida 0,48 (1,15), tiros 0,16 por minuto (0,47), goles 0,074 por minuto (0,071).

Baterías de referencia (L5 en los lugares del aprendiz, 512 episodios): control 0,92/0,67/0,83/0,91 (tocar, arco vacío, conducir y definir, recibir y definir); duelos: tiro contra el último hombre 0,23, defender 1v1 0,92, pelota dividida 0,37; 2v1 0,44 y pared 0,36, con pase en 98–99% de los goles; defensa 2v2/4v4/saques 0,88/0,86/0,90.

**Límites conocidos de RS-Pro** (no invalidan el aprendizaje, porque el RL debe superarlo y luego lo reemplaza el self-play):
- pasa menos y pierde más la pelota que los humanos (pases/pérdidas 0,47 contra 1,15);
- tira menos (0,18 contra 0,47 por minuto), aunque convierte igual que ellos;
- su ataque 1v1 es débil (0% contra el último hombre que contiene); el pase en profundidad, los centros y el 2v2 ofensivo también (≤6%), así que esas tareas no se juzgan en las compuertas (sí se entrenan);
- la separación entre L3, L4 y L5 es chica en puntos (0,53–0,58) porque hay muchos empates: los niveles altos se distinguen más en las baterías que en los partidos.

Las compuertas sólo juzgan tareas donde la referencia resuelve ≥15%.

## 6. Curriculum (`train/rs4z/stages.py`)

| Etapa | Tareas | Muestras (techo) | Rewards | Compuerta (pre-registrada) |
|---|---|---|---|---|
| S1 control | 1v0: tocar, arco vacío, conducir y definir, recibir y definir | 0,3B | gol/ejercicio + Φ_ball (se retira) + Φ_threat | control ≥ 0,90×L5 (cada tarea ≥ 0,75×) |
| S2 pelota y 1v1 | tiro contra último hombre, 1v1 ataque/defensa, pelota dividida, 1v1 completo | 0,5B | + Φ_access | duelos ≥ 0,85×L5; 1v1 contra L5 ≥ 55% (IC90 > 50%) |
| S3 tiro, pase, rebotes | 2v1, pared, profundidad, desvío, 2v2 ofensivo (2 aprendices); pared y profundidad exigen el pase | 0,7B | idem | batería de pases contra defensores L5 ≥ 0,80×L5 y, en 2v1, ≥50% de los goles con un pase entre aprendices |
| S4 defensa | 2v2, 4v4 y saques en contra; 2v2 completo | 0,8B | Φ_access → 0 | defensa ≥ 0,80×L5, goles recibidos ≤ 1,1×L5; detectores 2v2 |
| S5 cooperación 4v4 | 4v4 contra la escalera, 4v3, 4v2, 3v4, estados humanos, saques, transiciones; 30% espejo | 2,5B | + resultado ±0,3; Φ_threat → 0 | ≥60% contra L5 (IC90 >50%), ≥50% contra el estilo reservado; saques propios: ≤2% tardan ≥600 ticks y la mediana de cada tipo ≤2× la humana; detectores 4v4 |
| S6 liga | espejo 45%, PFSP 25%, RS-Pro 20%, exploiters 10%; 15% compañeros scripted | 7,5B | sólo goles y resultado | ≥75% contra L5, ≥65% contra el reservado, brecha visto−reservado ≤10 pp, detectores 4v4 en banda |
| S7 robustez | latencia de cliente 70%, compañeros ad-hoc, variantes | 1,0B | idem | ≥70% contra L5; pérdida ≤10 pp con 8 ticks de latencia (sólo el candidato) y ≤10 pp con la variante del mapa |

- **Los pesos de cada etapa son fracciones del tiempo simulado.** La probabilidad de iniciar una tarea es peso / duración media de su episodio (medida en línea). Sin esto, un partido de 3–10 min dura 10–30 veces lo que un ejercicio y tapaba a los ejercicios y al repaso (ver §9).
- Repaso de etapas anteriores ≥10% del tiempo (5% desde S6).
- Desde S5 los partidos duran 3–10 min de reloj, uniforme (la sala juega ~10 min; el actor no ve el reloj, el crítico sí ve el tiempo restante). S2–S4 mantienen 2–3 min.
- Un partido que pasa 2× su reloj + 1 min real sin terminar (reloj congelado porque nadie saca el inicial) se corta por truncación con bootstrap.
- Dificultad adaptativa por tarea (éxito buscado 60%).
- Una etapa sólo avanza por sus compuertas; si agota el presupuesto sin pasar, el entrenamiento se detiene para diagnóstico.
- 3v3 queda como contingencia (no aporta en una cancha de 2300×1340).

## 7. Rewards (`train/rs4z/rewards.py`)

| Término | Etapas | Escala | Por qué | Exploit y defensa | Retiro |
|---|---|---|---|---|---|
| Gol ±1 (equipo, suma cero) | todas | 1 | objetivo | — | nunca |
| Resultado ±0,3 al fin | S5+ | 0,3 | jugar para ganar; sólo el crítico ve marcador y reloj | suma cero: no repite la trampa anti-0-0 | nunca |
| Resultado de ejercicio (+1/0/−1) | ejercicios | 1 | define la tarea | éxitos exigen control sostenido; sacarla no cuenta | sale con el ejercicio |
| Φ_threat (xT humano) | S1–S5 | 0,25 | progreso denso | PBRS: los ciclos no rinden; suma cero | lineal a 0 en S5 |
| Φ_access (margen de llegada del equipo) | S2–S4 | 0,05 | presión/posicionamiento | sólo cuenta el más rápido: perseguir todos no suma | lineal a 0 en S4 |
| Φ_ball (distancia en 1v0) | S1 | 0,05 | exploración inicial | PBRS; sólo sin rivales | lineal a 0 en S1 |
| Φ_crowd | apagado | ≤0,02 | contingencia | sólo penaliza <120 px | sólo si el detector falla |

- xT (`reports/rs4z/xt.json`): 45 grabaciones de entrenamiento, 163 140 muestras; de −0,03 junto al arco propio a +0,40 frente al rival.
- Tests (`tests/rs4z/test_rs4z_rewards.py`): Σγᵗ·F = −Φ(s₀) con error <1e-6, suma cero por paso, y ninguna de siete políticas tramposas obtiene ganancia de shaping.
- Prohibidos por evidencia previa: posesión, pases, formación, proximidad a la pelota, multas anti-0-0, anclas KL.

## 8. Self-play y diversidad de oposición

- Parámetros compartidos: los 8 jugadores pueden ser el aprendiz (espejo), así que "uno se cuelga mientras otro trabaja" no es estable.
- Las filas congeladas o scripted nunca entran a la pérdida (test).
- **Liga** (`train/rs4z/league.py`):
  - instantáneas cada 250M muestras desde S5, con PFSP (1−p)²;
  - exploiters marcados en su propia fracción. `tools/rs4z_supervisor.py` corre el principal y, en S6–S7, cada 1B muestras copia el principal y entrena un exploiter de 200M muestras contra esa copia congelada (sólo partidos 4v4 contra él). Al terminar, el exploiter juega 128 partidos contra el principal y entra a la liga sólo si saca ≥60% de los puntos; el historial queda en `exploiters.json`;
  - los índices de miembro son estables: al superar 48 miembros, el más fácil se marca retirado en vez de borrarse (antes, borrarlo corría los índices de los partidos en curso);
  - RS-Pro L3–L5 con estilos de entrenamiento;
  - la región de estilos reservada (presión alta + directo + estrecho) nunca se muestrea al entrenar.
- Compañeros ad-hoc (15–20% en S6–S7): RS-Pro competente que infiere los roles de los compañeros que no controla.

## 9. Evaluaciones y evidencia de la infraestructura

- **Métricas comunes a la simulación y a los humanos** (`eval/rs4z/metrics.py`; referencia `reports/rs4z/human_reference.json`):
  - pases y pérdidas, tiros, goles;
  - estructura y perfil de distancias;
  - detectores de aglomeración, colgado, flotación junto al arco, quietud y oscilación;
  - duración de saques.
- **Baterías reproducibles** (`eval/rs4z/batteries.py`): mismas semillas para el candidato y la referencia. Las compuertas viven en `eval/rs4z/gates.py` y la calibración congelada en `reports/rs4z/gates.json` (huella de código `08ec29997c680485`).
- **Tests**: suite completa 876 aprobados, 44 omitidos al 2026-10-04 antes de §9.2; RS4-Z hoy 86 (793 anteriores + 83 RS4-Z en `tests/rs4z`: paridad del árbitro, reglas v2, obs y simetría, ejercicios, rewards y tramposos, RS-Pro, entrenador, liga, mezcla del curriculum y compuertas).
- **Tramposos contra RS-Pro L5** (128 partidos cada uno, versión final): quieto 1,00; oscilar 1,00; todos persiguen 0,96; pelotazos 1,00; sacarla afuera 0,78; bloque en el arco 0,97; delantero colgado 1,00; trabar saques 0,97; azar 0,99 (puntos de L5).
- **Humo en Pod** (RTX 3090, 1024 partidos, pesos aleatorios, `runs/rs4z/smoke_s1_gpu`):
  - **Corrección previa**: con minilotes de 32k (6 pasos por iteración) y Φ_ball = 0,05 la política no se movía (KL ~3e-4 a los 8,5M). Con minilotes de 8192 × 4 épocas y Φ_ball = 0,5 (PBRS, no cambia el óptimo) aprende.
  - Prueba aislada "tocar la pelota": éxito de 7% a 82–84% en 6M muestras; entropía de 2,89 a 0,66; KL 0,002–0,01.
  - **S1 completa**: a los 18,7M muestras (dificultad adaptativa) tocar 0,77, arco vacío 0,67, conducir y definir 0,81, recibir y definir 0,82.
  - **Compuerta S1 a los 20,05M**: batería de control a dificultad máxima 0,855 contra la referencia RS-Pro L5 de 0,832, con cada tarea por encima del 75% de L5 (tocar 0,84; arco vacío 0,71; conducir 0,95; recibir 0,92). **Aprobada**: el curriculum pasó solo a S2.
  - **S2** (17M muestras más, hasta que se detuvo a mano): 1v1 de ataque de 0,02 a 0,5–0,8; tiro contra el último hombre de 0,08 a 0,50; pelota dividida de 0,44 a ~0,6; los partidos 1v1 contra RS-Pro a la dificultad vigente los gana casi todos.
  - **Rendimiento**: 50–60k muestras/s en S1 (un aprendiz por partido) y 75–90k/s en S2.

### 9.1 Humo de liga, exploiters y compuertas (2026-10-04)

Tres corridas de infraestructura en el Pod desde pesos aleatorios (prueban la maquinaria, no el aprendizaje), con la liga llena, exploiters y las compuertas de S2–S7 sobre el checkpoint de humo S1/S2. Encontraron doce fallos, todos corregidos y con test de regresión salvo donde se indica:

| # | Fallo | Cómo apareció | Corrección |
|---|---|---|---|
| I1 | Los pesos de etapa se aplicaban por episodio; un partido dura 10–30 veces un ejercicio | En S6 no terminó ningún ejercicio después de la iteración 5; el repaso de S5 quedaba en ~1% del tiempo | Probabilidad de inicio ∝ peso / duración media; los pesos son fracciones del tiempo simulado |
| I2 | El estimador de duración se sesgaba: los partidos de 3 min terminan primero | Partidos al 76% del tiempo contra 63% buscado | En partidos se estima la razón ticks reales / ticks de reloj y se multiplica por la duración conocida |
| I3 | Los partidos de entrenamiento duraban fijo 2–3 min (el plan decía 3–10) | Revisión del código | Duración uniforme 3–10 min desde S5 |
| I4 | Un partido con el reloj congelado (nadie saca el inicial) no terminaba nunca | 0 partidos terminados en S6 con redes al azar | Corte por truncación a 2× el reloj + 1 min real |
| I5 | La compuerta de saques no podía fallar: la evaluación no tiene plazo y contaba los vencimientos de ambos equipos | Revisión del código | Mide los saques propios del candidato: ≤2% tardan ≥600 ticks y la mediana por tipo ≤2× la humana; un candidato quieto la reprueba (test) |
| I6 | S7 (robustez) la aprobaba una política débil (sólo medía caída) y la latencia se aplicaba también a RS-Pro | El checkpoint de S2 la aprobó | Piso ≥70% contra L5; latencia sólo en el candidato; variante del mapa |
| I7 | La compuerta 1v1 contra L3 no informaba (RS-Pro ataca mal en 1v1) | El checkpoint de S2 sacó 0,97 | Contra L5 |
| I8 | Retirar un miembro de la liga corría los índices de los partidos en curso | Revisión del código | Índices estables; el miembro se marca retirado |
| I9 | La liga guardaba 4 redes en memoria y releía del disco en cada paso; además hacía una pasada de red por rival | S6 cayó de ~170k a 14k muestras/s con ~11 miembros; con caché, 119k/s con 35 miembros | Todas las redes en el dispositivo (0,4M parámetros cada una); grupo activo de 8 rivales sorteado por PFSP y renovado cada 1000 partidos; todos los rivales en una sola llamada (vmap sobre los parámetros apilados, igualdad de logits verificada por test): 152k/s con 35 miembros |
| I10 | Nada lanzaba exploiters durante el entrenamiento largo | Revisión del plan | `tools/rs4z_supervisor.py` (§8) |
| I11 | La evaluación del exploiter fallaba si ninguna red sacaba el inicial | Excepción en el Pod | Los partidos trabados entre redes se cierran con el marcador vigente |
| I12 | Compuertas cada 50M fijos: ~150 evaluaciones en S5–S6 (~6 h en pausa y muchas oportunidades de aprobar por ruido); la huella de la calibración no se verificaba | Cálculo de costo | Cada 10% del presupuesto de la etapa (≥25M), aprobación confirmada con otra semilla, y la evaluación se niega a correr si la huella del código no coincide con `gates.json` (sin test: revisión) |

Resultados después de corregir:

- **Mezcla por tiempo** (S5, 1024 partidos, 80M muestras), objetivo partidos/situaciones/repaso 0,63/0,27/0,10. Último tercio: 0,646/0,258/0,096. El primer tercio sobrerrepresenta ejercicios porque al arrancar ningún partido largo terminó todavía (transitorio de ~35M muestras sobre 2,5B).
- **Rendimiento** (RTX 3090, 12 CPU): S5 157k muestras/s (1024 partidos); S6 152k/s con una liga de 35 miembros (512 partidos) y ~95–100k/s mientras entrena un exploiter en paralelo (el exploiter, ~60k/s).
- **Exploiters**: el supervisor lanzó tres, cada uno se evaluó contra su principal y se descartó (0,500: con redes al azar nadie saca el inicial y todos los partidos quedan 0-0 trabados). El circuito completo (copia del principal, entrenamiento, evaluación, `exploiters.json`, incorporación) funciona.
- **Compuertas nuevas sobre el checkpoint de humo S1/S2**: S2 no aprobada (duelos 0,542; 1v1 contra L5 0,891, así que el 1v1 de RS-Pro es débil incluso en L5 y la batería de duelos es la que decide); S5 no aprobada (0,328 contra L5; saques propios tarde 8,2% de 110, medianas dentro de la banda humana; detectores fuera de banda); S6 y S7 no aprobadas (0,328, piso 0,70). Antes de las correcciones, S7 y la compuerta de saques lo aprobaban.
- **Calibración** recalculada con la huella nueva: las referencias de RS-Pro son idénticas a las anteriores (los cambios no tocaron las baterías).

### 9.2 Hallazgo durante el entrenamiento largo: S3 aprobado sin pases (2026-10-04)

El entrenamiento largo arrancó con la auditoría aprobada y avanzó rápido: S1 y S2 aprobaron en la primera evaluación posible (control 0,90 contra 0,83 de L5; duelos 0,87 contra 0,43) y S3 a los 375M (0,92 contra 0,47). Al revisar repeticiones de cada etapa:

- **S2**: el 100% de ataque 1v1 venía de desbordar al defensor y definir al arco vacío. El defensor RS-Pro, siendo el último hombre, salía a presionar al contacto y un regate en diagonal lo dejaba pasado (en el momento del tiro nunca estaba entre la pelota y el arco).
- **S3**: en 40 episodios contra L5, sólo 1 gol vino de un pase entre los dos aprendices. En 2v1, pared, pase en profundidad y 2v2, el poseedor gambeteaba solo. Con dos defensores, el presionante pasado seguía siendo "el que llega antes" y el segundo se quedaba cubriendo sin salir nunca.
- **La compuerta no lo detectó** porque medía el éxito y no cómo se lograba. La etapa existía para enseñar el pase, y pasar poco fue uno de los fallos de v3.

Corrección (el entrenamiento se pausó a los 415M, en S4):

- **RS-Pro**: último hombre que contiene y relevo defensivo (§5). Con los mismos checkpoints, el desborde individual cayó de 100% a 42–48% en 1v1, de 8/8 a 1/8 en 2v1 y de 7/8 a 0/8 en 2v2; los ejercicios siguen siendo resolubles pasando (RS-Pro L5 atacando: 2v1 0,44, pared 0,36, 98–99% con pase). Escalera, estructura y tramposos revalidados (arriba). Tests de regresión: el último hombre no va al contacto, el presionante con cobertura sí, y el relevo.
- **Compuerta de S3**: además del éxito, en 2v1 y pared ≥50% de los goles deben incluir un pase entre aprendices. El checkpoint de S3 viejo la reprueba (30–33%).
- **Reanudación**: desde el checkpoint de S1 (ejercicios 1v0, sin defensores), rehaciendo S2–S4 (~1,5 h de GPU).

Segunda revisión (S2 retomado, a los ~25M de la etapa), dos fallos más:

- **Control de dificultad oscilante**: la dificultad se movía ±0,01 por episodio con un promedio de ~50 episodios. Con 1024 partidos terminan cientos por iteración, así que saltaba de 0 a 1 y de vuelta en dos iteraciones, alternando rivales L0 (que no contienen) y L4. El "éxito 0,85" del log venía de los L0. Ahora se actualiza una vez por rollout con el éxito del lote y un paso de ±0,05 como máximo (test: converge al 60% sin oscilar).
- **Engaño de soltar la pelota**: el atacante se alejaba de la pelota para que el último hombre saliera a buscarla y después se la robaba y lo desbordaba. Funcionaba en dos capas:
  - la regla "pelota suelta" de la contención: se quitó;
  - el último hombre salía a ganar cualquier pelota a la que llegara primero, aunque fuera por un tick, y RS-Pro protege mal la pelota recién ganada.
- **Correcciones**:
  - el último hombre sale a ganar la pelota sólo con 20 ticks de ventaja; si no, contiene;
  - en los ejercicios de ataque, si el rival toca la pelota sin ningún aprendiz a menos de 60 px, la ganó limpio y el ataque terminó (las disputas siguen la regla de control sostenido).
  - Con el mismo checkpoint, el ataque 1v1 contra L5 cayó de 0,29 a 0,04, y no quedan goles por engaño.
- **1v1 completo**: RS-Pro contra RS-Pro termina 0-0 en 94–98% de los partidos (su ataque 1v1 es muy débil). La red gana el 1v1 contra L5 (0,79) recuperando la pelota cuando RS-Pro ataca, así que la compuerta de S2 es alcanzable; en 4v4 RS-Pro tiene pases y apoyos y su estructura está validada.
- **Calibración** nueva con huella `c31a5318ce90e236`: 2v1 0,31 y pared 0,31, 100% con pase; defender 1v1 0,98.
- **Lección**: el RL encuentra en minutos cualquier regla rígida del scripted. Revisar repeticiones y la geometría de los éxitos en cada etapa (no sólo la tasa) es parte del procedimiento, no un extra.

Tercera revisión (corrida nocturna, 2026-10-04): S2 aprobó (duelos 0,855; 1v1 contra L5 1,00) y **S3 agotó su presupuesto sin aprobar**. Éxito 0,93–1,00 en 2v1, pared, profundidad y 2v2 a dificultad máxima, pero con 0 pases en las nueve evaluaciones: la regla de pases lo frenó, como debía.

- **Causa**: en zona de tiro (< 430 px) el último hombre iba al contacto, y un corte en diagonal lo dejaba pasado. La red ganaba así el 1v1 contra L5 el 81–95% y el 2v1 el 97–100%, sola.
- **RS-Pro**: en zona de tiro el último hombre tapa el ángulo a 30–50 px del lado del arco, leyendo la conducción del poseedor (su velocidad), y sólo mete la pierna si la pelota le queda al alcance.
  - Con la misma red, en solitario contra L5: 1v1 de 81–95% a 1–8%; 2v1 de 97–100% a 3–14%.
  - Contra L4 y L3 sigue siendo posible (32–79%).
  - RS-Pro L5 atacando con pases gana el 2v1 0,26–0,29 contra cualquier nivel, así que contra L5 pasar rinde más que gambetear.
- **Ejercicios**: la pared y el pase en profundidad exigen un pase entre aprendices antes del gol (la tarea es el pase; un gol en solitario no la cumple). El 2v1 y el 2v2 siguen abiertos.
- **Compuerta de S3**:
  - la batería de pases usa defensores L5 (contra L3 gambetear sigue siendo razonable y "elige pasar" era ambiguo);
  - el 2v1 exige ≥50% de goles con pase;
  - referencia: 2v1 0,30 y pared 0,27, 100% con pase;
  - calibración `08ec29997c680485`.

## 10. Riesgos e incertidumbres abiertas

1. **RS-Pro no es "profesional" en todo** (sección 5): su posesión y su 1v1 ofensivo son débiles. Riesgo: en S5 el agente podría sobreajustarse a explotar esas debilidades.
   - Mitigación: estilos variados, región reservada, espejo y PFSP desde S5. La meta de S6 se mide contra el estilo reservado e históricos, no sólo contra RS-Pro.
2. **Aprendizaje desde cero en S1**: la señal inicial es escasa (un aprendiz al azar toca la pelota en ~5% de los episodios). Ver el humo; si es lento, se sube Φ_ball (PBRS, seguro) sólo en S1.
3. **Latencia**: es un modelo aproximado del cliente. Falta medirla en sala con el bot nuevo.
4. **Competencia con humanos**: no queda demostrada hasta jugar con humanos. El modo rs4z de `deploy/bot.js` está pendiente (el contrato ya está portado y probado).
5. **Presupuesto** (medido): S1 ~55k muestras/s, S2 ~80k, S5 ~157k, S6 ~150k con la liga llena (≈100k mientras corre un exploiter); S3–S4 estimadas en ~100–120k. El plan completo (13,3B) lleva ~27 h de GPU sin evaluaciones; con exploiters (~9 de 200M) y ~56 evaluaciones de compuertas, ~32–36 h, dentro de las 50–100 h disponibles. Sobra margen para 2 semillas en S1–S3 (~+5 h).
6. **Ruido entre semillas**: S1–S3 están previstas con 2 semillas; las compuertas usan IC.
7. **Grabaciones**: la xT y las situaciones usan 45 grabaciones de entrenamiento. La partición de prueba no se usó para nada.

## 11. Por qué esta configuración no repite los fallos

| Fallo previo | Mecanismo nuevo |
|---|---|
| Simulador distinto del real | F1–F22 corregidos; conformidad de 1,39 M transiciones; contrato versionado |
| 0-0 eterno entre redes | ejercicios con éxito explícito, xT al inicio, RS-Pro que ataca, resultado de suma cero, detector de 0-0 |
| Bloque defensivo por la guía táctica | sin rewards de formación ni tácticas |
| Pasividad (BC/KL) | sin imitación ni anclas; inicio aleatorio |
| Delantero colgado por la multa anti-0-0 | sólo términos de suma cero; detector de colgado como compuerta |
| Córners trabados | el vencimiento entrega el saque al rival |
| Desconexión con compañeros congelados | parámetros compartidos, ≤20% de compañeros scripted, crítico centralizado |
| Ruido entre semillas | compuertas con IC, 2 semillas en S1–S3, muestras grandes |
| Sobreajuste al scripted o a familias conocidas | estilos continuos con región reservada, PFSP, exploiters |
| Reloj en el actor / latencia ignorada | actor sin reloj; latencia por jugador e historial de acciones |
| Ball-chasing y aglomeración | Φ_access no premia perseguidores extra; RS-Pro castiga los espacios; detector y contingencia |

## Cómo lanzar el entrenamiento largo (requiere aprobación)

```bash
python -m tools.pod_sync --extra reports/rs4z/gates.json reports/rs4z/xt.json
ssh -p 40900 <pod> "cd /workspace/HaxballRL && nohup .venv/bin/python -u -m tools.rs4z_supervisor --run rs4z_main --envs 1024 > pod_logs/rs4z_main_supervisor.log 2>&1 &"
```

- El supervisor corre el principal (`pod_logs/rs4z_main.log`, reanudable con `--resume`) y, en S6–S7, un exploiter cada 1B muestras (`pod_logs/rs4z_main_exploiter_KK.log`).
- Las compuertas se evalúan cada 10% del presupuesto de cada etapa y quedan en `runs/rs4z/rs4z_main/gates.jsonl`; la evaluación se niega a correr si el código no coincide con la calibración congelada.
- Si una etapa agota su presupuesto sin aprobar, el principal se detiene con el motivo en el log (no avanza a la fuerza).
