# Auditoría previa al entrenamiento RS4-Z (RS4 4v4 desde cero)

Fecha: 2026-10-03. Rama `rs4z`. Plan aprobado: `~/.claude/plans/fallamos-catastroficamente-en-muchas-agile-whisper.md`.

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
| Entrenador | PPO + crítico centralizado (MAPPO), inicio aleatorio, liga PFSP, compuertas | 66 tests RS4-Z; humo en Pod (sección 9) |
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
- **Defensa**: presión pegada del lado del arco con quite seguro, atajada en la línea de gol, seguridad que cubre el arco con la pelota cerca, despejes que nunca apuntan al arco propio.
- **Ejecución**: navegación de órbita alrededor de la pelota (no la toca al rodearla) y frenado por velocidad deseada.
- **Saques**: ejecutor por costo, espera de apoyos, impulsos reales del córner (×1,98) y del saque de arco (×2,71).

Validación (`reports/rs4z/rspro_ladder.json`, 256 partidos de 3 min por par, ambos colores, estilos al azar):

| Par | Puntos del nivel superior (IC95) |
|---|---|
| L1–L0 | 0,54 (0,50–0,58) |
| L2–L1 | 0,67 (0,64–0,71) |
| L3–L2 | 0,65 (0,61–0,69) |
| L4–L3 | 0,57 (0,53–0,61) |
| L5–L4 | 0,58 (0,55–0,62) |
| L5–L3 / L2 / L1 / L0 | 0,65 / 0,77 / 0,90 / 0,92 |

Estructura L5 contra L5 frente a humanos (69 grabaciones, mismas definiciones): dispersión 149 (humano 166), profundidad 324 (362), anchura 254 (260), distancias a la pelota 109/221/299/414 (96/203/283/392), colgado 0,7% (0,6%), aglomeración 6% (4%), flotación junto al arco 0%. Saques: laterales 117 ticks (117), córners 159 (153), saques de arco 192 (249).

Baterías de referencia (L5 en los lugares del aprendiz, 512 episodios): control 0,92/0,67/0,83/0,91 (tocar, arco vacío, conducir y definir, recibir y definir); defensa 2v2/4v4/saques 0,83/0,82/0,85; 2v1 0,53; pared 0,41.

**Límites conocidos de RS-Pro** (no invalidan el aprendizaje, porque el RL debe superarlo y luego lo reemplaza el self-play):
- pasa menos y pierde más la pelota que los humanos (pases/pérdidas 0,47 contra 1,15);
- tira menos (0,18 contra 0,47 por minuto), aunque convierte igual que ellos;
- su ataque 1v1 contra un defensor pegado es débil (0,2%); el pase en profundidad y los centros también (≤7%);
- la separación entre L3, L4 y L5 es moderada (57–65%).

Las compuertas sólo juzgan tareas donde la referencia resuelve ≥15%.

## 6. Curriculum (`train/rs4z/stages.py`)

| Etapa | Tareas | Muestras (techo) | Rewards | Compuerta (pre-registrada) |
|---|---|---|---|---|
| S1 control | 1v0: tocar, arco vacío, conducir y definir, recibir y definir | 0,3B | gol/ejercicio + Φ_ball (se retira) + Φ_threat | control ≥ 0,90×L5 (cada tarea ≥ 0,75×) |
| S2 pelota y 1v1 | tiro contra último hombre, 1v1 ataque/defensa, pelota dividida, 1v1 completo | 0,5B | + Φ_access | duelos ≥ 0,85×L5; 1v1 contra L3 ≥ 55% |
| S3 tiro, pase, rebotes | 2v1, pared, profundidad, desvío, 2v2 ofensivo (2 aprendices) | 0,7B | idem | pases ≥ 0,80×L5 |
| S4 defensa | 2v2, 4v4 y saques en contra; 2v2 completo | 0,8B | Φ_access → 0 | defensa ≥ 0,80×L5, goles recibidos ≤ 1,1×L5; detectores 2v2 |
| S5 cooperación 4v4 | 4v4 contra la escalera, 4v3, 4v2, 3v4, estados humanos, saques, transiciones; 30% espejo | 2,5B | + resultado ±0,3; Φ_threat → 0 | ≥60% contra L5 (IC90 >50%), ≥50% contra el estilo reservado, saques vencidos ≤2%, detectores 4v4 |
| S6 liga | espejo 45%, PFSP 25%, RS-Pro 20%, exploiters 10%; 15% compañeros scripted | 7,5B | sólo goles y resultado | ≥75% contra L5, ≥65% contra el reservado |
| S7 robustez | latencia de cliente 70%, compañeros ad-hoc, variantes | 1,0B | idem | pérdida ≤10 pp con 8 ticks de latencia |

- Repaso de etapas anteriores ≥10% (5% desde S6).
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
  - exploiters marcados en su propia fracción: `train.rs4z.run --exploiter-of <checkpoint principal>` parte del principal, juega sólo contra él congelado y se suma a su liga en el siguiente guardado;
  - RS-Pro L3–L5 con estilos de entrenamiento;
  - la región de estilos reservada (presión alta + directo + estrecho) nunca se muestrea al entrenar.
- Compañeros ad-hoc (15–20% en S6–S7): RS-Pro competente que infiere los roles de los compañeros que no controla.

## 9. Evaluaciones y evidencia de la infraestructura

- **Métricas comunes a la simulación y a los humanos** (`eval/rs4z/metrics.py`; referencia `reports/rs4z/human_reference.json`):
  - pases y pérdidas, tiros, goles;
  - estructura y perfil de distancias;
  - detectores de aglomeración, colgado, flotación junto al arco, quietud y oscilación;
  - duración de saques.
- **Baterías reproducibles** (`eval/rs4z/batteries.py`): mismas semillas para el candidato y la referencia. Las compuertas viven en `eval/rs4z/gates.py` y la calibración congelada en `reports/rs4z/gates.json` (huella de código `d7b610936c5cf1de`).
- **Tests**: suite completa 859 aprobados, 44 omitidos (793 anteriores + 66 RS4-Z en `tests/rs4z`: paridad del árbitro, reglas v2, obs y simetría, ejercicios, rewards y tramposos, RS-Pro, entrenador).
- **Tramposos contra RS-Pro L5** (128 partidos cada uno): quieto 1,00; oscilar 1,00; todos persiguen 0,95; pelotazos 0,99; sacarla afuera 0,81; bloque en el arco 0,98; delantero colgado 0,98; trabar saques 0,95; azar 0,99 (puntos de L5).
- **Humo en Pod** (RTX 3090, S1, 1024 partidos): ver el bloque de resultados al final.

## 10. Riesgos e incertidumbres abiertas

1. **RS-Pro no es "profesional" en todo** (sección 5): su posesión y su 1v1 ofensivo son débiles. Riesgo: en S5 el agente podría sobreajustarse a explotar esas debilidades.
   - Mitigación: estilos variados, región reservada, espejo y PFSP desde S5. La meta de S6 se mide contra el estilo reservado e históricos, no sólo contra RS-Pro.
2. **Aprendizaje desde cero en S1**: la señal inicial es escasa (un aprendiz al azar toca la pelota en ~5% de los episodios). Ver el humo; si es lento, se sube Φ_ball (PBRS, seguro) sólo en S1.
3. **Latencia**: es un modelo aproximado del cliente. Falta medirla en sala con el bot nuevo.
4. **Competencia con humanos**: no queda demostrada hasta jugar con humanos. El modo rs4z de `deploy/bot.js` está pendiente (el contrato ya está portado y probado).
5. **Presupuesto**: a ~55–65k muestras/s en S1, el plan completo (13,3B) lleva ~60–70 h de GPU. Las etapas con 4–8 aprendices por partido deberían rendir más por muestra; se mide al llegar.
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
python -m tools.pod_sync --extra reports/rs4z
ssh -p 40900 <pod> "cd /workspace/HaxballRL && nohup .venv/bin/python -u -m train.rs4z.run --run rs4z_main --envs 1024 --device cuda > pod_logs/rs4z_main.log 2>&1 &"
```
