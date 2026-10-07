# Pase como macro-acción X4 (pre-registro)

Pre-registro del 2026-10-07, escrito antes de lanzar. Corre en `runs/x4_pase` (pod A4000).

## 1. Por qué

- **Objetivo del usuario:** un bot competitivo y difícil de ganar contra humanos. Contra humanos, el juego individual no
  alcanza: en real soccer no se le gana a un humano 1v1.
- **Lo que ya se probó:** ninguna palanca de recompensa dio juego colectivo. Ni el bonus de pase, ni las franjas, ni el
  estilo humano (`docs/PROXIMA_CORRIDA.md` §7). Tampoco la liga barata (`docs/LIGA.md` §5).
- **Cómo está el agente hoy en duelos 1v1:** pasa el 10–13% (los humanos, el 22%) y pierde la pelota el 61–64% (los
  humanos, el 49%).
- **Por qué no aprende a pasar** (`docs/PLAN.md` §0.1): con controles crudos el pase aparece recién a ~1e10 pasos.
  Pasar es motricidad: ubicarse detrás de la pelota en la línea justa y patear en el tick justo, con 9–12 ticks de
  latencia. En Google Research Football y en TiZero, en cambio, el pase es una acción del juego.
- **La hipótesis:** si "pasarle al compañero k" es una acción, la política sólo tiene que decidir a quién y cuándo, y
  eso sí lo puede aprender el RL a nuestra escala.

## 2. El controlador y la prueba previa

`learn/x4_pass_macro.py`. El controlador:

1. predice al jugador con las teclas que ya están en cola (exacto contra el kernel: 0,0000 px de error; sin la cola,
   1,7–3,7 px);
2. busca entre las 18 acciones simulando 9 ticks;
3. patea sólo si la patada simulada sale a menos de 8° del compañero, adelantado según su velocidad.

Prueba (`tools/x4_pass_probe.py`): 256 momentos humanos con la pelota controlada, retardo de sala de 8 a 11 ticks, el
resto de los jugadores manejados por el control 2000. El receptor es el compañero más libre.

| | Patea | Completa lo que patea | Error | Pase total | Giro 0–45° | 45–90° | 90–180° |
|---|---|---|---|---|---|---|---|
| Controlador reactivo | 42% | 58% | 7,1° | 24% | 42% | 20% | 9% |
| **Controlador predictivo (8°)** | 33% | **68%** | **5,5°** | 22% | **46%** | 16% | 2% |
| Predictivo con 12° | 33% | 59% | 7,2° | 20% | 40% | 14% | 2% |
| Agente solo | – | – | – | 21% | 25% | 21% | 16% |

Precisión humana: 0,69. Con giro > 90° pierde la pelota antes de patear (79%). Por eso el pase sólo se ofrece con giro
≤ 90° y la pelota a ≤ 40 px del borde del jugador.

## 3. Diseño de la integración (`--pass-macro`)

| Pieza | Valor | Por qué |
|---|---|---|
| Acciones | 18 teclas + 3 "pasarle al compañero k" (orden de la obs v3) | el pase de GRF |
| Cabeza | compuerta p(pasar) = σ(g), con sesgo inicial −3 y peso 0; a quién, con un puntaje por compañero sobre su embedding (puntero) | arranca igual que la política de 18; el conjunto de compañeros no tiene orden fijo en la red |
| Máscara | pase ofrecido sólo con la pelota a ≤ 40 px y giro ≤ 90° (`pass_table` = −1 si no) | fuera de eso el controlador falla (§2) |
| Ejecución | compromiso: el controlador maneja al jugador hasta que patea, deja de ofrecerse o pasa 1 s (20 decisiones) | como una acción de GRF. Las decisiones intermedias no entran en la pérdida: la acción de la política fue empezar el pase |
| Ancla | KL(BC‖π_ef), donde π_ef es la distribución efectiva de teclas (cada pase suma su probabilidad a la tecla que ejecuta ahora) | la BC no tiene pases; así el ancla mide teclas, como antes |
| Rivales | igual que el control: 20% pool (BC y snapshots, con PFSP), 80% self-play; los snapshots con pases también los usan | |

## 4. A/B (decidido antes de ver resultados)

Dos ramas en paralelo en el A4000, desde `runs/x4_estilo/rl_control/best.pt` (actualización 2000). La configuración es
la del control (goles ±1, −0,1 por saques perdidos con razones 1–4, λ fijo 0,05, sin shaping, sin bonus, sin estilo,
40% de arranques humanos, 20% pool). La única diferencia es `--pass-macro`.

- `runs/x4_pase/control`: sin macro.
- `runs/x4_pase/pase`: con macro.

Las dos con 1500 actualizaciones de 1024 × 64 (~100M decisiones de partido), la misma semilla y evaluación cada 100.

**Veredicto** (con `last.pt` de cada rama):

1. **Primario:** enfrentamiento directo pase contra control, 128 partidos por lado (256), 3 min, retardo 8–11,
   muestreando. "El pase ayuda" si el puntaje de pase es ≥ 0,55. El error es de ~±0,03.
2. **Sin pérdida de fuerza:** pase contra la BC ≥ 0,8.
3. **Como dato:** pases por minuto y precisión en self-play contra humanos (cadena de pase), cuántas macros empieza y
   cuántas terminan en patada, y el enfrentamiento de cada rama contra el punto de partida.

Si se cumplen 1 y 2: portar el controlador al bot de sala (JS, con `pass_mask` en el ONNX) y probar contra humanos. Si
no: anotarlo, ver si la política usa los pases (si no los usa, el problema es de exploración y no del controlador), y no
seguir por este lado sin una idea nueva.

**Límite:** es simulación contra versiones del mismo agente. El controlador usa el estado verdadero con el retardo
verdadero; en la sala el bot estima su retardo, y un error ahí afecta la puntería. La prueba de verdad es la sala.

## 5. Medición de conducción (2026-10-07, durante el A/B)

Pregunta: ¿la política no pasa porque en self-play conducir no se castiga como lo castiga un humano?
`tools/x4_dribble_probe.py`, 1024 momentos humanos de test 4v4 con un portador claro, 5 s, retardo 8–11. Un equipo lo
maneja el atacante y el otro el defensor; la referencia humana es lo que pasó en la grabación desde los mismos estados.
Agente = `runs/x4_estilo/rl_control/best.pt`. Error de ~±0,015 en el total y ~±0,025 por rango de presión.

| Primer evento del portador | Humanos vs humanos | Agente vs agente | Agente vs BC | BC vs agente | BC vs BC |
|---|---|---|---|---|---|
| Pasa (lo toca un compañero) | **0,41** | 0,18 | 0,24 | 0,20 | 0,23 |
| Pierde conduciendo | 0,14 | 0,15 | 0,11 | 0,24 | 0,20 |
| Patea y la toma un rival | **0,36** | 0,50 | 0,45 | 0,39 | 0,38 |
| Sale (lateral, córner, saque de arco) | **0,05** | 0,14 | 0,14 | 0,14 | 0,14 |

Con presión (rival a < 90 px), pierde conduciendo: humanos 0,20, agente contra agente 0,26, BC contra agente 0,37. Sin
presión (> 160 px): humanos pasan 0,49 y patean a un rival 0,36; el agente, 0,22 y 0,60.

Lectura:
- **La hipótesis no se confirma.** El agente como defensor castiga conducir más que la BC y que los humanos entre sí
  (el agente pierde conduciendo contra sí mismo más que un humano contra humanos). Entrenar contra defensores tipo BC
  castigaría menos, no más.
- **La brecha está en las patadas:** cuando el portador patea, la pelota termina en un rival o afuera mucho más que con
  humanos, sobre todo sin presión, que es justo cuando un humano arma el pase. La BC tiene el mismo problema (0,23 de
  pases y 0,14 de salidas): no viene sólo del RL.
- Falta separar decisión (a dónde patea) de ejecución (puntería con latencia). El controlador de pase ataca la
  ejecución y, sin presión, tiene tiempo de acomodarse.

## 6. Mecánicas humanas: rifle, usar, cuerpear (2026-10-07)

El usuario aclaró que el problema es de decisión y describió tres mecánicas propias de HaxBall:

- **cuerpear:** usar el cuerpo para tapar al rival y conservar la pelota hasta que llegue un apoyo;
- **usar:** patear contra un rival buscando un saque a favor, un rebote propio o que su patada la mande a favor;
- **rifle:** dos compañeros llegan desde +y y −y, patean casi juntos y la pelota sale horizontal.

`tools/x4_mechanics.py` las detecta con la misma definición en grabaciones (test, 293 min de juego abierto) y en
self-play (48 partidos de 3 min por política). Las definiciones son una traducción mía de las del usuario y falta
validarlas con ejemplos.

| | Humanos | BC | Agente (control 2000) | A/B control (act. ~450) | A/B pase (act. ~330) |
|---|---|---|---|---|---|
| Rifle por min | 0,05 | 0,02 | 0,05 | 0,02 | 0,03 |
| Usar por min | 3,1 | 2,4 | 2,7 | 2,2 | 2,8 |
| Usar: resultado favorable | 0,44 | 0,38 | 0,41 | 0,42 | 0,44 |
| Cuerpear por min | **3,6** | 2,1 | 2,1 | 2,0 | 1,7 |
| Cuerpear: conserva (apoyo, patea, saque a favor) | **0,54** | 0,32 | 0,40 | 0,42 | 0,36 |
| Cuerpear: termina en saque en contra | **0,05** | 0,14 | 0,13 | 0,14 | 0,15 |

Lectura:
- **Rifle:** con esta detección es raro en humanos (14 en 293 min) y en el agente. O es raro en Sanguchito o el
  detector no lo agarra (por ejemplo, si la segunda patada llega más de 6 ticks después). Hay que validarlo.
- **Usar:** el agente lo hace casi tan seguido y con casi el mismo resultado que los humanos.
- **Cuerpear:** es la diferencia clara. Los humanos lo hacen 1,7 veces más seguido y conservan la pelota más (0,54
  contra 0,40). Cuando el agente cuerpea, termina en saque en contra casi 3 veces más (0,13 contra 0,05): coincide con
  las salidas de la §5. La BC tampoco lo aprendió (2,1 por min, conserva 0,32): no lo trae la imitación.

## 7. Resultado del A/B (2026-10-07 22:10 UTC): sin mejora

1500 actualizaciones por rama; enfrentamientos de 256 partidos (128 por lado), 3 min, retardo 8–11
(`runs/x4_pase/veredicto.json` en el pod).

| Enfrentamiento | Puntaje | G-E-P | Goles |
|---|---|---|---|
| pase contra control | **0,539** | 84-108-64 | 145-114 |
| pase contra BC | 0,856 | 186-66-4 | 318-15 |
| control contra BC | 0,852 | 183-70-3 | 339-6 |
| pase contra el punto de partida | 0,471 | 75-91-90 | 144-159 |
| control contra el punto de partida | 0,502 | 74-109-73 | 132-133 |

- Criterio 1 (pase ≥ 0,55 contra control): **no** (0,539, dentro del error de ±0,03). Criterio 2 (≥ 0,8 contra BC): sí.
  Veredicto: **sin mejora**.
- La política dejó el macro apenas empezó a entrenar (950 pases empezados por actualización con la compuerta inicial →
  ~5–15 desde la actualización 100). Más de la mitad de los pases empezados no llegaban a patear.
- Ninguna rama mejoró sobre el punto de partida en 1500 actualizaciones (0,47 y 0,50): con esta configuración el
  agente está estancado.
- Después del A/B apareció un error del controlador (§8): al rodear la pelota se quedaba quieto al costado. Afectaba
  sobre todo los giros grandes, así que el A/B midió una versión con un defecto.
