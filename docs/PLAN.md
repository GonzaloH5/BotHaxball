# Plan: juego colectivo de nivel humano en HaxBall RS 4v4

Borrador del 2026-10-06, pendiente de aprobación. Reemplaza todo el método anterior, que está archivado.

**Objetivo:** bots que jueguen 4v4 Real Soccer en las salas reales (HAXARG 2K23 y SANGUCHITO RS X4). Tienen que hacerlo con juego colectivo (pases, apoyos, estructura) y a nivel competitivo contra humanos, con la latencia real de la sala, de unos 15 ticks.

**Regla del plan:** cada decisión cita trabajos publicados.
- Lo que es deducción nuestra va marcado **[inferencia]**.
- Fuentes: la revisión bibliográfica del 2026-10-06, con unos 50 papers con el título comprobado en el PDF. Se citan por su arXiv ID.

## 0. Qué dice la literatura y qué implica para nosotros

1. **Desde cero, el RL multiagente de fútbol converge a juego individual.**
   - Un jugador encara y los demás hacen de señuelo, o se queda en posesión sin avanzar (Song 2023, 2309.12951).
   - El pase emerge muy tarde: Liu 2019 lo ve recién a 80e9 pasos (1902.07151). Liu 2022 llega a 6% de toques que son pases a 8e10 pasos (2105.12196). TiZero tardó 45 días con 800 CPUs (2302.07515).
   - Nuestro presupuesto es de 1e8–1e9 decisiones (1× RTX 3060, unos 8 cores efectivos), así que **el pase no va a emerger solo**.
2. **A escala chica rinde partir de imitación humana y mantener un ancla KL a esa política durante el RL.**
   - Lo respaldan AlphaStar (Nature 575), VPT (2206.11795), HR-PPO (2403.19648, λ=0.06 con 30 min de datos) y DiL-piKL (2210.05492).
   - BC→RL sin ancla "olvida" lo aprendido en los estados que no visita al principio (Wołczyk 2024, 2402.02868). Lo corrigen el kickstarting (1803.03835) y el replay o los arranques desde estados de demostración (1812.03381, 1807.06919).
3. **Shaping:** el de progreso, chico y con tope (GRF checkpoint, 1907.11180) ayuda.
   - El shaping de estilo de equipo tiene evidencia controlada en contra: en Liu 2022, "pelota hacia el compañero" y "territorio" rindieron peor.
   - TiZero usa +0.05 por pase, pero sin ablación.
4. **Latencia:** con delay constante, aumentar el estado con las acciones propias pendientes da un MDP exacto (Katsikopoulos & Engelbrecht 2003).
   - Se entrena con delay aleatorizado y crítico asimétrico sin retardo.
   - Opcional: un maestro sin retardo destilado en el alumno retardado (DIDA, 2205.05569).
5. **Evaluación:**
   - fuerza contra un pool fijo y diverso, con TrueSkill o Nash averaging (1806.02643);
   - parecido humano con métricas distribucionales contra los logs (WOSAC, 2305.12032) y con precisión prediciendo acciones humanas (Maia, 2006.01855);
   - todo medido con delay de sala.

## 1. Base del simulador (estado al 2026-10-06)

| Mapa | Física | Script de la sala | Grabaciones humanas |
|---|---|---|---|
| RS ONE | verificada (conformidad p90 a 60 ticks ~0,01 px) | verificado | 147 |
| SANGUCHITO RS X4 | verificada (p90 a 60 ticks 0,012 px) | verificado: 101/101 saques con pelota ≤ 0,05 px a 59 ticks (`reports/sanguchito_conformance.md`) | 7 |
| HAXARG 2K23 | verificada en 1 replay | **supuesto** igual a RS ONE desplazado 70 px | 2 |

**Pendiente antes de entrenar:**
- **B1.** Validar el script de 2K23 contra sus 2 grabaciones con `tools.rs4z_restart_conformance`, igual que se hizo con Sanguchito.
- **B2.** Medir la distribución de latencia en sala por bot (no solo la media de 15 ticks). La latencia en entrenamiento se muestrea de ahí (§5 de la revisión; Tan 2018, 1804.10332).

## 2. Etapas

Cada etapa tiene un criterio de avance pre-registrado. No se avanza por presupuesto agotado sin decisión explícita del usuario.

### E0. Evaluación antes que entrenamiento
Fuentes: Liu 2019, Balduzzi 2018, WOSAC, Maia.

- **Pool fijo de rivales:**
  - BC humano;
  - snapshots de E3 a medida que existan;
  - variantes con distintas latencias.
- **Fuerza:** TrueSkill o Nash averaging contra el pool, en greedy y en muestreado. Nunca un solo rival.
- **Parecido humano:** distancia distribucional contra las grabaciones de test, en:
  - pases/min, largo de pase y pases por pérdida;
  - profundidad, ancho y rangos de distancia a la pelota del equipo;
  - % de jugadores parados;
  - duración de la posesión;
  - ejecución de saques.
- **Imitación:** precisión prediciendo la acción humana en test (tipo Maia).
- **Sondas de coordinación:** 2v1 y 3v2 desde estados humanos grabados (Liu 2019).
- Todo se mide con delay de sala.

**Criterio:** las métricas reproducen a los humanos de test contra los humanos de train. Es el techo de referencia.

### E1. Imitación humana con latencia correcta
Fuentes: AlphaStar SL, HR-PPO, TiKick (2110.04507).

- **Datos:** todas las grabaciones 4v4, separadas por sesión. El mapa entra como condición (§6 de la revisión).
- **Arquitectura:**
  - política compartida por los 4 jugadores, con encoder de conjuntos para compañeros y rivales (Deep Sets 1703.06114 o atención; MARLadona 2409.20326; AlphaStar);
  - observación relativa a la geometría del mapa;
  - estado aumentado con las acciones propias pendientes.
  - Sin marcador ni reloj en el actor (preferencia del usuario); van solo al crítico.
- **Latencia de los humanos:** **[inferencia]** cada jugador grabado juega con su propio ping. Se estima el retardo por jugador como el que maximiza la verosimilitud de sus acciones, y la observación de BC se arma con ese retardo.
- **Criterio:** precisión de acción en test por encima de la línea base.
  - Se valida en closed-loop sabiendo que va a derivar (HR-PPO lo reporta).
  - La BC es **ancla y prior**, no rival.

### E2. Precalentar el crítico
**[inferencia]**, motivada por Wołczyk 2024.

- Crítico central asimétrico con el estado verdadero sin retardo (MAPPO 2103.01955; AlphaStar), entrenado con la BC congelada jugando antes de mover el actor.

### E3. RL desde la BC con ancla KL
Fuentes: HR-PPO, VPT, AlphaStar, DiL-piKL, MAPPO.

- **Algoritmo:** MAPPO con parámetros compartidos, valor normalizado, 5–10 épocas, clip ≤ 0,2 y batch grande.
- **Pérdida:** (1−λ)·L_PPO + λ·KL(BC‖π).
  - Barrido corto λ ∈ {0,02; 0,06; 0,1}.
  - Opción DiL-piKL: λ muestreado por partido.
- **Recompensa** (GRF, MARLadona, OpenAI Five 1912.06680):
  - gol ±1, suma cero;
  - shaping de progreso con tope, retirado al 75% contra el pool;
  - team spirit hacia 1,0.
  - El pase no se premia por defecto (Liu 2022). Si a 100–300M decisiones el pase cae por debajo del criterio, se prueba **como brazo pre-registrado** el término de TiZero (+0,05 por pase exitoso antes de un gol).
- **Oponentes** (TiZero, AlphaStar):
  - 80% contra versiones recientes y 20% contra el pool con PFSP (1−x)², incluida la BC;
  - un explotador barato cada 200–500M decisiones.
- **Arranques:** 30–50% de los episodios desde estados humanos grabados (juego abierto y saques), más arranques escalonados (1812.03381, 1807.06919, TiZero).
- **Variabilidad:**
  - delay aleatorio con la distribución de sala;
  - mezcla de RS ONE, 2K23 y Sanguchito con variaciones de geometría (Tobin 1703.06907, Cobbe 1812.02341).
- **[inferencia] Vigilar desde el día 1** los estados donde los humanos esperan, como el saque inicial: el ancla puede enseñar a quedarse quieto en greedy.
- **Criterio:** fuerza ≥ BC + margen contra el pool y parecido humano dentro del rango test-vs-train, todo a delay 15.

### E4. Sala real
- Solo cuando E3 pasa sus criterios en simulación.
- Prueba con el usuario en 2K23 y en Sanguchito, con grabación.
- La grabación entra como dato nuevo y como prueba de conformidad.

## 3. Qué no se va a hacer y por qué

- Ligas grandes o PBT (AlphaStar, Liu, FTW): fuera de presupuesto.
- MAT (2205.14953): decodifica en forma autorregresiva entre agentes y no encaja con clientes independientes con latencias distintas.
- Métodos value-based (QMIX/VDN): fallan en partido completo (Song 2023).
- AMP o GAIL como señal principal (2104.02180): tienen mode collapse y nosotros ya tenemos acciones.
- Búsqueda en tiempo real tipo piKL a 20 Hz.
- RL offline puro (TiKick): tenemos unos 156 partidos, no 22 000.

## 4. Cómputo

- **Pod:** 1× RTX 3060 Laptop, 12 GB.
- **CPU:** cuota de 12 CPUs en un host compartido; rinde mejor con 8 hilos numba.
- **Simulador:** ~195k decisiones de partido/s. El cuello real va a ser la inferencia o el PPO; se mide cuando exista el trainer.
- **Presupuesto por corrida:** 1e8–1e9 decisiones.

## 5. Próximos pasos concretos, en orden

1. B1: conformidad del script de 2K23 contra sus 2 grabaciones.
2. B2: distribución de latencia en sala.
3. E0: herramientas de evaluación (fuerza, parecido humano, imitación, sondas).
4. E1: dataset multimapa con observación retardada y política de conjuntos, y BC.
5. E2–E3: trainer MAPPO + ancla KL, con corridas cortas de barrido de λ.
