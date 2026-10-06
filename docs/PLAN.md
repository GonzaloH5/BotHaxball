# Plan: juego colectivo de nivel humano en HaxBall RS 4v4

Borrador del 2026-10-06, revisado esa noche con el dataset ampliado y las mediciones de la sesión autónoma (ver §6, "Cambios respecto del borrador"). Reemplaza todo el método anterior, que está archivado.

**Objetivo:** bots que jueguen 4v4 Real Soccer en las salas reales (HAXARG 2K23 y SANGUCHITO RS X4). Tienen que hacerlo con juego colectivo (pases, apoyos, estructura) y a nivel competitivo contra humanos, con la latencia real de la sala, de unos 9–12 ticks (§1, B2).

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
6. **[agregado] Imitación con historial de acciones propias:** riesgo de "copycat" (de Haan 2019, 1905.11979, "Causal Confusion in Imitation Learning"; Wen 2020, 2010.14876, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories"; título verificado en arXiv el 2026-10-06). El imitador aprende a repetir la tecla anterior en vez de leer el estado.
   - Los humanos mantienen la tecla el 84% de las decisiones: "repetir la tecla de hace 3 ticks" acierta el 84%.
   - Por eso se mide aparte la predicción en los cambios de tecla y se valida siempre en lazo cerrado.

## 1. Base del simulador y de los datos (estado al 2026-10-06, noche)

| Mapa | Física | Script de la sala | Grabaciones humanas (4v4) |
|---|---|---|---|
| SANGUCHITO RS X4 | verificada (p90 a 60 ticks 0,012 px, 7 grabaciones) | **verificado en 498 grabaciones** (`reports/x4/conformance_x4.md`) | **498** (2.822 min) |
| RS ONE | verificada (p90 a 60 ticks ~0,01 px) | **diferencias sin resolver** en saques con el dataset nuevo: el lateral se desvía 3,4 px p90 al liberar; v2 contra v2_lateral (`conformance_x4.md`) | 147 (2.745 min) |
| HAXARG 2K23 | verificada en 1 replay | **no coincide** en 13 saques de 2 pruebas con bots (B1) | 0 (las 2 grabaciones son pruebas con 7 bots) |

Correcciones del script de Sanguchito con el dataset nuevo:
- **Córner:** solo lo libera una patada del ejecutor hacia la cancha en x **y en y**.
- **Córner y saque de arco:** la dirección de la patada se juzga con la posición del pateador **al patear** (antes del tick).
- Con eso, las 2.746 patadas que liberan y las 346 ignoradas se clasifican sin errores.
- Laterales: tipo, punto, empujes y fin coinciden en el 99,7%.

**Pendiente antes de entrenar:**
- **B1.** Validar el script de 2K23 contra grabaciones de esa sala. Con las 2 pruebas con bots (13 saques) el supuesto "RS ONE desplazado 70 px" **no coincide**: la trayectoria de córner y saque de arco se desvía 46/26 px a +59 ticks, y el punto y la liberación del lateral también difieren (`reports/x4/conformance_x4.md`). Hacen falta más grabaciones de 2K23 para ajustar la regla.
- **B2.** ~~Medir la distribución de latencia en sala~~ **Hecho:**
  - dos sesiones con 7 bots dieron 11–12 y 9–11 ticks (mejor desfase por bot) entre el frame observado y el aplicado (`reports/room_latency/`). Se midieron con la contención de ONNX; hay que volver a medir en E4 con el arreglo;
  - en la convención del simulador, el retardo D es igual al lag menos 1 (§2, E1);
  - el entrenamiento sortea D de {8: 0,2; 9: 0,25; 10: 0,25; 11: 0,2; 14: 0,1} (lag − 1).
  - La contención de ONNX con 7 bots en una PC (decidían cada 9–10 ticks) quedó corregida: 1 hilo por bot.

## 2. Etapas

Cada etapa tiene un criterio de avance pre-registrado. No se avanza por presupuesto agotado sin decisión explícita del usuario.

### E0. Evaluación antes que entrenamiento
Fuentes: Liu 2019, Balduzzi 2018, WOSAC, Maia. **Implementado:** `tools/x4_metrics.py`, `learn/x4_eval.py`, `learn/x4_scripted.py`.

- **Pool fijo de rivales:**
  - BC humano;
  - scripted (presiona, apoya y cubre; independiente de los datos);
  - snapshots de E3 a medida que existan;
  - variantes con distintas latencias.
- **Fuerza:** Bradley–Terry (escala Elo) sobre victorias y empates, ambos lados, en greedy y en muestreado. Nunca un solo rival.
- **Parecido humano:** W1 normalizada contra las grabaciones de entrenamiento, en:
  - pases/min, largo de pase y pases por posesión;
  - profundidad, ancho y distancias a la pelota (1.º a 4.º);
  - % de jugadores parados y sin dirección;
  - duración de la posesión;
  - cambios de tecla/s y patadas/min;
  - espera del saque inicial y duración de los saques.
- **Imitación:** NLL y precisión en desarrollo, separadas en cambios de tecla y por contexto (abierto, saque propio/rival, saque inicial).
- **Sondas de coordinación:** 2v1 y 3v2 desde estados humanos grabados (Liu 2019). *Pendiente.*
- Todo se mide con delay de sala.

**Criterio (techo de referencia, `reports/x4/human_metrics_sanguchito.json`):** la W1 normalizada prueba-vs-entrenamiento de los humanos queda en:
- ≤ 0,05 en forma de equipo, posesión y largo de pase;
- ~0,1 en goles y espera del saque inicial;
- ~0,2 en pases/min y cambios de tecla/s.

### E1. Imitación humana con latencia correcta
Fuentes: AlphaStar SL, HR-PPO, TiKick (2110.04507). **Implementado:** `learn/x4_data.py`, `learn/x4_policy.py`, `learn/x4_bc.py`.

- **Datos:** todas las grabaciones 4v4, separadas por bloque de sesión (`reports/x4/splits.json`). El mapa entra como condición (§6 de la revisión).
  - Sanguchito: 2.072/307/443 min (entrenamiento/desarrollo/prueba).
  - RS ONE: 2.001/331/412 min.
- **Arquitectura:**
  - política compartida por los 4 jugadores, con encoder de conjuntos para compañeros y rivales: cada entidad lleva el contexto propio-pelota, encoder compartido por tipo y pooling de media y máximo (Deep Sets 1703.06114; MARLadona 2409.20326);
  - observación v3 (`env/rs4z/obs_v3.py`): geometría del mapa, mapa como one-hot, margen de contacto con la pelota y 5 decisiones propias (15 ticks);
  - la misma función arma la observación en el simulador y desde las grabaciones, y el bot de sala la reproduce en JS (paridad en 3.088 observaciones reales);
  - sin marcador ni reloj en el actor (preferencia del usuario); van solo al crítico.
- **Convención de retardo:**
  - una decisión tomada con el estado S_t y retardo D se aplica en los ticks que producen S_{t+D+1}..S_{t+D+3};
  - para la etiqueta humana del registro t_lab, la observación es t_lab − D − 1 y las acciones propias son las entradas de t_lab − 3(h+1).
- **Latencia de los humanos:**
  - **[inferencia, no confirmada]** la estimación por máxima verosimilitud no dio un codo: la NLL sube de forma monótona desde d=3 (`reports/x4/human_delay_probe.json`).
  - d=0 da mejor NLL por un artefacto: los humanos cambian de tecla en cualquier tick y el estado a d=0 ya muestra el cambio que empezó en ese tick.
  - **Decisión:** la imitación se entrena con el retardo de la sala (D ∈ [6, 15] con el retardo como feature), que es donde va a jugar el bot. Ver la convención en E1.
- **Criterio:**
  - en desarrollo: NLL por debajo de "repetir la tecla", y precisión en cambios de tecla por encima de la línea base;
  - en lazo cerrado: le gana al scripted, y las métricas humanas quedan a pocas veces el techo.
  - Se valida en closed-loop sabiendo que va a derivar (HR-PPO lo reporta).
  - La BC es **ancla y prior**, no rival.

### E2. Precalentar el crítico
**[inferencia]**, motivada por Wołczyk 2024. **Implementado:** `--critic-warmup` en `learn/x4_ppo.py`.

- Crítico central asimétrico entrenado con la BC congelada jugando antes de mover el actor (MAPPO 2103.01955; AlphaStar).
- Ve la observación del actor, reloj, marcador, la acción que cada uno de los 8 jugadores va a aplicar (privilegiada por la latencia) y λ.

### E3. RL desde la BC con ancla KL
Fuentes: HR-PPO, VPT, AlphaStar, DiL-piKL, MAPPO. **Implementado:** `learn/x4_ppo.py`, con prueba de humo en CPU. Corre en el pod (`docs/POD_RUNBOOK.md`).

- **Algoritmo:** MAPPO con parámetros compartidos, valor normalizado, 5–10 épocas, clip ≤ 0,2 y batch grande.
- **Pérdida:** (1−λ)·L_PPO + λ·KL(BC‖π).
  - Barrido corto λ ∈ {0,06; 0,2; 0,4}, con o sin decaimiento ×0,9995 (VPT). **Por defecto 0,2:** en CPU, con 0,06 la política derivó hacia la pasividad y perdió contra la BC; con 0,2 se mantuvo cerca de la BC el doble de tiempo (`docs/REPORTE_2026-10-06_noche.md`).
  - Criterio de corte pre-registrado en `docs/POD_RUNBOOK.md` §4.
  - Opción DiL-piKL: λ muestreado por partido. El actor no ve λ, así que con varios valores aprende el promedio.
- **Recompensa** (GRF, MARLadona, OpenAI Five 1912.06680):
  - gol ±1, suma cero;
  - shaping CHECKPOINT: 10 franjas del campo rival, +0,1 la primera vez por punto, el resto al marcar, suma cero. Se retira cuando le gana a la BC el 75% de los partidos;
  - team spirit 1,0 (todas las recompensas son de equipo);
  - **[inferencia, agregado]** −0,1 de suma cero al equipo que deja vencer un saque o el saque inicial (plazo de entrenamiento de 600 ticks).
    - Motivo: con el reloj congelado y el saque pasando al rival, que en self-play es la misma política, quedarse quieto vale exactamente 0.
    - Una corrida chica en CPU desde la BC dejó de sacar a las 50 actualizaciones; el ancla con λ=0,06 no lo impidió.
    - En la sala, quedarse quieto no es una opción.
  - El pase no se premia por defecto (Liu 2022). Si a 100–300M decisiones el pase cae por debajo del criterio, se prueba **como brazo pre-registrado** el término de TiZero: +0,05 por pase de la posesión que termina en gol, de suma cero (`--pass-bonus 0.05`).
  - **Criterio numérico de pase (agregado):** en self-play con latencia de sala, ≥ 6 pases/min y ≥ 0,28 de pases/(pases+pérdidas).
    - Referencia humana en Sanguchito: 9,2 y 0,36 (`reports/x4/pass_stats.json`).
    - La BC de partida está en 3,1 y 0,19.
    - Implicancia: hoy **el ancla no aporta por sí sola el pase humano**, solo un tercio. El pase va a tener que venir de una BC más fuerte (más entrenamiento en GPU), del RL o del brazo de TiZero.
- **Oponentes** (TiZero, AlphaStar):
  - 80% self-play y 20% contra el pool con PFSP (1−x)², incluida la BC; snapshots cada 100 actualizaciones;
  - un explotador barato cada 200–500M decisiones (*pendiente*).
- **Arranques:** 40% de los partidos desde estados humanos grabados (juego abierto y, en el 30% de esos, inicio de saque) del split de entrenamiento de Sanguchito.
- **Variabilidad:**
  - delay aleatorio con la distribución de sala;
  - **solo Sanguchito por defecto**: es el único mapa con física y script verificados de punta a punta. RS ONE y 2K23 se suman con `--maps` cuando se resuelvan sus saques (B1). *Pendiente:* variaciones de geometría.
- **[inferencia] Vigilar desde el día 1** los estados donde los humanos esperan, como el saque inicial. La evaluación periódica registra los saques iniciales que nadie ejecuta (`selfplay_safety`) y la espera contra la referencia humana.
- **Criterio:** fuerza ≥ BC + margen contra el pool y parecido humano dentro del rango test-vs-train, todo a delay de sala.

### E4. Sala real
- Solo cuando E3 pasa sus criterios en simulación.
- Prueba con el usuario en 2K23 y en Sanguchito, con grabación y trazas (`--trace`).
- La grabación entra como dato nuevo y como prueba de conformidad.

## 3. Qué no se va a hacer y por qué

- Ligas grandes o PBT (AlphaStar, Liu, FTW): fuera de presupuesto.
- MAT (2205.14953): decodifica en forma autorregresiva entre agentes y no encaja con clientes independientes con latencias distintas.
- Métodos value-based (QMIX/VDN): fallan en partido completo (Song 2023).
- AMP o GAIL como señal principal (2104.02180): tienen mode collapse y nosotros ya tenemos acciones.
- Búsqueda en tiempo real tipo piKL a 20 Hz.
- RL offline puro (TiKick): tenemos ~645 partidos (93 h), no 22 000.

## 4. Cómputo

- **Pod:** 1× RTX 3060 Laptop, 12 GB.
- **CPU del pod:** cuota de 12 CPUs en un host compartido; rinde mejor con 8 hilos numba.
- **Simulador:** ~195k decisiones de partido/s.
- **Presupuesto por corrida:** 1e8–1e9 decisiones.
- **Sesión del 2026-10-06 (noche):** corrió en un contenedor en la nube sin acceso al pod (el proxy de salida solo deja pasar HTTPS) y sin GPU, con 4 vCPU.
  - La imitación entrena ahí a ~28k muestras/s.
  - El RL solo se probó en miniatura.

## 5. Próximos pasos concretos, en orden

1. En el pod: rehacer el dataset (`docs/POD_RUNBOOK.md` §1) y entrenar la imitación final en GPU (§3). Criterio de E1.
2. E2–E3 en el pod: barrido corto de λ con evaluación periódica (§4 del runbook).
3. B1: grabar un partido humano en 2K23 y correr la conformidad de saques.
4. Sondas de coordinación 2v1/3v2 desde estados humanos (E0).
5. E4 cuando E3 pase sus criterios.

## 6. Cambios respecto del borrador (2026-10-06, noche)

| Cambio | Por qué |
|---|---|
| Sanguchito pasa a ser el mapa principal de imitación y de RL (70% de los partidos) | El dataset pasó de 7 a 498 grabaciones de Sanguchito |
| Latencia objetivo de 9–12 ticks en lugar de 15 | Medida en sala en dos sesiones; el "15" venía de antes del arreglo de hilos de ONNX |
| Imitación con el retardo de sala como feature, sin retardo humano por jugador | La sonda de verosimilitud no mostró codo (§2, E1); el bot juega con su propio retardo |
| Obs v3: 5 decisiones pendientes, geometría del mapa, mapa one-hot y margen de contacto | 3 decisiones no cubren 9–12 ticks; la lateral de 2K23 está en 600 y no en 670; la imitación pateaba la mitad que los humanos |
| Scripted en el pool de evaluación | Referencia de fuerza independiente de los datos humanos |
| Métricas de imitación en cambios de tecla | Riesgo de copycat (de Haan 2019, Wen 2020): "repetir la tecla" ya acierta el 84% |
| Correcciones del script de Sanguchito (córner en y, dirección al patear) | Conformidad en 498 grabaciones |
| Penalización por dejar vencer saques en el RL | Equilibrio degenerado observado: el RL dejó de sacar a las 50 actualizaciones |
| RL solo en Sanguchito por defecto (antes: mezcla de 3 mapas) | Prioridad 1, fidelidad: los saques de 2K23 y RS ONE no coinciden con sus grabaciones |
| λ = 0,2 por defecto (antes 0,06) | Dos corridas chicas de CPU: con 0,06 hay deriva hacia la pasividad desde la actualización 50 (VPT usa 0,2) |
