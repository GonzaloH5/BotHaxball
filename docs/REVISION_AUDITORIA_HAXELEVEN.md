# Revisión de la auditoría del plan HaxEleven

Fecha: 8 de octubre de 2026 (tarde). Contrasta [`AUDITORIA_PLAN_HAXELEVEN_2026-10-08.md`](AUDITORIA_PLAN_HAXELEVEN_2026-10-08.md) con [`PLAN_HAXELEVEN.md`](PLAN_HAXELEVEN.md), los reportes y el código, y con evidencia nueva medida en esta sesión sobre los 399 partidos del estadio actual.

**Pedido del usuario:** incorporar la auditoría sin tratarla como verdad absoluta, conservar lo que sigue siendo lo mejor, priorizar y llegar a prototipos jugables.

**Cada cambio del plan lleva un identificador `P#`** que remite a la tabla de §4 y a la evidencia que lo motiva.

**Convención de evidencia (la misma de la auditoría, más una):**
- **[R]** medido en este repositorio, con archivo y número;
- **[L]** resultado publicado;
- **[H]** hipótesis sin validar;
- **[X]** recomendación que requiere un experimento antes de adoptarse.

---

## 1. Estado real recuperado

Verificado directamente:
- **D0:** 399 partidos.
- **D1:** 399 partidos.
- **D2:** 47 partidos. El fallo ocurrió en `BOV_AV_HaxEleven-7-9-2026-23h13m`.
- **D3:** sin correr.
- **Procesos:** ningún proceso HaxEleven corriendo.
- **Repositorio:** cambios sin commitear, iguales a los del traspaso.

La causa del fallo de D2 no era un caso raro de código: esa grabación es un **1 contra 1** de 14 minutos dentro del banco del estadio actual.

---

## 2. Evidencia nueva de esta sesión

| ID | Hallazgo | Número | Fuente |
|---|---|---|---|
| N1 | Hay grabaciones del estadio actual que no son 11v11 | 5 de 399 (equipo más chico con mediana 1, 2, 6, 6 y 7 jugadores). Quedan **394 aptos**: 290 de entrenamiento, 40 de desarrollo y 64 de prueba. | `tools/hx11_d0.py plantel`, `manifest.json` (`apto`) |
| N2 | D1 pegaba un anuncio de powershot a dos patadas seguidas | De 101,6% a **98,4%** (8.950 de 9.091). Goles 642/642 en los aptos. | `reports/hx11/d1_eventos.md` |
| N3 | Formaciones con los 394 partidos | Familia 4-3-3 (4-1-4-1, 4-3-3, 4-2-3-1) **63%**; 3-4-3 20%; 4-4-2 7%; **3-5-2 5% (40 equipos)** | `reports/hx11/d2_puestos.md` |
| N4 | Validación interna de D2, sin etiquetas | Rol dominante igual entre mitades **83,4%**; contra una asignación independiente por anclas **95,6%**. Por tick: 63% en su rol, 11% en permuta directa, 25% en otro. El ancla de 3 min identifica al jugador en el **76,5%** de las ventanas. | `reports/hx11/d2_validacion.md` |
| N5 | Modos del script por partido | **Powershot** prendido en 394. **Offside:** 313 prendido, 4 apagado, 77 indeterminados. **Faltas:** 185 prendidas, 31 apagadas, 178 desconocidas. Con faltas prendidas, la sala cobra el 78,5% de los contactos de ≥ 110 ticks (1,26 faltas por partido). El contacto antes de un cobro es estable: p50 115 ticks. 9 partidos cambian el offside a mitad del partido. | `reports/hx11/e2_modos.md` |
| N6 | Validez del valor (E3) | **Predice:** VAEP A AUC 0,905 y B 0,906; estado A 0,907 y B (grilla) 0,870. **Calibración:** ECE 0,001–0,004, pero A sobreestima ~30% en el decil más alto. **Fiabilidad entre partidos del valor por jugador:** VAEP A r = 0,20 y B r = 0,28, contra **0,50 del pase completo**. Acuerdo A–B en Φ: Spearman 0,90. | `reports/hx11/d3_valor.md` §4 |
| N7 | Control del espacio de D3 | Vuelve a elegir σ = 4, el borde de la grilla. Está sobreconfiado en la zona media (P 0,35 → 12,5% completados). AUC 0,915, optimista porque mide donde termina el pase. | `reports/hx11/d3_valor.md` §1 |
| N8 | La "marca perdida" no se sostiene a escala | 167.506 eventos; remate o gol en 5 s 6,2% contra 4,7% al azar (× 1,3). El 64% de un solo partido (n = 14) era ruido. El seguimiento βa sí se sostiene: p50 0,28, marca mayormente zonal. | `reports/hx11/d2_marcas.md` |
| N9 | E4, potencia | σ de la diferencia de gol 1,33: de 88 (δ = 0,5) a 2.179 (δ = 0,1) partidos por brazo. Métricas de proceso en sala: 16–36 partidos del bot por línea para medir su media con 1/4 del rango humano. | `reports/hx11/e4_potencia.md` |
| N10 | L0-mini en lazo cerrado (CPU, desarrollo) | Hay que decidir cada 3 ticks (decidiendo cada tick, 45 cambios de tecla por segundo contra 2,7). La historia propia causa copycat: sin ella, la distancia al ancla es humana (219 contra 217) y las patadas suben × 3,7 (1,61 contra 0,43 por minuto; humanos 4,7), pero el bot vibra. La escala sube las patadas × 2,6. **En todos los brazos el bot toca la pelota de 3,4 a 4,7 veces menos que el humano** (tasas totales sobre tiempo total). Con los 22 imitando, nadie va a la pelota: el más cercano queda a 260–296 px (humanos 57) y la velocidad a 0,8–1,8 px/tick (humanos 2,26). | `docs/TRASPASO_HAXELEVEN_D0_D3.md`, `reports/hx11/l0_brazos.md` |

---

## 3. Contraste, punto por punto

| # | Lo que dice la auditoría | Clase | Evaluación | Decisión |
|---|---|---|---|---|
| A1 | Hay 88,4 h, no 142 h, y 61,2 h de entrenamiento | [R] | Confirmado y afinado: 394 partidos aptos, 290 de entrenamiento (N1) | Corregir los números del plan (**P1**) |
| A2 | Pocos goles, juego caótico | [R] | Confirmado: 1,63 goles por partido en los aptos. La consecuencia sobre la señal de RL es [H] | Riesgo de L2; se mide con E14 antes de cualquier L2 (**P7**) |
| A3 | Las reglas se validaron en una muestra que no representa al banco | [R] | **Matizado con N5.** El umbral de la falta es estable; lo que cambia es el **modo**. Con faltas prendidas hay 1,26 por partido, cerca de la muestra de S2 (1,5). Las faltas están apagadas en al menos 31 partidos (8%) y en 178 no se puede saber. En esos 178 casi no hubo contactos largos (0–1 por partido), así que la imitación no aprende de ellos que "sostener el contacto es gratis". | Condicionar la imitación por el modo (`faltas_l0`, `offside_on`) (**P1**). Árbitro en sombra en otros meses (E2b) antes de L2, no antes de L0 |
| A4 | El etiquetador de puestos no tiene verdad y es inestable | [R] + [X] | **Lo inestable es el rol por tick; el puesto por partido es estable** (N4: 83% entre mitades). La "contradicción" con las formaciones del usuario es sobre todo de nombres: la familia 4-3-3 es el 63%. El 3-5-2 no está ausente (40 equipos). Falta la verdad humana, pero puede salir del lazo de entrenamiento. | **P2:** condicionar por el **ancla medida** del jugador en el tramo y la **línea** de su rol dominante, no por el rol por tick. El etiquetador queda para el vocabulario de `--pos` y los reportes. E1 se achica a una validación liviana (nivel 2) |
| A5 | El EPV sería shaping y juez a la vez | diseño | Correcto. N6 muestra además que A y B rankean casi igual el estado (0,90): B no es un juez fuerte solo | **P6:** shaping con A (si hay L2); juez con goles con potencia, B y métricas de proceso. Nunca el mismo modelo |
| A6 | El lazo abierto no predice el lazo cerrado | [L] Dauner 2023, Montali 2023 + [R] X4 | Confirmado | **P3:** puerta de L0 en lazo cerrado no reactivo; el lazo abierto queda como diagnóstico |
| A7 | El banco mezcla modos y versiones | [R] | Medido (N5). La mezcla real es sobre todo de faltas; powershot y offside casi uniformes | Igual que A3 (**P1**) |
| A8 | L2 no resuelve el crédito: el bot es 1 de 22 | [R] + [H] | Plausible; no medido | E14 (señal/ruido) antes de L2 (**P7**) |
| A9 | Una población de imitaciones juega otro fútbol | [R] X4 | Confirmado en X4; en 11v11 es [H] | **P5:** puerta de realismo antes de usar L1 para RL o para el valor de reemplazo |
| A10 | Pesar demostraciones por valor agregado o por resultado | [L] Paster 2022 + **[R] N6** | **Reforzado:** el VAEP por jugador casi no se repite entre partidos (r 0,20–0,28). Pesar por jugador con VAEP sería pesar por ruido | **P4:** se elimina esa variante. L0.5 solo con habilidad estimada con métricas fiables (por ejemplo pase completo, r 0,50) o ventaja por acción, y solo si mejora a L0 en pruebas independientes |
| A11 | El ancla KL no protege el puesto | [R] X4 (miniatura) | En HX11 es [H] | **P7:** medir la fidelidad de ubicación aparte del KL. Restricción sobre la distribución de estados como brazo, adoptada solo si el KL no alcanza [X] |
| A12 | Un aprendiz por equipo, no 2–3 | [L] Carroll 2019 | De acuerdo; además es el pedido del usuario | **P7** |
| A13 | El 40% de arranques humanos no tiene justificación | [L] Lu 2022 | De acuerdo | Ablación 0 / 40 / 80% / críticos [X] (**P7**) |
| A14 | Memoria en la política | [L] Knott 2021 | Útil solo si fallan las pruebas de compañeros AFK o con otro estilo | Nivel 3 [X] |
| A15 | Sin potencia estadística no se puede declarar nada | [R] σ = 1,36 | De acuerdo; es barato | E4 en nivel 1 para cualquier afirmación de "mejor" (**P8**) |
| A16 | Piloto temprano en sala | razonamiento + [L] | De acuerdo: es la única verdad sobre la respuesta humana | **P9:** C0 apenas haya una L0 que pase la puerta de lazo cerrado |
| A17 | Latencia en ráfagas | [R] S6 | De acuerdo | Probar L0 en lazo cerrado con trazas de S6 (E5), dentro de la puerta de L0 (**P3**) |
| A18 | Penales: 6 casos | [R] | De acuerdo | Penal con guion |
| A19 | Goles en contra: 20% | [H] | No verificado; puede ser la regla del último toque | Revisarlo en la muestra de D1 del usuario |
| A20 | FCP (Strouse) superó a la imitación humana | [L] | Matiz válido | Modelos humanos **más** diversidad anclada en datos (estilos de jugadores reales) (**P5**) |

**Lo que se conserva del plan original sin cambios:**
- **imitación como base**, con PPO para el refinamiento (la auditoría no encontró algo mejor);
- **una sola red** para todos los puestos;
- **el ancla continua como condición** (§4 del plan). Se refuerza: ahora es medida por tramo, no aprendida por rol;
- **no premiar estilo;**
- **actor sin marcador ni reloj;**
- **valor de reemplazo** como marco;
- **la referencia programada** como control;
- **la prueba con una formación reservada:** con N3 sí hay datos (40 equipos de 3-5-2), contra lo que dijo la auditoría con 47 partidos.

---

## 4. Cambios al plan, con su evidencia

| ID | Cambio | Evidencia | Sección del plan |
|---|---|---|---|
| P1 | Datos: solo partidos aptos (394). Modos por partido y por tick como observación: faltas (prendido, apagado o desconocido) y offside. En la sala del usuario se usa "prendido". | N1, N5; A1, A3, A7 | §1, §3, §5 L0 |
| P2 | Condición de puesto: ancla medida del jugador en la mitad del partido (con y sin pelota, relativa al centroide de campo) más la línea de su rol dominante. No el rol por tick. El usuario elige el puesto; `--pos` se traduce a un ancla y una línea. | N4; A4; auditoría C2 | §3 D2, §4 |
| P3 | Puerta de L0 en lazo cerrado no reactivo, desde estados reales de prueba, con trazas de latencia de S6. El lazo abierto queda como diagnóstico. Hay que medir pasividad, zona, deriva, distancia a la pelota y acciones con pelota contra el rango humano. | A6, A17; [R] X4 0,55 contra 3,2 patadas por minuto | §5 L0, §6 A |
| P4 | Se elimina "pesar por valor agregado o filtrar a los mejores". L0.5 es opcional: brazo de habilidad (con métricas fiables) o de ventaja por acción, adoptado solo si mejora a L0 en lazo cerrado y en escenarios críticos. | N6; A10 | §5 L0 |
| P5 | L1 tiene puerta de realismo propia (métricas colectivas, tácticas y de mecánicas dentro del techo humano contra humano; C2ST). Diversidad anclada en estilos de jugadores reales. Compañeros "malos" realistas como prueba de robustez. | A9, A20 | §5 L1 |
| P6 | El valor A (MLP de estado) es candidato a shaping. El juez es otro: goles con potencia, el modelo B y métricas de proceso. A se calibra antes de usarse, porque sobreestima en el decil alto. | N6; A5 | §3 D3, §5 L2, §6 |
| P7 | L2: un solo protagonista por partido (no 2–3 aprendices); reglas aleatorizadas (falta, offside, plazos de saque); fidelidad de ubicación medida aparte del KL; restricción de ubicación como brazo; ablación de arranques reales; E14 antes de cualquier campaña; auditoría de exploits en los bordes de las reglas (contacto 100–114 ticks: 288 casos humanos en todo el banco). **No hay campaña larga hasta pasar las puertas.** | A2, A8, A11–A13; N5 | §5 L2 |
| P8 | Tabla de potencia (E4) antes de afirmar que algo es mejor. | A15 | §6 |
| P9 | Piloto en sala (C0) apenas L0 pase la puerta de lazo cerrado; valida a L1 en la interacción con el bot. | A16 | §6 C, §8 |
| P10 | El control del espacio de D3 se recalibra (grilla más chica para σ y calibración en desarrollo) antes de usar el "peligro" de las marcas como objetivo. Como análisis vale, porque la marca perdida se valida con su resultado. | N7 | §3 D3 |
| P11 | L0 decide cada 3 ticks y sostiene la acción, como X4. | N10 | §5 L0 |
| P12 | La historia propia de teclas no entra tal cual: o se quita, o se entrena con dropout. Lo decide la comparación de brazos en lazo cerrado. Hay rasgos explícitos de "voy yo" como brazo. | N10; [L] de Haan 2019 | §5 L0 |
| P13 | **L1 no puede ser la población de L0 tal cual.** Con los 22 imitando, nadie va a la pelota. Antes de cualquier L1 hace falta un arreglo estructural, a probar con la puerta de realismo: ajuste en lazo cerrado con discriminador (Igl 2022), estructura programada para quién va a la pelota, o compañeros entrenados con RL hacia el realismo. **El bot entre 21 humanos se evalúa primero con el lazo no reactivo y en la sala (C0).** | N10; A9 | §5 L1, §8 |
| P14 | "La marca perdida" no es métrica de evaluación hasta rediseñarla y validarla a escala. Sí lo es el seguimiento βa. | N8 | §3 D2–D3, §6 |

---

## 5. Experimentos en tres niveles

### Nivel 1: bloqueantes
Sin esto se entrenaría o se juzgaría con datos, etiquetas o entornos inválidos.

| ID | Qué | Estado |
|---|---|---|
| B1 | Higiene del banco (partidos aptos) | **hecho** (N1) |
| B2 | Modos por partido (E2a) | **hecho** (N5); entra como condición |
| B3 | Condición de puesto válida sin verdad humana | **hecho** en lo interno (N4); P2 |
| B4 | Validez del valor (E3) | **hecho** (N6): sirve para análisis y como candidato a shaping; no sirve para pesar jugadores |
| B5 | Banco de evaluación de L0 en lazo cerrado no reactivo y su puerta | **siguiente** |
| B6 | Tabla de potencia (E4) | pendiente; barato |
| B7 | Puerta de realismo de L1 (E10, pruebas 1–2). Los techos humano contra humano se pueden calcular ya | antes de cualquier RL o valor de reemplazo |
| B8 | Árbitro aleatorizado para L2 | antes de L2 |

### Nivel 2: necesarios para llegar a competencia humana
Comprueban o mejoran capacidades que el agente todavía no demuestra.

- **E6:** curva de escala de L0 (25, 50 y 100%) y semillas.
- **E8:** intercambio de puesto en lazo cerrado.
- **Escenarios críticos** desde estados reales: 2 contra 1, espalda ganada, contraataque y marca perdida.
- **C0:** piloto en sala (E12) y respuesta humana al bot (E13).
- **E1 liviano:** el usuario valida el vocabulario de `--pos` y unos 10 equipos-partido.
- **L0.5 (E9):** solo si L0 falla en los escenarios críticos.
- **L2:** E14 primero; luego el piloto corto con ablaciones (E15) y la auditoría de exploits (E16).
- **E2b:** árbitro en sombra en otros meses.
- **E5:** latencia con trazas reales.

### Nivel 3: opcionales
Solo si la evidencia señala una necesidad concreta.

- memoria en la política (E7);
- cabezas auxiliares (ghosting, próximo evento, acciones en bloque);
- ventaja contrafáctica;
- liga de explotadores y compañeros HSP (primero como sondas);
- jueces humanos a ciegas y comparaciones por pares;
- mapas viejos con el mapa como condición, si la curva de E6 no se aplana;
- desarrollos tipo WOSAC.

---

## 6. Camino corto a algo jugable

```
D0–D3 validados (hecho)
 └─ L0-mini en CPU local (prototipo: datos, red de conjuntos, condición P2, modos P1)
     └─ Banco de lazo cerrado no reactivo (B5) → puerta de L0 (P3)
         └─ L0 completo en el pod  ← pide aprobación (GPU)
             └─ C0: piloto en sala con L0 (P9)
                 └─ L1 = población desde L0 + puerta de realismo (P5)
                     └─ [L0.5 si L0 falla en escenarios críticos (P4)]
                         └─ L2 piloto: un protagonista, E14, ablaciones, exploits (P7)
```

**Qué requiere la aprobación del usuario:**
1. Usar la GPU del pod para el L0 completo.
2. Organizar el piloto en sala.
3. Una hora de etiquetas livianas (E1).
