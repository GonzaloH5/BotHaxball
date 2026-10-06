# Revisión pre-lanzamiento del RL X4 (2026-10-06)

Qué tiene que pasar antes, durante y después de la corrida larga en el pod, y con qué evidencia se llega a ella. El objetivo
del usuario es explícito: **la cadena de pase completa del agente tiene que llegar, como mínimo, al nivel del humano promedio
de la modalidad**, medida objetivamente contra grabaciones reales, antes de considerarlo competitivo.

> Este documento es la referencia de la corrida: los umbrales y las acciones de la cola (`learn/x4_queue.py`) salen de acá.
> Lo marcado **[inferencia]** es razonamiento propio sin verificación experimental.

## 1. La vara: la cadena de pase contra humanos

`tools/x4_pass_chain.py` mide 39 métricas con **el mismo código** sobre grabaciones humanas y sobre partidos simulados
(muestreo cada 3 ticks, estado antes del paso, mismas definiciones de toque, pase, pérdida y posesión que
`tools/x4_metrics.py`). Todo en el marco del equipo que tiene la pelota. El valor de cada acción sale de un modelo de valor
de posesión aprendido de humanos (`learn/x4_epv.py`: P(gol propio en 10 s) − P(gol rival); AUC 0,873 y logloss skill 0,28
en la partición de prueba, `reports/x4/epv.json`).

| Eslabón del usuario | Métricas (tipo) |
|---|---|
| posicionarse, generar líneas | compañeros con línea abierta (cono de intercepción), línea progresiva disponible, apoyo cercano no adelantado, control del espacio (eficacia); separación de la marca (estilo) |
| desmarcarse, ofrecer apoyo | desmarques que abren línea por minuto de posesión, desmarques que terminan en recepción, reofrecerse tras pasar (eficacia) |
| elegir y ejecutar el pase | pases/min, precisión (completados / intentos dirigidos), precisión bajo presión, progresivos, rompe líneas, al espacio, cambios de orientación, paredes, salidas de presión, asistencias, valor por intento (eficacia); fracción hacia atrás, largo y velocidad del pase (estilo) |
| anticipar, recibir, controlar, orientarse | retención 2 s tras recibir, retención bajo presión, control orientado hacia adelante, anticipación del receptor (eficacia); de primera, tiempo con la pelota (estilo) |
| aprovechar la ventaja | avance de la pelota en 3 s tras un pase progresivo o que rompe líneas, tiro en 5 s (eficacia); devolución inmediata hacia atrás (estilo) |
| posesión y valor de la secuencia | valor por posesión, pases por posesión, tiros/min (eficacia); circulación inútil (≥ 3 pases sin avanzar ni sumar valor; menos es mejor) |
| cuándo pasar y cuándo no | valor de cada decisión del portador (pase, conducción, aguantar, tiro, pérdida), pérdida sin pase bajo presión (menos es mejor), pasar cuando hay línea progresiva (estilo) |
| contexto defensivo | presión sobre el portador (estilo): evita que el pase "mejore" porque la defensa dejó de defender |

**Referencias humanas** (`reports/x4/pass_chain_human.json`):
- `sanguchito_test`: 77 grabaciones de prueba de la sala Sanguchito (376 min). Es la referencia del gate: mismo mapa,
  misma sala y jugadores que el agente no vio.
- `liga_rs_one`: 116 grabaciones de la liga HAXARG (equipos de torneo; RS ONE, misma cancha y física). Es la vara de
  "jugador competente": más pases (16,1 contra 12,8 por minuto de juego abierto), más precisión (0,73 contra 0,69), más
  paredes (1,9 contra 0,8) y defensas más cerradas (presión al portador 77 contra 98 px).

**Gate** (`tools.x4_pass_chain.gate`): cada métrica de eficacia del agente ≥ el promedio humano (≤ en las "menos"); cada
métrica de estilo dentro del p10–p90 humano entre grabaciones. Una métrica sólo se juzga con datos suficientes (≥ 20
eventos esperados al ritmo humano o ≥ 20 observaciones); el **índice de la cadena** es la media geométrica de
agente/humano (1 = promedio humano) sobre las métricas juzgables, con intervalo por bootstrap de partidos.

## 2. Dónde estamos: la imitación (BC) de partida

Self-play de la BC con la latencia de sala (`reports/x4/pass_chain_bc_final.json`): **índice 0,45**.
- Al nivel humano: líneas disponibles, apoyo, línea progresiva, anticipación del receptor, presión defensiva.
- Muy por debajo: pases/min 5,7 contra 12,8; precisión 0,49 contra 0,69; retención tras recibir 0,40 contra 0,58;
  tiempo con la pelota 0,10 contra 0,40 s; paredes 0,12 contra 0,77; valor por intento ~0 contra 0,022.

Diagnóstico (`reports/x4/bc_diagnose.json`, `pass_chain_bc_final_lowdelay.json`):
- En estados humanos (lazo abierto), la probabilidad de patear de la BC está **calibrada** con la humana en todas las
  distancias a la pelota.
- En sus propios partidos (lazo cerrado), alguien está al alcance de la pelota el 13% del tiempo contra el 29% humano: la
  BC toca la pelota y la suelta, no la controla. Con latencia baja (3–5 ticks) mejora poco (0,53), así que no es sobre todo
  la latencia. Bajar la temperatura empeora (0,42 a 0,7; 0,29 a 0,5).
- [inferencia] El déficit es de control en lazo cerrado (errores que se acumulan, intención que la imitación por decisión
  no sostiene), no de "no saber cuándo pasar". Eso es justo lo que el RL corrige bien: perder la pelota baja el valor de
  posesión en el paso siguiente.

## 3. Cómo empuja el entrenamiento hacia cada eslabón

| Mecanismo (learn/x4_ppo.py) | Qué empuja |
|---|---|
| Ancla KL a la BC (λ 0,2 → 0,05) | conserva lo que la BC ya hace como humano (posicionamiento, líneas, apoyo, estilo de pase) mientras el RL corrige el control |
| Shaping de potencial con el EPV, F = γ·φ(s') − φ(s) (suma cero) | cada pérdida de pelota se paga en el paso; cada pase que acerca al gol o rompe una línea suma; una posición sin pelota que los humanos asocian a goles sube φ. No cambia la política óptima (Ng 1999; equilibrios de Nash en juegos: Devlin y Kudenko 2011), así que no se puede explotar a la larga |
| Gol ±1 | el objetivo real |
| Arranques desde estados humanos (40%) | situaciones reales de posesión, presión y saque desde el primer minuto |
| Pool con PFSP (BC + snapshots) | defensas distintas; la BC es una defensa humano-símil |
| Brazo de pases (actualización 1000) | si el índice sigue < 0,85, +0,05 por pase de las posesiones que terminan en gol (TiZero); no premia circulación inútil |
| Penalización por espera en saques | evita el equilibrio "nadie saca" (sólo esperas; el mal lateral ya cuesta la pelota) |

## 4. Plan de corrida (cola pre-registrada)

(se completa con la revisión)

## 5. Problemas encontrados y corregidos

(se completa con la revisión)

## 6. Riesgos que siguen

(se completa con la revisión)

## 7. Evidencia

(se completa con la revisión)
