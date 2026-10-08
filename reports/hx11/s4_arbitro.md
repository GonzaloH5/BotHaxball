# S4: conformidad del árbitro (árbitro en sombra sobre las grabaciones)

Fecha: 8 de octubre de 2026. Etapa S4 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md).
- **Muestra:** 31 partidos enteros de la versión actual, [1800, 1200]. 1,2 millones de ticks de juego.
- **Datos completos:** [`s4_arbitro.json`](s4_arbitro.json).

```
python -m tools.hx11_conformance arbitro data/hx11/npz/*.npz --out reports/hx11/s4_arbitro.json
```

## Método

El árbitro del simulador (`env/hx11/kernel.py`, el mismo que va a entrenar) corre en sombra sobre la grabación:
- **Cada tick:** se cargan posiciones y velocidades de la pelota y de los jugadores, las patadas y los goles de la sala, y se corre solo el árbitro.
- **Emparejamiento:** cada decisión suya se empareja con una de la sala del mismo tipo y equipo, a ≤ 4 ticks para colocaciones y offsides y ≤ 12 para faltas.
- **Desacuerdo:** la sala decidió lo mismo con otro equipo o tipo de saque. Se impone la decisión de la sala.
- **Falso positivo:** la sala no decidió nada parecido. Se deshace y se vuelve a juego abierto.
- **No visto:** la sala decidió algo que el árbitro no decidió a tiempo. Se fuerza la decisión de la sala.
- **Así cada regla se mide sola,** sin arrastrar errores de las anteriores.
- **Comparación por tick:** masa de la pelota, gravedad (comba), barrera, disco del punto y jugadores sin masa.

## Decisiones

| Decisión | De la sala | Iguales | Desacuerdos | Falsos | No vistos | Recall | Precisión |
|---|---|---|---|---|---|---|---|
| Colocaciones (lateral, córner, saque de arco, tiro libre, penal) | ~1.560 | 1.490 | 20 | 53 | 49 | **95,6%** | **95,3%** |
| Faltas | 49 | 35 | 0 | 1 | 14 | **71%** | **97%** |
| Offsides | 138 | 129 | 0 | 14 | 9 | **93,5%** | **90%** |

### Tick y punto de las decisiones correctas

| | Desfase (árbitro − sala) | Error del punto |
|---|---|---|
| Lateral | p50 +1, máx 4 | p50 0; **99,9% exacto** |
| Córner | p50 0, p90 1 | **100% exacto** |
| Saque de arco | p50 0, p90 1 | **100% exacto** |
| Tiro libre | p50 −1 | p50 0, p90 2,6 px; 68% exacto |
| Falta (congelamiento) | p50 0, p90 1 | p50 0, p90 2,8 px |
| Offside (congelamiento) | 0 en todos | p50 0, p90 2,6 px |

- Los desacuerdos de colocación son casi todos laterales con el equipo invertido (16 de 20). Viene de la atribución del último toque en roces de dos jugadores.
- Los no vistos de colocación incluyen re-colocaciones de la sala: laterales mal sacados y saques entregados por hacer tiempo, que el árbitro modela con supuestos.

## Estado de la pelota y de los discos, por tick

| | Error p50 | Error p90 | Ticks exactos |
|---|---|---|---|
| Masa en juego (1,65) | 0 | 0 | 100% |
| Masa del lateral (1,8) | 0 | 0 | 99,7% |
| Masa del penal (2,7) | 0 | 0 | 99,3% |
| Masa en la carga del powershot | 0 | 0,14 | 71% |
| Masa en la barra del córner | 0,025 | 0,20 | — |
| Masa en la barra del saque de arco | 0,043 | 0,19 | — |
| Masa en la barra del tiro libre | 0,058 | 0,13 | — |

Los errores de la barra vienen del valor fijo inicial: en la sala varía entre 6 y 9 ticks, porque el script usa temporizadores de reloj. Un tick de diferencia cambia mucho la masa en las partes empinadas de la onda.
- **En el entrenamiento:** el simulador sortea esa duración con la distribución medida.
- **En la sombra:** se usa la típica, 7 ticks.

| Acuerdo por tick | Lateral | Córner | Tiro libre | Saque de arco | Penal |
|---|---|---|---|---|---|
| Barrera (radio y equipo) | 99,4% | 99,9% | 99,96% | 99,5% | 100% |
| Disco del punto | 99,3% | 99,6% | 98,9% | 99,5% | 100% |
| Jugadores sin masa | 100% | 99,6% | 99,3% | 99,6% | 99,3% |

### Comba: gravedad de la pelota del árbitro contra la de la sala

| Después de | Error p50 / p90 | Gravedad de la sala p50 | Error relativo (p50) |
|---|---|---|---|
| Patada cargada | 0,0008 / 0,0030 | 0,0074 | 11% |
| Tiro libre | 0,0012 / 0,0053 | 0,0074 | 16% |
| Saque de arco | 0,0017 / 0,0058 | 0,0105 | 16% |
| Córner | 0,0035 / 0,0120 | 0,0152 | 23% |

En 3.196 ticks la sala pone comba después de una patada sin carga, que el modelo no tiene.

## Lo que corrigió la sombra (comparado con S3)

| Regla | Antes | Ahora | Evidencia |
|---|---|---|---|
| Detección del lateral y su x | pelota entera afuera (1208); x del cruce de la línea | **\|y\| > 1210**; x de ese tick | 97% exacto en 219 laterales; error del punto de 6 px a 0 |
| Fin del lateral (barrera y masa 1,8) | 3 y 5 ticks después de patear | cuando la pelota vuelve a \|y\| < 1210 | barrera del lateral de 73% a 99,4% |
| Punto de la falta | la pelota | **el jugador que la recibió** | 82% exacto; punto de 20 px a 0 |
| Punto del offside | la pelota | **el adelantado + 15 px hacia el arco rival** | exacto en 44; punto de 26 px a 0 |
| Umbral de la falta | 113 ticks | **115** | desfase de −2 a 0 |
| Carga del powershot | 2 px y 90 ticks | **3 px y 92 ticks** (91 seguidos antes) | 98,5% de las cargas; error p90 de la masa de 0,43 a 0,14 |
| Penal | barrera contra el rival | barrera contra **los compañeros del que patea** y disco del punto, 76 ticks | 1 penal (débil); barrera de 1% a 100% |
| Disco del punto en falta y offside | desde el tiro libre | desde el congelamiento | — |
| Valor fijo de la barra | 7 ticks | sorteado entre 6 y 9 con la distribución medida | 653 saques |
| Comba | una patada no la cortaba | **cualquier toque la corta** | — |
| Barrera del saque | solo en el tick 1 | una vez, desde el tick 1 | barrera del córner de 73% a 99,9% |

## Contra los criterios del plan (tabla de S4)

| Prueba | Criterio propuesto | Resultado | ¿Pasa? |
|---|---|---|---|
| Física en juego abierto | p90 a 60 ticks ≤ 0,02 px | 0,014 / 0,003 (S1) | **sí** |
| Powershot | % igual en ≥ 99% de los ticks con contacto | masa exacta en el 71% de los ticks de carga; inicio de la carga en ±1 tick en el 98,5% | no del todo |
| Faltas | F1 ≥ 0,99 | F1 0,83 (recall 71%, precisión 97%) | **no** |
| Offside | ≥ 99% | recall 93,5%, precisión 90% | **no** |
| Pelota parada | tipo, punto y equipo al 100% | 95,5%; puntos exactos en córner, saque de arco y lateral | casi |
| Discos y masas | — | barreras, disco del punto y jugadores sin masa ≥ 98,9% | sí |
| Comba | trayectoria p90 ≤ 2 px a +60 | gravedad con error relativo del 11–23% (no se midió la trayectoria) | parcial |

### Lectura

- **Exacto o casi:** la mecánica que toca la pelota en cada saque (colocación, masa, barreras, jugadores sin masa) y la física.
- **Lo que falta:** son cobros que dependen de casos dudosos.
  - Faltas en forcejeos donde la posesión cambia por un roce.
  - Offsides con desvíos ambiguos.
  - Laterales con el último toque en disputa.
- **Por qué es difícil de cerrar:** para cerrar esos casos habría que reproducir el criterio exacto de "toque" del script, que no es legible.
- **[inferencia] Efecto en el aprendizaje:**
  - Un árbitro que cobra el 71% de las faltas con 97% de precisión enseña la regla correcta: despegarse antes de los 2 s. Un bot que cuerpea 1,9 s sale un poco más favorecido que en la sala.
  - Con el offside al 93% de recall, el bot aprende a no quedar adelantado.

## Decisión para el usuario

1. **Aceptar el árbitro como está.** Los criterios del plan eran propuestas y los de faltas y offside no se cumplen. Las etapas de datos y etiquetas (D0–D3) no dependen del árbitro y pueden empezar.
2. **Seguir afinando antes de seguir.** Una opción es una sesión con bots cliente en la sala (S5): forcejeos controlados y pases con el receptor a distancias conocidas de la línea, para cerrar las faltas y los offsides que la grabación no resuelve.
