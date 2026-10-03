# Potencia del criterio "sin retroceso" (RS4-b1)

Fecha: 2026-10-03. Insumo: `reports/rs4_b1/eval/v3_2568M_final.json` (referencia de las compuertas: 32 estados iniciales × 9 pares compañero-rival × 2 colores × 4 puestos = 2304 partidos).

## Método

Simulación nula: un candidato **de igual fuerza** que la referencia, con resultados re-muestreados por celda de los propios partidos de la referencia (independientes de los de la referencia).
- Se cuenta cuántas veces el criterio declara algún retroceso falso.
- Para la potencia, se resta 0,5 punto con probabilidad 0,3 a cada partido de una celda (retroceso real de ~−12 a −15 pp) y se cuenta cuántas veces se detecta.
- La regla es la del plan: diferencia < −5 pp y cota superior del intervalo de 90% < 0.
- 300 repeticiones por fila, con error estándar agrupado por estado inicial.

| Celdas del criterio | Falso rechazo por semilla | Las dos semillas pasan | Detecta el retroceso real |
|---|---|---|---|
| 72 (compañero, rival, color, puesto), 32 partidos cada una: implementación actual | 0,68 | 0,10 | 0,13 |
| 72 con Bonferroni (10% familiar) | 0,00 | 0,99 | 0,00 |
| 9 pares compañero × rival, 256 partidos cada uno | 0,03 | 0,95 | 0,72 |
| 9 pares con Bonferroni (10% familiar) | 0,00 | 1,00 | 0,57 |

Una aproximación previa con bootstrap por filas daba 0,53 de falso rechazo para la regla actual.

## Lectura

Con 72 celdas de 32 partidos, la regla actual tiene dos problemas a la vez:
- **Rechaza por ruido:** un candidato idéntico en fuerza falla en 2 de cada 3 evaluaciones y pasa las dos semillas solo el 10% de las veces.
- **No detecta retrocesos reales:** cada celda es demasiado chica (13% de potencia).

Si se mantiene, "sin retroceso" decidiría el bloque por azar.

La versión por pares compañero × rival:
- conserva la intención del plan: los promedios no ocultan una regresión con un compañero o un rival concreto;
- falla por ruido solo el 3% de las veces;
- detecta un retroceso de ~−15 pp en un par con probabilidad 0,72.

Los resultados por color y puesto se siguen informando.

## Propuesta (pendiente de decisión del usuario)

Definir "celda crítica" como el par compañero × rival:

  python -m tools.rs4_b1_gates --cell-level pair ...

La propuesta se presenta antes de evaluar ningún candidato, para no elegir la regla viendo los resultados.
