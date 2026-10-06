# 06. Pelota parada

[Volver al índice](README.md) · [Reglas y mapas](01_modalidad_reglas_mapas.md)

Los saques combinan ejecución, posiciones y reglas específicas. Una jugada preparada debe probarse en la misma variante en la que se pretende usar.

## Qué está verificado en el proyecto

**[verificado: archivo local]** El [contrato del árbitro](../../../env/rs4z/contract.py) implementa diferencias entre RS ONE, 2K23 y Sanguchito. **[medido: reporte local]** Sanguchito tiene validación con siete grabaciones y comparación de 101 reinicios evaluables: 69 laterales, 20 córners y 12 saques de arco. Esto valida los mecanismos descritos en esa muestra; no mide el éxito táctico de las jugadas. [Reporte](../../../reports/sanguchito_conformance.md).

| Reinicio | Diferencia relevante documentada | Consecuencia táctica propuesta |
|---|---|---|
| Lateral | Puntos de colocación, barreras y criterios de liberación distintos | Ajustar aproximación, ubicación del receptor y tiempo de ejecución |
| Córner | Discos de exclusión y mecanismo de impulso según script/mapa | Ensayar la trayectoria y cómo responder al rebote |
| Saque de arco | Restricción del área y mecanismos de salida distintos | Preparar recepción con la trayectoria propia de la variante |
| Saque inicial | Sanguchito reposiciona en rombo; el reloj local se congela durante la espera | No interpretar la colocación inicial como roles permanentes |

Las consecuencias de la tercera columna son **[inferencia táctica]**.

**[verificado: archivo local]** El script puede añadir curva e impulso en córners y saques de arco. Esos efectos no deben atribuirse automáticamente a una «comba» manual universal de juego abierto. Algunos parámetros de 2K23 y el plazo del lateral de Sanguchito siguen señalados como supuestos. Consultar el contrato antes de usar un número exacto.

## Organización común [inferencia táctica]

Con cuatro jugadores, acordar ejecutor, opción principal, alternativa y protección de la pérdida. No son necesariamente cuatro posiciones inmóviles: un jugador puede cambiar de función cuando viaja la pelota.

Antes de ejecutar, comprobar:

1. Que la pelota está en la fase de saque y se puede liberar legalmente.
2. Qué rivales están excluidos y desde cuándo pueden intervenir.
3. Qué receptor puede llegar a la trayectoria real.
4. Dónde se ganaría o perdería la segunda pelota.
5. Quién protege un despeje o recuperación rival.

## Lateral propio [inferencia táctica]

**Opción corta:** un receptor ofrece ángulo interior y una devolución. Útil si se puede continuar sin quedar encerrado. Falla cuando el rival anticipa la devolución y el ejecutor no tiene otra salida.

**Opción adelantada:** el receptor amenaza profundidad o disputa. Necesita una segunda pelota organizada y protección detrás.

**Reinicio hacia atrás:** puede cambiar el ángulo y conservar; no garantiza seguridad si la trayectoria cruza una zona interceptable.

La barrera rival no garantiza que el primer pase sea útil. Evitar ejecutar solo porque hay espacio visual: hay que prever quién podrá intervenir después de la liberación.

## Defender un lateral [inferencia táctica]

Respetar las restricciones y repartir amenazas. Un jugador atiende al receptor probable, otro la devolución o pase interior y otro protege profundidad, con ajuste del GK. La disposición exacta cambia con la ubicación del lateral.

Presionar el punto del saque puede ser ineficaz o imposible por la barrera. Preparar el siguiente contacto suele ser más útil que esperar una recuperación inmediata que el script no permite.

## Córner propio [inferencia táctica]

Dos familias de jugadas:

- **Envío para finalización:** un receptor prepara el contacto, otro la segunda pelota y otro la cobertura, contando al GK cuando corresponda.
- **Córner corto:** buscar una devolución o un ángulo diferente; requiere que la geometría y el reglamento lo permitan.

La potencia y la curva del script pueden hacer que una aproximación pequeña cambie el destino. Practicar con varias posiciones de llegada rival; una jugada que funciona sin oposición todavía no está validada para partido.

## Defender un córner [inferencia táctica]

Coordinar línea de tiro, receptor y rebote sin que todos ataquen la misma trayectoria. El disco de exclusión puede limitar la aproximación inicial: ubicar a cada jugador en función de lo que podrá defender cuando la pelota esté activa.

No presuponer que un toque corta la curva en todas las variantes; el contrato local distingue mecanismos diferentes. El GK debe anticipar la trayectoria de esa sala, no la de otro mapa.

## Saque de arco [inferencia táctica]

Una salida corta permite conservar si existe un receptor con continuidad. Una salida larga puede superar presión, siempre que haya disputa y segunda pelota. Preparar ambas vuelve menos fácil anticipar al equipo.

Los receptores deben revisar tiempo de llegada, posible velocidad elevada y ángulo de contacto. El ejecutor conserva una función después de sacar: reorganizar el fondo y ofrecer devolución si la jugada lo requiere.

## Saque inicial [inferencia táctica]

Definir quién toca y qué harán los otros tres. Una devolución, una salida lateral o un envío profundo tienen condiciones distintas. Evitar una espera indefinida o un primer pase cuya única salida sea perder contra una presión preparada.

## Ficha para una jugada preparada

Registrar nombre, variante y versión; tipo de saque; posición del ejecutor; destino y contacto esperado; alternativa si se cierra; responsables de rebote y cobertura; restricciones; señales de abandono; resultados con oposición. Una jugada debe tener una salida de emergencia.

## Límite importante de simulación

**[verificado: archivo local]** `TRAINING_DEADLINE` introduce un plazo de entrenamiento de 600 ticks. No equivale a una regla universal de partido. El simulador además tiene salvaguardas contra un saque inicial interminable. Distinguir restricciones del script real de mecanismos destinados a que un episodio de entrenamiento termine. [Contrato](../../../env/rs4z/contract.py).
