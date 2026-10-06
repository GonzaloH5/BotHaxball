# 05. Defensa y transiciones

[Volver al índice](README.md) · [Manual de decisiones](11_manual_de_decisiones.md)

**[inferencia táctica]** Las prioridades y respuestas de este capítulo son propuestas para resolver amenazas, no restricciones del árbitro.

## Prioridad defensiva

Primero resolver la amenaza inmediata de gol. Después, dificultar el pase o progresión que produciría esa amenaza. Recuperar con una salida jugable es deseable cuando no compromete lo anterior.

Esto no obliga a defender siempre al lado del arco. Ceder un receptor libre puede permitir un remate mejor que una presión coordinada más lejos.

## Presión, cobertura y equilibrio

| Función | Trabajo |
|---|---|
| Presión | Reducir tiempo y opciones del portador; orientar su salida |
| Cobertura | Resolver la amenaza que aparece si el primer defensor es superado |
| Equilibrio | Proteger zonas y receptores que no están en el primer duelo |

El jugador que presiona debe hacerlo desde un ángulo que dificulte una opción peligrosa. Correr de frente sin apoyo puede permitir al rival elegir ambos lados.

La cobertura no consiste en seguir al primer defensor unos pasos detrás. Debe proteger la trayectoria que quedaría abierta. El GK coordina la profundidad y la protección del arco; también puede salir cuando la lectura de llegada lo justifica.

## Defender líneas de pase

Una marca debe considerar pelota, rival y destino. Estar al lado de un delantero puede impedir un remate y permitir un pase al espacio que recibe sin presión. Estar en una línea de pase puede impedir esa recepción y conceder otra.

El equipo decide qué opción tolera y cómo reaccionará. Orientar hacia la banda tiene sentido si hay una trampa preparada allí; puede ser perjudicial si el rival puede progresar por fuera y nadie cubre.

## Cuándo saltar a presionar

Disparadores propuestos:

- Una recepción aleja la pelota del control rival.
- El portador queda contra la línea con pocas salidas.
- Viaja un pase que permite llegar antes del siguiente contacto.
- El rival necesita corregir su ángulo y dispone de menos tiempo.
- Un compañero ya puede cubrir la salida más peligrosa.

Ningún disparador garantiza recuperación. Comparar tiempo de llegada, salida probable y protección detrás. Si el atacante puede rematar antes, la prioridad vuelve al arco.

## Cuándo contener

Contener puede ser mejor si no llegás a disputar con ventaja y un salto dejaría un pase fácil. Se busca retrasar o restringir al rival mientras se reconstruye la cobertura.

Esperar sin cerrar ninguna opción permite al rival mejorar la jugada. Una contención útil cambia su decisión o compra tiempo para el equipo.

## Transición tras pérdida

Evaluar dos respuestas:

**Recuperar cerca:** si hay jugadores próximos con buena llegada y cobertura, uno disputa y otros cierran salidas. Si la pelota o el rival ya escaparon, seguir persiguiendo puede abrir más espacio.

**Replegar:** proteger primero la vía directa al arco, asignar el portador y cubrir el receptor más peligroso. Retroceder todos hasta la línea de gol puede conceder demasiado tiempo al rival.

La primera reacción no siempre debe hacerla quien perdió la pelota. Un compañero puede tener mejor ángulo y permitir que el primero recupere cobertura.

## Transición tras recuperación

Mirar hacia adelante antes de reciclar por costumbre. Si existe un receptor con ventaja y trayectoria viable, acelerar puede explotar al rival desorganizado. Si la recuperación dejó una pelota incómoda o no hay apoyo, asegurar un contacto puede ser mejor.

El compañero libre debe prepararse para recibir en movimiento o devolver de primera. El jugador más retrasado sostiene la seguridad del contraataque, salvo que se decida asumir otro riesgo.

## Defender un 2v1

El defensor intenta dificultar dos amenazas el tiempo suficiente para permitir retorno o intervención del GK. Puede posicionarse entre líneas, retrasar al portador y ajustar cuándo comprometerse.

Si el portador tiene un tiro inmediato, negar únicamente el pase no alcanza. Si el GK ya cierra al portador, cubrir al receptor puede ser más útil. No hay una obligación universal de marcar siempre al mismo atacante.

## Duelos cerca del arco

Evaluar dónde sale la pelota si el contacto se gana, se pierde o rebota. Un despeje hacia el centro puede dejar un segundo remate. Un contacto lateral puede reducir el peligro, aunque entregue un reinicio.

La elección también depende del script y del valor del saque concedido. Evitar un gol inmediato suele justificar un despeje menos ambicioso.

## Coordinación GK–DEF

Caso típico: el delantero rival recibe por banda. Si DEF puede negar el pase interior, GK protege la línea de remate y prepara su reacción. Si DEF queda superado y GK debe salir, un compañero intenta cubrir la portería o el receptor, según la amenaza real.

Salir ambos sin acuerdo puede hacer que una devolución elimine a los dos. Quedarse ambos puede conceder un tiro sin presión. La solución depende de quién llega y qué función queda descubierta.

## Gestión del riesgo

Una ventaja en el marcador y poco tiempo disponible pueden aumentar el valor de conservar estructura. Estar perdiendo puede justificar más presión y participación ofensiva, con mayor exposición. La calidad de la ocasión sigue importando: apurarse no crea por sí solo una ventaja.

Para el bot del proyecto, el acceso a marcador y reloj tiene restricciones de diseño propias; ver [capítulo 12](12_aplicacion_a_haxballrl.md).

## Diagnóstico de un gol recibido

Retroceder hasta la primera decisión que hizo difícil defender: pérdida con mala cobertura, pase que eliminó una línea, salto sin relevo, llegada tardía al rebote o intervención fallida del GK. Registrar varios factores si los hay. Culpar al último contacto impide mejorar la secuencia que produjo el tiro.
