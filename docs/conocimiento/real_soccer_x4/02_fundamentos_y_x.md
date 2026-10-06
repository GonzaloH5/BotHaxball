# 02. Fundamentos y uso de X

[Volver al índice](README.md) · [Roles](03_roles_y_estructura.md)

## Movimiento y contacto

**[verificado: archivo local]** Los mapas objetivo usan aceleración normal 0,12, aceleración durante el estado de patada preparada 0,07 y amortiguación 0,96 para el jugador. El movimiento tiene inercia: soltar una dirección no detiene el disco instantáneamente. El simulador normaliza las diagonales; no les asigna una aceleración total mayor. [Motor físico](../../../sim/physics.py), [cargador de mapas](../../../sim/stadium.py).

**[inferencia táctica]** Para llegar bien a la pelota importa dónde vas a estar al contactar, no solamente cuánto te acercaste. Empezar a frenar o corregir antes ayuda a recibir con salida, mantener un ángulo y evitar empujar la pelota fuera.

Tres capacidades diferentes:

- **Llegada:** alcanzar a tiempo una trayectoria.
- **Orientación geométrica:** ubicar tu centro del lado que produce el siguiente contacto deseado. El disco no tiene una orientación corporal como un futbolista.
- **Control:** convertir el contacto en una pelota jugable para vos o un compañero.

## Qué hace X y por qué importa

«X» se usa como abreviatura de la entrada de patear, aunque el jugador utilice otra tecla equivalente.

**[verificado: archivo local]** La patada de juego abierto añade un impulso dirigido desde el centro del jugador hacia el centro de la pelota. La velocidad anterior de la pelota sigue formando parte de la trayectoria. En el motor local, el impulso es `kickStrength × invMass_pelota × dirección`; los saques pueden tener intervenciones adicionales del script. [Implementación de juego abierto](../../../env/rs4z/kernel.py).

**[verificado: fuente pública]** HaxBall permite configurar aceleración y amortiguación específicas durante la patada. El host también permite limitar la frecuencia de patadas. [PlayerPhysics](https://github.com/haxball/haxball-issues/wiki/Stadium-(.hbs)-File), [setKickRateLimit](https://github.com/haxball/haxball-issues/wiki/Headless-Host#setkickratelimit).

**[verificado: archivo local]** Una patada válida cancela el estado preparado; soltar la entrada lo rearma. Mantener X después de esa patada no genera por sí solo una sucesión ilimitada de patadas. Antes del contacto, mantener el estado preparado utiliza la aceleración de patada. Hay que distinguir tecla sostenida, patada armada y evento de patada efectivamente ejecutada. [Kernel RS4-Z](../../../env/rs4z/kernel.py).

**[inferencia táctica]** La importancia de X viene de combinar geometría, preparación y momento del contacto. Preparar demasiado temprano puede perjudicar una carrera; demasiado tarde puede hacer perder una ventana. Patear inmediatamente puede resolver una amenaza o desperdiciar una recepción que necesitaba control.

| Situación | Uso de X propuesto | Riesgo que hay que revisar |
|---|---|---|
| Carrera hacia una pelota lejana | Priorizar llegar; preparar cuando se acerca la ventana | Perder aceleración antes de tiempo |
| Pase de primera | Llegar con ángulo y patada disponible | Que el impulso salga hacia un rival |
| Recepción con espacio | Evaluar control antes de patear | Que la pelota se aleje por un contacto precipitado |
| Despeje ante gol inminente | Priorizar una salida segura y viable | Despejar al centro del área |
| Duelo de choque y patada | Preparar según el contacto esperado y su salida | Entrar sin cobertura o regalar el rebote |
| Preparar un segundo toque | Soltar y rearmar según la secuencia | Intentar patear con el estado cancelado |

Estas opciones son **inferencias tácticas**, no una pauta de pulsaciones fija para todas las jugadas.

## Pase

**[inferencia táctica]** Un buen pase no se define solo por tocar al compañero. Debe considerar:

1. **Destino:** a su posición actual o al espacio que puede alcanzar.
2. **Trayectoria:** rivales que pueden interceptar antes de la recepción.
3. **Velocidad:** una pelota demasiado fuerte puede volver inútil un receptor libre.
4. **Siguiente acción:** posibilidad de devolver, conducir, cambiar de lado o rematar.
5. **Consecuencia de la pérdida:** quién cubre si el rival gana el contacto.

El pase atrás tiene valor cuando cambia el ángulo o permite salir de la presión. Repetirlo sin mover al rival puede conservar la pelota y no producir ninguna ventaja.

## Recepción y primer contacto

**[inferencia táctica]** Identificá tu siguiente acción antes de recibir. Si te presionan, una devolución de primera puede ser mejor que intentar girar. Con espacio, un control puede mejorar el ángulo. Un receptor disponible durante un instante puede dejar de estarlo mientras viaja la pelota: hay que anticipar su trayectoria y la del rival.

La recepción útil deja pelota y jugador en una relación desde la cual se puede continuar. Recibir debajo de la trayectoria o empujarla contra la línea puede cerrar tus propias opciones.

## Remate

**[inferencia táctica]** Evaluá la abertura real entre arquero, defensores, postes y trayectoria. Un remate fuerte sin línea libre puede convertirse en salida rival. La colocación depende del contacto; apretar una dirección no garantiza que la pelota salga exactamente hacia allí.

Antes de rematar, comparar tiro, pase y control. Tener una ventana clara y corta puede justificar patear de primera. Una defensa ya alineada puede hacer preferible un pase lateral que cambie el punto del remate.

## Duelos y segundas pelotas

**[inferencia táctica]** En un choque, ganar es producir una salida favorable al equipo. Rebotar más fuerte que el rival pero dejarle la segunda pelota no resuelve el duelo. Revisá ángulo de entrada, velocidad relativa, disponibilidad de patada y ubicación del compañero que recogería el rebote.

«No ir» puede ser la decisión correcta si el contacto es desfavorable y esperar permite interceptar la salida. También puede ser un error si concede un remate libre: el contexto determina el valor.

## Latencia y anticipación

**[verificado: fuente pública]** El input lag es el retraso entre una entrada y su manifestación en pantalla; el navegador contribuye a ese retraso. [Documentación de input lag](https://github.com/haxball/haxball-issues/wiki/Input-Lag).

**[medido: reporte local]** El README del proyecto registra unos quince ticks, aproximadamente 250 ms, entre observación y actuación de sus bots cliente. Es una medición del sistema local, no el ping universal de jugadores humanos. [Estado del proyecto](../../../README.md), [herramienta de medición](../../../bridge/latency_probe.js).

**[inferencia táctica]** Con retraso, mirar únicamente la posición instantánea favorece llegar tarde. Anticipá trayectorias cortas y prepará márgenes. Una conexión inestable puede volver poco fiable una combinación que funcionaba con menor retardo.

## Errores frecuentes

Patear por costumbre; preparar X durante toda una carrera; perseguir la posición pasada de la pelota; recibir sin siguiente acción; rematar hacia el bloque; confundir una trayectoria modificada por el script con una técnica universal de efecto.
