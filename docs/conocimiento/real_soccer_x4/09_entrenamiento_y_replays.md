# 09. Entrenamiento y revisión de replays

[Volver al índice](README.md) · [Cualidades](08_cualidades_y_evaluacion.md)

**[propuesta de entrenamiento y análisis]** Los ejercicios y métricas de este capítulo son herramientas de trabajo. No se presentan como un programa experimentalmente validado ni como estándares oficiales.

## Entrenar una falla concreta

Identificar situación, decisión y ejecución. Si se pierde al recibir presionado, practicar únicamente remates sin oposición no aborda el problema. Si dos jugadores abandonan la cobertura, un ejercicio individual de control tampoco basta.

Elegir un objetivo por bloque, repetir con variaciones y revisar el resultado bajo oposición. La progresión depende de la calidad que se observa, no de completar un número arbitrario de días.

## Biblioteca de ejercicios

| Ejercicio | Objetivo | Cómo hacerlo | Señal de mejora |
|---|---|---|---|
| Contacto orientado | Llegar con geometría útil | Recibir trayectorias desde varios lados y continuar hacia un destino elegido | Menos contactos que cierran la siguiente acción |
| Preparar y rearmar X | Coordinar llegada y patada | Alternar carrera, primer contacto y segunda patada | Menos ventanas perdidas por estado de patada |
| Pase y nueva función | Salir del hábito de quedarse tras pasar | Dos o tres jugadores pasan y cambian apoyo o profundidad | El siguiente receptor dispone de alternativas |
| 2v1 con GK opcional | Leer decisión del defensor | Variar distancias y posiciones; añadir arquero cuando corresponda | Mejor elección entre tiro y pase |
| Salida bajo presión | Conectar sin perder cobertura | Repetir salidas desde posiciones de partido con distintos ángulos rivales | Más progresiones jugables y menos pérdidas peligrosas |
| Defensa de 2v1 | Retrasar y coordinar protección | Variar quién tiene tiro y dónde puede intervenir el GK | Menos remates libres, incluso sin recuperar |
| Duelo y segunda pelota | Preparar continuidad | Un jugador disputa; otros ajustan apoyo y cobertura | La salida del contacto encuentra a alguien preparado |
| Pérdida con retorno | Elegir presión o repliegue | Iniciar desde una pérdida real y alternar ventajas de llegada | Se protege la amenaza antes de perseguir |
| Reinicio con oposición | Validar jugada preparada | Repetir lateral, córner o saque de arco en el mapa exacto | Recepción útil y cobertura de despejes |
| Partido condicionado | Transferir fundamentos | Observar un objetivo concreto sin convertirlo en obligación durante toda jugada | El comportamiento aparece también en partido libre |

Una condición como «dar tres pases antes de tirar» puede practicar paciencia, pero puede enseñar a ignorar un remate claro. Retirarla al comprobar transferencia al juego libre.

## Sesión de ejemplo

Sesenta minutos ajustables: diez de contactos y patada; quince de una situación pequeña; quince de reinicios o transiciones; quince de partido; cinco de revisión. Cambiar la distribución según la falla prioritaria.

Con pocos jugadores, trabajar técnica y pequeños duelos. Para evaluar funciones de los cuatro puestos, se necesitan situaciones donde existan esas responsabilidades, no únicamente pruebas 1v1.

## Cómo revisar una jugada

1. Confirmar variante y fase del partido.
2. Pausar antes de la decisión, cuando todavía no se conoce el desenlace.
3. Anotar las opciones realmente viables, no soluciones imposibles por velocidad o distancia.
4. Observar decisión, contacto y reorganización.
5. Separar consecuencia inmediata de continuidad: recibir no significa conservar después.
6. Identificar un cambio concreto y su posible coste.
7. Buscar otras jugadas similares, incluidas las que sí funcionaron.

Revisar solo errores puede hacer que un equipo abandone una buena estrategia por unas pocas ejecuciones fallidas. Revisar solo highlights oculta las pérdidas y amenazas concedidas.

## Ficha de anotación

```text
Replay y fecha:
Mapa / script / tramo:
Equipo y rol:
Tiempo y fase:
Posiciones y trayectorias relevantes:
Opciones viables antes del contacto:
Decisión observada:
Calidad de ejecución:
Cobertura y apoyos disponibles:
Resultado inmediato y continuidad:
Alternativa propuesta y riesgo:
Confianza: alta / media / baja:
```

La confianza refleja cuánto se puede concluir del replay. El estado grabado no revela perfectamente la información que vio cada humano bajo su latencia.

## Métricas propuestas y sus límites

| Métrica | Definición operacional propuesta | Límite |
|---|---|---|
| Pase completado | Envío compatible con pase recibido por compañero antes de intervención rival o salida | La trayectoria sola no revela intención; separar contactos dudosos |
| Continuidad de recepción | El receptor conserva o produce una siguiente acción favorable | Requiere definir ventana y criterio antes de comparar |
| Pérdida peligrosa | Pérdida tras la que el rival obtiene una amenaza directa según criterio predefinido | No todas las pérdidas son responsabilidad exclusiva del portador |
| Ocasión clara | Remate o recepción de remate con trayectoria viable y oposición insuficiente | Es una etiqueta de revisión; no equivale a un modelo xG calibrado |
| Segunda pelota ganada | Primer control favorable después de un contacto dividido o despeje | «Control» exige criterio explícito |
| Reinicio útil | Saque legal que produce recepción o continuidad favorable | Comparar por tipo de saque y variante |
| Presión útil | Presión que restringe opciones o contribuye a recuperar sin conceder una amenaza mayor | Puede requerir revisión manual |
| Fallo de cobertura | Amenaza facilitada por abandonar una función sin relevo viable | Necesita considerar responsabilidades acordadas |

Separar toques, patadas, pases y despejes. Un contacto con el cuerpo puede pasar la pelota; una patada puede ser remate o despeje. Si no se puede clasificar, usar «desconocido» en vez de forzar la etiqueta.

## Comparaciones válidas

Normalizar por minutos efectivos u oportunidades cuando corresponda. Si hay muchos reinicios, los pases por minuto total pueden disminuir sin empeorar la asociación. Comparar por puesto, variante, rival y fase; informar numerador y denominador.

No usar el porcentaje de posesión, el ancho del equipo o la cantidad de pases como prueba única de calidad. Más amplitud puede abrir líneas y también dejar a un jugador fuera de alcance. Más pases pueden construir una ocasión o evitar indefinidamente un ataque viable.

## Criterio de progreso

La falla aparece menos en situaciones comparables y la solución se mantiene con oposición y en partido libre. Si mejora el ejercicio pero no el partido, revisar si el ejercicio reproduce decisión, presión, latencia y consecuencias reales.
