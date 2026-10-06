# Biblioteca de conocimiento: HaxBall Real Soccer X4

Versión 1.0 · 6 de octubre de 2026 · Idioma: español.

Una base para aprender, dirigir, analizar y modelar el juego de **cuatro jugadores por equipo, incluido el arquero**. El foco está en las variantes del proyecto: HAXARG 2K23, Real Soccer ONE y SANGUCHITO RS X4.

**Idea central [inferencia táctica]:** jugar bien consiste en crear una siguiente acción favorable sin perder la capacidad de defender una pérdida. La técnica importa porque permite ejecutar esa decisión. La coordinación importa porque determina qué decisiones están disponibles.

## Índice

| Capítulo | Contenido | Sirve para |
|---|---|---|
| [01. Modalidad, reglas y mapas](01_modalidad_reglas_mapas.md) | Qué es RS X4, diferencias entre variantes y límites de lo verificado | Entender el contexto correcto |
| [02. Fundamentos y uso de X](02_fundamentos_y_x.md) | Movimiento, inercia, contactos, pase, remate, duelos y patada | Mejorar la ejecución |
| [03. Roles y estructura](03_roles_y_estructura.md) | Arquero, defensor, volante, delantero y relevos | Organizar los cuatro puestos |
| [04. Ataque colectivo](04_ataque_colectivo.md) | Salida, apoyos, amplitud, profundidad, superioridades y definición | Crear ocasiones |
| [05. Defensa y transiciones](05_defensa_y_transiciones.md) | Presión, cobertura, líneas de pase, repliegue y contraataque | Reducir ocasiones rivales |
| [06. Pelota parada](06_pelota_parada.md) | Laterales, córners, saques de arco y saque inicial | Preparar reinicios |
| [07. Estilos y meta](07_estilos_y_meta.md) | Posesión, juego directo, presión, bloque bajo, híbridos y adaptaciones | Elegir cómo competir |
| [08. Cualidades y evaluación](08_cualidades_y_evaluacion.md) | Qué hace bueno a un jugador; importancia por rol y contexto | Evaluar sin depender de goles |
| [09. Entrenamiento y replays](09_entrenamiento_y_replays.md) | Ejercicios, progresión, métricas y revisión de partidos | Convertir conocimiento en mejora |
| [10. Glosario y preguntas frecuentes](10_glosario_y_preguntas.md) | Terminología, errores de interpretación y mitos | Consultar conceptos |
| [11. Manual de decisiones](11_manual_de_decisiones.md) | Situaciones concretas, respuestas y excepciones | Aplicar los principios durante el juego |
| [12. Aplicación a HaxballRL](12_aplicacion_a_haxballrl.md) | Traducción a observaciones y evaluación; límites de los datos | Usar la biblioteca en el proyecto |
| [Fuentes y grado de evidencia](FUENTES.md) | Documentación oficial, reglamento y evidencia local | Auditar las afirmaciones |

## Cómo interpretar las afirmaciones

- **[verificado: fuente pública]:** comprobado en documentación del juego o de la organización citada.
- **[verificado: archivo local]:** presente en el mapa o en la implementación del proyecto. Describe esa versión, no todas las salas.
- **[medido: reporte local]:** resultado documentado de una comparación con grabaciones. Su alcance es la muestra utilizada.
- **[inferencia táctica]:** propuesta razonada a partir de la geometría y las opciones de juego. Puede tener excepciones.
- **[hipótesis por validar]:** afirmación que necesita contrastarse con partidos o experimentos.
- **[pendiente]:** dato que todavía no se pudo establecer con la evidencia disponible.

En los capítulos tácticos, la etiqueta de apertura se aplica a sus recomendaciones salvo que un bloque indique otra evidencia. Esta biblioteca no atribuye sus consejos a una supuesta estadística de jugadores profesionales.

## Los fundamentos que conviene aprender primero

Estos diez principios son **inferencias tácticas**, no reglas de la sala:

1. Mirá pelota, compañeros, rivales y espacio antes de actuar.
2. Prepará el ángulo de contacto antes de patear.
3. Ofrecé una recepción que permita continuar la jugada.
4. Después de pasar, buscá una nueva función: apoyo, profundidad o cobertura.
5. Evitá que dos compañeros compitan por el mismo contacto sin un motivo concreto.
6. Protegé la vía central hacia tu arco y coordiná quién sale a presionar.
7. Cuando un compañero abandona una zona importante, alguien debe asumir su función.
8. Al recuperar, comprobá si hay ventaja para acelerar; al perder, comprobá si hay condiciones para recuperar o conviene replegar.
9. Prepará las segundas pelotas y la defensa de tus propios ataques.
10. Ajustá el riesgo a la situación, la variante y la capacidad real del equipo.

## Rutas de lectura

**Para empezar:** 01 → 02 → 03 → 11. Después, ataque y defensa.

**Para un jugador con experiencia:** 02 → 08 → 09, más el capítulo de su rol y las situaciones que falla.

**Para un capitán o DT:** 03 → 04 → 05 → 06 → 07 → 09.

**Para HaxballRL:** leer primero el juego y después 12. La biblioteca describe comportamientos deseables y preguntas de evaluación; no reemplaza el [plan de aprendizaje del proyecto](../../PLAN.md).

## Qué significa «meta» aquí

El meta es el conjunto de estrategias que resultan eficaces y de las respuestas que generan en una comunidad y un período determinados. **No se verificó un meta competitivo universal vigente en octubre de 2026:** consultar fuentes y física no equivale a analizar una muestra representativa de partidos recientes. El capítulo 07 ofrece un marco de estilos y un procedimiento para comprobar tendencias.

## Mantenimiento

Actualizar fecha y fuentes cuando cambien el mapa, el bot oficial o el reglamento. Un resultado de una sala pública no debe extrapolarse automáticamente a una liga. Para añadir una tendencia al meta, registrar variante, competencia, fechas, rivales, tamaño de muestra y contraejemplos.
