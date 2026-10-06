# 12. Aplicación al proyecto HaxballRL

[Volver al índice](README.md) · [Plan del proyecto](../../PLAN.md)

Esta biblioteca aporta vocabulario y preguntas sobre el juego. Su contenido táctico no establece por sí solo una arquitectura, una recompensa o un método de aprendizaje.

## Estado relevante del proyecto

**[verificado: archivos locales]** El [README](../../../README.md) describe un reinicio del método de aprendizaje y conserva física, simulador, conversión de grabaciones y herramientas de conformidad. El [plan](../../PLAN.md) es un borrador y exige justificar decisiones de diseño con trabajos publicados, marcando las inferencias.

El motor, el árbitro y las observaciones son capas distintas. Un agente necesita actuar en la variante y con el retardo que se evalúa; aprender una combinación sin esa condición no demuestra que pueda ejecutarla en la sala.

## Traducir conceptos a observaciones y evaluación

**[propuesta de análisis, no implementación]**

| Concepto del juego | Información relevante | Qué se podría evaluar |
|---|---|---|
| Pase viable | Posición y movimiento de pelota, receptor e interceptores | Recepción y continuidad, separadas por contexto |
| Recepción útil | Trayectoria, geometría de contacto y presión | Siguiente acción favorable, no solo primer toque |
| Cobertura | Quién sale, amenazas abiertas y jugadores que pueden intervenir | Ocasiones concedidas tras pérdida o salto |
| Uso de X | Estado preparado, velocidad, distancia y contacto previsto | Ventanas perdidas y calidad de salida |
| Apoyo | Línea alcanzable y opciones posteriores del receptor | Conexiones que facilitan progresión o salida |
| Profundidad | Amenaza adelantada y respuesta defensiva | Ventajas creadas con y sin recepción |
| Estilo | Secuencias de decisiones en cada fase | Distribuciones de conducta y rendimiento contra distintos rivales |
| Reinicio | Tipo, equipo, fase, geometría y restricciones | Ejecución válida y continuidad en cada variante |

Inferir intención únicamente por trayectoria puede clasificar mal pases, despejes y rebotes. Si se usa anotación automática, conservar incertidumbre y comprobar ejemplos manuales.

## Métricas y objetivos no son equivalentes

**[inferencia]** Una métrica útil para diagnosticar puede ser perjudicial como objetivo aislado. Más pases pueden significar circulación sin ocasión. Más cercanía a la pelota puede destruir apoyos. Más amplitud puede alejar a los receptores. Una preferencia de estilo puede impedir una respuesta buena ante otro rival.

Por eso los capítulos 08 y 09 proponen observar consecuencias y condiciones. Cualquier cambio de aprendizaje debe seguir el procedimiento y evidencia del plan; aquí no se añade una recompensa táctica nueva.

## Datos humanos y alcance del conocimiento

**[verificado: documentación local]** El README indica que las grabaciones disponibles se concentran en RS ONE y que contienen cambios de estadio. El reporte de Sanguchito advierte que penales y otros mapas deben separarse de los tramos objetivo. [README](../../../README.md), [reporte de conformidad](../../../reports/sanguchito_conformance.md).

**[inferencia]** Reproducir las acciones de esa muestra no demuestra dominar una competición distinta ni su meta. Hace falta distinguir sala pública, práctica y liga; roles, rivales, versiones y retardo también pueden influir.

## Información del actor

**[verificado: archivo local]** [obs_v2](../../../env/rs4z/obs_v2.py) incluye información pública de jugador, pelota y otras entidades, historial y características del reinicio. Su documentación excluye marcador y reloj del actor y los coloca en las características del crítico.

La recomendación humana de ajustar riesgo al marcador no se convierte automáticamente en una conducta condicionada del actor actual. Hay que distinguir conocimiento táctico humano y decisiones de observación del proyecto.

## Dos puntos de consistencia para futuros análisis

**[verificado: archivos locales]** En `rs_one.hbs`, `bg.height` es 670, el contrato del árbitro usa 670 y el metadato `haxballrl.field_half_h` es 600. [Mapa](../../../stadiums/rs_one.hbs), [contrato](../../../env/rs4z/contract.py), [cargador](../../../sim/stadium.py). Registrar qué dimensión consume cada herramienta antes de interpretar ancho, distancias o normalizaciones. Esta biblioteca no modifica el código ni resuelve la discrepancia.

**[verificado: archivo local]** El contrato mantiene supuestos pendientes en partes del script de 2K23 y en el plazo del lateral de Sanguchito. La conformidad física de juego abierto no demuestra conformidad total del árbitro, del reloj y de todos los reinicios.

## Uso recomendado de la biblioteca

**[propuesta]** Usarla para formular situaciones de evaluación: 2v1, salida presionada, relevo de cobertura, pérdida central, duelo con segunda pelota y reinicio con alternativas. Guardar estado inicial, variante, retardo, responsabilidades, resultado y contraejemplos.

Un agente competitivo debe resolver situaciones distintas contra rivales distintos. Un agente que se parece a los humanos debe compararse además con conductas humanas representativas. Son preguntas relacionadas, con evidencia diferente.
