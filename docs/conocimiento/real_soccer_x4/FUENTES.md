# Fuentes y grado de evidencia

[Volver al índice](README.md)

Consulta realizada el **6 de octubre de 2026**. Las recomendaciones tácticas están identificadas como inferencias o propuestas; no se presentan como conclusiones de estas fuentes oficiales.

## Fuentes públicas primarias

| ID | Fuente | Qué permite comprobar |
|---|---|---|
| P1 | [HaxBall: About](https://www.haxball.com/about) | Naturaleza del juego y relevancia del trabajo de equipo |
| P2 | [Stadium (.hbs) File, wiki oficial](https://github.com/haxball/haxball-issues/wiki/Stadium-(.hbs)-File) | Propiedades configurables de mapas, discos y jugadores |
| P3 | [Headless Host, wiki oficial](https://github.com/haxball/haxball-issues/wiki/Headless-Host) | Cambios dinámicos de propiedades y límites de patadas |
| P4 | [Input Lag, wiki oficial](https://github.com/haxball/haxball-issues/wiki/Input-Lag) | Concepto de retardo de entrada y contribución del navegador |
| P5 | [Bot oficial de HaxARG](https://haxarg.com/botoficial) | Modalidades y cambios publicados del bot |
| P6 | [Reglamento oficial de HaxARG](https://docs.google.com/document/d/16eDMrzs8D8B4hvKmGitLcLRVr0lfi4FJXUGcoXtWCCs/edit), enlazado desde [HaxARG](https://haxarg.com/reglamento) | Requisitos de competición; especialmente capítulo 4 y apartado de problemas del temporizador |

P6 se leyó mediante la exportación pública de texto del documento enlazado por la organización. Los anuncios específicos de cada competición y las actualizaciones posteriores deben consultarse antes de aplicar una regla.

La página oficial de input lag también incluye recomendaciones técnicas fechadas; esta biblioteca utiliza su definición del fenómeno. No prescribe una configuración universal del navegador.

## Fuentes locales primarias y documentación del proyecto

| ID | Archivo | Alcance |
|---|---|---|
| L1 | [HAXARG 2K23](../../../stadiums/haxarg_2k23.hbs) | Configuración de la copia del mapa guardada |
| L2 | [Real Soccer ONE](../../../stadiums/rs_one.hbs) | Geometría, física y metadatos locales |
| L3 | [SANGUCHITO RS X4](../../../stadiums/sanguchito_rs_x4.hbs) | Configuración de la variante exportada |
| L4 | [Cargador de estadios](../../../sim/stadium.py) | Valores heredados y resolución de parámetros |
| L5 | [Motor físico](../../../sim/physics.py) | Movimiento, contacto y patada del simulador |
| L6 | [Contrato del árbitro](../../../env/rs4z/contract.py) | Mecanismos por variante; distingue medidas y supuestos |
| L7 | [Kernel RS4-Z](../../../env/rs4z/kernel.py) | Implementación de juego abierto y reinicios |
| L8 | [Reporte de Sanguchito](../../../reports/sanguchito_conformance.md) | Comparaciones medidas contra grabaciones y límites declarados |
| L9 | [Observación v2](../../../env/rs4z/obs_v2.py) | Información del actor y del crítico |
| L10 | [README del proyecto](../../../README.md) | Estado documentado, corpus disponible y medida de retardo |
| L11 | [Medidor de latencia](../../../bridge/latency_probe.js) | Definición y medición del retraso efectivo del cliente |
| L12 | [Plan de aprendizaje](../../PLAN.md) | Borrador de diseño y criterio de evidencia del proyecto |

Un archivo local prueba qué está representado en esa versión. La correspondencia con una sala se fundamenta mediante grabaciones y conformidad, con los límites de la muestra. Los resultados del reporte se citan como resultados ya documentados; esta tarea no vuelve a ejecutar la conformidad.

## Mapa de afirmaciones

| Afirmación | Evidencia | Límite |
|---|---|---|
| Las variantes difieren en radio, impulso o geometría | L1–L4 | Copias locales, no todos los mapas con el mismo nombre |
| El script modifica propiedades durante saques | P3, L6–L8 | Cada script tiene su propia implementación |
| Mantener y rearmar la patada son estados diferentes | L5, L7 | El host puede añadir límites de frecuencia |
| Preparar X cambia la aceleración en el motor local | P2, L4–L7 | Distinguir tecla, estado preparado y patada ejecutada |
| Sanguchito tiene conformidad documentada con grabaciones | L8 | Muestra pequeña y condiciones de la comparación |
| Los bots del proyecto tienen retardo medido | L10–L11 | No extrapolar a todos los clientes humanos |
| Es útil repartir presión, cobertura y apoyo | Inferencia táctica de la biblioteca | Validar frente a situaciones y rivales concretos |
| Un estilo domina el meta actual de RS X4 | Pendiente | Falta una muestra competitiva representativa |

## Pendientes de conocimiento

1. Establecer tendencias competitivas por liga, mapa y período mediante replays representativos.
2. Medir eficacia de estilos controlando nivel del equipo y rival.
3. Validar partes pendientes del árbitro de 2K23 y el plazo del lateral de Sanguchito.
4. Contrastar cualidades por rol con anotaciones y situaciones, evitando valoraciones basadas solo en goles.
5. Medir retardo y variabilidad por cliente en vez de reutilizar una cifra general.

## Cómo añadir conocimiento

Registrar afirmación, etiqueta de evidencia, fuente, fecha, versión, contexto, muestra y contraejemplos. Si cambia el script, marcar qué capítulos necesitan revisión. Si se aporta un consejo personal, mantenerlo como inferencia o hipótesis hasta disponer de evidencia suficiente.
