# 01. Modalidad, reglas y mapas

[Volver al índice](README.md) · [Fuentes](FUENTES.md)

## Qué es Real Soccer X4

**[verificado: fuente pública]** HaxARG distingue Real Soccer X4, Real Soccer ONE X4 y otras modalidades en su bot oficial. HaxBall describe su juego como una combinación de fútbol y hockey de aire y destaca el trabajo de equipo en 3v3 y 4v4. [Bot oficial de HaxARG](https://haxarg.com/botoficial), [HaxBall: About](https://www.haxball.com/about).

En esta biblioteca, X4 significa cuatro contra cuatro **contando al arquero**. La estructura de referencia tiene arquero, defensor, volante y delantero, aunque las funciones pueden intercambiarse.

**[verificado: archivo local]** Las variantes implementadas en el proyecto distinguen juego abierto y reinicios por lateral, córner y saque de arco. El árbitro de la sala se modela aparte del motor físico. [Contrato local de la sala](../../../env/rs4z/contract.py).

**[inferencia táctica]** La posibilidad de salir de la cancha cambia el valor de la banda: una trayectoria que en un mapa cerrado devolvería la pelota mediante una pared puede terminar en un reinicio. También aumenta la importancia de conservar un receptor disponible y decidir quién disputa la segunda pelota. Los arcos, postes y otros objetos siguen teniendo sus propias colisiones; «cancha abierta» no significa ausencia de geometría física.

## Tres capas que no conviene confundir

| Capa | Determina | Ejemplo |
|---|---|---|
| Motor y mapa | Radios, impulso, aceleración, colisiones y líneas de gol | Tamaño de pelota y rebote en un poste |
| Script de sala | Cobro y ejecución de saques, barreras y modificaciones dinámicas | Posicionar la pelota y restringir rivales en un lateral |
| Reglamento de competencia | Validez del partido y procedimientos deportivos | Host autorizado, firmas, pausas y grabación |

**[verificado: fuente pública]** El formato `.hbs` permite configurar propiedades físicas; la API del host permite modificar discos y jugadores durante el partido. Por eso leer únicamente el mapa no reconstruye todas las reglas de una sala. [Formato de estadios](https://github.com/haxball/haxball-issues/wiki/Stadium-(.hbs)-File), [API Headless Host](https://github.com/haxball/haxball-issues/wiki/Headless-Host).

## Variantes presentes en HaxballRL

**[verificado: archivos locales]** Valores de las versiones guardadas en el repositorio. Las dimensiones corresponden al rectángulo dibujado por `bg`; son unidades de coordenadas del juego, no metros ni píxeles constantes de pantalla.

| Propiedad | HAXARG 2K23 | RS ONE | SANGUCHITO RS X4 |
|---|---:|---:|---:|
| Cancha: largo × ancho completos | 2300 × 1200 | 2300 × 1340 | 2300 × 1340 |
| Límites del fondo de cancha | x = ±1150; y = ±600 | x = ±1150; y = ±670 | x = ±1150; y = ±670 |
| Línea de gol | x = ±1162 | x = ±1162 | x = ±1159 |
| Semiabertura del arco | 124 | 124 | 123,95 |
| Radio de pelota | 9 | 8,325 | 8,325 |
| Radio del jugador | 15 | 15 | 15 |
| `kickStrength` | 5,65 | 5,85 | 5,85 |
| `invMass` de pelota en el mapa | 1,05 | 1,05 | 1,05 |

Fuentes: [2K23](../../../stadiums/haxarg_2k23.hbs), [RS ONE](../../../stadiums/rs_one.hbs), [Sanguchito](../../../stadiums/sanguchito_rs_x4.hbs). Radios y parámetros heredados del jugador se resuelven mediante [el cargador local](../../../sim/stadium.py).

**[inferencia táctica]** Una cancha más angosta reduce el espacio transversal y altera apoyos y cambios de lado. Una pelota de otro radio cambia las distancias de contacto. Un impulso distinto modifica ventanas de intercepción. Por eso conviene practicar en la variante de competencia y repetir allí las jugadas preparadas.

**[verificado: archivo local]** En RS ONE el metadato adicional `haxballrl.field_half_h` vale 600, mientras `bg.height` y el contrato del árbitro usan 670. Ese metadato tiene uso interno y no redefine por sí solo las líneas del partido. Para analizar geometría, registrar qué capa se está usando; la discrepancia está documentada en el [capítulo 12](12_aplicacion_a_haxballrl.md).

## Reglas que se deben comprobar por sala

La etiqueta «RS X4» no basta para resolver estas preguntas:

- ¿Cuánto dura cada tiempo y quién controla el adicional?
- ¿Cuándo se puede pausar, sustituir o reanudar?
- ¿Cómo se ejecuta cada saque y qué ocurre si se demora o se ejecuta mal?
- ¿Qué restricciones existen para rivales y compañeros durante el saque?
- ¿Se permite gol directo desde ese reinicio?
- ¿Hay reglas adicionales de defensa, contacto o fuera de juego?
- ¿Qué límites de patadas, modificaciones del cliente o ayudas externas se permiten?
- ¿Qué mapa y qué versión del bot son obligatorios?

**[pendiente]** No se establece aquí una respuesta universal sobre offside, faltas, 3def o goles directos de lateral. Las restricciones de otra modalidad y las reglas del fútbol de once no se transfieren automáticamente a RS X4.

## Información relevante para competir en HaxARG

**[verificado: fuente pública, consulta 06/10/2026]** El reglamento enlazado por HaxARG exige el bot oficial o un sistema asociado autorizado, con alojamiento en Argentina. Contempla identificación mediante firmas, condiciones de pausa y grabación completa del encuentro. En el apartado de problemas del temporizador describe tiempos controlados por el bot de diez minutos más agregado; no debe confundirse con poner diez minutos en el temporizador nativo. Verificar además el anuncio particular del torneo. [Reglamento oficial, capítulo 4](https://docs.google.com/document/d/16eDMrzs8D8B4hvKmGitLcLRVr0lfi4FJXUGcoXtWCCs/edit).

**[verificado: fuente pública]** El historial del bot registra cambios de RS ONE en julio y agosto de 2026 relacionados con laterales. Una grabación anterior puede reflejar otra implementación. [Historial del bot oficial](https://haxarg.com/botoficial).

## Checklist de contexto para un replay

Registrar nombre del mapa, versión del script si se conoce, cantidad de jugadores, tipo de partido, fecha, duración efectiva y condiciones de conexión. Si el replay cambia de estadio para una tanda de penales o entrenamiento, separar esos tramos del análisis de juego abierto.
