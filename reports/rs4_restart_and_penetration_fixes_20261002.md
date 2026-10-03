# Saques con un ejecutor y receptores cerca del arco

Observación del usuario en el checkpoint de 2300M: tres jugadores se acercan al
saque lateral; en ataque los apoyos esperan cerca del borde del área y no crean
suficientes opciones interiores. Los replays muestran organización prometedora,
pero los duelos contra el campeón de 1400M acabaron 0–0. No se atribuye causalidad
a una única imagen ni se considera demostrado que el actual sea superior.

## Incentivos encontrados y cambio

La guía del saque otorgaba 70% del potencial a la proximidad/alineación del
ejecutor y 30% a los tres apoyos. Un grupo junto al balón aún podía conservar una
puntuación elevada. Ahora se reparte 55%/45% y se reduce la puntuación según la
proximidad de los tres jugadores que no ejecutan. Se conserva la asignación
intercambiable: ningún jugador recibe un rol fijo ni una acción obligatoria.

La guía ofensiva premiaba dos líneas de pase sin valorar específicamente un
receptor interior cerca del arco. La formación v4 incorpora profundidad,
centralidad, espacio frente a defensores y una línea de pase libre. Permite el
pase atrás cuando el balón está cerca de la línea de fondo. La modificación sólo
actúa con ventaja geométrica de acceso al balón y en campo de ataque. Acerca
una de las referencias de apoyo al carril de finalización; no mueve la referencia
del arquero ni elimina la cobertura. Las versiones anteriores de la formación
siguen disponibles; el entrenador RS4 v3 selecciona v4 y lo imprime en sus logs.

Los ejercicios difíciles incluyen grupos de tres cerca de la banda y un portador
abierto frente a defensa compacta, con compañeros que aún deben entrar en juego.
Se mantienen ejercicios fáciles y difíciles, la fracción de partidos completos,
los plazos de viaje para saques y el resto de configuraciones de fase D.

No se añaden premios por entrar repetidamente al área: los términos siguen siendo
potenciales acotados y compartidos, con el mismo descuento y tratamiento terminal.
No se cambian observaciones, pesos, optimizador, calendario, fase o presupuesto.
Los checkpoints históricos no adquieren estas conductas por instalar el código;
el siguiente tramo PPO debe aprenderlas y luego validarse en partidos.

## Validación

79 pruebas locales: guía táctica, ejercicios, entrenador, registro de goles,
simetrías de color/eje/identidad, acotación, terminales y ciclos sin recompensa
positiva. Nuevas regresiones: receptor interior accesible frente a apoyo de
rebote, pase bloqueado, receptor marcado, cobertura trasera, defensa sin incentivo
de invasión, un ejecutor frente a tres, y presencia de las situaciones nuevas en
ambos colores. Las pruebas no demuestran todavía mejora competitiva del modelo.

El entrenamiento permanece detenido en 2300M. No se inicia otro tramo ni una
evaluación completa como parte de esta corrección.
