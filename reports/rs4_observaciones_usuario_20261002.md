# Observaciones de juego aportadas por el usuario

Fuente: siete capturas y relato de una reproducción. El rótulo visible indica `runs\multi\rs4.pt` contra `scripted(policy=r2,style=-1,eps=0.0)`. No se ha confirmado el hash del checkpoint ni la versión del simulador que generó esa reproducción. No atribuir automáticamente estos fallos al checkpoint control de 2.100M que se evalúa en el Pod. Una captura permite ver posiciones; duración, oscilación, trayectoria y secuencias proceden del relato del usuario.

| Situación | Evidencia aportada | Comportamiento deseado | Comprobación propuesta |
| --- | --- | --- | --- |
| Saque inicial; imágenes 1 y 5 | Avance individual, balón hacia adelante, compañeros permanecen cerca de mitad de cancha | Ofrecer receptores, conservar posesión y progresar con apoyos; presión y cobertura tras pérdida | Receptor distinto del ejecutor, retención tras saque, opciones de pase y reacción de compañeros después de habilitarse el juego |
| Ataque; imagen 2 | Jugadores próximos y sin opciones claras; rival poco exigente según el relato | Anchura, profundidad, desmarque y líneas de pase utilizables | Distancias, líneas bloqueadas, movimiento para abrir recepción; comparar rivales R2, R3 y checkpoints |
| Transición defensiva; imagen 3 | Espera hasta campo propio; dos jugadores oscilan sin intervenir según el relato | Presión útil al poseedor, cierre de líneas y cobertura coordinada | Tiempo hasta presión/intercepción y contribución de cada jugador después de perder posesión |
| Córner; imagen 4 y último episodio descrito | Ambos equipos dejan vencer saques en episodios distintos y el balón vuelve al centro | Ejecutor llega, se coloca y saca legalmente; compañeros ofrecen continuación | Propietario del saque, ejecutor, restricciones, acciones de kick, distancia, temporizador y motivo de reinicio |
| Pase rival; imagen 6 | Jugador cercano deja pasar el balón según el relato | Interceptar cuando sea alcanzable sin desproteger una amenaza mayor | Trayectoria y velocidades, ventana alcanzable de intercepción, decisión efectiva y cobertura |
| Ataque; imagen 7 | Dos atacantes frente a cuatro defensores y dos compañeros alejados | Incorporación de un apoyo útil cuando el riesgo lo permita, conservando cobertura del contraataque | Opciones de recepción, ocupación de espacios y riesgo de pérdida antes de exigir una incorporación |

## Hallazgos del código revisado

- `bots/scripted.py` distingue R2 histórico y R3 táctico. Ambos pasan por `_restart_actions`, que selecciona un ejecutor próximo y trata el córner con dirección hacia dentro. Esto no prueba que la rec usara este código ni que todos los saques funcionen: hace falta reproducir el caso.
- La rutina del scripted orienta el saque inicial hacia el arco rival; no selecciona un compañero receptor para ese saque. Es una limitación concreta del rival, no una explicación demostrada de la política neuronal roja.
- El entorno RS ONE 4v4 sin árbitro Pegeche configura `restart_timeout_terminal`: un saque vencido puede terminar el episodio. El render procesa reinicios de episodio, por lo que volver al centro puede ser un reset del entorno. Confirmar configuración y eventos de la rec antes de concluir que reproduce una regla competitiva.
- `env/rs4_tactics.py` contiene una guía geométrica de arquero, conductor/presionante y dos apoyos. Una guía o recompensa de posición no garantiza desmarques, coordinación temporal ni circulación útil.
- El contrato actual mide éxito de saques y córners, defensa, salida y ataque. Debe verificarse si esas pruebas detectan también los casos relatados durante partidos completos. Superar una prueba aislada no demuestra resolver todos los reinicios ni jugar con coordinación competitiva.

## Orden de trabajo

1. Identificar la rec, checkpoint y configuración; reproducir córners vencidos y documentar la causa antes de ajustar recompensas.
2. Contrastar las mismas situaciones con el checkpoint actual y rivales de la evaluación vigente.
3. Medir participación sin balón, receptores y reacción a pérdida/recuperación en secuencias completas.
4. Preparar cambios según causas reproducidas. No convertir una formación fija ni un número obligatorio de atacantes en objetivo universal.

No se modificó el entrenamiento, la evaluación activa, los pesos ni las recompensas al registrar estas observaciones.
