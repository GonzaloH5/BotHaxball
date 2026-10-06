# 10. Glosario y preguntas frecuentes

[Volver al índice](README.md)

Los términos tácticos se usan aquí como convenciones de análisis. Los nombres y abreviaturas pueden variar entre comunidades.

## Glosario

| Término | Significado en esta biblioteca |
|---|---|
| RS / RS X4 | Real Soccer, aquí cuatro jugadores por equipo contando al GK |
| GK / arquero | Función principal de protección del arco |
| DEF | Defensor que sostiene cobertura y participa en salida |
| MED / volante | Jugador que conecta apoyos, progresión y equilibrio |
| DEL / FW | Referencia de profundidad, apoyo ofensivo y finalización |
| X | Entrada de patada, aunque se use otra tecla |
| Patada armada | Estado que permite efectuar una patada cuando existe alcance |
| Contacto orientado | Contacto cuya geometría produce una salida útil |
| Apoyo | Posición que permite ayudar al portador mediante recepción u otra función |
| Amplitud | Separación transversal que busca abrir líneas o marcas |
| Profundidad | Amenaza hacia espacios más adelantados |
| Fijar | Mantener ocupado a un rival mediante una amenaza creíble |
| Desmarque | Movimiento para obtener una recepción o modificar la marca |
| Pared | Pase y devolución coordinados para superar una oposición |
| Tercer jugador | Uso de un receptor intermedio para habilitar a otro |
| Cambio de orientación | Trasladar el ataque a otro lado de la cancha |
| Superioridad numérica | Más jugadores relevantes que rivales en una situación local |
| Superioridad posicional | Ubicación que permite una acción favorable aunque no sobren jugadores |
| Presión | Acción para reducir tiempo y opciones del rival |
| Cobertura | Protección de la amenaza que puede abrir otro movimiento |
| Relevo | Asumir la función que abandona un compañero |
| Rotación | Intercambio coordinado de funciones o posiciones |
| Bloque | Organización defensiva del conjunto |
| Repliegue | Recuperar posiciones defensivas después de una progresión o pérdida |
| Transición | Cambio de comportamiento al ganar o perder control |
| Contraataque | Ataque que intenta aprovechar la desorganización rival tras recuperar |
| Segunda pelota | Salida posterior a una disputa, despeje o rebote |
| Duelo / choque | Contacto disputado cuyo valor depende de su salida y continuidad |
| Reinicio / pelota parada | Fase específica para volver a poner la pelota en juego |
| Lateral | Reinicio tras una salida por banda según el script |
| Córner / CK | Reinicio en esquina según el último toque y el árbitro |
| Saque de arco | Reinicio defensivo; «GK» también puede nombrarlo en algunos scripts |
| Comba | Trayectoria curvada; distinguir efecto del script de juego abierto |
| Ping | Medida de tiempo de comunicación; no resume todo el retardo efectivo |
| Input lag | Retraso entre una entrada y su manifestación |
| Jitter | Variación del retraso, importante para repetir contactos ajustados |
| Replay | Grabación de estados y eventos del partido |
| Meta | Estrategias eficaces y sus respuestas en un entorno y período concretos |
| 3def | Regla defensiva de ciertos formatos; no asumir que aplica a RS X4 |
| xG | Probabilidad de gol estimada por un modelo calibrado; no un nombre para cualquier conteo de tiros |

## Preguntas frecuentes

### ¿Qué es lo más importante?

**[inferencia táctica]** Resolver tu función con regularidad: leer, posicionarte, elegir y ejecutar. El cuello de botella cambia por jugador. A uno le faltará primer contacto; a otro, cobertura; a otro, reconocer cuándo pasar.

### ¿Es mejor jugar simple o gambetear?

**[inferencia táctica]** Una conducción que atrae un defensor puede crear un pase. Una gambeta sin salida puede perder una ocasión de asociarse. Un pase simple puede resolver la jugada o entregarle presión al compañero. Evaluar la ventaja que produce y el riesgo que deja.

### ¿Hay que pasar siempre?

**[inferencia táctica]** No. Pasar, conducir, controlar y rematar compiten según las ventanas disponibles. La asociación necesita pases útiles, no una obligación de pasar en cada posesión.

### ¿Hay que presionar siempre?

**[inferencia táctica]** No. Presionar exige llegada, ángulo y cobertura. Una presión individual superada puede ser peor que contener mientras vuelve el equipo.

### ¿El arquero tiene que quedarse pegado al arco?

**[inferencia táctica]** Su distancia cambia con las amenazas. Debe poder defender la trayectoria y decidir si salir evita más peligro del que crea.

### ¿Mantener X me permite patear continuamente?

**[verificado: archivo local]** El motor del proyecto cancela la preparación después de una patada válida y la rearma al soltar. El host puede imponer límites de frecuencia adicionales. Ver [fundamentos](02_fundamentos_y_x.md).

### ¿El movimiento diagonal es más rápido?

**[verificado: archivo local]** La entrada diagonal está normalizada en el simulador. La velocidad que observás también depende de la velocidad previa, amortiguación y colisiones; no hay una bonificación de aceleración diagonal en esta implementación.

### ¿Todos los RS tienen la misma física?

**[verificado: archivos locales]** No: 2K23 usa otro radio de pelota y otra potencia de patada frente a las versiones RS ONE y Sanguchito guardadas. Además, el script cambia los saques. Ver [variantes](01_modalidad_reglas_mapas.md).

### ¿Tener más posesión demuestra que jugamos mejor?

**[inferencia táctica]** Hace falta mirar ocasiones creadas, amenazas concedidas y continuidad. Un equipo puede conservar sin progresar o atacar con eficacia mediante menos posesiones.

### ¿Cuatro jugadores muy buenos forman necesariamente un gran equipo?

**[inferencia táctica]** Necesitan repartir funciones y leer movimientos compatibles. Cuatro jugadores que buscan el mismo contacto pueden dejar sin protección al equipo.

### ¿Se puede afirmar un meta actual a partir de esta biblioteca?

**[pendiente]** La biblioteca ofrece fundamentos y un marco de estilos. Falta un análisis representativo de partidos competitivos recientes para establecer tendencias dominantes. Ver [estilos y meta](07_estilos_y_meta.md).
