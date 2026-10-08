# Inventario del banco HaxEleven (S0)

Fecha: 7 de octubre de 2026. Etapa S0 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md).
- Corrió en local, sin pod.
- Metadatos: `bridge/hx11_meta.js`, unos 16 min para los 753 archivos con 3 procesos.
- Agregado: `tools/hx11_inventory.py`.
- Datos completos: [`inventario.json`](inventario.json) y particiones en [`splits.json`](splits.json).
- Lo versionado no tiene nombres, chat, IDs de Discord ni títulos de sala. Eso queda en `data/hx11/`, que no se versiona, con claves `p_<sha1>`, como en X4.

```
node bridge/hx11_meta.js --stadiums data/hx11/stadiums <grabaciones...> > data/hx11/meta_k.jsonl
python -m tools.hx11_inventory
```

## Resumen

| | |
|---|---|
| Archivos | 753; **711 únicos**; 42 copias exactas (mismo SHA-256); 0 errores |
| Qué es cada archivo | **un partido entero** de un solo tiempo, de unos 12,7 min (p10 12,2; p90 14,8). 693 de 711 terminan con "Partido terminado". Mediana de 2 goles; 1.410 goles en total |
| 11v11 en mapas x11 | **8.535 min (142 h)** de 11v11 exacto. Con ≥ 9 por lado son 9.065 min. Equivale a unas 3.100 h de jugador |
| Fuera de x11 | x7: 15 archivos, 167 min. Entrenamiento: 1 archivo. Se separan |
| Jugadores | 1.948 nombres distintos; mediana de 24 por partido, contando cambios |
| Sesiones | 313 |
| Particiones | entrenamiento 521 partidos (6.366 min 11v11); desarrollo 73 (801); prueba 105 (1.251); **prueba de grupos** 12 (117: LPM y VB, nunca vistos en entrenamiento) |

## Versiones del mapa

El estadio se llama "HaxEleven RSx11 [ancho, alto]". Hay 81 tamaños distintos.

| Versión | Partidos | Minutos | Cuándo | Diferencias |
|---|---|---|---|---|
| [1800, 1200] | 399 | 5.275 | mayormente septiembre y octubre (la actual) | arco en x = ±1810 (medio arco 170); aceleración al patear 0,1. Algunas exportaciones traen invMass del jugador 0,69 |
| [1875, 1064] | 212 | 2.948 | junio a agosto | arco en x = ±1885; aceleración al patear 0,085 |
| Tamaño aleatorio por partido | 77, cada tamaño en un solo archivo | 902 | agosto | de 1801 a 1985 de ancho y de 1001 a 1080 de alto |
| [1985, 1200] y [1800, 1080] | 7 | 80 | | |

- Las fechas salen del nombre del archivo, que mezcla mes-día y día-mes. Se descartan las lecturas futuras. 227 quedan ambiguas, incluidas las 39 que caen de enero a mayo: probablemente son de septiembre u octubre.
- **Para S1:** el simulador tiene que aceptar el tamaño de la cancha como parámetro. La versión [1800, 1200] es la referencia: es la de la sala actual.
- **Para S2:** cada etapa (junio a agosto, agosto, septiembre y octubre) puede ser otra versión del script. Las reglas se comparan por versión.

## Lo que toca el script (`setDiscProperties`, por minuto jugado)

| Qué | Por minuto | Lectura probable (a verificar en S2) |
|---|---|---|
| invMass de la pelota | 1.046 | carga del powershot |
| color del disco 26 | 680 | indicador de la carga o de la barra de fuerza |
| color de la pelota | 597 | color de la carga |
| invMass de los jugadores | 446 | ¿masa del poseedor o del que empuja? |
| gravedad de la pelota | 231 | curva (comba) |
| gravedad de los jugadores | 15 | ¿frenado en saques o faltas? |
| cGroup de los jugadores | 13 | barreras en los saques |
| bCoef de los jugadores | 13 | ? |
| disco 1: posición, radio y cMask | 6,7 | barrera de la pelota parada |
| disco 2: posición | 6,9 | ¿punto del saque? |
| discos 28–46: posición y velocidad | 2,5 | ¿línea de offside o barra? |
| pelota: posición, velocidad y gravedad | 3,7 | colocación en saques y faltas |

## Los anuncios de la sala son el registro del árbitro

Quedan grabados con su tick. Son etiquetas listas para S2:

| Anuncio | Veces | Valores |
|---|---|---|
| "Powershot utilizado. X%" | 16.587 | p50 33%. Se concentra en 30–40% (11.399), 60–70% (972) y 90–100% (580): los **3 niveles**. El resto está por debajo de 33% |
| "Potencia del corner: X%" | 5.186 | p10 86%, p50 96%: casi siempre cerca del máximo |
| "Potencia del saque: X%" | 3.605 | p10 68%, p50 93% |
| "Potencia del tiro libre: X%" | 2.878 | p10 58%, p50 83%: más repartido |
| "X es el Arquero del Red/Blue" (`!gk`) | ~900 | arquero declarado por equipo |
| "El arquero podrá utilizar !lado o !centro" | 3.732 | el arquero elige dónde sacar |
| "¡SE ACABA EL TIEMPO PARA SACAR EL CORNER / LATERAL / ARCO / TIRO LIBRE!" | 163 / 157 / 48 / 27 | plazo de los saques |
| "¡… entregado al equipo contrario por hacer tiempo!" | 7 / 5 / 4 | saque de arco, lateral y córner que pasan al rival |
| "El árbitro decide seguir el partido con esta jugada." | 254 | ¿ventaja tras falta? |
| "Gol de X con asistencia de Y \| N km/h" | ~1.400 | goleador, asistente y velocidad del tiro |
| "Tiempo adicionado", "Última jugada del partido", "Partido terminado" | uno por partido | cierre del partido |
| Rankings de la sala (OFFSIDES, FALTAS RECIBIDAS/GENERADAS, PENALES, PASES, INTERCEPCIONES…) | | el script lleva esas estadísticas por dentro |

**No hay anuncio cuando se cobra una falta o un offside.** Esos cobros hay que detectarlos en S2 por lo que pasa en la cancha: pelota recolocada, barrera, quietud. "El árbitro decide seguir" puede ayudar a detectar la ventaja.

### Reglas que anuncia la propia sala (hipótesis para S2)

- "El Offside cuenta a partir del anteultimo jugador sobre el campo rival, sin importar si es arquero o no."
- "En x11 el Offside siempre se cobrará a partir de mitad de cancha. En x7 es a partir de los 3/4 de cancha."
- "Acordate! En los Saques de Arco, Laterales y Corners, no hay OFFSIDE."
- "En los Corners y Saques de Arco, según el movimiento que hagas antes de patear, podrás patear con COMBA."
- "Mientras más tiempo lleves la pelota, con más fuerza patearas! Hay hasta 3 niveles de POWERSHOT."
- "En caso de que esté activado, al momento de empujar a un jugador con la pelota por **2 segundos**, se cobrará FALTA."
  - **No coincide con los 6 s que describió el usuario.** Se mide en S2. Puede depender de la versión del script o ser un texto desactualizado.

### Modos que se activan y desactivan

- La sala anuncia "Modos: PowerShot [✅] • Offside [✅] • Faltas [✅]" en 76 partidos. En 4 de ellos el offside está apagado.
- Hay cambios a mitad de partido:
  - offside activado o desactivado en unos 31 partidos;
  - powershot y faltas, en 1;
  - una "Gambeta" para todos los jugadores, en 3.
- En el resto de los partidos el estado no se anuncia: S2 tiene que inferirlo, por ejemplo por si hay cobros de offside.
- Antes de usar un partido para el árbitro o para la imitación, hay que saber qué modos tenía activos.

## Pistas de puesto (para D2)

- **Avatares:** son números de camiseta (5.713 jugadores-partido con número) y no indican el puesto. De los arqueros declarados con `!gk`, solo 5 tenían el avatar 1.
- **Arquero:** es la única etiqueta confiable. Se declara con `!gk` y la sala lo anuncia.
- **Chat:** de 82.207 líneas reenviadas por la sala, 1.615 (2%) mencionan una palabra de puesto. Es una pista débil.
- **Etiquetas de los puestos de campo:** hace falta la muestra etiquetada por el usuario, o las formaciones que publiquen los grupos en Discord.

## Correcciones al plan

- Cada archivo es un partido, no un tiempo. Las sesiones agrupan partidos seguidos de los mismos jugadores.
- El simulador tiene que aceptar varios tamaños de cancha y dos físicas de jugador (aceleración al patear 0,1 y 0,085).
- La regla de offside de x11 queda precisada: cuenta desde mitad de cancha y desde el anteúltimo jugador, sea arquero o no.
- Hay que medir el umbral de la falta: 2 s según la sala, 6 s según el usuario.
- Hay modos que se prenden y se apagan: faltas, offside, powershot y gambeta.
- Faltas y offsides no se anuncian. Powershot, barra y arquero sí, y son etiquetas directas.
