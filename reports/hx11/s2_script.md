# S2: reglas del script de HaxEleven medidas en las grabaciones (avance)

Fecha: 7 de octubre de 2026. Etapa S2 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md).
- **Muestra:** 31 partidos enteros de la versión actual del estadio, [1800, 1200]. Son del split de entrenamiento, del 27 de septiembre al 5 de octubre.
- **Método:** el script está ofuscado y no se ejecuta. Las reglas salen de lo que hace en la cancha (lo que la física no explica: `setDiscProperties`) y de sus anuncios.
- **Datos completos:**
  - [`s2_faltas_ajuste.json`](s2_faltas_ajuste.json)
  - [`s2_offside.json`](s2_offside.json)
  - [`s2_powershot.json`](s2_powershot.json)
  - [`s2_barra.json`](s2_barra.json)

```
node bridge/hx11_ticks.js <grabación.hbr2> --max-minutes 0 --stdout | python -m tools.hx11_npz - --out data/hx11/npz/<nombre>
python -m tools.hx11_script {saques|faltas|ajuste_faltas|offside|powershot|barra} data/hx11/npz/*.npz --out ...
```

## Resumen

| Mecánica | Regla medida | Evidencia |
|---|---|---|
| **Falta** | Un rival pegado al poseedor durante **≈ 113 ticks (1,9 s)** es tiro libre, o penal en el área. Detalle abajo | 38 de 50 faltas predichas con 5 disparos falsos (≈ 88% de precisión) |
| **Offside** | La regla del fútbol, cobrada cuando el adelantado toca la pelota. Detalle abajo | 120 de 138 cobros explicados; el 91% de las predicciones se cobran |
| **Powershot** | Tras **90 ticks** de contacto (pelota a ≤ 1–2 px), la masa inversa sube de 1,65 a **2,43**, después a **3,21** y a **4,0**. La patada vale fuerza × masa inversa | 1.113 cargas, 917 patadas anunciadas |
| **Comba del powershot** | Gravedad en y = **−0,0171 × velocidad vertical del jugador** al patear. Empieza al tick siguiente y se atenúa ×0,982 por tick hasta unos 150 ticks | 894 patadas |
| **Pelota parada** | Tipo según el punto de colocación; masa fija o barra; jugadores sin masa; barreras. Detalle abajo | 1.746 saques |

## Pelota parada

| Saque | Punto de colocación | Masa inversa de la pelota mientras espera | Barrera (disco 1, solo contra el rival) |
|---|---|---|---|
| Lateral | (x de salida, ±1230) | **1,8 fija**, sin barra (893 de 893) | radio 373, centrado en la pelota |
| Córner | (±1765, ±1165) | barra: 3,8 durante unos 10 ticks y después oscila entre **0,76 y 4,29** | radio **750** |
| Tiro libre | donde estaba la pelota | igual que el córner, con una oscilación más suave | radio 373 |
| Saque de arco | (±1572, 0); el arquero puede usar `!lado` o `!centro` | barra: 4,33 al empezar y después de **0,77 a 4,81** | radio 373 a 300 px de la pelota, y los rivales reciben c0 o c1: zona prohibida de x = ±1050 a ±1800, \|y\| ≤ 675 |
| Penal | (±1275, 0) | **2,7 fija** | — (2 penales en la muestra) |

- **Período de la barra:** ≈ 150 ticks (2,5 s) en córner, tiro libre y saque de arco.
- **% anunciado:** es lineal con la masa al patear en córner y saque de arco (error p90 de 1,2 y 1,4 puntos). En el tiro libre no (error p90 de 11 puntos): **pendiente**.
- **Jugadores en córner, tiro libre y saque de arco:** todos pasan a masa inversa 999999, es decir, sin masa. Nadie puede mover la pelota empujándola: solo sale con una patada. Vuelven a 0,69 en el tick siguiente a la patada.
- **Lateral:** las masas no cambian.
- **Disco 2** (radio 17,5, choca con los dos equipos): se ubica en la pelota al colocarla. Falta medir cuánto dura.
- **Color de la pelota mientras espera:** es el del equipo que saca, que puede ser una camiseta personalizada. La señal confiable del saque es la masa, no el color.
- **Tiro libre por falta u offside:** la primera colocación congela la pelota donde está, y conserva la carga del poseedor. Unos 15 ticks después, la segunda la deja lista para el tiro. Si la falta es en el área, la segunda colocación va al punto de penal.
- **Comba en saques:**
  - córner: gravedad en x según el movimiento del jugador, más 0,01 en y;
  - tiro libre y saque de arco: gravedad en y proporcional a la velocidad vertical del que patea, con un coeficiente de entre 0,018 y 0,028.
  - Hace falta una regresión en S3. La sala lo anuncia: "según el movimiento que hagas antes de patear".

## Falta

**Regla medida:**
- **Posesión:** la tiene el último jugador que tocó la pelota.
- **Condición:** un rival se pega al poseedor (centros a ≤ 30 + 2–3 px) mientras el poseedor tiene la pelota a ≤ 25 px de su borde.
- **Temporizador:** cuenta desde que se cumplen las dos cosas. Los cortes de hasta 6 ticks no lo reinician; cambiar de poseedor o perder la pelota sí.
- **Cobro:** a los **≈ 113 ticks (1,9 s)** hay falta. Al cobrar, el temporizador marca p50 de 113–114 ticks y p90 de 114–115.
- **Coincide con el anuncio de la sala:** "al momento de empujar a un jugador con la pelota por 2 segundos, se cobrará FALTA". **No con los 6 s.** Que se perciba entre 5 y 8 s puede deberse a que la cuenta se reinicia cuando el poseedor pierde la pelota o se aleja más de 25 px.

**Ajuste** (50 faltas en 31 partidos; mejor combinación: tolerancia 2–3 px, cortes de 6 ticks, pelota a 25 px):

| Distancia máxima pelota–poseedor | Faltas predichas | Disparos falsos |
|---|---|---|
| 25 px | 37–38 de 50 | 4–5 |
| 50 px | 36–37 | 11–13 |
| 200 px (sin condición práctica) | 34–37 | 21–29 |

El primer ajuste, sin la condición de la pelota, daba falsos positivos con la pelota a 300–800 px.

**Pendiente:** el 24% de las faltas no se explica. Parece posesión mal atribuida, por ejemplo cuando el rival roza la pelota en el forcejeo, o faltas de otro tipo.

## Offside

**Regla medida:**
- **Pase:** una **patada** de un compañero.
- **Posición:** en el tick de esa patada, el que recibe estaba:
  - por delante del **anteúltimo rival**, arquero incluido;
  - en **campo rival**;
  - **por delante de la pelota**.
- **Cobro:** cuando ese jugador **toca** la pelota (a veces con una patada). El tiro libre es para el rival, en ese punto.
- **Desvíos:** un toque de un rival sin patear no anula el offside. Una patada de otro jugador corta la jugada.
- **Exentos:** el lateral, el córner y el saque de arco. Antes de exentarlos, 15 de los 19 adelantados no cobrados de 6 partidos venían de un lateral.

| Variante (31 partidos, 138 cobros) | Predichos | Precisión | Cobros explicados |
|---|---|---|---|
| Pase = patada; receptor = siguiente poseedor; sin exenciones (6 partidos) | 38 | 0,50 | 72% |
| … con las exenciones | 110 | 0,81 | 63% |
| … y por delante de la pelota | 99 | 0,90 | 63% |
| Pase = cualquier toque, con o sin desvíos | 56–168 | 0,28–0,66 | 27–34% |
| **Pase = patada, con desvíos, exenciones y por delante de la pelota** | **132** | **0,91** | **87%** |

- **Pendiente:** 18 cobros no explicados (unos 6 parecen faltas) y 12 predichos sin cobro.
- **El modo offside se puede apagar** (en unos 31 de 711 partidos se cambió en el medio): `offside_on` lo sigue con los anuncios.

## Powershot

- **Inicio:** la carga empieza con **90 ticks** de contacto continuo, con la pelota a ≤ 1–2 px del borde del jugador (p10 84, p50 91, p90 92 con 2 px).
- **Niveles**, contados desde el inicio de la carga:

  | Nivel | Masa inversa | Desde el inicio | Contacto total | Color |
  |---|---|---|---|---|
  | 1 | 2,43 | 24 ticks | ≈ 1,9 s | verde 0x52fa58 |
  | 2 | 3,21 | 209 | ≈ 5 s | amarillo 0xfefe00 |
  | 3 | 4,0 | 359 | ≈ 7,5 s | rojo 0xff4746 |

  - La subida al nivel 1 dura unos 24 ticks, con el color pasando de blanco a verde.
  - Cada nivel suma 0,78 de masa inversa, es decir, patadas un 47%, 95% y 142% más fuertes.
- **Uso humano:** de las 1.113 cargas, 323 se patearon antes del nivel 1, 720 en el nivel 1, 61 en el 2 y 9 en el 3.
- **Fin de la carga:**
  - 910 por una patada;
  - 177 por perder el contacto o porque la tocó otro;
  - 26 por un saque o una pausa.
- **% anunciado:** ≈ (masa − 1,65) / 2,35 × 100 (33% ↔ 2,43, 66% ↔ 3,21, 100% ↔ 4,0). En las cargas a medio subir el ajuste lineal tiene un residuo p90 de 2,4 puntos.

## Lo pendiente de S2 (para cerrar antes o durante S3)

1. **Tiempos de la pelota parada:**
   - cuánto dura el disco 2;
   - plazo del saque ("¡SE ACABA EL TIEMPO…!") y cesión "por hacer tiempo";
   - lateral mal sacado: se vuelve a colocar para el rival.
2. **Comba en córner, tiro libre y saque de arco:** regresión contra la velocidad del jugador y la dirección de la patada.
3. **% del tiro libre:** por qué no es lineal con la masa.
4. **Penal:** la ejecución completa (solo 2 en la muestra; hace falta buscar más en el banco).
5. **Saque inicial:**
   - el script lleva al arquero a x = ±1760;
   - qué pasa con el 12.º punto de saque.
6. **Jugadores:**
   - gravedad (unas 15 veces por minuto);
   - cambios de bCoef (unas 13 veces por minuto).
7. **Modos por partido:** faltas, offside, powershot y "gambeta". Hay que reconstruirlos para cada partido de entrenamiento.
8. **Las faltas (24%) y los offsides (13%) que la regla todavía no explica.**

## Qué implica para el bot (observación y conducta)

- **Ve:**
  - la masa de la pelota: carga y nivel en juego, fase de la barra en los saques;
  - los ticks de contacto propios y del rival con el poseedor (temporizador de falta, en los dos sentidos);
  - la línea de offside de los dos equipos y si él mismo está adelantado en la última patada de un compañero;
  - el tipo de saque y el tiempo desde la colocación.
- **Aprende de los humanos** (no se premia):
  - despegarse antes de 1,9 s si marca;
  - proteger la pelota para provocar la falta;
  - cargar o no el powershot según la presión: el nivel 1 cuesta 1,9 s pegado a la pelota;
  - elegir la fase de la barra: los humanos patean los córners casi siempre al 85–100%;
  - moverse en y al patear para darle comba.
