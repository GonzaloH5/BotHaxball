# S3: contrato HX11-1 y árbitro en el simulador (`env/hx11`)

Fecha: 8 de octubre de 2026. Etapa S3 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md). Se trabajó solo en local y con el estadio actual, [1800, 1200].

| Archivo | Qué es |
|---|---|
| `env/hx11/contract.py` | Contrato **HX11-1**: cada regla del script con su evidencia, [medido] o [supuesto] |
| `env/hx11/tables.json` | Tablas medidas: onda de la barra por tipo de saque (150 ticks, valor fijo inicial) y rampa del powershot (394 ticks). Se regeneran con `python -m tools.hx11_script tablas` |
| `env/hx11/kernel.py` | Kernel numba: física de HaxBall (copia exacta de `sim.physics`, con arreglos por partido) y árbitro al final de cada tick |
| `env/hx11/core.py` | `HX11Env(N)`: N partidos 11v11, `reset()` con saque inicial y `step(acciones, frames)`, que devuelve los eventos de cada tick |
| `tests/test_hx11_env.py` | 8 pruebas, una por regla, en escenarios armados a mano |

## Lo que hace el árbitro, en el orden del script (al final de cada tick)

| Regla | Comportamiento | Fuente |
|---|---|---|
| Gol | El reposicionamiento llega a los 150 ticks; el saque inicial es del que recibió el gol; los arqueros van a (±1760, 0) | medido: 31 goles, 16 saques iniciales por equipo |
| Toque y posesión | Pelota a ≤ 0,5 px del borde del jugador, o patada | S2 |
| Offside | En cada patada se marca a los compañeros que están por delante del anteúltimo rival, en campo rival y por delante de la pelota. Se cobra si uno de ellos toca la pelota, y saca el rival. Un toque rival sin patear no anula; cualquier patada corta. Exentas las ejecuciones de lateral, córner y saque de arco | S2: 87% de los cobros, 91% de precisión |
| Powershot | 90 ticks seguidos de contacto (≤ 2 px) y después la rampa medida (2,43 / 3,21 / 4,0 a los +24 / +209 / +359). Separarse lo corta. La patada cargada deja la masa en 1,65 en el tick siguiente y una comba en y de −0,0171 × vy del jugador, que se atenúa ×0,982 durante 150 ticks | S2: 1.113 cargas, 894 patadas |
| Falta | Un rival a ≤ 32,5 px del poseedor, con la pelota a ≤ 25 px de este, durante 113 ticks (cortes de hasta 6 no reinician). La pelota queda congelada 14 ticks; después, tiro libre para el poseedor, o penal en (±1275, 0) si fue en el área | S2: 38/50 faltas, 5 disparos falsos; congelado medido en 78 |
| Salida | Pelota entera afuera: el lateral se coloca a los 6 ticks, en (x del cruce, ±1230); córner y saque de arco a los 15 ticks. Saca el rival del último que tocó. Córner si el último toque fue del que defiende ese arco | S3: 289 laterales, 94 córners, 62 saques de arco |
| Pelota parada: masa | Lateral 1,8 fijo. Córner y tiro libre: 3,8 durante 7 ticks y después la onda medida (0,76–4,29). Saque de arco: 4,33 y su onda (0,77–4,81). Penal: 2,7 fijo | S2–S3: 893 laterales; barra en 649 saques |
| Pelota parada: jugadores | Córner, tiro libre, saque de arco y penal: todos sin masa (invMass 999999) hasta el tick siguiente a la patada. Saque de arco: los rivales quedan con cGroup c0 o c1, son fantasmas fuera de la zona | S3: 1.175 saques, 68/68 saques de arco |
| Pelota parada: discos | **Barrera contra el rival**, desde el tick siguiente a la colocación: radio 373, o 750 en el córner. Se quita 3 ticks después de la patada en el lateral, 1 en los demás; en el saque de arco va al punto de penal y dura 121 ticks. **Disco del punto** (radio 17,5): 30 ticks en el lateral y 120 en los demás; mientras está, nadie llega a patear | S3 |
| Pelota parada: después de patear | La masa de la pelota vuelve a 1,65 al tick siguiente, o a los 5 ticks en el lateral. La comba está aproximada (ver supuestos) | S3 |
| Modos | Offside, faltas y powershot se pueden apagar por entorno (`HX11Env(offside=..., fouls=..., powershot=...)`) | — |

## Supuestos: lo que no está medido o es aproximado

- **Plazo del saque, 600 ticks:** al vencer, el saque pasa al rival en el mismo punto. La sala anuncia "¡SE ACABA EL TIEMPO…!" a los ~510–600 ticks, y hubo 16 cesiones "por hacer tiempo" en el banco.
- **Lateral mal sacado:** si la pelota no entra a la cancha en 120 ticks, se repite para el rival.
- **Comba de la pelota parada:** en tiro libre y saque de arco, gravedad en y de −0,024 × vy; en el córner, −0,021 × vx en x y 0,01 en y hacia la cancha. Falta la regresión.
- **Penal:** solo 2 en la muestra. Se modela como un saque con barrera de 373 y jugadores sin masa; no se sabe dónde se ubican los demás.
- **`!lado` / `!centro` del arquero en el saque de arco:** no está modelado.
- **Orden de los jugadores en el motor:** es fijo (rojos y después azules). En la sala es por orden de creación. Cambia el detalle de los choques múltiples, no su distribución.
- **Reloj:** corre fuera del gol y del saque inicial. La duración del partido es un parámetro (`match_ticks`).
- **Faltas y offsides que la regla no explica:** un 24% de las faltas y un 13% de los offsides de las grabaciones.

## Verificación

- **Física idéntica a `sim.physics.BatchSim`** (diferencia < 1e-9 en 240 ticks con acciones al azar), que S1 validó contra las grabaciones: error máximo de 0,001 px a 1 tick.
- **Pruebas de cada regla:** lateral (demora, punto, equipo, masa 1,8, barrera), córner y saque de arco (asignación, punto, barrera de 750 y en el punto de penal, jugadores sin masa, rivales fantasma), falta y penal (113 + 14 ticks), offside, rampa y patada del powershot, restauraciones después de patear el saque.
- **Prueba de humo** con 16 partidos de 18.000 ticks y jugadores que persiguen la pelota: goles, saques iniciales, 403 salidas, 426 colocaciones, 415 patadas de saque y 23 offsides, sin errores.
- **Velocidad (esta PC, 16 hilos):** unos 500.000 ticks de partido por segundo con 256 partidos (11 M ticks de jugador/s), con física y árbitro.

## Siguiente: S4, conformidad completa

**Re-arbitrar las grabaciones:**
- desde cada tramo de juego abierto, con las entradas grabadas, el árbitro del simulador tiene que producir los mismos eventos que la sala, en tipo, equipo, punto y tick;
- tiene que dejar la misma masa de la pelota (carga y barra) y las mismas barreras.

**Métricas por regla**, con los criterios de la tabla de S4 del plan:
- acuerdo en los cobros de falta y de offside;
- error de la trayectoria de los saques (comba);
- error de la masa en la carga y en la barra.
