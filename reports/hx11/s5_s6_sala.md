# S5 y S6: pruebas en la sala del usuario (8 de octubre de 2026)

Etapas S5 y S6 de [`docs/PLAN_HAXELEVEN.md`](../../docs/PLAN_HAXELEVEN.md).
- **Sala:** "⪦ 🏆 [H11] 🔴 HaxEleven 🟡 ⪧", con el script de HaxEleven. El host es la pestaña del usuario; el ping local es de unos 2 ms.
- **Bots:** cuatro clientes de node (`bridge/hx11_probe.js`). Los arqueros escriben `!gk`, porque la sala no arranca sin arquero; fuera de eso no usan el chat.
- **Grabaciones del host:**
  - `HBReplay-2026-10-08-01h30m`: 3.703 ticks de juego, 3v2;
  - `HBReplay-2026-10-08-01h38m`: 2.363 ticks, 2v2, con la pestaña visible.
- **Datos:** [`s6_latencia.json`](s6_latencia.json).

```
node bridge/hx11_probe.js <sala> --role lat|holder|presser [--gk] --trace runs/hx11_room/<sesión>/trace_<rol>.jsonl
python -m tools.hx11_room_latency --recording data/hx11/room/<grabación> --meta data/hx11/room/<meta>.jsonl \
    --traces 'runs/hx11_room/<sesión>/trace_*.jsonl'
```

## S6: latencia

La latencia es lo que pasa entre el estado que el bot veía al decidir (buscado en la grabación del host por su posición y la de la pelota) y el tick en que el host aplica su tecla. Se mide en ticks del host.

| Sesión | Decisiones emparejadas | p10 | p50 | p90 | Eco que midió el propio bot (p50) | Ritmo de frames del cliente |
|---|---|---|---|---|---|---|
| 1 | 390 | 9 | 21 | 26 | 17–21 | 40–84 fps |
| 2 (pestaña visible) | 230 | 10 | 25 | 31 | 26 | 44–110 fps |

- **Es bimodal:** un pico a 8–13 ticks (lo que se midió en X4 con un host que anda bien) y otro a 21–28.
- **Los frames llegan a ráfagas:** el cliente ve entre 40 y 110 fps en lugar de 60. A los cuatro bots les pasa lo mismo y en el mismo momento, así que la causa es la sala, no un bot.
- **Causa:**
  - **[inferencia]** El host no sostiene un ritmo de 60 frames parejo. El script de HaxEleven, que cambia propiedades de la pelota y los discos muchas veces por segundo con 22 jugadores, corre en una pestaña del navegador.
  - Con 2 ms de ping, no es la red.
- **El eco que mide el bot coincide con la latencia real:** p50 de 26 contra 25. El bot puede medir su propia latencia en vivo, en cualquier sala y sin grabación del host.
- **Salas de otras personas:** el ping de 30–90 ms suma unos 2–6 ticks (ida y vuelta) a lo que dé el host de esa sala.

### Qué se usa para entrenar

- **[decisión, a confirmar en la sala real]** Retardo D sorteado en un rango amplio, **8–36 ticks**, con la forma medida: un pico cerca de 10 y otro cerca de 25. Así quedan cubiertos un host parejo, uno a ráfagas y el ping de una sala ajena.
- **La latencia entra como observación.** Es lo mismo que se hizo en X4 (el retardo como dato de entrada del bot).
- **En la sala, el bot mide su eco y lo usa.** Eso también sirve para un host que cambia de ritmo durante el partido.
- **Falta:** una medición en la sala donde va a jugar el bot. Alcanza con un bot y 1–2 minutos, usando su propio eco; no hace falta la grabación del host.

## S5: prueba de la falta

**Resultado: no se pudo hacer con bots.**
- En la primera sesión, el bot que presionaba se alejaba en las pausas y el que sostenía la pelota la perseguía: no hubo cuerpeo, como notó el usuario.
- En la segunda, con un bot quieto con la pelota y el otro pegado sin empujar, el contacto más largo fue de 49 ticks. El que sostenía la pelota la tenía a 40 px de mediana. No hubo ninguna falta.
- **[inferencia]** Con 10–30 ticks de latencia, un control reactivo simple sobrecorrige y no sostiene un contacto estable. Los humanos lo hacen anticipando.

**Decisión:** la regla queda como se midió en las grabaciones (S2–S4): ≈ 115 ticks de contacto con la pelota a ≤ 25 px, aceptada por el usuario. Si alguna vez se quiere confirmar, la prueba más simple es manual:
- un bot se queda quieto con la pelota;
- una persona se le pega 1, 2 y 3 s;
- se graba.
