# HaxballRL

Bot de HaxBall 4v4 (Real Soccer) con juego colectivo de nivel humano competitivo.

**Estado (2026-10-06):** reinicio desde cero. Todo el método anterior de aprendizaje está archivado: entrenadores, recompensas, currículos, RS-Pro, gates, modelos y reportes. Desde ahora, cada decisión de diseño se justifica con trabajos publicados (ver `docs/PLAN.md` cuando esté listo).

## Qué queda en el repo

Solo infraestructura verificada contra el juego real. No contiene ningún método de aprendizaje.

| Ruta | Qué es |
|---|---|
| `sim/physics.py`, `sim/stadium.py` | Motor de física de HaxBall (numba) y carga de mapas `.hbs`. |
| `env/rs4z/` | Simulador vectorizado de partidos: física + árbitro del script de la sala (contrato RS4-Z-2), latencia por jugador y observación `obs_v2`. |
| `stadiums/` | Mapas objetivo: `haxarg_2k23.hbs` (liga HaxArg) y `sanguchito_rs_x4.hbs` (sala SANGUCHITO RS X4, exportado de su grabación). `rs_one.hbs` es el mapa de la mayoría de las grabaciones. `classic*.hbs` son para los tests de física. |
| `replays_real/stadiums/` | Grabaciones humanas 4v4 (gitignored). `rsx4` tiene 156: unas 147 en RS ONE y 7 en SANGUCHITO RS X4 (`SanguREC-*`). `haxarg2k23` tiene 2. |
| `data/rs4_jsonl`, `data/haxarg_jsonl` | Grabaciones convertidas a JSONL por tick (gitignored; se regeneran con `tools.rs4_jsonl_cache`). |
| `bridge/` | Node: `replay_to_jsonl.js` (grabación → JSONL), `record.js`, `latency_probe.js` y `compare_sim.py` (sim vs grabación). |
| `tools/rs4z_conformance.py` | Conformidad del simulador contra las grabaciones (física a 1 y 60 ticks, córners, reloj). |
| `tools/rs4z_bc_dataset.py`, `rs4z_bc_splits.py` | Dataset de imitación humana: observación exacta del simulador + acción humana, con splits por sesión. |
| `deploy/` | Cliente de sala: `bot.js`, `rs4z/join_bots.js`, `rs4z/room_state.js`, extensión hybrid/manager. Sin modelos. |

## Comandos

```bash
python -m pytest -q tests
python -m tools.rs4z_conformance --out conformance.json
node deploy/rs4z/join_bots.js --join <link> --count 7 --model <modelo.onnx>
```

Los trabajos pesados (datasets, entrenamiento, evaluación) se corren en el pod. El código se sube con `python -m tools.pod_sync`, usando las variables `HAXBALL_POD` y `HAXBALL_POD_PORT`.

## Hechos medidos del juego real

- La física y el árbitro del simulador reproducen las grabaciones: en el replay de 2K23 del usuario, el error p90 a 60 ticks es de 0,009 px.
- Después de cada reposicionamiento, la masa inversa de los jugadores es 0,5. El script la pasa a 0,3 en la primera patada de saque.
- En una sala real, los bots cliente actúan unos 15 ticks (250 ms) detrás del estado del host.
- SANGUCHITO RS X4 tiene la misma física que RS ONE (pelota 8,325, patada 5,85, cancha 1150×670), pero otras paredes, arcos (x = ±1159), discos y script. El simulador todavía no lo soporta: falta cargarlo en `contract.MAPS` y validar su conformidad contra las 7 grabaciones.

## Archivo

- Rama git `archivo/rs4z-20261006`: el estado completo antes del reinicio, incluidos los cambios sin commitear.
- `..\HaxballRL_archivo_20261006\`: runs, datos derivados, backups del pod, reportes y modelos (~9,4 GB). Se puede borrar.
