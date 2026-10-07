# Liga barata X4 (pre-registro)

Pre-registro del 2026-10-07, escrito antes de lanzar. Lo corre `learn/x4_league.py` en `runs/x4_liga`.

## 1. Por qué y qué se espera

- **Objetivo del usuario:** que el bot sea competitivo y difícil de ganar contra humanos. Parecerse a un humano no es un
  objetivo en sí.
- Ninguna palanca de las corridas del 2026-10-07 dio juego colectivo: bonus de pase, franjas y estilo humano
  (`docs/PROXIMA_CORRIDA.md` §7). Todas mejoraron la fuerza contra la BC, pero ese es un rival flojo (le gana el 85–90%).
- `docs/PLAN.md`:
  - §2 E3: "un explotador barato cada 200–500M decisiones (*pendiente*)".
  - §0.5: fuerza contra un pool fijo y variado, "nunca un solo rival".
  - §3: "Ligas grandes o PBT: fuera de presupuesto".
  - §0.1: desde cero el pase no emerge a nuestra escala. Por eso la liga de octubre, que arrancaba desde cero, no
    aprendió juego colectivo. Esta arranca del agente ya entrenado, con la imitación de ancla.
- **Qué se espera** [inferencia]: un principal más difícil de explotar, porque los explotadores encuentran sus agujeros y
  el principal aprende a taparlos. **No se espera** que aparezca juego colectivo: la respuesta a un explotador puede ser
  otra forma de juego individual. El pase se informa pero no decide.

## 2. Diseño

| Pieza | Valor | Por qué |
|---|---|---|
| Principal | arranca de `runs/x4_estilo/rl_control/best.pt` (actualización 2000) | certificado 0,776 de índice y 0,867 contra la BC; red sólida y laterales aprendidos |
| Recompensa | goles ±1 y −0,1 por saques perdidos (lateral mal ejecutado incluido); sin franjas, sin bonus de pase, sin estilo | la configuración que ganó el A/B del 2026-10-07 |
| Ancla | λ fijo 0,05 a la BC | el principal ya está a esa distancia; los explotadores también, para que busquen debilidades que un humano podría aprovechar y no trucos del simulador |
| Tramos del principal | 2 × 3000 actualizaciones (~200M decisiones cada uno) | el mínimo de "cada 200–500M" del plan |
| Rivales del principal | 50% contra el pool (BC, explotadores aceptados y sus snapshots, con PFSP) y 50% self-play | AlphaStar mezcla ~35% self-play y el resto contra la liga; antes era 20% pool |
| Explotadores | 3 (antes del tramo 1, entre tramos y al final), 600 actualizaciones con 1024 partidos (~39M decisiones) cada uno | "barato": ~20% de un tramo |
| Explotador | arranca del objetivo congelado y juega sólo contra él (`--no-pool-bc --pool-frac 1`) | mejor respuesta a ese objetivo |
| Aceptación | el explotador entra al pool si le gana a su objetivo con puntaje ≥ 0,55 | uno más flojo no aporta presión |
| Enfrentamientos | 128 partidos por lado (256), 3 min, retardo de sala 8–11, muestreando | error de ~±0,03 por enfrentamiento |

Orden: objetivo 0 = punto de partida → explotador 0 → tramo 1 → objetivo 1 → explotador 1 → tramo 2 → objetivo 2 →
explotador 2 → evaluación final → certificación y export.

## 3. Cómo se juzga (decidido antes de ver resultados)

- **Explotabilidad:** el puntaje de cada explotador contra su objetivo. Los tres explotadores tienen el mismo presupuesto,
  así que es comparable.
- **Evaluación final:** el principal al final contra la BC, el rival scripted, el punto de partida, las referencias
  (`best.pt` y `best_pase.pt` de la corrida de franjas) y cada explotador.
- **Veredicto "más difícil de explotar"**, sólo si se cumplen las tres:
  1. la explotabilidad final es al menos 0,05 menor que la inicial (explotador 2 contra explotador 0);
  2. el principal final les gana a los explotadores anteriores (puntaje ≥ 0,5 contra el 0 y el 1);
  3. no pierde fuerza: puntaje ≥ 0,8 contra la BC.
- Si no, "sin mejora de robustez".
- La certificación de pase y el veredicto "competitivo_en_pases" siguen igual que en la cola, como dato.
- **Límite:** con ±0,03 por enfrentamiento, una diferencia de 0,05 entre dos explotabilidades está a poco más de un
  desvío. Es un chequeo de dirección, no una prueba fuerte. La prueba de verdad sigue siendo la sala contra humanos.

## 4. Lanzamiento

```bash
python -m learn.x4_league --root runs/x4_liga --device cuda --init runs/x4_estilo/rl_control/best.pt \
  --segments 2 --segment-updates 3000 --exploiter-updates 600 --accept 0.55 --h2h-matches 128 \
  --main-pool-frac 0.5 --refs runs/x4_cola/rl_checkpoint/best.pt,runs/x4_cola/rl_checkpoint/best_pase.pt \
  --rl-extra="--shaping-kind none --auto-pass-arm-update 0 --forfeit-reasons 1,2,3,4 --lambda-dist 0.05 --lambda-decay 1 --lambda-min 0.05"
```

Duración estimada: ~9–10 h (tramos de ~3,5 h y explotadores de ~40 min).
