# RS4 v5: por qué v3 no aprendía y qué cambia

Fecha: 2026-10-03. Fuentes: logs y evaluaciones del Pod (`pod_logs/rs4_direct_100m.log`,
`runs/rs4_v3_public/evaluations_public_joints_v1/control_phase_C_recovered_2100000000.json`),
replays de 2568M y `tools/rs4_style_gap.py` sobre los 60 partidos humanos de `data/bc_rs4_v2/rsx4`.

## Evidencia

- **Estancamiento.** El Elo por goles oscila entre 2.000 y 2.400 desde los 5.000M pasos.
  KL final ~0,001, clip 0,005, lr 1e-4 fijo: el controlador de LR exigía KL máx < 0,0008 y nunca subió.
- **Recompensa ≈ solo gol.** El reward táctico medio era −0,00003 por paso.
  La guía potencial (≤3,5% de un gol) no cambia la estrategia óptima.
- **Empates entre redes.** Contra parent y anchors el 91% de los partidos terminó 0-0, con ~0,07 goles por partido de 2 minutos.
  Contra R3: 99% de victorias y 0 goles recibidos, es decir, nada que aprender.
- **Juego individual.** 5 pérdidas por pase. Con compañeros distintos, el éxito era del 4%.
  El 45% de las asignaciones usaba compañeros congelados heterogéneos (script, teacher o snapshots).
- **Estilo** (greedy, partidos de 2 minutos, 16 por fila; en humanos, media de 60 grabaciones):

  | Métrica | Humanos | v3 1,94B espejo | v3 2568M (replays contra campeón) | R3 contra R3 | BC v5 (muestreado) |
  |---|---|---|---|---|---|
  | 3+ detrás del balón | 47% | 61% | 84–91% | 81% | 52% |
  | Dispersión (px) | 170 | 199 | 122–137 | 331 | 249 |
  | Profundidad (px) | 375 | 467 | 204–269 | 975 | 593 |
  | Goles por partido | — | 0,06 | 0 | 0,44 | 0,00 |
  | Partidos 0-0 | — | 94% | 100% | 56% | 100% |
  | Pases / pérdidas | — | 0,20 | — | 0,16 | 0,40 |

  - El BC se ubica como humano (3+ detrás 52%, anchura 265 contra 262), pero casi no toca la pelota (0,1 pases/min).
  - Greedy, el BC no ejecuta nunca el saque inicial: espera, como hacen los humanos antes del silbato.
  - v3 tiene técnica de balón, pero termina en un equilibrio sin goles.
- **Timeouts "centrales" de 2568M.** Eran saques iniciales de R3: su pase orbitaba sin alinearse.
  En una simulación de 64 partidos, 63 saques vencieron sin arreglo y 0 con el arreglo.

## Causas

1. **Sin señal.** El 85% de los partidos eran contra redes y terminaban 0-0; el 15% contra R3 ya estaba ganado.
2. **El equilibrio de self-play con recompensa escasa es defender.** El 0-0 no costaba nada.
3. **Compañeros congelados y aleatorios.** Hacían que pasar fuera arriesgado e impedían formar convenciones de equipo.
4. **Conocimiento humano débil.** Era solo un ancla KL de 0,005 a 0,001 y no se usaba para inicializar.

## Cambios v5 (`train/config_rs4_v5.yaml`, `train/rs4_program_v5.py`)

- **Arranque desde BC humano nuevo:** `runs/bc_rs4_v5_attn64/bc.pt`, con `attentive_meanmax`, entidades de 64 y reglas privadas enmascaradas.
  - Validación sobre los mismos 10 replays / 429.952 decisiones: acierto 49,1%, top-3 82,2%, 28,3% en cambios de decisión (BC v2: 48,9%, 81,8%, 27,1%).
  - Pérdida de validación 1,436 (BC v2: 1,457).
- **Ancla KL al BC** de 0,05 a 0,01, con piso en 0,01.
- **Recompensa:**
  - gol ±1;
  - progreso del balón 0,10, presión del más cercano 0,02 y espaciado 0,05, todos potenciales;
  - pases con tope de 0,05 por posesión;
  - recuperación ±0,01 (doble si se pierde en campo propio), de suma cero;
  - **−0,3 a ambos equipos si el partido termina sin goles**, como terminal real sin bootstrap;
  - sin guía de formación.
- **Self-play:** 90% de los compañeros son aprendices (10% congelados). Rivales: 50% espejo, 40% liga, 10% R3.
- **PPO:** 4 épocas, γ 0,995, λ 0,95, entropía 0,005 a 0,003.
  LR adaptativo por media de KL (sube si < 0,003, baja si > 0,008, corte de época a 0,015), con techo de 4e-4 a 1,5e-4.
- **Medición:** log de recompensa por término (por cada 1000 decisiones) y partidos espejo/liga/R3 con goles por partido y porcentaje de 0-0.
- **Arreglos menores:**
  - R3 abandona el pase de saque inicial a los 240 ticks y saca hacia adelante;
  - el log de potenciales usa la versión de formación del entorno.

En el smoke local (CPU, 0,4M pasos) la KL final quedó entre 0,005 y 0,010 y el clip entre 0,04 y 0,08: la política se mueve.
Esto no demuestra que juegue mejor; eso lo decide el piloto en el Pod.

## Piloto y criterios

300M pasos. Criterios de éxito:

| Métrica | Objetivo |
|---|---|
| Goles por partido en espejo | > 1,0 |
| Tiempo con 3+ detrás del balón | < 70% |
| Dispersión | ≥ 150 px |
| Pases / pérdidas | > 0,4 |
| Puntos contra el campeón de 1400M | ≥ 55% |
| Victorias contra R3 | ≥ 95% |

Plan B: si a los 300M no supera al campeón, aplicar la recompensa y los compañeros de v5 sobre el checkpoint de control v3.

```bash
python -m tools.rs4_style_gap --human data/bc_rs4_v2/rsx4 --red runs/rs4_v5/latest.pt --red runs/rs4_v5/latest.pt --blue runs/rs4_v5/latest.pt --blue runs/rs4_v3_public/champion.pt --games 32
```

## Resultados de los pilotos (2026-10-03)

| Run | Arranque | Ancla KL | Resultado |
|---|---|---|---|
| v5 | BC humano | BC 0,05 → 0,01 | A 96M: espejo con 96–98% de 0-0, pierde contra R3 y contra la liga, 0 pases. No aprende a manejar la pelota. Detenido. |
| v5b | Campeón v3 | BC 0,02 → 0,005 | Mejora inicial: espejo con 54–75% de 0-0. Después pierde técnica (detalle abajo). |
| v5c | Campeón v3 | ninguna | En curso, con evaluación greedy cada 50M (`tools/rs4_eval_watch.py`). |

### v5b

Evaluación greedy a los 150M, 32 partidos por cruce:

| Cruce | Puntos | Goles por partido | 0-0 |
|---|---|---|---|
| Contra R3 | 0,61 | — | — |
| Contra el campeón | 0,16 | 1,88 | — |
| Espejo | — | 0,22 | 78% |
| Referencia: campeón contra R3 | 1,00 | 1,91 | — |

A los 293M, en el log de entrenamiento, quedó en 88–95% de 0-0 en espejo y 0,12–0,17 de puntos contra R3.

**Corrección del diagnóstico.** El campeón juega en bloque también contra R3, y gana todos los partidos:

| Métrica | Campeón contra R3 |
|---|---|
| 3+ detrás del balón | 90% |
| Dispersión | 94 px |
| Profundidad | 155 px |

En este simulador el bloque es eficaz. Parecerse a los humanos en posiciones (la dispersión del v5b subió a ~200 px) **no** mejora la fuerza; lo debilitó.
Por eso los criterios de estilo dejan de ser objetivo. El criterio pasa a ser fuerza:
- puntos contra el campeón ≥ 55%;
- contra R3 sin regresión;
- más goles en espejo.

**Por qué v5b perdió técnica:**
1. El ancla KL densa hacia un BC pasivo se impone a la señal escasa de los goles: el BC-KL bajó de 0,61 a 0,15.
2. El crítico estaba calibrado para la recompensa v3.
3. El controlador subió el LR hasta 3,5e-4.
4. La entropía era mayor que en v3.

**v5c ataca esas causas:**
- sin ancla al BC;
- LR de 1e-4 con techo de 1,2e-4;
- entropía 0,002 → 0,001;
- 15M pasos de **calentamiento del crítico** (solo aprende la cabeza de valor; política y encoder congelados).

Se mantienen la penalización por partido sin goles, pases y posesión, y los compañeros aprendices.
