# Próxima corrida X4: recompensa de estilo humano (pre-registro)

Pre-registro del 2026-10-07, escrito al cerrar la corrida de ese día (`runs/x4_cola`, ver `docs/PRELANZAMIENTO.md` §5).
El lanzamiento está en §6.

## 1. Por qué

La corrida del 2026-10-07 mejoró todo menos el pase:
- Definición: 0,37 → 1,31 tiros por minuto (humanos: 1,38). Le gana a la imitación hasta en el 84% de los partidos.
- Laterales mal sacados: 40% → 8,5% (humanos: 6,5%).
- Pase: el índice de la cadena quedó en una meseta de ~0,74 desde la actualización 1450 (el gate pide 0,93). Pases por
  posesión 0,21 contra 0,59 humanos; precisión 0,46 contra 0,69; salidas de presión 1,1 por minuto contra 3,1.

Diagnóstico de los duelos 1v1 (el que lleva la pelota con un solo rival delante; 2 s después):

| | Humanos | Imitación | Agente ~2000 |
|---|---|---|---|
| Supera al rival | 2% | 2% | 3% |
| Pierde la pelota | 49% | 53% | 61% |
| Pasa | 22% | 15% | 13% |

- La defensa del self-play no es blanda: al rival no se lo pasa 1v1, como dice el usuario de Real Soccer. No hace falta
  entrenar contra mejores defensores.
- Lo que falta es salir del duelo con un pase. La imitación ya pasaba poco y el RL no lo corrigió, porque nada de la
  recompensa lo pide: los goles salen jugando directo.
- El bonus de pase de TiZero no sirvió: es ~1% de la señal (0,018 por minuto contra ~2 de las franjas y ~0,5 de los
  goles) y no movió el índice. El ancla KL copia a la imitación decisión por decisión, con sus defectos.

## 2. La recompensa de estilo (`learn/x4_style.py`)

AMP (Peng et al. 2021) sobre lo que hace el equipo, no sobre cada decisión:
- Un discriminador separa grabaciones humanas de Sanguchito (+1) del agente (−1). Ve, en el marco de cada equipo:
  - las posiciones y velocidades de la pelota y los 8 jugadores hace 1 s, 0,5 s, 0,25 s y ahora;
  - la cadena de los últimos 6 toques de los últimos 4 s: de quién, dónde, cómo salió la pelota, cuánto tardó, presión y
    apoyo.
- Cada decisión con los últimos 2 s en juego abierto paga `coef·(r − r̄)` al equipo, con r = max(0, 1 − ¼(D − 1)²). Los
  saques valen 0. No es de suma cero.
- Discriminador: lr 1e-3, penalización de gradiente 0,1, 8 pasos de 4096 por actualización, 20 actualizaciones de
  calentamiento sin pagar.
- Los rasgos del entrenamiento son idénticos a los de las grabaciones (test sobre un partido del simulador).

Prueba offline (discriminador entrenado con partidos guardados de las actualizaciones 1950–2250 y medido en los de
2000–2300 y en humanos de desarrollo, que no vio):

| | Valor |
|---|---|
| AUC humanos contra agente | 0,85 |
| r medio: humanos / agente | 0,87 / 0,58 |
| r del segundo siguiente a un duelo, humanos: pasa / pierde | 0,90 / 0,85 |
| r del segundo siguiente a un duelo, agente: pasa / pierde | 0,65 / 0,63 |
| Momentos con 2 s de juego abierto por minuto y equipo (agente) | 514 |

[inferencia] La preferencia por pasar en el duelo es chica en el agente. Contra un agente fijo, el discriminador
separa sobre todo por el estilo general (espaciado, movimiento). En línea, cuando el agente se parezca en eso, el
discriminador debería concentrarse en lo que quede distinto, como la cadena. Por eso se mide con un A/B y no se da por
hecho.

## 3. Qué cambia desde el arranque

| Cambio | Por qué |
|---|---|
| Red sólida para los jugadores (`solid_net`, por defecto) | el agente cruzaba la red 6 veces más que un humano; en las salas con red sólida queda trabado |
| Penalización del lateral mal ejecutado (`--forfeit-reasons 1,2,3,4`) | con ella, 40% → 8,5% de laterales mal sacados |
| Sin franjas (`--shaping-kind none`) | se arranca de un agente que ya avanza y define; sin franjas la fuerza siguió subiendo |
| Sin brazo de pases (`--auto-pass-arm-update 0`) | ~1% de la señal y sin efecto; incentiva pases inútiles frente al arco |
| Arranque desde `best.pt` de la corrida del 2026-10-07 (`--init`), con la imitación de ancla | ya define, saca bien los laterales y no cruza la red |
| λ fijo en 0,05 (`--lambda-dist 0.05 --lambda-decay 1`) | el agente de partida ya está a esa distancia de la imitación; volver a 0,2 lo empujaría hacia atrás |

## 4. A/B pre-registrado

Brazos en paralelo, con los 1024 partidos repartidos en tercios y 600 actualizaciones cada uno (~13M decisiones por
brazo, como el A/B de las franjas):

| Brazo | `--style-coef` | Qué prueba |
|---|---|---|
| A, control | 0 | lo mismo sin estilo: separa el efecto del estilo del de los otros cambios de §3 |
| B, estilo | 0,002 | jugar como humano vale ~0,3 goles por minuto (0,002 × 0,29 de diferencia de r × 514 momentos) |
| C, estilo fuerte | 0,006 | el triple, por si la señal en línea es más débil que offline |

Regla de elección, decidida antes de ver resultados:
1. Gana el brazo de mayor índice de la cadena de pase, con la media de sus últimas 3 evaluaciones.
2. Un brazo queda afuera si en su última evaluación le gana a la imitación por menos de 0,6 de puntaje (el punto de
   partida está en 0,85–0,91), o si tiene deriva.
3. El estilo tiene que demostrar algo: si B o C no superan al control por al menos 0,03, gana el control.

Se informan, pero no deciden:
- en duelos, el % de pases y de pérdidas (humanos: 22% y 49%);
- pases por posesión, salidas de presión y tiros por minuto (que no caigan);
- `estilo.agente` contra `estilo.humanos_dev` en `eval.jsonl`.

Señales de que el agente engaña al discriminador. Si aparecen, el brazo queda en "revisar" y no puede ganar:
- `style_d_agent` sube hacia `style_d_human` mientras el índice no sube o el parecido humano (`human_w1_mean`) empeora;
- suben `still_frac` o `dist_ball_1` (quedarse quieto o lejos "parece" humano en alguna situación);
- la circulación inútil (`circulacion_inutil_frac`) pasa la humana (0,15);
- el discriminador colapsa: `style_d_loss` clavado en ~2 (no separa) o cerca de 0 (separa todo y la recompensa se
  anula).

## 5. Corrida principal

El brazo ganador sigue en su carpeta hasta 3000 actualizaciones con los 1024 partidos. Después viene la misma
certificación contra humanos que en la corrida del 2026-10-07, con confirmación con otras semillas, y el export a ONNX.

## 6. Lanzamiento

- Punto de partida: `runs/x4_cola/rl_checkpoint/best.pt`, actualización 2500 (certificación: índice 0,731 y 0,90
  contra la BC; red sólida y penalización del lateral ya aprendidas). `best_pase.pt` (1700) tenía más índice (0,833)
  pero es de antes de la red sólida y de la penalización: cruzaba la red y sacaba mal el 21% de los laterales.
- La cola acepta brazos (`--ab-arms`), desempata a favor del control (`--ab-default control`) y aplica los chequeos de
  engaño de §4 (`--ab-guards`): circulación inútil > 0,15, discriminador colapsado, quietos > 1,5 × el control o
  distancia a la pelota > 1,2 × el control. La subida de `style_d_agent` hacia `style_d_human` sin mejora del índice se
  informa (`ab_temprano`) pero no excluye: se revisa a mano.
- El preflight verifica el checkpoint de `--init` y el banco de estilo, y mide su costo en una actualización real.

```bash
python -m learn.x4_queue --root runs/x4_estilo --device cuda --ab-updates 600 --updates 3000 --extend-to 3000 \
  --ab-arms "control=--style-coef 0;estilo=--style-coef 0.002;estilo3=--style-coef 0.006" \
  --ab-default control --ab-margin 0.03 --ab-min-score 0.6 --ab-guards --ab-max-useless 0.15 \
  --rl-extra "--init runs/x4_cola/rl_checkpoint/best.pt --shaping-kind none --auto-pass-arm-update 0 \
              --forfeit-reasons 1,2,3,4 --lambda-dist 0.05 --lambda-decay 1 --lambda-min 0.05"
```

Sin extensión: la corrida principal termina en la 3000 y pasa a la certificación.

## 7. Resultado (2026-10-07)

- **A/B (600 actualizaciones):** ganó el **control**. Media de las últimas 3 evaluaciones: control 0,733, estilo 0,709,
  estilo fuerte 0,697. Ninguno quedó afuera por los chequeos de engaño. La recompensa de estilo no mejoró el pase: el
  puntaje de estilo del agente quedó plano (~0,59 contra ~0,83 de humanos que el discriminador no vio). Mantuvo el juego
  más parecido al humano en las métricas generales (W1 0,38–0,41 contra 0,44–0,48 del control), que no era el objetivo.
- **Principal (control, hasta la 3000, sin extensión: pendiente −0,001):** el índice subió al principio (0,76 en
  50–600) y volvió a ~0,72 desde la 1250. Fuerza contra la BC 0,82 → 0,86. Pases por posesión ~0,20, pérdida bajo
  presión ~0,65, salidas de presión ~1 por minuto: sin cambios. Duelos 1v1 al final: pasa 10% y pierde 64% (antes 13%
  y 61%; humanos 22% y 49%). Sin deriva.
- **Certificación:** `best.pt` (2000) índice 0,776 y 0,867 contra la BC (97 ganados, 28 empatados, 3 perdidos);
  `best_pase.pt` índice 0,765 y 0,867. Ninguno aprueba. Veredicto "no_competitivo".
- **Export:** la cola lo marcó como fallido por una diferencia de logits de 1,2e-4 (con logits de hasta 355; en
  probabilidades, 1,8e-6). El chequeo ahora compara probabilidades y logits relativos (`export/to_onnx_x4.py`).
  Modelo: `deploy/rs4z/x4_rl_c2000.onnx`.
- **Conclusión:** con imitación + ancla + RL contra sí mismo, ni el bonus de pase, ni las franjas, ni el estilo humano
  produjeron juego colectivo; mejoran la fuerza contra la BC y los detalles (laterales, red). Lo siguiente es medir
  contra humanos en la sala (E4) y, si es explotable, el explotador barato de `docs/PLAN.md` §2 (E3).
