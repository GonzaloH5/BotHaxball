# Rama RS4: guía táctica opt-in, 01/10/2026

## Auditoría de la propuesta

Se conservaron la idea de guía conjunta, roles intercambiables, amenaza/cobertura
y la necesidad de reducir shaping con el tiempo. No se copió la lista de pagos
por tick/evento ni los porcentajes arbitrarios de reward total.

- Diferencia correcta: gamma * Phi_siguiente - Phi_anterior. El coeficiente
  temporal va dentro del potencial de cada instante, conservando el anterior.
- No pagar continuamente por estar en una posición, presionar, tocar, atajar,
  despejar o dar cualquier pase. No existen detectores fiables de todas esas
  acciones en esta modalidad; nombres plausibles no son evidencia de eventos.
- No se convirtió Danger en una probabilidad de gol. Es una heurística acotada
  de apertura, proximidad, control geométrico y tres carriles cubiertos.
- RS4 actual es rs_one 4v4/soccer con saques simplificados. No se añadieron reglas
  del script Pegeche/otras modalidades ni premios por faltas o tiros libres.

Referencia formal: [Ng, Harada y Russell, 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
La garantía teórica no constituye una promesa sobre PPO aproximado contra una
liga cambiante ni sobre los otros bonuses existentes. Se validan invariantes
concretos; la calidad necesita partidos contra rivales congelados.

## Implementación

env/rs4_tactics.py: kernel serial Numba float64 sin fastmath ni RNG. Evalúa las
24 asignaciones de cuatro jugadores a arquero/líbero, presión/conductor y dos
apoyos diagonales. Las zonas dependen de pelota/control; hay tolerancia espacial,
amplitud ofensiva y compacidad defensiva. No se fija una identidad ni se entregan
IDs de rol al modelo. No hay memoria de roles ni una táctica óptima demostrada.

Phi = 0,65 estructura + 0,35 balance de amenaza, en [0,1], compartida entre
compañeros. Coeficiente 0,12 -> 0 durante 200M pasos adicionales desde la
activación, independiente de LR/entropía/BC. Goles/stall usan terminal 0;
truncaciones conservan Phi final para el bootstrap. Pelotas paradas tienen Phi 0.
El paso al coeficiente cero descarga la Phi anterior una sola vez.

Opt-in en un run exclusivo RS4: configure_rs4_rewards respalda la config y cambia
sólo ajustes de rewards/guía/estilo. Desactiva near_ball, kick_to_goal y spread
genérico para no apilar señales contradictorias. Conserva goles, saques, outs,
córners y cooperación existente. La cooperación tiene utilidad/retención, control
de devoluciones y cap por posesión; no se aumentó su escala.

R3 ya tenía estilos 0 equilibrado, 1 agresivo, 2 conservador. Sólo la rama activa
los mantiene durante cada partido completo en lugar de cambiar tras cada gol.
No cambian R3, liga, PFSP, proporciones de rivales, PPO ni pesos/Adam al configurar.
evaluate_rs4 --styles compara ambos candidatos contra cada estilo y el padre,
con semillas comunes, ambos colores, sin shaping de evaluación. No selecciona
ni despliega automáticamente un ganador.

## Verificación y límites

Suite local completa: **498 passed, 42 skipped**, seis avisos ONNX previos,
61,96 s. CUDA no está disponible aquí; tests omitidos no se consideran validados.

Pruebas nuevas: forma/bounds y referencia, simetrías por identidad/color/eje y,
cobertura versus amontonamiento, bloqueo de tiros, igualdad de guía por equipo,
no guía en saques, no farm quieto/cíclico descontado, gol terminal, truncación,
annealing a cero, estilos por partido, opt-in y exclusión del generalista,
continuación PPO real en CPU, config respaldada y checkpoints intactos.

Microbenchmark ilustrativo local del kernel, 144 partidos, datos aleatorios,
una llamada de warmup y 300 llamadas: ~0,458 ms/llamada. No es benchmark del
rollout ni del Pod y se ejecutó mientras la suite podía estar activa. No permite
prometer igual velocidad de entrenamiento ni una ganancia de rendimiento.

No se modificó ningún run real ni se inició entrenamiento. Comandos completos
de activación/reanudación/evaluación en README. Si el run exige gate R3, la
modificación de haxball_env invalida su hash y hay que regenerar el gate, nunca
saltarlo. Restore de config no revierte pesos ya aprendidos; el padre congelado
permanece disponible.
