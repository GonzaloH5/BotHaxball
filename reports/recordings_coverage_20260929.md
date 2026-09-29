# Cobertura de grabaciones humanas — 2026-09-29

Se incorporaron 22 grabaciones nuevas compatibles de `replays_real/stadiums/rsx6`:
328.374 muestras de observación/acción adicionales. Se conservaron los datos y
pesos anteriores. No se reemplazó el checkpoint PPO `runs/multi/latest.pt`.

## Dataset utilizable

Las muestras son decisiones de jugadores, no partidos independientes. La duración
bruta de una grabación no indica cuánto material aprovechable contiene.

| Carpeta | Recs originales | Recs convertidas | Muestras | Recs de validación |
| --- | ---: | ---: | ---: | ---: |
| bigx3 | 30 | 28 | 681.474 | 3 |
| futsalx1 | 32 | 31 | 38.798 | 3 |
| futsalx3 | 60 | 51 | 1.173.322 | 5 |
| futsalx4 | 12 | 11 | 273.728 | 1 |
| futsalx7 | 14 | 12 | 901.566 | 2 |
| rfx7 | 7 | 6 | 618.280 | 3 |
| rsx4 | 12 | 12 | 602.844 | 5 |
| rsx6 | 35 | 34 | 514.508 | 3 |
| Total | 202 | 185 | 4.804.520 | 25 |

El entrenamiento reserva partidos completos por hash del nombre: 160 recs y
3.945.696 muestras para entrenamiento, 25 recs y 858.824 muestras para validación.

## Qué conviene conseguir después

1. **RS x6 del mapa/host de destino**, especialmente partidos completos que
   incluyan saques de banda, arco, córners y reinicios después de gol. En `rsx6`,
   478.248 de las 514.508 muestras son de 6v6; sólo 3.096 son de 3v3. En `rsx4`
   hay otras 8.928 muestras de 3v3. Esos tramos con equipos pequeños no implican
   que exista una modalidad objetivo RS x3: el usuario confirmó que el objetivo
   real es RS x6. No confundir la familia del mapa con el número de jugadores de
   cada tramo del partido.
2. **Futsal x4 y Real Futsal**, con distintos equipos y partidos: sólo hay 11 y 6
   recs convertidas, respectivamente. La validación de Futsal x4 contiene un solo
   partido; Real Futsal tiene tres partidos de entrenamiento y tres de validación.
3. **1v1**, si la prioridad sigue siendo la técnica individual: actualmente hay
   sólo 38.798 muestras, pese a tener 31 grabaciones convertidas.

Futsal x3 tiene una base comparativamente amplia (51 recs, 1,17 millones de
muestras). No hace falta considerarlo "incompleto" por cantidad: más ejemplos
diversos pueden ayudar, pero no existe un número de recs que garantice que el
bot domine una modalidad. Priorizar errores concretos y evaluar en partidos.

## Exclusiones y alcance

- Una rec nueva de RS usa `JJRS RS4 compact (RS5V4 family)`, fuera del catálogo.
  No se convirtió como si fuera el JJRS x6: la geometría no es equivalente.
- En Futsal x3 hay siete recs de mapas fuera del catálogo/entrenamiento y otras
  dos compatibles que no produjeron muestras válidas al intentar convertirlas.
- Las recs de JJRS x6 no sustituyen ejemplos específicos de RS One o del mapa
  Pegeche, ni corrigen por sí solas errores de reglas, física o scripted.
- El currículo al realizar esta auditoría contenía tareas artificiales de equipos pequeños:
  `rs4_3v3` usa `rs_one`; `rs_2v2`, `rs_3v3` y `rs_4v4` usan `x6_half` con el
  árbitro Pegeche. `jjrs_6v6` y `rs_6v6` usan `jjrs_x6` y `x6`, respectivamente,
  con seis jugadores por equipo. En la actualización posterior se retiraron
  las variantes pequeñas del currículo y se incorporó `jjrs_6v6` desde la etapa
  B junto a `rs4_4v4`. Ambas usan saques simplificados sin powershot/slide/faltas.
  Las tareas Pegeche siguen en el catálogo pero fuera del perfil multitarea.
- Incorporar recs al dataset no modifica una política PPO ya guardada. Hace
  falta entrenar un imitador con estos datos y, si se decide usarlo, cargarlo
  como referencia de imitación o como inicialización de una corrida nueva.

Auditoría reproducible: `python -m tools.audit_bc_dataset --out data/bc/coverage_report.json`.
El JSON y los datasets son artefactos locales ignorados por Git.

## Imitador actualizado y comparación

Se entrenó `runs/bc_rsx6_20260929/bc.pt` desde cero durante cuatro épocas, con
la misma arquitectura que `bc2`, etiqueta `act_lag6`, semilla 0 y ocho hilos CPU.
La comparación siguiente evalúa ambos sobre **la misma validación actualizada**.
No se compara el resultado nuevo con métricas históricas de otro conjunto.

| Métrica de imitación | bc2 | Nuevo |
| --- | ---: | ---: |
| Entropía cruzada (menor es mejor) | 1,7142 | 1,7088 |
| Acción exacta | 42,14% | 42,24% |
| Acción entre las tres primeras | 74,33% | 74,46% |
| Acción cuando el humano cambia de tecla | 23,28% | 23,37% |
| JJRS x6: acción exacta | 48,69% | 48,99% |
| Futsal x3: acción exacta | 40,92% | 41,06% |
| 1v1: acción exacta | 45,72% | 45,25% |

La mejora es pequeña y no uniforme: 1v1 empeoró ligeramente. Estos resultados
no demuestran una mejora en victorias, cooperación o ejecución de saques.
Se conserva `bc2` como referencia predeterminada, además del PPO original.
El nuevo imitador se publica aparte como opción para evaluar, no como un
reemplazo obligatorio ni como una continuación de los millones de pasos PPO.

Para probarlo en Runpod como referencia de imitación del PPO ya entrenado:

```bash
git pull --ff-only
python -m train.multitask --config train/config_runpod.yaml --run multi --resume --override bc_reference=runs/bc_rsx6_20260929/bc.pt
```

Si existe un entrenamiento activo, interrumpirlo una vez con Ctrl+C y esperar
el mensaje de guardado antes de reanudar. `--resume` conserva la política
actual: no usar `--init-from` para reemplazarla por el imitador.
