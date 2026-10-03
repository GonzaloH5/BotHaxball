# Reconstruir el entrenamiento de RS4 con pruebas que decidan si funciona

Versión 2 (2026-10-03). Parte del plan original del usuario y agrega las correcciones que pide la evidencia de los experimentos v5 a v5d.
El detalle de esos experimentos está en `reports/rs4_v5_diagnostico_20261003.md`.
Los cambios respecto del original están al final, en "Correcciones respecto de la versión 1".

## 1. Objetivo y decisiones

**Objetivo final:** un bot que pueda ocupar un puesto junto a tres humanos habituales en **Real Soccer ONE 4v4** sin ser una desventaja evidente.
Movimientos, pases, defensa y saques los aprende la red.

**Primer bloque:** hasta **24 horas de GPU acumuladas**, con experimentos separados y cortes por resultados.
No se lanza otro entrenamiento de miles de millones de pasos por adelantado.

- Con las grabaciones disponibles se puede demostrar progreso y descartar estrategias fallidas.
- **La categoría "competitivo con humanos" sigue pendiente hasta probar partidas con humanos.** Ni las grabaciones ni los compañeros simulados sustituyen esa prueba.
- El trabajo se concentra en entrenamiento, simulador, datos y evaluación. El panel queda fuera de este bloque.
- **La GPU se alquila solo para entrenar.** Inventario, auditorías, datos, imitación en lazo cerrado y evaluaciones corren en CPU (PC local) con el Pod apagado.

## 2. Reconstruir la base antes de gastar GPU

### 2.1 Conservar lo aprendido y reproducir el fracaso

- **Inventario** (`tools/rs4_inventory.py`): todos los checkpoints locales y del Pod con hash, configuración efectiva, pasos, código y evaluaciones.
  Los originales se conservan en `runs/pod_archive_20261003/` y no se sobrescriben.
- **Congelamiento del código** (`tools/rs4_freeze.py`):
  - huella SHA-256 de todo el código de simulación, entorno, entrenamiento y evaluación, más el parche respecto del commit base;
  - cada experimento registra esa huella y el lanzador se niega a entrenar si el código difiere;
  - cualquier cambio posterior es un experimento distinto, con huella nueva.
- **Reevaluar con el mismo protocolo** (sección 4):
  - el campeón v3 (1400M), la referencia;
  - el mejor candidato posterior disponible (v5c 300M, el único con 0,56 contra el campeón);
  - el último v5d;
  - los imitadores existentes (BC v2 y BC v5).
- **Convertir los fallos observados en casos reproducibles** de la batería técnica y en detectores del informe:
  - saques vencidos;
  - bloqueo sin goles (porcentaje de 0-0);
  - pérdidas al pasar (pases/pérdidas);
  - falta de apoyos y jugadores desconectados (jugador "colgado" a más de 800 px de la pelota, dos jugadores pegados al arco, jugadores quietos);
  - mala intercepción;
  - regresión con compañeros externos.
- Reutilizar las correcciones factoriales de evaluación existentes y comprobar en cada informe a qué checkpoint y versión de código se aplicaron.
  No volver a confundir una corrección implementada con una mejora demostrada.

### 2.2 Verificar el juego que está aprendiendo

La auditoría actual (`reports/rs4_v3_simulator_audit_sample.json`) cubre una muestra de física con el mapa grabado.
Excluye las modificaciones del script y no acredita el mapa del catálogo.
**Ya muestra una diferencia:** la grabación usa `kickStrength` 5,75 y el catálogo 5,85.

La auditoría ampliada (`tools/rs4_rules_audit.py`, sobre `data/rs4_jsonl` generado con el motor original) cubre:

| Área | Qué se audita |
|---|---|
| Física (todas las grabaciones, no solo una muestra) | movimiento, frenado, colisiones, patada mantenida, liberación de patada y contacto; comparaciones de 1 tick y secuencias de 60 ticks con los mismos estados y entradas |
| Mapas | agrupar las grabaciones por mapa y parámetros físicos; elegir como contrato la variante predominante; no mezclar variantes incompatibles |
| Reglas del script, reconstruidas desde los datos | dónde se coloca la pelota en lateral, córner y saque de arco; quién saca; cuánto tarda; qué pasa si nadie saca; protección de rivales; saque inicial |

Las reglas reconstruidas se comparan con el árbitro simplificado del simulador:
- posiciones de colocación;
- 420 ticks de límite;
- impulso del saque de arco;
- distancia de protección.

**Límite de tiempo de la primera pasada:**
- se audita lo anterior;
- lo que no pueda reconstruirse se documenta como pendiente, sin excluirlo para declarar el simulador validado.

**Condición para entrenar:**
- los eventos discretos de las pruebas de conformidad coinciden;
- las diferencias físicas quedan dentro de tolerancias publicadas y congeladas antes de evaluar candidatos.

Si una regla indispensable no puede reconstruirse con las grabaciones, conseguir evidencia adicional precede al entrenamiento largo.
El contrato resultante (reglas, observaciones, acciones y latencia) queda versionado en `reports/rs4_b1/contract.json`.

### 2.3 Corregir los datos humanos

- **Deduplicar por contenido** (SHA-256) y separar entrenamiento, desarrollo y prueba **por partida y por sesión** (sala y fecha), no por nombre de archivo.
  Si una sesión tiene varias partidas, todas van a la misma partición.
- **Reservar como prueba grabaciones no usadas para ajustar modelos**, cuando existan. Se identifica expresamente la contaminación histórica:
  - el BC v2 y el BC v5 usaron 50 de las 60 grabaciones convertidas;
  - esas 50 no pueden ser prueba.
- **Auditar la alineación temporal entre observación y acción.** Hoy conviven etiquetas desplazadas (`act_lag6`) y secuencias con entradas del mismo frame.
  Se mide qué desfase predice mejor la tecla y se usa uno solo, documentado.
- **Separar** juego activo, pausas, espera previa al saque y restricciones del árbitro, para no enseñar "quedarse quieto" como respuesta universal.
- **Evaluar por separado** decisiones de patada, cambios de dirección, recepción y saques. El acierto global de imitación no es criterio de aprobación.
- **Reconstruir secuencias con continuidad de memoria y contexto previo**, sin atravesar cambios de jugador, equipo, mapa o episodio.

La imitación se prueba actuando en el entorno: sus errores cambian las situaciones que encuentra después, algo que la precisión sobre grabaciones no mide ([Ross et al. 2011](https://proceedings.mlr.press/v15/ross11a)).

## 3. Tres candidatos, una comparación controlada

No se elige de antemano "más PPO", "más imitación" o "una red más grande".

| Candidato | Inicialización | Qué permite comprobar |
|---|---|---|
| A | Mejor política existente reevaluada | Cuánto mejora corrigiendo entorno, objetivo y compañeros |
| B | Misma política con GRU residual de 64 unidades, inicialmente neutra | Si la memoria mejora decisiones secuenciales |
| C | Misma arquitectura recurrente, aprendida desde las demostraciones corregidas | Si reiniciar desde conocimiento humano supera conservar la política anterior |

- Se usan los componentes existentes de atención y memoria.
- B debe reproducir inicialmente las acciones de A antes de aprender (test de paridad).
- C no recibe autoridad de "experto" por tener buena precisión de imitación.
- **Compuerta previa de C, en CPU:** el imitador corregido juega partidos en el simulador. Entra a la comparación solo si:
  - ejecuta ≥ 80% de sus saques;
  - patea al arco al menos una vez por minuto;
  - pasa con una relación pases/pérdidas ≥ 50% de la humana medida con la misma herramienta.

  Si no la pasa, C queda documentado como fallido y su presupuesto no se gasta. El v5 de hoy (BC 49% de acierto, 0 goles en el simulador) no la habría pasado.

### Condiciones comunes

- **PPO con crítico privilegiado.** Durante el entrenamiento ve reloj, marcador, saques y último toque; el actor solo recibe lo disponible en la sala.
  La separación se verifica también en el modelo recurrente y en ONNX: el ONNX no tiene la entrada privilegiada.
- **Optimización nueva** en cada experimento y **calentamiento inicial del crítico** con el actor (y el encoder compartido) congelados.
- **Mismos hiperparámetros entre candidatos.** No se cambian a la vez arquitectura, recompensas y tasa de aprendizaje para rescatar a uno.
- **Comparación a igual experiencia:** cada candidato recibe el mismo número de muestras de aprendizaje (filas que entran a PPO), con un techo de tiempo.
  Se informa también el tiempo usado. Así una red más lenta (B, C) no queda penalizada por reloj.
- **80% de asignaciones a partidos completos y 20% a situaciones técnicas.**
  - Las situaciones técnicas arrancan en estados reconstruidos de grabaciones humanas de la partición de entrenamiento: ataques, defensas, transiciones y saques.
  - Duran hasta gol, salida, resolución del saque o 12 s.
  - **Son la fuente explícita de señal de gol**: en partidos entre redes fuertes casi no hay goles (91% de 0-0 en v3).
  - Antes de comparar candidatos se mide que producen goles con frecuencia suficiente: ≥ 1 gol cada 10 situaciones de ataque con la política A.
- **Composición de los partidos:** 70% con un aprendiz y tres compañeros congelados, 20% con dos o tres aprendices y 10% con cuatro.
- **Compañeros:** varias políticas y estilos, congelados durante cada bloque. Los imitadores pasivos no son la única aproximación a humanos.
  **Se separan dos familias:**
  - compañeros de entrenamiento;
  - compañeros reservados para la evaluación.

  Sin esa separación no se puede detectar la explotación de compañeros conocidos ([Carroll et al. 2019](https://arxiv.org/abs/1910.05789)).
- **Rivales:** históricos, población diversa y scripted como control básico. Vencer al scripted no permite ascender de fase.

### Recompensas

- **Señal principal:** goles a favor y en contra (±1).
- **Se retiran del experimento** las multas por 0-0 (de fin de partido o por tiempo), los premios por formación, por acumular pases o posesión, y por generar goles totales.
  Los cuatro experimentos de hoy mostraron trampas con cada uno:

  | Recompensa | Trampa observada |
  |---|---|
  | Guía táctica (v3) | Bloque defensivo |
  | Ancla humana (v5b) | Pasividad |
  | Presión anti-0-0 (v5d) | Delantero colgado aceptando goles en contra |

- **Sin guía potencial** en el experimento inicial. La falta de goles se ataca con las situaciones técnicas, no con la recompensa.
- Se conservan solo las reglas del juego definidas por el contrato. Por ejemplo, la consecuencia de dejar vencer un saque se ajusta a lo que hace la sala real.
- Pases, apoyos y posesión son métricas de diagnóstico.

### Presupuesto máximo

En muestras de aprendizaje, con un techo de tiempo por corrida. A ~100k pasos simulados/s, con ~25% de filas aprendiendo por la composición de compañeros:

| Etapa | Muestras | Tiempo de GPU |
|---|---|---|
| Preparación de candidatos e imitación (B: paridad; C: imitación recurrente y su compuerta en CPU) | — | 2 h |
| Selección: 3 candidatos × 2 semillas | 100M por corrida | ~1,1 h por corrida, techo 1,5 h; ≈ 7 h |
| Ampliación de los dos mejores: 2 × 2 semillas | +300M por corrida | ~3,2 h por corrida; ≈ 13 h |
| Verificaciones finales con GPU y exportación | — | 2 h |

Si C no pasa su compuerta, la selección usa solo A y B.
El presupuesto sobrante no obliga a continuar. Las auditorías y evaluaciones corren en CPU sin una GPU alquilada ociosa.

## 4. Evaluación y reglas para continuar

### Protocolo fijo (`eval/rs4_b1.py`)

- Partidos de **tres minutos**, ambos colores y el candidato **rotando por los cuatro puestos**.
- **Comparación principal:** un candidato con tres compañeros **externos y reservados** contra varios rivales congelados.
- **Misma batería de estados iniciales** para candidato y referencia, extraída de las grabaciones de desarrollo con pequeñas perturbaciones.
  - Se mide la diversidad.
  - Se detectan trayectorias duplicadas por hash: se exige ≥ 95% de partidos distintos.
- **Desarrollo y prueba final separados.** La prueba reservada se abre una sola vez, después de elegir el candidato.
- **Greedy** en la comparación principal, como `deploy/bot.js --temp 0`. El muestreo se informa por separado.
- **Resultados por combinación** de compañeros, rival, color y puesto. Los promedios no pueden ocultar una regresión importante.
- **Al menos 32 partidos por celda crítica** en la evaluación final, con incertidumbre por bootstrap agrupando partidas relacionadas (mismo estado inicial y semilla).
- **Además de victorias, empates y derrotas:** diferencia de gol, remates, pases/pérdidas y los detectores de trampas de 2.1.
  Con 90% de empates, el resultado por puntos solo tiene poca potencia estadística.
- **La evaluación corre en CPU**, fuera de la máquina de entrenamiento. Evaluar en el Pod durante el entrenamiento lo frenaba de ~100k a ~27k pasos/s.
- Se informa la robustez a latencia de despliegue (1–2 decisiones de retraso), sin que sea compuerta en la selección.

### Condiciones para ampliar un candidato

1. **Saques:** resolver al menos el 95% de los saques legales de la batería técnica, sin automatismos programados.
2. **Sin retroceso:** no retroceder más de cinco puntos porcentuales respecto de la referencia en ninguna celda crítica.
   Una celda falla si la diferencia estimada es menor que −5 pp **y** el intervalo de 90% no incluye el cero (retroceso detectable).
   Las celdas con menos de 32 partidos no deciden: se amplían.
3. **Mejora:** al menos +5 pp en el resultado equilibrado de una instancia (un candidato con tres compañeros externos), con un intervalo de confianza de 95% positivo.
4. **Sin atajos:** la mejora no puede venir de errores del simulador, autogoles, bloqueo de saques, explotación de compañeros o jugadores desconectados.
   Los detectores de trampas no pueden empeorar respecto de la referencia.
5. **Reproducible:** la mejora aparece en las dos semillas de entrenamiento.

Los umbrales son criterios operativos de este piloto, no una certificación de nivel humano.

**Si ningún candidato cumple, el bloque termina.** Se entrega qué hipótesis falló y qué evidencia falta.
No se inicia automáticamente otra variante ni se aumenta el presupuesto para intentar alcanzar el resultado.

## 5. Entrega y definición de terminado

El bloque entrega:
- contrato versionado de reglas, observaciones, acciones y latencia;
- dataset reproducible con particiones y alineación temporal documentadas;
- manifiesto por experimento: hashes, huella de código, parámetros, semillas, cómputo consumido y métricas;
- comparación reproducible de las referencias y los candidatos;
- grabaciones completas de victorias, derrotas y casos difíciles, con selección aleatoria además de ejemplos elegidos;
- ONNX del candidato seleccionado, solo si supera las condiciones, con paridad de acciones y memoria verificada frente a Python y sin entradas privilegiadas.

La revisión visual de las grabaciones explica resultados, nunca los sustituye.
El informe final distingue **funcionamiento técnico**, **mejora en simulación** y **competencia con humanos**.
La primera meta es conseguir evidencia de aprendizaje transferible y un candidato justificable.
La meta final solo se cierra con partidos reales donde el bot sustituya a un humano, con compañeros, rivales y puesto comparables.

## Correcciones respecto de la versión 1

1. **Fuente explícita de señal de gol.** Se retiran las multas anti-0-0, pero algo tiene que producir goles: v3 tenía 91% de 0-0 entre redes y v5c volvió al 81%.
   Las situaciones técnicas pasan a ser estados reales de ataque y defensa sacados de grabaciones, con un umbral de goles medido antes de comparar.
2. **Comparación a igual experiencia, no a igual reloj.** Con 70/20/10 de compañeros, solo ~25% de las filas aprenden; hoy, en self-play, era ~75%.
   Una hora rinde ~4 veces menos aprendizaje que en v5. El presupuesto se expresa en muestras, con un techo de tiempo.
3. **Compuerta previa de C en CPU.** Evita gastar GPU en una inicialización que ya falló hoy por falta de técnica con la pelota (v5).
4. **Compañeros y rivales de evaluación reservados**, distintos de los de entrenamiento, para detectar la explotación de compañeros conocidos.
5. **Evaluación en CPU, fuera del Pod**, con su costo considerado.
6. **Auditoría del simulador con alcance y límite de tiempo.** Incluye la diferencia ya conocida de `kickStrength` (5,75 contra 5,85) y la reconstrucción de las reglas del script desde los eventos grabados.
7. **Criterio de retroceso estadísticamente definido.** Con 32 partidos, un umbral de 5 pp sobre la estimación puntual es casi solo ruido.
8. **Métricas además de los puntos** (diferencia de gol, remates, detectores de trampas), porque con 90% de empates los puntos solos no distinguen candidatos.
9. **Contaminación de datos declarada:** las 50 grabaciones usadas por los BC no pueden ser prueba.
10. **Congelamiento por huella de código** verificada al lanzar, sin depender de commits manuales.
