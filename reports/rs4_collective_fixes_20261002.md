# Ajustes a partir de las siete situaciones de juego

## Cambios implementados

- R3 busca un receptor con línea libre en el saque inicial, incluyendo pases laterales o atrás. Rodea la pelota antes de patear para evitar que un contacto accidental termine el saque y descarte el pase. R2 conserva su comportamiento histórico.
- La guía de saques del aprendiz distingue llegar al balón de estar colocado para patear hacia dentro. En el centro también acepta orientarse hacia compañeros con línea libre. Sigue siendo un potencial compartido y acotado, sin añadir información privada a las observaciones ni sobrescribir acciones de la red.
- La guía colectiva exige dos opciones distintas de pase con líneas libres y distancia razonable. La máxima separación entre dos compañeros ya no basta para considerar que el equipo ofrece apoyos. Se conserva el componente de cobertura defensiva.
- A mayor dificultad, algunos ejercicios de ataque empiezan con los apoyos retrasados y algunos saques con todo el equipo disperso. Así hay que incorporarse o desplazarse hasta el reinicio. Se conserva la mezcla con ejercicios fáciles y el plazo de saque calculado por distancia.
- La referencia del evaluador se cachea por los parámetros efectivos de la suite. Las comparaciones con el campeón reutilizan la misma referencia aunque entren por la ruta denominada control. Se reconocen las cachés antiguas equivalentes y se conservan las validaciones del evaluador.

## Evidencia y límites

Se reprodujeron saques desde posiciones dispersas con ambos colores y con decisiones cada uno o tres ticks, incluidos R2 y R3. El bloqueo de córner descrito en la rec no apareció con el código actual: no se afirma haber identificado su causa histórica. Se agregaron regresiones para mantener esta cobertura.

Los tests verifican dirección inicial del pase hacia un compañero, ejecución del saque, potenciales acotados y simétricos, líneas bloqueadas, apoyos amontonados, colocación del ejecutor y variedad de ejercicios. No demuestran una tasa de recepción de pases ni mejora competitiva de un checkpoint entrenado.

La mala presión, las intercepciones omitidas y los desmarques insuficientes siguen requiriendo medir secuencias de la política después del entrenamiento. Estos cambios no programan roles fijos en la red ni garantizan resolver esas conductas de inmediato.

## Aplicación

Validación: batería completa con 733 pruebas aprobadas, 43 omitidas y 12 avisos de deprecación ONNX. Después del último ajuste de la guía de pase hacia atrás se repitieron las pruebas colectivas y del entorno: 52 aprobadas. `git diff --check` y la comprobación inversa del parche también pasaron.

Código publicado en `origin/main`, commit `fdb186d`. Nueva versión descargada y validada por separado en `/workspace/rs4_update_fdb186d` del Pod: 31 pruebas aprobadas, una omitida; CUDA RTX 3090 y configuraciones CUDA/Torch 2/física 4 comprobadas. El checkout usado por el proceso activo permanece en la versión anterior. Incorporar la versión después de que termine el runner activo; cambiar el evaluador durante la prueba mezclaría versiones de la evidencia. Los cambios alteran la firma de fuentes: las evaluaciones previas se conservan pero no se reutilizan como si fueran de esta versión.

El archivo `rs4_collective_fixes_20261002.patch` contiene sólo estos cambios de código y tests, incluyendo el nuevo archivo de regresiones. Se puede comprobar con `git apply --check` y aplicar sobre el código de partida. Los pesos y el optimizador se conservan; se necesita entrenamiento posterior para medir el efecto en el aprendiz.

### Comando preparado en el Pod

Cuando haya terminado el runner actual:

```bash
bash /workspace/rs4_update_fdb186d/pod_resume_collective_fixes.sh
```

El script verifica que no haya runner, evaluador o entrenador activo en este proyecto, actualiza por fast-forward al commit fijado y comprueba hashes de los pesos, configuraciones y ledger. Después comprueba CUDA y ejecuta el dry-run antes de lanzar, mediante nohup, la reevaluación seguida de un tramo real de 100M pasos. Si queda un proceso activo, sale con código 75 sin actualizar ni lanzar otro.

Para ver el log del nuevo arranque:

```bash
cd /workspace/HaxballRL
tail -f "$(cat pod_logs/rs4_collective_latest_log.txt)"
```

Los scripts también están guardados en este repositorio bajo `reports/pod_apply_collective_fixes.sh` y `reports/pod_resume_collective_fixes.sh`. La protección contra actualización durante un proceso activo fue comprobada en el Pod. El dry-run posterior a la actualización y el arranque se ejecutarán cuando se invoque el comando después de terminar el proceso actual; no se afirma que esos pasos ya hayan ocurrido.
