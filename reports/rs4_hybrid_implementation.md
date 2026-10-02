# RS4 híbrido — entrega técnica, 2026-10-02

## Estado

Implementados servicio local ONNX, extensión MV3 Brave/Chrome, adaptador versionado
del cliente oficial, árbitro de entradas, panel, emparejamiento y pruebas.
**La aceptación en una sala real autorizada sigue pendiente.** No se ha conectado
automáticamente a una sala ni se afirma que los resultados aislados demuestren
30 minutos de juego real o fidelidad completa del adaptador.

No se modificaron entrenamiento, currículo, checkpoints ni pesos de usuario. Los
exports temporales de pruebas fueron modelos sintéticos FF/GRU con proyección
pública no nula. La suite que reexporta fixtures existentes se ejecutó conservando
y restaurando exactamente sus archivos de despliegue; no se publica un modelo.

## Componentes

- `deploy/hybrid_server.js`: ONNX local autenticado, dueño único, puerto loopback,
  cache de geometría, recarga manual del mismo archivo y reset generacional.
- `deploy/hybrid/engine.js`: observación pública existente, greedy, memoria
  recurrente aislada, cola de una muestra, confirmación de la acción enviada.
- `deploy/hybrid/official_profile.js`: correspondencias de la versión `0349dd60`
  con huella SHA256 completa y validación estructural. Sin copia del cliente oficial.
- `deploy/hybrid/arbiter.js`: HUMANO/BOT/SUSPENDIDO, generaciones, teclas retenidas,
  neutralización, límite 100 ms y watchdog 250 ms.
- `deploy/hybrid_extension`: MAIN captura controlador; contexto aislado muestra
  panel; worker/popup privilegiados manejan credenciales y WS. Sin permiso `debugger`.
- `deploy/runtime.js`: geometría y muestreo compartidos con `deploy/bot.js`; el bot
  por terminal conserva su ejecutable/ruta anterior, con pruebas de regresión.
- `deploy/hybrid/README.md`: exportación, instalación descomprimida, emparejamiento,
  diagnóstico, limitaciones y aceptación manual.

## Evidencia local

Suite completa: **701 passed, 43 skipped, 12 warnings**, 107.72 s. Los omitidos son
tests condicionados al entorno; no se presentan como aprobados. Avisos: exportador
ONNX TorchScript legado. Pruebas específicas Python: 3 aprobadas; incluyen exports
feedforward/recurrente, observaciones/logits de fixtures y ejecución ONNX real.

Doce grupos de contratos Node verifican segundo plano con guardas de frescura, atajos elegidos por el usuario, cambios de dueño, teclas retenidas,
respuestas viejas, watchdog, captura transparente/removible, metadatos, historia
manual sin inferencia, memoria sólo después de confirmación, cola latest-only,
estados inválidos, outputs no finitos, autenticación/origen/loopback, exclusión de
pestañas y recarga restringida. Los tests preexistentes de memoria del bot por
terminal y de señales públicas también pasan.

Probe del cliente: script público descargado sólo en memoria y controlador nativo
de inputs ejecutado en entorno aislado. Valida máscara de movimiento/disparo,
neutralización y 100 cambios, con estado visible de fixture. La lectura de estado
completo no se prueba contra una conexión real en esta entrega.

Brave **154.1.96.60** y Chrome **154.0.8037.95**: extensiones realmente cargadas en
perfiles temporales, con servicio WS local y ONNX real. Solicitudes HaxBall
interceptadas; escenario propio con clase nativa de inputs, sin WebRTC de sala.
100 alternancias, identidad fija de fixture, segunda pestaña rechazada, chat/atajo, pérdida de foco por eventos
despachados, fallo de servicio neutralizado y versión desconocida con control manual
operativo. Los JSON y capturas del arnés son la evidencia de cada navegador:

- `hybrid_brave_offline.json` / `.png` (modelo recurrente sintético).
- `hybrid_chrome_offline.json` / `.png` (modelo feedforward sintético).

En pasadas aisladas: p95 estado→propuesta alrededor de **5–7 ms**, relación de
medianas de intervalos de frame aproximadamente **1.000×**. No son cifras del
checkpoint entrenado, del Pod ni de una sala real. El preflight compara relay /
relay + inferencia con extensión ya instalada, no cliente sin extensión.

## Límites y próximo paso autorizado

Pendiente: prueba real de 30 minutos en RS ONE propia/autorizada, ambos navegadores,
colores, reinicios, cambios de equipo/espectador, corners/laterales, y foco/ocultación
reales del sistema operativo. Comparación alternada contra cliente sin extensión
para el límite formal de 5%; fixtures no demuestran esta aceptación.

Si cliente/mapa/contrato no es reconocido, el bot queda bloqueado. Si la prueba real
no alcanza p95 ≤50 ms o degrada >5% el tiempo de frame, no considerarlo aceptado:
mantener HUMANO y presentar diagnóstico antes de cambiar arquitectura/permisos.
El dueño de saque no se toma del árbitro privado; señales ambiguas siguen ambiguas.
En mapas con exportación nativa protegida se respeta el bloqueo, sin reconstruir
una segunda conexión ni pedir permisos adicionales.

## Atajos personalizados

El popup captura una tecla/combinación propia, incluidos modificadores exactos;
permite «Guardar atajo» sin emparejar de nuevo y «Sin atajo» para desactivarlo.
Sin default F en instalaciones nuevas; configuraciones antiguas se conservan
hasta que el usuario elija otro. Esc mantiene la salida de emergencia. El teclado
del chat, composición de texto y auto-repeat no alternan el control. Los atajos
reservados por Windows/navegador que nunca llegan al DOM no se pueden interceptar;
hay que elegir otra combinación o usar el botón. No se instalan hotkeys globales.

## Continuación al minimizar (0.1.2)

A petición del usuario cambia la política anterior de foco: BOT conserva control,
generación y memoria al minimizar/ocultar/cambiar ventana y al regresar. HUMANO
neutraliza teclas al perder foco sin quedarse suspendido. Errores/desincronización,
congelación de pestaña o falta de respuestas actuales sí suspenden; no se desactiva
el límite 100 ms ni el watchdog 250 ms.

Servicio local envía pulsos a la cadencia del modelo; worker despierta el adaptador,
que avanza `client.A()` sobre **la conexión existente** cuando no hay rAF/foco.
No conecta otro jugador, instala hotkeys ni cambia flags del navegador habitual.
`test_browser.js` detiene el rAF del fixture, simula hidden/blur y exige que avance
el frame y se confirmen ≥20 acciones nuevas, conservando BOT al recuperar foco.
Estos ensayos no prueban la minimización real de Windows ni evitan congelación/
descarte/suspensión del equipo. La aceptación real de 30 minutos sigue pendiente.

## Diagnóstico de botones bloqueados (0.1.3)

El panel informa requisitos separados: conexión, controlador, jugador activo,
partido en marcha, plantel 4v4, mapa preparado y consentimiento. La casilla de
permiso queda visible fuera del desplegable. El mensaje «Sin foco» no oculta
estos bloqueos. La activación explícita verifica el foco al hacer clic, sin
deshabilitar por falta de foco un botón ya validado. No se fuerza habilitación
en salas, planteles ni modelos incompatibles. No se modificó ningún checkpoint.

Pruebas Node: 13 grupos aprobados, más inferencia/confirmación con el ONNX local
del usuario. El arnés verifica mensajes de espectador, partido detenido, plantel
incompleto y permiso faltante; también activación mediante clic después de blur.
Evidencia de navegador: `hybrid_readiness_brave.json` / `.png` y
`hybrid_readiness_chrome.json` / `.png`, en perfiles aislados sin conexión a salas.

## Recuperación del transporte (0.1.4)

El panel protege todos los envíos contra ports cerrados, descarta resultados del
canal anterior y no encola frames obsoletos. Reintenta con backoff acotado cuando
el contexto sigue válido; si fue invalidado por recarga de extensión, pide recargar
HaxBall. Desconectar neutraliza entradas e invalida preflight; reconectar restaura
HUMANO, nunca BOT. El worker acepta las dos rutas oficiales `/play` / iframe
cacheado sólo desde esta extensión y origen. No se cambiaron permisos de inyección.
Se protegen las aperturas tardías y envíos fallidos de WebSocket; el popup muestra
diagnóstico de emparejamiento sin exponer credenciales al contexto MAIN.

Pruebas: 14 grupos Node más el ONNX local. Regresiones con mocks de Chromium:
port cerrado en evento y en envío síncrono, reconexión sin replay de frames,
respuestas antiguas ignoradas, contexto invalidado, máximo de reintentos, cierre
de página, URLs oficiales/ajenas, otro ID de extensión, duplicado y WebSocket viejo.
Evidencia de navegador aislado: `hybrid_transport_brave.json` / `.png` y
`hybrid_transport_chrome.json` / `.png`. La aceptación real de sala sigue pendiente.

## Teclado manual sin conexión (0.1.5)

Se corrigió el bloqueo reproducible: `resetGame`/fallos de preparación llevaban
HUMANO a SUSPENDIDO mientras los wrappers nativos aceptaban teclado únicamente
en HUMANO. Ahora preparación/nueva sesión conservan HUMANO, invalidan respuestas
sin neutralizar entradas humanas y los fallos repetidos no anulan movimiento.
Tras fallo de BOT, una pulsación deportiva nueva fuera del chat recupera HUMANO;
las respuestas antiguas siguen rechazadas y el bot exige validación explícita.
Los eventos de teclado humano actualizan foco real si las notificaciones de un
iframe/popup llegaron tarde.

Mapa no admitido/exportación protegida muestran servicio conectado y control
humano disponible. No se habilitó Sanguchito por nombre ni se modificó entrenamiento.
15 grupos Node y ONNX local; el arnés comprueba WASD + disparo antes de emparejar,
sin permiso, en mapa no admitido/protegido y después de fallo del servicio.
Evidencia aislada: `hybrid_manual_brave.json` / `.png` y
`hybrid_manual_chrome.json` / `.png`. No corresponde a una sala real.

Evidencia de esta variante: `hybrid_background_brave.json` / `.png` y
`hybrid_background_chrome.json` / `.png`; estos ensayos leen el ONNX local exportado
por el usuario, sin modificarlo. No corresponden a una sala conectada ni a rendimiento
del entrenamiento. Pruebas Node: 12 grupos; pruebas específicas Python: 3 aprobadas.
