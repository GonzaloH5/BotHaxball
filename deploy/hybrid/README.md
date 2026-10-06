# RS4 Control 0.5: instalación y aceptación

## Presets y preparación 0.5.0

En **Salas y bots → Tus presets**, prepará la sala y el plantel, elegí un nombre
 y pulsá **Guardar preset**. Se guardan acción (entrar/crear), enlace o ID, nombre
 de sala, mapa, capacidad, límites, visibilidad y nombre/avatar/país/equipo/modelo
 de cada bot. Los presets son locales a ese perfil de navegador.

- **Cargar preset** aplica la selección sin iniciar bots. Al reabrir se restaura
  el último guardado o cargado. Si ya tenías cambios, **Deshacer carga** recupera
  la configuración anterior durante esa sesión.
- **Actualizar preset** guarda los cambios y permite renombrarlo. El texto
  «En edición» identifica cuál se modifica; elegir otro en la lista no lo carga
  hasta pulsar Cargar. **Guardar como copia** requiere un nombre distinto.
- **Eliminar** mantiene el formulario actual y ofrece **Deshacer eliminación**.
- El indicador avisa de cambios sin guardar. Los cambios no guardados se pierden
  al cerrar la página; el preset guardado se conserva.
- No se incluyen contraseña, token headless ni token local. Al cargar se limpian
  las credenciales del formulario. Para crear sala se necesita un token reciente.
- Guardado y carga funcionan sin servicio conectado. Un modelo o mapa faltante
  se conserva y bloquea el inicio con una explicación; nunca se sustituye solo.
- El gestor explica falta de conexión/cupo, espera el catálogo antes de habilitar
  inicio, evita envíos simultáneos y deshabilita campos de creación al entrar a
  una sala. Un fallo al recordar el plantel después de iniciar no se informa como
  fallo de creación del grupo.

Pruebas: `node deploy/test_presets.js` y `node deploy/test_manager_browser.js`.
Resultados de navegador y capturas: `reports/rs4_presets_manager*` (salas simuladas).

## Interfaz 0.4.0

La configuración y el gestor usan una paleta clara de papel y verde cancha,
con navegación compartida, separadores y controles compactos. El panel del
partido mantiene fondo oscuro para integrarse con el juego. Identifica la
modalidad `rs4_4v4` / Real Soccer 4 vs 4; esa etiqueta no selecciona otro modelo
ni instala las reglas de una sala.

La barra principal presenta siempre **Conectar → Comprobar configuración →
Ceder al bot**. El paso disponible se destaca; la comprobación ya no está
oculta en el diagnóstico. Cuando el bot juega, el último botón pasa a
«Tomar control». Esc y el atajo personalizado conservan su comportamiento.
Los mensajes reflejan la preparación vigente después de comprobar y recuperar
el control. En suspensión se indica recuperar primero el control humano.

En Conexión, «Cómo iniciar el servicio» abre directamente la guía con comando
copiable. Al conectar aparece el acceso a HaxBall con el siguiente paso.
El atajo se guarda por separado. En Salas y bots se muestra el total activo,
los nombres legibles de modelos y Real Soccer ONE, estados de conexión y
controles de desconexión diferenciados. En pantallas angostas los campos de
modelo y equipo ocupan todo el ancho del perfil.

Para actualizar: recargar `deploy/hybrid_extension` en el gestor de extensiones
y luego recargar HaxBall. No cambian permisos, protocolo ni contratos del modelo.

Validación del rework: `reports/rs4_rework_20261003.md`.

## Gestor de salas y múltiples bots (0.3.0)

Además del control híbrido de tu jugador, la extensión ahora abre una consola
«Salas y bots». Está disponible en el panel de HaxBall y en la configuración de
la extensión. Reiniciar el servicio local, recargar la extensión y emparejar con
el token nuevo antes de usarla. El comando del servicio sigue siendo:

```powershell
node deploy/hybrid_server.js --model deploy/rs4_public/model.onnx
```

- **Agregar a sala:** pegar el enlace oficial o ID y la contraseña si corresponde.
- **Crear sala:** elegir nombre, mapa, capacidad, límites de goles/minutos y
  visibilidad pública. Introducir un token reciente obtenido manualmente en
  `https://www.haxball.com/headlesstoken`. Es distinto del token local de emparejamiento.
- **Perfiles:** hasta ocho bots activos por servicio, con nombres distintos por
  grupo, avatar de hasta dos unidades UTF-16, país de dos letras, equipo solicitado
  y modelo local. Los perfiles se recuerdan al iniciar un grupo. No se guardan el
  token headless ni la contraseña de la sala en el almacenamiento de la extensión.
- **Modelos:** el modelo pasado al servicio y los modelos Clásico/Multitarea si
  existen en `deploy/model.onnx` y `deploy/multi/model.onnx`, con su JSON asociado.
  Se permiten sólo esos archivos y los mapas `.hbs` descubiertos en `stadiums`.
- **Instancias:** pausar/reanudar inferencia, solicitar equipo, desconectar un bot
  o detener todos. El primer perfil al crear sala actúa de host, permite iniciar,
  pausar/continuar y detener el partido. Los demás esperan el enlace para entrar.
  Los equipos mostrados son los observados; en salas ajenas el host conserva la
  autoridad para asignarlos. No se evitan límites por IP, capacidad o admisión.

Cada instancia ejecuta `deploy/bot.js` en su propio proceso, con memoria de política
independiente y un hilo de inferencia. Cerrar la consola no desconecta los bots.
Cerrar el servicio detiene sus procesos; una desconexión del padre también termina
el bot. Hay timeout de conexión y cancelación de los bots todavía en cola. No se
reconectan automáticamente instancias que hayan terminado.

El gestor usa `/fleet` con autenticación local, separado del protocolo híbrido
`/rs4`. No recibe comandos desde el puente de la página de HaxBall. Los bots
administrados tampoco aceptan los comandos de chat del antiguo bot por terminal.
Los secretos viajan al proceso por entorno, no por argumentos de línea de comandos,
y no se incluyen en los estados del gestor. No se agregan permisos de navegador.

El mapa no instala el script de reglas de una sala: cargar Real Soccer ONE por sí
solo no reproduce poderes ni arbitraje del servidor original. Elegir el modelo
apropiado. La prueba automatizada del gestor usa procesos y salas simulados;
hosting WebRTC real, aceptación del token y varias conexiones desde la misma IP
requieren validación con una sala real.

```powershell
node deploy/test_bot_manager.js
node deploy/test_bot_managed_runtime.js
node deploy/test_bot_memory.js
node deploy/test_manager_browser.js
```

## Panel de control 0.2.0

El panel compacto muestra modo, atajo, conexión y el motivo que impide activar el
bot. «Opciones y diagnóstico» abre una consola con altura limitada para conservar
espacio de juego, requisitos de preparación, inferencia y edad de la última
propuesta. Las métricas sin muestras durante tres segundos se muestran como «—».
El último resultado del preflight se conserva como referencia; «Preflight vigente»
indica si todavía es válido para la configuración actual.

El historial mantiene hasta 60 cambios de estado en memoria por página.
«Descargar diagnóstico» exporta un JSON con versión, estado, modelo, requisitos,
métricas y eventos. No exporta credenciales, chat ni frames de juego. «Limpiar
eventos» borra sólo el historial visible; no cambia el control ni la validación.

La configuración incorpora estado del servicio y pestaña propietaria, visibilidad
opcional del token, guardado independiente del atajo y una guía con comando
copiable. No se agregan permisos de navegador. Para actualizar, recargar la
extensión en `brave://extensions` y luego la página de HaxBall.

## Qué hace y qué no

En el modo híbrido hay un solo jugador y una sola conexión oficial del navegador.
La extensión alterna las entradas del humano y del modelo por el controlador
nativo. El protocolo de inferencia sólo recibe geometría y estado seleccionado.
El gestor adicional crea conexiones WebRTC propias únicamente al iniciar bots
desde su consola. Ninguno de estos modos entrena ni modifica checkpoints.
Usar exclusivamente en salas propias o con autorización.

Primera versión: **Real Soccer ONE**, 4 jugadores por color, modelos del programa
RS4 v3 públicos v1, feedforward o recurrentes. Un nombre igual no garantiza una
réplica del script de la sala. Se mantienen los juegos/mapas originales del cliente.
Mapas con exportación nativa deshabilitada no se fuerzan: el bot queda bloqueado.

## Preparación en Windows

Desde la raíz de HaxballRL, usar Node 22+ y la `.venv` del proyecto:

```powershell
npm ci --prefix deploy
node deploy/hybrid/build_extension.js
```

Copiar manualmente del Pod el checkpoint público que quieras jugar, sin modificar
sus originales ni detener su entrenamiento por este servicio. Si la copia local
está en otro sitio, cambiar solamente la ruta de entrada:

```powershell
.venv\Scripts\python.exe -m export.to_onnx runs/rs4_v3_public/control/latest.pt --out deploy/rs4_public/model
node deploy/hybrid_server.js --model deploy/rs4_public/model.onnx
```

La exportación lee el checkpoint y genera `model.onnx`, `model.json`, `fixture.json`
en una carpeta independiente; no actualiza pesos PPO. No usar `runs/multi/rs4.pt`
si no tiene el contrato público RS4 v3. El servicio rechaza contratos incompatibles.

El servidor muestra puerto y un token nuevo en cada arranque. Por defecto escucha
**sólo `127.0.0.1:17841`**; no requiere GPU, Pod, port forwarding ni abrir firewall
para conexiones remotas. `--port 17842` permite otro puerto local. Si Python no
está en `.venv`, configurar `HAXBALL_PYTHON` con su ejecutable para preparar geometría.

## Extensión y flujo de juego

1. Abrir `brave://extensions` o `chrome://extensions` y activar modo desarrollador.
2. «Cargar descomprimida» → `deploy/hybrid_extension`. No se publica en una tienda.
3. Abrir el popup de la extensión. Introducir puerto y token; «Guardar y conectar».
   Para elegir atajo, hacer clic en «Atajo propio», presionar tu tecla/combinación
   y «Guardar atajo». Se puede guardar sin reconectar ni introducir el token.
   Las instalaciones nuevas empiezan sin atajo; «Sin atajo» permite desactivarlo.
   Evitar teclas de juego y combinaciones que Windows/navegador capturen antes de
   enviarlas a la página: la extensión no puede apropiarse de esos atajos del sistema.
4. Recargar [la web oficial](https://www.haxball.com/play) si ya estaba abierta,
   para capturar los manejadores desde el inicio. Entrar normalmente y por tu cuenta.
5. El panel se coloca antes del área de juego, no sobre la cancha. Marcar la casilla
   visible de autorización; pulsar «Comprobar configuración» en la barra principal.
6. Tras aprobar el preflight, «Ceder al bot» o tu atajo elegido. «Tomar control» o Esc vuelve al humano.

Inicio y reconexión siempre manuales. El servicio **no entra en la sala**. La
pantalla, equipos, chat y menús siguen siendo los oficiales. En BOT no se combinan
movimientos humanos; el atajo no funciona mientras se escribe y no repite al mantenerlo.
Una tecla retenida en un cambio debe soltarse antes de volver a actuar como humano.

En BOT, minimizar/ocultar la pestaña o cambiar de ventana **conserva el bot activo**;
al recuperar foco sigue en BOT. En HUMANO, perder foco libera teclas y conserva
HUMANO. Esc también conserva su comportamiento oficial. Desincronizarse, cambiar
jugador/equipo/plantel/mapa o fallar inferencia sí libera movimiento/disparo y suspende.
Tras un error sin cambio de foco, Esc recupera HUMANO; comprobar nuevamente si se
invalidó la configuración. Goles y reinicios deportivos neutralizan y reinician
memoria, sin reconectar. Cambiar de sesión exige confirmar autorización otra vez.

Sólo una pestaña posee el servicio. Las restantes muestran «Otra pestaña activa»;
cerrar/salir de la propietaria libera el propietario y «Tomar esta pestaña» permite
reclamarlo. No se roba el control de una propietaria conectada.

### Si los botones están deshabilitados (0.1.3)

El panel muestra la condición que falta, no sólo el último mensaje de foco.
«Comprobar configuración» requiere servicio, controlador capturado, jugador activo
en rojo/azul, partido en marcha sin pausa/desincronización, **4 jugadores por color**,
mapa preparado y autorización marcada. Ser espectador, estar en el menú después
del final o jugar 3v3 no cumple el contrato de este modelo. El diagnóstico enumera
servicio, ID/equipo, cantidades, nombre/preparación del mapa y permiso.

«Ceder al bot» requiere además preflight aprobado. Perder foco no bloquea el botón
de una configuración validada: el clic recupera/verifica foco antes del cambio.
No se desactiva ningún requisito de compatibilidad ni se inicia el bot solo.
Después de recargar la extensión en `brave://extensions`, recargar también HaxBall
para capturar el cliente desde el inicio; esto puede requerir volver a entrar a la sala.

### Canal desconectado (0.1.4)

No se envían frames ni acciones por un `Port` cerrado, ni se almacenan para más
tarde. Si el contexto sigue válido, el panel reintenta conectar al worker con
espera incremental y un máximo de cinco intentos. Una recuperación exige mapa y
preflight nuevos y vuelve a HUMANO; nunca reactiva BOT automáticamente. Si la
extensión fue recargada y el contexto anterior está invalidado, hay que recargar
también HaxBall. Se indica explícitamente en el panel. Esc permite recuperar el
control manual mientras tanto.

El worker admite el documento oficial `/play` y el iframe cacheado, siempre del
origen HaxBall y de esta misma extensión. No se amplían los permisos ni los
patrones de inyección. Aperturas tardías de WebSockets viejos y envíos sobre una
conexión cerrada se descartan sin excepciones no capturadas. El popup distingue
servicio caído, puerto inválido, falta de token y autenticación rechazada.

### Control manual independiente (0.1.5)

Entrar a una sala, detectar jugador/equipo y preparar mapa/servicio mantienen
HUMANO. No hace falta token, ONNX, preflight ni permiso para moverse o patear
manualmente. La preparación invalida respuestas del modelo sin neutralizar una
tecla manual que ya estaba presionada. Errores repetidos de observación tampoco
liberan continuamente el teclado humano.

Si falla BOT, se neutraliza y queda SUSPENDIDO; una nueva pulsación de una tecla
de juego (WASD, flechas, X, espacio, Ctrl, Shift, Numpad0) fuera del chat, o Esc,
recupera explícitamente HUMANO. No reactiva el bot: requiere nueva validación.
Las teclas ya retenidas al cambiar de controlador deben soltarse primero.

La conexión del servicio y la compatibilidad del mapa son estados distintos.
Esta versión habilita BOT sólo con el contrato `Real Soccer ONE`. Otro nombre,
por ejemplo «Sanguchito x4rs», queda rechazado por este contrato aun estando
conectado el servicio; el juego manual sigue disponible. No se asegura que ese
mapa tenga física diferente sólo por el nombre: hay que auditar su `.hbs` antes
de admitirlo, no saltarse la validación ni renombrarlo a ciegas.

«Recargar modelo» sólo funciona en HUMANO: relee **la misma ruta local**, invalida
acciones/memoria y exige mapa y preflight nuevos. No descarga modelos del Pod.
Cerrar el servicio con Ctrl+C provoca suspensión y neutralización en la extensión.

## Contratos y límites

- Adaptador `0349dd60`, SHA256 del script oficial
  `322d1904fc09896e5a16482f53e35a6e046c50a40c0270646305700b7420e885`.
  Las correspondencias minificadas están aisladas en `official_profile.js`.
  No se distribuye el código oficial ni se reemplaza su archivo/conexión.
- `MAIN` captura enlaces transparentes de manejadores nativos y envuelve el
  controlador; el contexto aislado gestiona panel/puente. El service worker y el
  popup son los únicos contextos con acceso al token (`TRUSTED_CONTEXTS`).
  [Separación de mundos documentada por Chrome](https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts).
- El servicio valida origen `chrome-extension://…`, token, propietaria y mensajes.
  El puente de la página no es una frontera frente a una página maliciosa del
  mismo origen: una web comprometida podría falsificar estados/comandos de su propia
  sesión. No obtiene el token ni permisos sobre otras páginas; no usar en sitios distintos.
- Estado: posiciones, velocidades, radios, colores, equipos, identificador local,
  máscara nativa, reloj/resultado visibles, gravedad y joints. Sin claves, chat,
  roles privados, dueño interno de saque o etiquetas del currículo. La identidad
  del saque se deduce de colores/joints y contacto público. El dueño de un kickoff
  inicial desconocido queda desconocido, no se lee el árbitro privado.
- La geometría se exporta/prepara al cambiar el mapa, no por decisión; caché local
  de cuatro geometrías. Observaciones/raycasting, señales, memoria y acciones
  reutilizan componentes del bot por terminal.
- Un ONNX pendiente por sesión; cola de tamaño uno. La siguiente decisión espera
  confirmación del comando emitido por el controlador nativo. Sólo entonces se
  acepta la memoria y se registra la acción anterior. Esto confirma envío nativo,
  **no una garantía de entrega de red o de contacto con la pelota**.
- Identificadores de sesión, generación, frame y tiempo monotónico local invalidan
  resultados viejos. No se ejecutan propuestas mayores de 100 ms, desordenadas o
  ajenas. Watchdog 250 ms neutraliza y suspende.
- Inferencia greedy y `frame_skip` del JSON. HUMANO actualiza historia sin inferir;
  la comprobación explícita realiza inferencias secas alternadas, sin ejecutarlas.
- Segundo plano: el servicio envía pulsos a la cadencia de decisiones mediante el
  worker; la página avanza el mismo cliente nativo y captura estado sin depender de
  `requestAnimationFrame`. No añade una conexión de sala, hotkeys globales o flags
  al navegador habitual. Se mantienen los límites de antigüedad y watchdog.
  Si el navegador **congela/descarta** la pestaña o la PC duerme, no se garantiza
  ejecución: `freeze` suspende y cualquier salto grande/estado obsoleto se rechaza.
  [Límites del ciclo de vida del navegador](https://developer.chrome.com/docs/web-platform/page-lifecycle-api).

## Comprobación y aceptación pendiente

«Comprobar configuración» alterna seis bloques de 120 frames con relay manual /
relay + inferencia seca. Exige 30 respuestas, p95 estado→propuesta ≤50 ms y mediana
de intervalo de frame no más de 1.05× la baseline. Permanece deshabilitado si falla.
**Esta baseline tiene la extensión instalada:** es un preflight, no la comparación
formal con el cliente sin extensión. El navegador throttling puede afectar la medida.

No se ha entrado automáticamente en una sala real. Antes de considerar aceptada
la integración hay que validar, tanto en Brave como en Chrome, en sala propia:

- 30 minutos, ambos colores, 100 cambios sin variar `playerId`/sesión/conexión.
- Movimiento/disparo retenidos, chat, Esc, ocultar pestaña/cambiar ventana y
  apagar servicio. BOT debe seguir con estados frescos en segundo plano; HUMANO
  debe neutralizar teclas. Un fallo/freeze del navegador sí requiere recuperación manual.
- Dos pestañas, espectador, cambios de equipo/mapa, gol, reinicio y reconexión.
- Laterales/córners de ambos colores y joints extendidos/colapsados; ninguna
  propuesta >100 ms aplicada; p95 ≤50 ms durante la prueba real.
- Comparar condiciones alternadas sin extensión / con extensión: regresión de
  tiempo de frame ≤5%. El preflight no demuestra este punto.

Si algo falla, mantener HUMANO/BOT deshabilitado y entregar diagnóstico, sin
ampliar permisos ni sustituir el cliente. Una versión oficial nueva queda bloqueada
hasta inspección/validación del adaptador; no se actualiza la huella a ciegas.

## Pruebas reproducibles sin entrar a una sala

```powershell
.venv\Scripts\python.exe -m pytest tests/test_hybrid.py -q
node deploy/hybrid/build_extension.js
node deploy/test_hybrid.js
node deploy/test_bot_memory.js
node deploy/test_public_signals.js
node --use-system-ca deploy/hybrid/probe_official.js
```

La prueba del cliente sólo descarga código público en memoria. No lo empaqueta,
no abre ninguna conexión de sala. Los exports de tests son desechables, no pesos
de usuario. `test_obs.js` compara fixtures Python/ONNX/JavaScript de los dos modelos.

Para el arnés de navegador se requiere Playwright disponible en `NODE_PATH` o
instalado en un entorno de pruebas. Usa perfil temporal independiente y solicitudes
HaxBall interceptadas: únicamente escenario propio y controlador nativo de entradas.

```powershell
node --use-system-ca deploy/hybrid/test_browser.js --browser brave --model deploy/rs4_public/model.onnx --out reports/hybrid_brave_offline.json
node --use-system-ca deploy/hybrid/test_browser.js --browser chrome --model deploy/rs4_public/model.onnx --out reports/hybrid_chrome_offline.json
```

Chrome de marca eliminó `--load-extension`. El arnés usa el dominio CDP Extensions
**sólo en ese perfil aislado**. La extensión distribuida no pide `debugger`; para
jugar se carga manualmente desde `chrome://extensions`, sin flags de depuración.
[Aviso oficial sobre el cambio](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/1-g8EFx2BBY/m/S0ET5wPjCAAJ).
El foco del arnés headless se prueba con eventos blur/focus despachados; la pérdida
real de foco y el rendimiento de la sala siguen pendientes de aceptación manual.

Resultados verificables y limitaciones: `reports/rs4_hybrid_implementation.md`.
