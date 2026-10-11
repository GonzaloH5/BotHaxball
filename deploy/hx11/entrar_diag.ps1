# Un intento de entrada del bot con diagnóstico de WebRTC (HAXBALL_DEBUG): para ver por qué falla en una sala.
$env:HAXBALL_DEBUG = '1'
node --use-system-ca deploy/hx11/bot.js --join $args[0] --pos MCD --name GonzaloH4 --avatar 5 --flag uy --lat -34.9 --lon -56.17
