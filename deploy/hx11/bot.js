// Bot de HaxEleven X11 en una sala real, como cliente (docs/PLAN_HAXELEVEN.md §4; piloto C0).
//
//   node --use-system-ca deploy/hx11/bot.js --join <id o link de la sala> --pos LI [--name HX-Bot] [--password x]
//   (--use-system-ca: en la PC del usuario Avast reemplaza los certificados HTTPS; sin eso, "Error de conexión Master")
//        [--model runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt] [--temp 0.7] [--renovar-patada 0.33]
//        [--sin-saques corner,tiro_libre,saque_arco | ninguno] [--flag uy --lat -34.9 --lon -56.17] [--avatar x]
//
// * El puesto lo elige SOLO el usuario con --pos (o escribiendo "pos MC" en esta consola). Nada del chat lo cambia.
// * No elige equipo: lo pone el admin de la sala. Mientras no esté en un equipo, no juega.
// * Cada ejecución genera una clave de autenticación nueva (el script de la sala echa a dos bots con los mismos datos).
// * Decide cada 3 ticks con el cerebro en Python (learn/hx11_cerebro.py, misma observación que el entrenamiento). El
//   retardo que ve la red es el eco medido de sus propias teclas (S6: el eco coincide con la latencia real).
// * Consola: "pos <PUESTO>" cambia el puesto, "pausa" deja de apretar teclas, "sigue" vuelve, "salir" se va.
// Úsalo solo en salas propias o con permiso del dueño.
"use strict";

const path = require("path");
const readline = require("readline");
let API;
try { API = require("../../bridge/haxball")(); } catch (e) { API = require("node-haxball")(); }
const { Room, Utils, Plugin, AllowFlags } = API;
const { FrameBuilder } = require("./frame");
const { startCerebro } = require("./cerebro_proc");

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const joinArg = arg("--join", null);
let pos = arg("--pos", null);
if (!joinArg || !pos) {
  console.error("uso: node deploy/hx11/bot.js --join <id o link> --pos <LI|LD|DFC|CRI|CRD|MCD|MC|MCO|MI|MD|EI|ED|DC|ARQ>");
  process.exit(1);
}
const joinId = joinArg.includes("?c=") ? joinArg.split("?c=")[1].split("&")[0] : joinArg;
const model = arg("--model", "runs/hx11_l0/l0_ajuste_corner_h2/model_final.pt");
const temp = parseFloat(arg("--temp", "0.7"));
const D_MIN = 8, D_MAX = 36;

const fb = new FrameBuilder();
let room = null, paused = false, ready = false;
const delays = [];
let pendingInput = null;
const echo = () => {
  if (!delays.length) return 22;   // S6: p50 21–25 ticks en la sala del usuario
  const d = [...delays].sort((a, b) => a - b);
  return Math.min(D_MAX, Math.max(D_MIN, d[d.length >> 1]));
};

const cer = startCerebro({
  model, pos, temp, ens: Number(arg("--ens", "0")), ensM: parseFloat(arg("--ens-m", "0")),
  renovarPatada: parseFloat(arg("--renovar-patada", "0.33")),   // patada persistente en juego abierto (puerta P3)
  reglaSaques: args.includes("--regla-saques"),   // si saca córners/tiros libres: desde atrás y con la barra alta
  // juego abierto: no patea hacia afuera pegado a la banda ni atrás en campo propio sin compañero (los dos errores del
  // partido real del 2026-10-09; puerta P3 completa con la regla). --sin-regla-juego la apaga
  reglaJuego: !args.includes("--sin-regla-juego"),
  // en pelotas paradas y saques iniciales que saca otro no se acerca a la pelota (partido real del 2026-10-10: se paraba
  // al lado del que sacaba el córner y tomaba el saque inicial). --sin-distancia-saques la apaga
  distanciaSaques: !args.includes("--sin-distancia-saques"),
  renovar: parseFloat(arg("--renovar", "0.33")),   // dirección persistente lejos de la pelota: 10 → 4,7 cambios de tecla/s (puerta P3 pasa)
  // con humanos: el bot no patea córners, tiros libres ni saques de arco de su equipo (los saca un humano); "ninguno"
  // para que patee todo
  sinSaques: arg("--sin-saques", "corner,tiro_libre,saque_arco,saque_inicial") === "ninguno" ? ""
    : arg("--sin-saques", "corner,tiro_libre,saque_arco,saque_inicial"),
  onReady: (m) => { ready = true; console.log(`cerebro listo: puesto ${m.pos} (línea ${m.linea})`); },
  onDecision: (msg) => {
    if (!room || paused) return;
    const state = msg.input;
    if (!pendingInput && state !== room.currentPlayer?.input) pendingInput = { state, frame: room.currentFrameNo };
    room.setKeyState(state);
  },
});

// node-haxball exige que el plugin sea un Plugin (como deploy/bot.js)
function HX11Plugin() {
  Object.setPrototypeOf(this, Plugin.prototype);
  Plugin.call(this, "hx11Bot", true, {
    version: "1.0", author: "HaxballRL", description: "Bot HaxEleven X11 (imitación por puesto)",
    allowFlags: AllowFlags.JoinRoom,
  });
  Object.assign(this, callbacks);
}

const callbacks = {
  onGameTick: () => {
    if (!ready || !room) return;
    const me = room.currentPlayer;
    if (!me) return;
    const fr = fb.build(room, room.currentFrameNo, me.id, echo());
    if (fr) cer.send(fr);
    if (pendingInput && room.currentFrameNo - pendingInput.frame > 120) pendingInput = null;
  },
  onPlayerInputChange: (id, value) => {
    if (!pendingInput || !room || id !== room.currentPlayerId || value !== pendingInput.state) return;
    const d = room.currentFrameNo - pendingInput.frame;
    pendingInput = null;
    if (d >= 0 && d < 120) {
      delays.push(d);
      if (delays.length > 60) delays.shift();
      if (delays.length === 10) console.log(`latencia medida (eco): ${echo()} ticks`);
    }
  },
  onSetDiscProperties: (id, type, d1) => fb.onSetDiscProperties(id, type, d1),
  onPlayerBallKick: (id) => fb.onPlayerBallKick(id),
  onTeamGoal: (t) => fb.onTeamGoal(t),
  onPlayerJoin: (p) => fb.onPlayerJoin(p),
  onAnnouncement: (m) => fb.onAnnouncement(m),
};
const plugin = new HX11Plugin();

readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const [cmd, val] = line.trim().split(/\s+/);
  if (cmd === "pos" && val) { pos = val.toUpperCase(); cer.setPos(pos); console.log(`puesto pedido: ${pos}`); }
  else if (cmd === "pausa") { paused = true; room && room.setKeyState(0); console.log("en pausa (no aprieta teclas)"); }
  else if (cmd === "sigue") { paused = false; console.log("juega"); }
  else if (cmd === "salir") { room && room.leave(); cer.stop(); process.exit(0); }
});

(async () => {
  if (API.ready) await API.ready;
  const [, authObj] = await Utils.generateAuth();
  Room.join({ id: joinId, authObj, password: arg("--password", undefined) }, {
    storage: { player_name: arg("--name", "HX-Bot"), avatar: arg("--avatar", "🤖"),
               geo: { lat: parseFloat(arg("--lat", "-34.6")), lon: parseFloat(arg("--lon", "-58.4")), flag: arg("--flag", "ar") } },
    plugins: [plugin],
    onOpen: (r) => { room = r; console.log(`conectado a "${r.name}" como ${arg("--name", "HX-Bot")}; puesto ${pos}`); },
    onClose: (e) => {
      let det = "";
      try { det = JSON.stringify({ code: e && e.code, params: e && e.params }); } catch {}
      console.log("salió de la sala", e ? String(e) : "", det); cer.stop(); process.exit(0);
    },
  });
})().catch((e) => { console.error(e.message); cer.stop(); process.exit(1); });
