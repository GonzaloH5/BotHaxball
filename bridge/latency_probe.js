// Mide el retraso real entre que el bot aprieta una tecla y que el juego la aplica,
// jugando como cliente en una sala (para calibrar action_delay_max del entrenamiento).
//
// Uso:
//   node bridge/latency_probe.js <link o id de sala> [--name "[BOT] lag-test"] [--password x] [--samples 60] [--minutes N]
//
// El bot tiene que estar en un equipo (rojo o azul) con el partido en marcha: pedile al admin que lo mueva.
// Mientras mide, sólo da pasitos izquierda/derecha cada ~0.75 s (no patea).
//
// Para cada cambio de tecla guarda:
//   obsFrame   = frame del estado que el bot estaba viendo al decidir (room.currentFrameNo)
//   applyFrame = frame en que el juego aplicó ese input (onPlayerInputChange del propio jugador)
//   delay      = applyFrame - obsFrame  -> retraso efectivo en ticks entre observación y acción
// Resumen al final + JSON en runs/real/latency_<fecha>_<sala>.json

const fs = require("fs");
const path = require("path");
const API = require("./haxball")(); // node-haxball + arreglo para salas remotas
const { Utils, Room } = API;

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const target = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!target) {
  console.error('uso: node bridge/latency_probe.js <link o id de sala> [--name "[BOT] lag-test"] [--password x] [--samples 60] [--minutes N]');
  process.exit(1);
}
const m = target.match(/[?&]c=([\w-]+)/);
const roomId = m ? m[1] : target;
const NAME = opt("--name", "[BOT] lag-test");
const SAMPLES = parseInt(opt("--samples", "60"), 10);
const EVERY = 45; // ticks entre cambios de tecla
const PATTERN = [4, 0, 8, 0]; // izq, nada, der, nada: cada cambio es distinguible del anterior

const AUTH_FILE = path.join(__dirname, ".auth_key");
async function loadAuth() {
  if (fs.existsSync(AUTH_FILE)) {
    const key = fs.readFileSync(AUTH_FILE, "utf8").trim();
    return [key, await Utils.authFromKey(key)];
  }
  const [key, obj] = await Utils.generateAuth();
  fs.writeFileSync(AUTH_FILE, key);
  return [key, obj];
}

const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};

(async () => {
  await API.ready; // fija la interfaz de red antes de conectar
  const [authKey, authObj] = await loadAuth();
  const samples = [];
  const pings = [];
  let pending = null; // { value, obsFrame, t }
  let ticks = 0;
  let step = 0;
  let roomRef = null;
  let warned = false;
  let finished = false;

  const finish = (why) => {
    if (finished) return;
    finished = true;
    console.log(`\nterminado (${why}).`);
    try { roomRef && roomRef.setKeyState(0); } catch (_) {}
    if (samples.length) {
      const d = samples.map((s) => s.delay);
      const ms = samples.map((s) => s.ms);
      const summary = {
        room: roomRef ? roomRef.name : null, roomId, when: new Date().toISOString(), n: samples.length,
        delayTicks: { min: Math.min(...d), median: pct(d, 0.5), p90: pct(d, 0.9), max: Math.max(...d) },
        delayMs: { median: +pct(ms, 0.5).toFixed(1), p90: +pct(ms, 0.9).toFixed(1) },
        pingMs: pings.length ? { median: +pct(pings, 0.5).toFixed(1), max: +Math.max(...pings).toFixed(1) } : null,
        samples, pings,
      };
      console.log(`retraso en ticks: min ${summary.delayTicks.min}, mediana ${summary.delayTicks.median}, ` +
        `p90 ${summary.delayTicks.p90}, máx ${summary.delayTicks.max}`);
      console.log(`retraso en ms: mediana ${summary.delayMs.median}, p90 ${summary.delayMs.p90}; ping ${JSON.stringify(summary.pingMs)}`);
      const outDir = path.join(__dirname, "..", "runs", "real");
      fs.mkdirSync(outDir, { recursive: true });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const file = path.join(outDir, `latency_${stamp}_${roomId}.json`);
      fs.writeFileSync(file, JSON.stringify(summary, null, 1));
      console.log("guardado en", file);
    } else {
      console.log("no se tomó ninguna muestra (¿el bot estuvo en un equipo con el partido en marcha?)");
    }
    try { roomRef && roomRef.leave(); } catch (_) {}
    setTimeout(() => process.exit(0), 300);
  };
  process.on("SIGINT", () => finish("Ctrl+C"));
  const minutes = parseFloat(opt("--minutes", "0"));
  if (minutes > 0) setTimeout(() => finish("tiempo cumplido"), minutes * 60e3);

  console.log(`uniéndome a ${roomId} como "${NAME}"...`);
  Room.join({ id: roomId, password: opt("--password", null), authObj }, {
    storage: { player_name: NAME, avatar: "⏱", player_auth_key: authKey },
    onOpen: (room) => {
      roomRef = room;
      console.log(`dentro de "${room.name}". Moveme a un equipo y arrancá el partido para medir.`);

      room.onPingChange = (inst, avg) => pings.push(avg);

      room.onPlayerInputChange = (id, value) => {
        if (id !== room.currentPlayerId || !pending || value !== pending.value) return;
        const s = { value, obsFrame: pending.obsFrame, applyFrame: room.currentFrameNo,
                    delay: room.currentFrameNo - pending.obsFrame, ms: performance.now() - pending.t };
        samples.push(s);
        pending = null;
        process.stdout.write(`\rmuestras ${samples.length}/${SAMPLES}  último retraso ${s.delay} ticks (${s.ms.toFixed(0)} ms)   `);
        if (samples.length >= SAMPLES) finish("muestras completas");
      };

      room.onGameTick = () => {
        const me = room.currentPlayer;
        if (!room.gameState || !me || !me.team || me.team.id === 0) {
          if (!warned && room.gameState) { console.log("estoy en espectadores: moveme a rojo o azul"); warned = true; }
          return;
        }
        if (++ticks % EVERY !== 0) return;
        pending = null; // si el anterior no llegó (p.ej. el admin me movió), se descarta
        const value = PATTERN[step++ % PATTERN.length];
        if (value === me.input) return;
        pending = { value, obsFrame: room.currentFrameNo, t: performance.now() };
        room.setKeyState(value, true);
      };
    },
    onClose: (msg) => {
      console.log("\nsalí de la sala:", msg ? msg.toString() : "");
      finish("desconectado");
    },
  });
})().catch((e) => { console.error(e); process.exit(1); });
