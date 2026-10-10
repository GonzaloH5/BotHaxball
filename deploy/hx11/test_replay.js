// Prueba de punta a punta del bot de HaxEleven sin sala: una grabación pasa por el mismo armado de ticks
// (deploy/hx11/frame.js) y por el cerebro (learn/hx11_cerebro.py), como si fuera la sala en vivo. Se mide:
//   * que el cerebro responda una decisión cada 3 ticks con el jugador elegido en la cancha;
//   * el tiempo de respuesta del cerebro (tiene que ser bastante menor que 3 ticks = 50 ms);
//   * el acuerdo de la dirección con la del humano que ocupa ese lugar en la grabación (diagnóstico).
//
//   node deploy/hx11/test_replay.js <grabación.hbr2> [--model runs/hx11_l0/l0_sinhist_full/model.pt] [--pos DFC]
//        [--player <id>] [--minutos 3]
"use strict";

const fs = require("fs");
const path = require("path");
const API = require("../../bridge/haxball")({}, { language: false });
const { Replay } = API;
const { FrameBuilder } = require("./frame");
const { startCerebro } = require("./cerebro_proc");

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const file = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
const model = opt("--model", "runs/hx11_l0/l0_sinhist_full/model.pt");
const pos = opt("--pos", "DFC");
const minutes = parseFloat(opt("--minutos", "3"));
let meId = opt("--player", null) != null ? Number(opt("--player")) : null;

const fb = new FrameBuilder();
const sent = new Map();   // t -> ms de envío
const lat = [];
let decisions = 0, agree = 0, compared = 0, frames = 0;
const human = new Map();  // t -> input del humano en ese lugar
const MOVE = (b) => [((b >> 3) & 1) - ((b >> 2) & 1), ((b >> 1) & 1) - (b & 1)];

let ready = false;
const cer = startCerebro({
  model, pos, temp: 0.7, renovarPatada: 0.33, sinSaques: "corner,tiro_libre,saque_arco",
  reglaJuego: argv.includes("--regla-juego"),
  onReady: () => { ready = true; },
  onDecision: (msg) => {
    decisions++;
    const t0 = sent.get(msg.t);
    if (t0 != null) lat.push(Date.now() - t0);
    const h = human.get(msg.t);
    if (h != null) {
      const [ax, ay] = MOVE(msg.input), [hx, hy] = MOVE(h);
      compared++;
      if (ax === hx && ay === hy) agree++;
    }
  },
});

let reader;
function tick() {
  if (!ready) return;
  const gs = reader.gameState;
  if (!gs) return;
  if (meId == null) {
    const p = reader.state.players.find((q) => q.team && q.team.id === 1 && q.disc);
    if (!p) return;
    meId = p.id;
  }
  const t = reader.getCurrentFrameNo();
  const me = reader.state.players.find((q) => q.id === meId);
  if (me) human.set(t, me.input | 0);
  const fr = fb.build(reader, t, meId, 20);
  if (!fr) return;
  sent.set(t, Date.now());
  cer.send(fr);
  if (opt("--volcar", null)) fs.appendFileSync(opt("--volcar"), JSON.stringify(fr) + "\n");   // ticks para perfilar
  frames++;
  if (frames >= minutes * 3600) finish();
}

function finish() {
  try { reader.setSpeed(0); } catch {}
  setTimeout(() => {
    lat.sort((a, b) => a - b);
    const q = (p) => lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null;
    console.log(JSON.stringify({ archivo: path.basename(file), jugador: meId, ticks: frames, decisiones: decisions,
      decisiones_por_tick: +(decisions / Math.max(frames, 1)).toFixed(3), latencia_ms_p50: q(0.5), latencia_ms_p95: q(0.95),
      acuerdo_direccion_con_humano: compared ? +(agree / compared).toFixed(3) : null }));
    cer.stop();
    process.exit(0);
  }, 3000);
}

const data = new Uint8Array(fs.readFileSync(file));
reader = Replay.read(data, {
  onGameTick: tick,
  onSetDiscProperties: (id, type, d1) => fb.onSetDiscProperties(id, type, d1),
  onPlayerBallKick: (id) => fb.onPlayerBallKick(id),
  onTeamGoal: (t) => fb.onTeamGoal(t),
  onPlayerJoin: (p) => fb.onPlayerJoin(p),
  onAnnouncement: (m) => fb.onAnnouncement(m),
  onEnd: () => finish(),
}, {
  requestAnimationFrame: (cb) => setTimeout(() => cb(performance.now()), 16),
  cancelAnimationFrame: (id) => clearTimeout(id),
});
// a velocidad real (1×): así se mide el tiempo de respuesta del cerebro como en la sala
const wait = setInterval(() => { if (ready) { clearInterval(wait); reader.setSpeed(1); } }, 200);
setTimeout(() => finish(), (minutes * 60 + 90) * 1000);
