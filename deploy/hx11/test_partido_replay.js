// Prueba sin sala del cerebro de deploy/hx11/partido.js: una grabación pasa por el armado de ticks (frame.js) y el
// cerebro en modo de varios bots decide por los 22 jugadores de la grabación (roles 4-3-3 / 4-4-2 por orden), como en
// la sala. Mide decisiones por segundo, cuántos bots reciben teclas, el tiempo de respuesta y la parte de decisiones
// con dirección (si da ~0, el cerebro está trabado como en la sala del 2026-10-09).
//
//   node deploy/hx11/test_partido_replay.js <grabación.hbr2> [--segundos 60] [--cada-ticks 3]
"use strict";

const fs = require("fs");
const API = require("../../bridge/haxball")({}, { language: false });
const { Replay } = API;
const { FrameBuilder } = require("./frame");
const { startCerebro } = require("./cerebro_proc");

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const file = argv[0];
const seconds = parseFloat(opt("--segundos", "60"));
const RED = ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MED-I", "MED-C", "MED-D", "DEL-I", "DEL-C", "DEL-D"];
const BLUE = ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MED-I", "MED-CI", "MED-CD", "MED-D", "DEL-CI", "DEL-CD"];

const fb = new FrameBuilder();
const sent = new Map();
const lat = [];
const registered = new Set();
const got = new Set();
let ready = false, msgs = 0, moves = 0, total = 0, frames = 0, t0 = null;

const cer = startCerebro({ model: opt("--model", "runs/hx11_l0/l0_sinhist_ancho/model_final.pt"), pos: null, temp: 0.7,
                           every: Number(opt("--cada-ticks", "3")), renovarPatada: parseFloat(opt("--renovar-patada", "0.33")),
                           onReady: () => { ready = true; } });
require("readline").createInterface({ input: cer.proc.stdout }).on("line", (line) => {
  let m; try { m = JSON.parse(line); } catch { return; }
  if (!m.inputs) return;
  msgs++;
  const s = sent.get(m.t);
  if (s != null) lat.push(Date.now() - s);
  for (const [id, bits] of Object.entries(m.inputs)) { got.add(id); total++; if (bits & 15) moves++; }
});

let reader;
function tick() {
  if (!ready) return;
  const st = reader.state;
  if (!reader.gameState) return;
  // registro: los jugadores de cada equipo por orden de aparición, con los roles de la formación
  for (const team of [1, 2]) {
    const ps = st.players.filter((p) => p.team && p.team.id === team && p.disc);
    ps.forEach((p, i) => {
      if (registered.has(p.id) || i >= 11) return;
      registered.add(p.id);
      cer.send({ register: { id: p.id, pos: `${team === 1 ? "4-3-3" : "4-4-2"}:${(team === 1 ? RED : BLUE)[i]}` } });
    });
  }
  const t = reader.getCurrentFrameNo();
  const fr = fb.build(reader, t, [...registered][0], 22);
  if (!fr) return;
  fr.bots = [...registered].map((id) => ({ me: id, delay: 22 }));
  sent.set(fr.t, Date.now());
  cer.send(fr);
  frames++;
  if (t0 == null) t0 = Date.now();
  if (frames >= seconds * 60) finish();
}

function finish() {
  try { reader.setSpeed(0); } catch {}
  setTimeout(() => {
    lat.sort((a, b) => a - b);
    const q = (p) => lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null;
    const secs = (Date.now() - t0) / 1000;
    console.log(JSON.stringify({ ticks: frames, segundos: +secs.toFixed(1), registrados: registered.size,
      con_teclas: got.size, decisiones_s: +(msgs / secs).toFixed(1), resp_ms_p50: q(0.5), resp_ms_p95: q(0.95),
      parte_con_direccion: total ? +(moves / total).toFixed(3) : null }));
    cer.stop();
    process.exit(0);
  }, 2000);
}

reader = Replay.read(new Uint8Array(fs.readFileSync(file)), {
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
const wait = setInterval(() => { if (ready) { clearInterval(wait); reader.setSpeed(1); } }, 200);
setTimeout(() => finish(), (seconds + 120) * 1000);
