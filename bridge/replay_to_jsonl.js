// Convierte un replay de HaxBall (.hbr2) al mismo formato JSONL que record.js, reproduciéndolo
// con el motor de node-haxball (el del juego original). Sirve para validar sim/ contra partidas reales
// de cualquier modo sin entrar a una sala.
//
// Uso: node bridge/replay_to_jsonl.js <replay.hbr2> [--out runs/real] [--max-minutes 5]
//
// Salida: <out>/replay_<nombre>.jsonl + <out>/replay_<nombre>_0.hbs
// Los cambios de discos que hace el script de la sala (laterales, córners, powershot...) quedan como
// eventos "disc_props" en su frame: compare_sim.py excluye esas transiciones.

const fs = require("fs");
const path = require("path");
const API = require("./haxball")({}, { language: false });
const { Replay, Utils } = API;

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const file = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!file) {
  console.error("uso: node bridge/replay_to_jsonl.js <replay.hbr2> [--out runs/real] [--max-minutes 5]");
  process.exit(1);
}
const outDir = opt("--out", path.join(__dirname, "..", "runs", "real"));
const maxFrames = Math.round(parseFloat(opt("--max-minutes", "5")) * 3600);
fs.mkdirSync(outDir, { recursive: true });
const base = path.join(outDir, "replay_" + path.basename(file).replace(/\.hbr2?$/, "").replace(/[^\w.-]+/g, "_"));
// escritura sincrónica: el replay se reproduce más rápido de lo que un stream alcanza a vaciarse
const fd = fs.openSync(base + ".jsonl", "w");
const write = (o) => fs.writeSync(fd, JSON.stringify(o) + "\n");
const r = (v) => Math.round(v * 1e4) / 1e4;

let reader = null;
let nTicks = 0;
let nStadium = 0;
let finished = false;

function saveStadium(st) {
  const f = `${base}_${nStadium++}.hbs`;
  try { fs.writeFileSync(f, Utils.exportStadium(st)); } catch (e) { return null; }
  return path.basename(f);
}

function frame() { return reader.getCurrentFrameNo(); }

function tick() {
  const gs = reader.gameState;
  if (!gs) return;
  const discs = gs.physicsState.discs;
  const players = reader.state.players
    .filter((p) => p.team && p.team.id !== 0 && p.disc)
    .map((p) => ({ id: p.id, team: p.team.id, input: p.input, kicking: p.isKicking, disc: discs.indexOf(p.disc) }));
  write({
    type: "tick", frame: frame(), state: gs.state,
    ko: gs.goalConcedingTeam ? gs.goalConcedingTeam.id : null,
    time: r(gs.timeElapsed), score: [gs.redScore, gs.blueScore],
    discs: discs.map((d) => ({ x: r(d.pos.x), y: r(d.pos.y), vx: r(d.speed.x), vy: r(d.speed.y), r: d.radius, pid: d.playerId ?? null })),
    // pistas observables del powershot del script: masa y gravedad de la pelota
    ball: { im: discs[0].invMass, gx: r(discs[0].gravity ? discs[0].gravity.x : 0), gy: r(discs[0].gravity ? discs[0].gravity.y : 0) },
    players,
  });
  nTicks++;
  if (frame() >= maxFrames) finish("límite de minutos");
}

function finish(why) {
  if (finished) return;
  finished = true;
  try { reader.setSpeed(0); } catch (_) {}
  fs.closeSync(fd);
  console.log(`${path.basename(file)}: ${nTicks} ticks (${why}) -> ${base}.jsonl`);
  process.exit(0);
}

const data = new Uint8Array(fs.readFileSync(file));
const ev = (name, extra) => write({ type: "event", name, frame: reader ? frame() : 0, ...extra });
reader = Replay.read(data, {
  onGameTick: tick,
  onTeamGoal: (teamId) => ev("goal", { team: teamId }),
  onPositionsReset: () => ev("positions_reset", {}),
  onPlayerBallKick: (playerId) => ev("kick", { playerId }),
  onGameStart: () => ev("game_start", {}),
  onGameStop: () => ev("game_stop", {}),
  onStadiumChange: (st) => ev("stadium_change", { stadium: st.name, stadiumFile: saveStadium(st) }),
  onSetDiscProperties: (id, type) => ev("disc_props", { id, kind: type }),
  onPlayerDiscPropertiesChange: (id) => ev("disc_props", { id, kind: "player" }),
  onBallDiscPropertiesChange: () => ev("disc_props", { id: 0, kind: "ball" }),
  onGamePauseChange: (paused) => ev("pause", { paused }),
  onTimeIsUp: () => ev("time_up", {}),
  onEnd: () => finish("fin del replay"),
}, {
  // sin animación: avanzar tan rápido como se pueda
  requestAnimationFrame: (cb) => setImmediate(() => cb(performance.now())),
  cancelAnimationFrame: (id) => clearImmediate(id),
});

const st = reader.state.stadium;
write({ type: "header", room: reader.state.name, source: path.basename(file), stadium: st.name,
        stadiumFile: saveStadium(st), maxFrameNo: reader.maxFrameNo });
reader.setSpeed(1000);
// vigilante: si el replay llegó al último frame o dejó de avanzar (partido detenido antes del final del
// archivo, onEnd a veces no llega) se termina en vez de esperar al timeout de seguridad
let lastFrame = -1, still = 0;
setInterval(() => {
  const f = reader.getCurrentFrameNo();
  if (f >= reader.maxFrameNo - 1) return finish("fin del replay");
  still = f === lastFrame ? still + 1 : 0;
  lastFrame = f;
  if (still >= 5) finish("el replay dejó de avanzar");
}, 1000);
setTimeout(() => finish("timeout de seguridad"), 10 * 60e3);
