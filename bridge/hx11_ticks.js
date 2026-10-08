// Grabación HaxEleven -> JSONL por tick con el estado completo que hace falta para simular (docs/PLAN_HAXELEVEN.md, S1).
//
// A diferencia de replay_to_jsonl.js, guarda las PROPIEDADES de los discos que cambia el script de la sala (invMass de
// la pelota con la carga del powershot, invMass/bCoef/gravedad/cGroup de los jugadores, barreras): el simulador las
// necesita en cada tramo, no alcanza con saltear el tick en que cambian. Para que el archivo no sea enorme:
//   * "m": discos del estadio que se movieron este tick (los quietos no se repiten): {índice: [x, y, vx, vy]}
//   * "p": propiedades que cambiaron respecto del tick anterior: {índice: {campo: valor}} (el primer tick trae todas)
//   * la pelota (0) y los jugadores van siempre, con su entrada (input) y si la patada está activa.
// Los anuncios del host van como eventos (traen nombres: la salida es para data/, no se versiona).
//
// Uso: node bridge/hx11_ticks.js <grabación.hbr2> --out data/hx11/ticks [--max-minutes 3] [--skip-minutes 0]

const fs = require("fs");
const path = require("path");
const API = require("./haxball")({}, { language: false });
const { Replay, Utils } = API;

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const file = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!file) {
  console.error("uso: node bridge/hx11_ticks.js <grabación.hbr2> --out data/hx11/ticks [--max-minutes 3] [--skip-minutes 0]");
  process.exit(1);
}
const outDir = opt("--out", path.join(__dirname, "..", "data", "hx11", "ticks"));
const maxFrames = Math.round(parseFloat(opt("--max-minutes", "3")) * 3600);
const skipFrames = Math.round(parseFloat(opt("--skip-minutes", "0")) * 3600);
fs.mkdirSync(outDir, { recursive: true });
const base = path.join(outDir, path.basename(file).replace(/\.hbr2?$/, "").replace(/[^\w.-]+/g, "_"));
const fd = fs.openSync(base + ".jsonl", "w");
const write = (o) => fs.writeSync(fd, JSON.stringify(o) + "\n");
const r4 = (v) => (v == null || Number.isNaN(v) ? null : Math.round(v * 1e4) / 1e4);

let reader = null;
let finished = false;
let nTicks = 0;
let nStadium = 0;
let firstFrame = null;
const frame = () => reader.getCurrentFrameNo();
// último estado conocido por disco: [x, y, vx, vy] y propiedades
const lastPos = [];
const lastProps = [];
const PROPS = ["radius", "invMass", "bCoef", "damping", "cMask", "cGroup", "gx", "gy"];

function propsOf(d) {
  return { radius: d.radius, invMass: d.invMass, bCoef: d.bCoef, damping: d.damping, cMask: d.cMask,
           cGroup: d.cGroup, gx: r4(d.gravity ? d.gravity.x : 0), gy: r4(d.gravity ? d.gravity.y : 0) };
}

function saveStadium(st) {
  const f = `${base}_${nStadium++}.hbs`;
  try { fs.writeFileSync(f, Utils.exportStadium(st)); } catch (e) { return null; }
  return path.basename(f);
}

function tick() {
  const gs = reader.gameState;
  if (!gs) return;
  const f = frame();
  if (f < skipFrames) return;
  if (firstFrame === null) firstFrame = f;
  const discs = gs.physicsState.discs;
  const playerDisc = new Set();
  const players = [];
  for (const p of reader.state.players) {
    if (!p.team || p.team.id === 0 || !p.disc) continue;
    const di = discs.indexOf(p.disc);
    playerDisc.add(di);
    players.push({ id: p.id, team: p.team.id, input: p.input, kicking: p.isKicking, disc: di });
  }
  const moved = {};
  const props = {};
  const out = {};
  for (let i = 0; i < discs.length; i++) {
    const d = discs[i];
    const pv = [r4(d.pos.x), r4(d.pos.y), r4(d.speed.x), r4(d.speed.y)];
    if (i === 0 || playerDisc.has(i)) out[i] = pv;
    else {
      const lp = lastPos[i];
      if (!lp || lp[0] !== pv[0] || lp[1] !== pv[1] || lp[2] !== pv[2] || lp[3] !== pv[3]) moved[i] = pv;
    }
    lastPos[i] = pv;
    const pr = propsOf(d);
    const lpr = lastProps[i];
    let ch = null;
    for (const k of PROPS) {
      if (!lpr || lpr[k] !== pr[k]) (ch = ch || {})[k] = pr[k];
    }
    if (ch) props[i] = ch;
    lastProps[i] = pr;
  }
  write({ type: "tick", frame: f, state: gs.state, ko: gs.goalConcedingTeam ? gs.goalConcedingTeam.id : null,
          time: r4(gs.timeElapsed), score: [gs.redScore, gs.blueScore], nd: discs.length,
          d: out, m: moved, p: props, players });
  nTicks++;
  if (f - firstFrame >= maxFrames) finish("límite de minutos");
}

function finish(why) {
  if (finished) return;
  finished = true;
  try { reader.setSpeed(0); } catch (_) {}
  fs.closeSync(fd);
  console.log(`${path.basename(file)}: ${nTicks} ticks (${why}) -> ${base}.jsonl`);
  process.exit(0);
}

const plain = (v) => (v == null ? null : Array.from(v, (x) => (x == null || Number.isNaN(x) ? null : x)));
const ev = (name, extra) => write({ type: "event", name, frame: reader ? frame() : 0, ...extra });
const data = new Uint8Array(fs.readFileSync(file));
reader = Replay.read(data, {
  onGameTick: tick,
  onTeamGoal: (team) => ev("goal", { team }),
  onPositionsReset: () => ev("positions_reset", {}),
  onPlayerBallKick: (playerId) => ev("kick", { playerId }),
  onGameStart: () => ev("game_start", {}),
  onGameStop: () => ev("game_stop", {}),
  onStadiumChange: (st) => ev("stadium_change", { stadium: st.name, stadiumFile: saveStadium(st) }),
  // data1: [x, y, vx, vy, gx, gy, radius, bCoef, invMass, damping]; data2: [color, cMask, cGroup]
  onSetDiscProperties: (id, type, d1, d2) => ev("disc_props", { id, player: !!type, d1: plain(d1), d2: plain(d2) }),
  onPlayerTeamChange: (id, team) => ev("team_change", { id, team }),
  onAnnouncement: (msg) => ev("announcement", { msg }),
  onGamePauseChange: (paused) => ev("pause", { paused }),
  onTimeIsUp: () => ev("time_up", {}),
  onEnd: () => finish("fin del replay"),
}, {
  requestAnimationFrame: (cb) => setImmediate(() => cb(performance.now())),
  cancelAnimationFrame: (id) => clearImmediate(id),
});

const st = reader.state.stadium;
write({ type: "header", source: path.basename(file), stadium: st.name, stadiumFile: saveStadium(st),
        maxFrameNo: reader.maxFrameNo });
reader.setSpeed(1000);
let lastF = -1, still = 0;
setInterval(() => {
  const f = reader.getCurrentFrameNo();
  if (f >= reader.maxFrameNo - 1) return finish("fin del replay");
  still = f === lastF ? still + 1 : 0;
  lastF = f;
  if (still >= 5) finish("el replay dejó de avanzar");
}, 1000);
setTimeout(() => finish("timeout de seguridad"), 15 * 60e3);
