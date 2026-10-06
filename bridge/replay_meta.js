// Metadatos de grabaciones .hbr2 sin volcar la física: sala, estadios por tramo, planteles con nombres,
// goles, inicio/fin de partidos y frames jugados por estadio y por plantel 4v4. Sirve para indexar el
// dataset (sesiones, jugadores, splits sin fuga) sin convertir a JSONL.
//
// Uso: node bridge/replay_meta.js <a.hbr2> [<b.hbr2> ...]  > meta.jsonl   (una línea JSON por archivo)

const fs = require("fs");
const path = require("path");
const API = require("./haxball")({}, { language: false });
const { Replay } = API;

function metaOf(file) {
  return new Promise((resolve) => {
    const out = { file: path.basename(file), bytes: fs.statSync(file).size, error: null };
    let reader = null;
    let done = false;
    const stadiums = [];       // {frame, name}
    const events = [];         // goles, start/stop, cambios de plantel
    const names = {};          // id -> último nombre visto
    let lastRoster = "";
    // frames jugados (state 1 = en juego, 0 = saque inicial) por estadio y si el plantel es 4v4
    const played = {};
    let playedFrames = 0;
    const frame = () => reader.getCurrentFrameNo();
    const finish = (why) => {
      if (done) return;
      done = true;
      try { reader.setSpeed(0); } catch (_) {}
      out.end = why;
      out.frames = reader ? reader.maxFrameNo : 0;
      out.stadiums = stadiums;
      out.events = events;
      out.names = names;
      out.played = played;
      out.played_frames = playedFrames;
      resolve(out);
    };
    const roster = () => {
      const ps = reader.state.players.filter((p) => p.team && p.team.id !== 0);
      const red = ps.filter((p) => p.team.id === 1).map((p) => p.id).sort((a, b) => a - b);
      const blue = ps.filter((p) => p.team.id === 2).map((p) => p.id).sort((a, b) => a - b);
      for (const p of reader.state.players) names[p.id] = p.name;
      return { red, blue };
    };
    const onTick = () => {
      const gs = reader.gameState;
      if (!gs) return;
      const r = roster();
      const key = JSON.stringify(r);
      if (key !== lastRoster) {
        lastRoster = key;
        events.push({ t: "roster", f: frame(), red: r.red, blue: r.blue });
      }
      if (gs.state === 0 || gs.state === 1) {
        const st = reader.state.stadium ? reader.state.stadium.name : "?";
        const k = `${st}|${r.red.length}v${r.blue.length}`;
        played[k] = (played[k] || 0) + 1;
        playedFrames++;
      }
    };
    try {
      const data = new Uint8Array(fs.readFileSync(file));
      reader = Replay.read(data, {
        onGameTick: onTick,
        onTeamGoal: (teamId) => {
          const gs = reader.gameState;
          events.push({ t: "goal", f: frame(), team: teamId, score: gs ? [gs.redScore, gs.blueScore] : null,
                        time: gs ? Math.round(gs.timeElapsed * 100) / 100 : null });
        },
        onGameStart: () => events.push({ t: "start", f: frame() }),
        onGameStop: () => {
          const gs = reader.gameState;
          events.push({ t: "stop", f: frame(), score: gs ? [gs.redScore, gs.blueScore] : null,
                        time: gs ? Math.round(gs.timeElapsed * 100) / 100 : null });
        },
        onTimeIsUp: () => events.push({ t: "time_up", f: frame() }),
        onGamePauseChange: (paused) => events.push({ t: "pause", f: frame(), paused }),
        onStadiumChange: (st) => stadiums.push({ f: frame(), name: st.name }),
        onEnd: () => finish("fin"),
      }, {
        requestAnimationFrame: (cb) => setImmediate(() => cb(performance.now())),
        cancelAnimationFrame: (id) => clearImmediate(id),
      });
      out.room = reader.state.name;
      stadiums.push({ f: 0, name: reader.state.stadium.name });
      for (const p of reader.state.players) names[p.id] = p.name;
      reader.setSpeed(1000);
    } catch (e) {
      out.error = String(e && e.message || e);
      resolve(out);
      return;
    }
    let lastFrame = -1, still = 0;
    const iv = setInterval(() => {
      if (done) return clearInterval(iv);
      const f = reader.getCurrentFrameNo();
      if (f >= reader.maxFrameNo - 1) { clearInterval(iv); return finish("fin"); }
      still = f === lastFrame ? still + 1 : 0;
      lastFrame = f;
      if (still >= 5) { clearInterval(iv); finish("detenido"); }
    }, 500);
    setTimeout(() => { clearInterval(iv); finish("timeout"); }, 5 * 60e3);
  });
}

(async () => {
  const files = process.argv.slice(2);
  for (const f of files) {
    const m = await metaOf(f);
    process.stdout.write(JSON.stringify(m) + "\n");
  }
  process.exit(0);
})();
