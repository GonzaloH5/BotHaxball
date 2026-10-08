// Inventario de grabaciones HaxEleven 11v11 (docs/PLAN_HAXELEVEN.md, S0), sin volcar la física.
// Por archivo: hash, sala, estadios por tramo, frames jugados por estadio y plantel, entradas/salidas y
// cambios de equipo, nombres y avatares, chat, anuncios del host, resumen de setDiscProperties (qué campos
// toca el script y en qué discos) y patadas. Cada estadio distinto se exporta una vez a <dir estadios>.
//
// Uso: node bridge/hx11_meta.js [--stadiums data/hx11/stadiums] <a.hbr2> [<b.hbr2> ...] > data/hx11/meta.jsonl
// (una línea JSON por archivo; nombres y chat quedan en data/, que no se versiona)

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const API = require("./haxball")({}, { language: false });
const { Replay, Utils } = API;

const argv = process.argv.slice(2);
const si = argv.indexOf("--stadiums");
const stadiumDir = si >= 0 ? argv[si + 1] : null;
const files = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1] === "--stadiums"));
if (stadiumDir) fs.mkdirSync(stadiumDir, { recursive: true });

// data1: [x, y, vx, vy, gx, gy, radius, bCoef, invMass, damping]; data2: [color, cMask, cGroup] (null = sin cambio)
const F1 = ["x", "y", "vx", "vy", "gx", "gy", "radius", "bCoef", "invMass", "damping"];
const F2 = ["color", "cMask", "cGroup"];
const MAX_CHAT = 5000;
const MAX_ANN = 3000;
const MAX_SDP_SAMPLES = 40;

function exportStadium(st) {
  if (!stadiumDir) return null;
  let txt;
  try { txt = Utils.exportStadium(st); } catch (_) { return null; }
  const h = crypto.createHash("sha256").update(txt).digest("hex").slice(0, 12);
  const name = `${String(st.name || "sin_nombre").replace(/[^\w.-]+/g, "_").slice(0, 60)}_${h}.hbs`;
  const f = path.join(stadiumDir, name);
  if (!fs.existsSync(f)) fs.writeFileSync(f, txt);
  return name;
}

function metaOf(file) {
  return new Promise((resolve) => {
    const buf = fs.readFileSync(file);
    const out = { file: path.basename(file), bytes: buf.length,
                  sha256: crypto.createHash("sha256").update(buf).digest("hex"), error: null };
    let reader = null;
    let done = false;
    const t0 = Date.now();
    const stadiums = [];      // {f, name, file}
    const events = [];        // start/stop/goal/time_up/pause/kickoff/positions_reset/limits
    const teamChanges = [];   // {f, id, team, by}
    const joins = [];         // {f, id, name}
    const leaves = [];        // {f, id, reason, banned, by}
    const names = {};         // id -> nombre
    const avatars = {};       // id -> [{f, v}]
    const headless = {};      // id -> [{f, v}] (los pone el host)
    const chat = [];          // {f, id, m}
    const ann = new Map();    // texto -> {n, f0, color, style}
    let annTotal = 0;
    const sdp = {};           // "tipo:id:campos" -> n
    const sdpSamples = [];
    let sdpTotal = 0;
    let kicks = 0;
    const played = {};        // "estadio|RvB" -> frames en estado 0/1
    let playedFrames = 0;
    let curStadium = "?";
    const frame = () => (reader ? reader.getCurrentFrameNo() : 0);

    const finish = (why) => {
      if (done) return;
      done = true;
      try { reader.setSpeed(0); } catch (_) {}
      Object.assign(out, {
        end: why, secs: Math.round((Date.now() - t0) / 100) / 10,
        frames: reader ? reader.maxFrameNo : 0, played_frames: playedFrames, played,
        stadiums, events, team_changes: teamChanges, joins, leaves, names, avatars, headless_avatars: headless,
        chat, chat_truncated: chat.length >= MAX_CHAT,
        announcements: [...ann.entries()].map(([m, v]) => ({ m, ...v })), announcements_total: annTotal,
        sdp, sdp_total: sdpTotal, sdp_samples: sdpSamples, kicks,
      });
      resolve(out);
    };

    const onTick = () => {
      const gs = reader.gameState;
      if (!gs || (gs.state !== 0 && gs.state !== 1)) return;
      let red = 0, blue = 0;
      for (const p of reader.state.players) {
        const t = p.team ? p.team.id : 0;
        if (t === 1) red++;
        else if (t === 2) blue++;
      }
      const k = `${curStadium}|${red}v${blue}`;
      played[k] = (played[k] || 0) + 1;
      playedFrames++;
    };
    const fieldsOf = (d1, d2) => {
      const s = [];
      if (d1) for (let i = 0; i < F1.length; i++) if (d1[i] != null && !Number.isNaN(d1[i])) s.push(F1[i]);
      if (d2) for (let i = 0; i < F2.length; i++) if (d2[i] != null && !Number.isNaN(d2[i])) s.push(F2[i]);
      return s.join(",");
    };
    const plain = (v) => (v == null ? null : Array.from(v, (x) => (x == null || Number.isNaN(x) ? null : x)));

    try {
      const data = new Uint8Array(buf);
      reader = Replay.read(data, {
        onGameTick: onTick,
        onTeamGoal: (team) => {
          const gs = reader.gameState;
          events.push({ t: "goal", f: frame(), team, score: gs ? [gs.redScore, gs.blueScore] : null,
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
        onKickOff: () => events.push({ t: "kickoff", f: frame() }),
        onPositionsReset: () => events.push({ t: "positions_reset", f: frame() }),
        onTimeLimitChange: (v) => events.push({ t: "time_limit", f: frame(), v }),
        onScoreLimitChange: (v) => events.push({ t: "score_limit", f: frame(), v }),
        onStadiumChange: (st) => {
          curStadium = st.name;
          stadiums.push({ f: frame(), name: st.name, file: exportStadium(st) });
        },
        onPlayerTeamChange: (id, team, by) => teamChanges.push({ f: frame(), id, team, by }),
        onPlayerJoin: (p) => { names[p.id] = p.name; joins.push({ f: frame(), id: p.id, name: p.name }); },
        onPlayerLeave: (p, reason, banned, by) => leaves.push({ f: frame(), id: p.id, reason: reason ?? null, banned: !!banned, by }),
        onPlayerAvatarChange: (id, v) => (avatars[id] = avatars[id] || []).push({ f: frame(), v }),
        onPlayerHeadlessAvatarChange: (id, v) => (headless[id] = headless[id] || []).push({ f: frame(), v }),
        onPlayerChat: (id, m) => { if (chat.length < MAX_CHAT) chat.push({ f: frame(), id, m }); },
        onAnnouncement: (m, color, style) => {
          annTotal++;
          const e = ann.get(m);
          if (e) e.n++;
          else if (ann.size < MAX_ANN) ann.set(m, { n: 1, f0: frame(), color, style });
        },
        onSetDiscProperties: (id, type, d1, d2) => {
          sdpTotal++;
          const k = `${type}:${id}:${fieldsOf(d1, d2)}`;
          sdp[k] = (sdp[k] || 0) + 1;
          if (sdpSamples.length < MAX_SDP_SAMPLES) sdpSamples.push({ f: frame(), id, type, d1: plain(d1), d2: plain(d2) });
        },
        onPlayerBallKick: () => kicks++,
        onEnd: () => finish("fin"),
      }, {
        requestAnimationFrame: (cb) => setImmediate(() => cb(performance.now())),
        cancelAnimationFrame: (id) => clearImmediate(id),
      });
      out.room = reader.state.name;
      curStadium = reader.state.stadium.name;
      stadiums.push({ f: 0, name: curStadium, file: exportStadium(reader.state.stadium) });
      for (const p of reader.state.players) {
        names[p.id] = p.name;
        if (p.avatar) (avatars[p.id] = avatars[p.id] || []).push({ f: 0, v: p.avatar });
      }
      out.initial_players = reader.state.players.map((p) => ({ id: p.id, team: p.team ? p.team.id : 0 }));
      reader.setSpeed(1000);
    } catch (e) {
      out.error = String((e && e.message) || e);
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
    setTimeout(() => { clearInterval(iv); finish("timeout"); }, 10 * 60e3);
  });
}

(async () => {
  for (const f of files) {
    const m = await metaOf(f);
    process.stdout.write(JSON.stringify(m) + "\n");
  }
  process.exit(0);
})();
