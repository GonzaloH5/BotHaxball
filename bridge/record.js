// Grabador de partidas reales de HaxBall.
//
// Se une a una sala (como espectador, no toca nada) y guarda el estado de cada tick en JSONL,
// para comparar después con sim/physics.py y medir la latencia real.
//
// Uso:
//   node record.js <link o id de sala> [--name "[BOT] rec"] [--password x] [--out ../runs/real] [--minutes 10]
//
// Salida (en --out, un archivo por sesión):
//   <fecha>_<sala>.jsonl   primera línea {"type":"header",...}; luego {"type":"tick",...} y eventos
//   <fecha>_<sala>_<n>.hbs estadio en uso (se vuelve a guardar si el admin lo cambia)
//
// Si la sala pide reCAPTCHA, node-haxball no puede entrar solo: hay que usar otra sala o una sin captcha.

const fs = require("fs");
const path = require("path");
const API = require("./haxball")(); // node-haxball + arreglo para salas remotas
const { Utils, Room } = API;

// ------------------------------------------------------------------ argumentos
function parseArgs(argv) {
  const args = { name: "[BOT] rec", password: null, out: path.join(__dirname, "..", "runs", "real"), minutes: 0, antiAfk: 0 };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--name") args.name = argv[++i];
    else if (a === "--password") args.password = argv[++i];
    else if (a === "--out") args.out = argv[++i];
    else if (a === "--minutes") args.minutes = parseFloat(argv[++i]);
    else if (a === "--anti-afk") args.antiAfk = parseFloat(argv[++i]);
    else rest.push(a);
  }
  if (rest.length !== 1) {
    console.error('uso: node record.js <link o id de sala> [--name "[BOT] rec"] [--password x] [--out dir] [--minutes N] [--anti-afk segundos]');
    process.exit(1);
  }
  const m = rest[0].match(/[?&]c=([\w-]+)/);
  args.roomId = m ? m[1] : rest[0];
  return args;
}

// la misma identidad entre ejecuciones (así los admins ven siempre al mismo "jugador")
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

// ------------------------------------------------------------------ serialización
const r = (v) => Math.round(v * 1e4) / 1e4; // 4 decimales alcanzan y el archivo queda chico

function discJson(d) {
  return { x: r(d.pos.x), y: r(d.pos.y), vx: r(d.speed.x), vy: r(d.speed.y), r: d.radius, pid: d.playerId ?? null };
}

function tickJson(room, t0) {
  const gs = room.gameState;
  const discs = gs.physicsState.discs;
  const players = room.players
    .filter((p) => p.team && p.team.id !== 0)
    .map((p) => ({
      id: p.id,
      team: p.team.id,            // 1 rojo, 2 azul
      input: p.input,             // bits: 1 arriba, 2 abajo, 4 izq, 8 der, 16 patear
      kicking: p.isKicking,
      disc: p.disc ? discs.indexOf(p.disc) : -1,
    }));
  return {
    type: "tick",
    frame: room.currentFrameNo,
    ms: r(performance.now() - t0),
    state: gs.state,              // GamePlayState: 0 antes del saque, 1 jugando, 2 gol, 3 fin
    ko: gs.goalConcedingTeam ? gs.goalConcedingTeam.id : null, // equipo que saca (1 rojo, 2 azul)
    time: r(gs.timeElapsed),
    score: [gs.redScore, gs.blueScore],
    discs: discs.map(discJson),   // índice 0 = pelota en los mapas normales
    players,
  };
}

// ------------------------------------------------------------------ main
async function main() {
  const args = parseArgs(process.argv.slice(2));
  fs.mkdirSync(args.out, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const base = path.join(args.out, `${stamp}_${args.roomId}`);
  const out = fs.createWriteStream(base + ".jsonl");
  const write = (obj) => out.write(JSON.stringify(obj) + "\n");

  await API.ready; // fija la interfaz de red antes de conectar
  const [authKey, authObj] = await loadAuth();
  const t0 = performance.now();
  let nTicks = 0;
  let nStadium = 0;
  let ping = null;

  const saveStadium = (room) => {
    const file = `${base}_${nStadium++}.hbs`;
    try {
      fs.writeFileSync(file, Utils.exportStadium(room.stadium));
    } catch (e) {
      console.warn("no pude exportar el estadio:", e.message);
      return null;
    }
    return path.basename(file);
  };

  let stopping = false;
  let roomRef = null;
  const stop = (why) => {
    if (stopping) return;
    stopping = true;
    console.log(`\ncerrando (${why}); ${nTicks} ticks guardados en ${base}.jsonl`);
    const st = require("./haxball").stats;
    console.log(`mensajes retenidos hasta abrir todos los canales: ${st.heldMessages}`);
    try { roomRef && roomRef.leave(); } catch (_) {}
    out.end(() => process.exit(0));
  };
  process.on("SIGINT", () => stop("Ctrl+C"));
  if (args.minutes > 0) setTimeout(() => stop("tiempo cumplido"), args.minutes * 60e3);

  console.log(`uniéndome a ${args.roomId} como "${args.name}"...`);
  Room.join(
    { id: args.roomId, password: args.password, authObj },
    {
      // crappy_router: espera 10 s (en vez de 4) al conectar por WebRTC con hosts remotos
      storage: { player_name: args.name, avatar: "🤖", player_auth_key: authKey, crappy_router: true },
      onConnInfo: (state, extra) => {
        if (extra && typeof extra === "object" && extra.toString) console.log("conexión:", state, String(extra).slice(0, 120));
        else console.log("conexión:", state);
      },
      onOpen: (room) => {
        roomRef = room;
        console.log(`dentro de "${room.name}". Grabando (Ctrl+C para terminar).`);
        write({
          type: "header",
          room: room.name,
          roomId: args.roomId,
          me: room.currentPlayerId,
          stadium: room.stadium.name,
          stadiumFile: saveStadium(room),
          started: new Date().toISOString(),
        });

        // anti-AFK (salas "juegan todos" que kickean inactivos): si estoy en un equipo,
        // cada N segundos un pasito de 6 ticks alternando izquierda/derecha. Queda grabado como input.
        let afkTick = 0;
        let afkDir = 4;
        const antiAfk = () => {
          const me = room.currentPlayer;
          if (!args.antiAfk || !me || !me.team || me.team.id === 0) return;
          afkTick++;
          const period = Math.round(args.antiAfk * 60);
          if (afkTick % period === 0) {
            room.setKeyState(afkDir, true);
            afkDir = afkDir === 4 ? 8 : 4;
          } else if (afkTick % period === 6) {
            room.setKeyState(0, true);
          }
        };

        room.onGameTick = () => {
          if (!room.gameState) return;
          antiAfk();
          write(tickJson(room, t0));
          if (++nTicks % 600 === 0) {
            process.stdout.write(`\r${nTicks} ticks (${(nTicks / 60).toFixed(0)} s de juego), ping ${ping ?? "?"} ms   `);
          }
        };
        const ev = (name, data) => write({ type: "event", name, frame: room.currentFrameNo, ms: r(performance.now() - t0), ...data });
        room.onGameStart = (byId) => ev("game_start", { byId });
        room.onGameStop = (byId) => ev("game_stop", { byId });
        room.onTeamGoal = (teamId) => ev("goal", { team: teamId });
        room.onPositionsReset = () => ev("positions_reset", {});
        room.onPlayerBallKick = (playerId) => ev("kick", { playerId });
        room.onPlayerTeamChange = (id, teamId) => ev("team_change", { id, team: teamId });
        room.onStadiumChange = () => ev("stadium_change", { stadium: room.stadium.name, stadiumFile: saveStadium(room) });
        room.onPingChange = (inst, avg, max) => {
          ping = avg;
          ev("ping", { inst, avg, max });
        };
      },
      onClose: (msg) => {
        let why = "";
        try { why = msg ? msg.toString() : ""; } catch (_) {}
        console.log("\nsalí de la sala:", why || msg);
        stop("desconectado");
      },
    }
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
