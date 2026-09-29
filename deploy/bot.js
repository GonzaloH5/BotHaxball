// Bot de HaxBall entrenado con RL, jugando en una sala real vía node-haxball.
//
// Crear sala propia (el bot es el host y juega):
//   HAXBALL_TOKEN=thr1.xxxx node bot.js --name "Sala RL" --password 1234
//   (el token se saca a mano en https://www.haxball.com/headlesstoken, dura unos minutos)
// Unirse a una sala existente (sin recaptcha):
//   node bot.js --join <roomId> [--password 1234] [--extrap 80]
//   (como cliente el bot mide en vivo el retraso de sus inputs y extrapola el estado esa cantidad;
//    ideal usar un modelo afinado con env.action_delay_max=3 para el jitter restante)
//
// Comandos en el chat: !bot red | !bot blue | !bot spec | !greedy | !temp 0.7 | !start
//
// Úsalo sólo en tus salas o con permiso: en salas públicas/competitivas contra gente es trampa.
const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-node");
// bridge/haxball.js (sesión del puente) arregla WebRTC para salas remotas: node-haxball usa
// setRemoteDescription con callbacks y el polyfill de node-datachannel los ignora, y además fija la
// interfaz de red. Si no está, se usa node-haxball tal cual (sólo sirve en local).
let API;
try {
  API = require("../bridge/haxball")();
} catch (e) {
  console.warn("bridge/haxball.js no disponible, usando node-haxball sin arreglos:", e.message);
  API = require("node-haxball")();
}
const { Room, Plugin, Utils, AllowFlags } = API;
const { buildObs, decodeAction } = require("./obs");
const { buildObsUniversal } = require("./obs_universal");
const { assertObservationContract } = require("./observation_contract");
const { PolicyMemory } = require("./policy_memory");
const os = require("os");
const { spawnSync } = require("child_process");

// ------------------------------------------------------------------ args
const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const MODEL = arg("--model", path.join(__dirname, "model.onnx"));
const META = JSON.parse(fs.readFileSync(MODEL.replace(/\.onnx$/, ".json"), "utf8"));
let temperature = parseFloat(arg("--temp", "0.0"));
const joinId = arg("--join", null);
// ms de extrapolación como cliente; por defecto se mide en vivo (ver inputDelayTicks)
const extrapMs = arg("--extrap", null) != null ? parseFloat(arg("--extrap")) : null;
const STADIUM_ARG = arg("--stadium", null);

// Modelo multi-tarea (obs "universal", train/multitask.py): juega cualquier mapa y formato.
// La geometría del mapa de la sala se calcula con el mismo código Python del entrenamiento
// (export/stadium_geom.py). --rules plain|real|auto: si la sala tiene script de powershot + pelotas
// paradas (Real Soccer, Real Futsal, HaxEleven). auto = por el nombre del mapa.
const UNIVERSAL = META.layout === "universal";
const RULES = arg("--rules", "auto");
const REPO = path.resolve(__dirname, "..");
const PYTHON = process.env.HAXBALL_PYTHON ||
  path.join(REPO, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

function stadiumGeometry(stadium) {
  const file = path.join(os.tmpdir(), `haxballrl_stadium_${process.pid}.hbs`);
  fs.writeFileSync(file, Utils.exportStadium(stadium));
  const r = spawnSync(PYTHON, ["-m", "export.stadium_geom", file], { cwd: REPO, encoding: "utf8", maxBuffer: 64 << 20 });
  if (r.status !== 0) throw new Error(`export.stadium_geom falló: ${r.stderr || r.error}`);
  return JSON.parse(r.stdout);
}

function rulesFor(name) {
  const real = RULES === "real" || (RULES === "auto" && /real|rs ?x|x6|x7|haxeleven|rsx/i.test(name || ""));
  return { psOn: real, outOfBounds: real };
}

const R_PLAYER = 15, R_BALL = 10, KICK_REACH = 4;

function sample(logits) {
  if (temperature <= 0) return logits.indexOf(Math.max(...logits));
  const mx = Math.max(...logits);
  const e = logits.map((l) => Math.exp((l - mx) / temperature));
  const s = e.reduce((a, b) => a + b, 0);
  let r = Math.random() * s;
  for (let i = 0; i < e.length; i++) { r -= e[i]; if (r <= 0) return i; }
  return e.length - 1;
}

function BotPlugin(session) {
  Object.setPrototypeOf(this, Plugin.prototype);
  Plugin.call(this, "rlBot", true, {
    version: "1.0", author: "HaxballRL", description: "Bot PPO self-play",
    allowFlags: AllowFlags.CreateRoom | AllowFlags.JoinRoom,
  });
  const that = this;
  let tick = 0;
  let ticksSinceKickoff = 0;
  let kickCancel = false;
  let lastKey = { dirX: 0, dirY: 0, kick: false };

  // Estado manual del saque después de un gol.
  let pendingKickoffTeam = -1;
  let forcedKickoff = false;
  let forcedKickoffTeam = -1;
  let busy = false;
  const policyMemory = new PolicyMemory(META);
  let policyActive = false;
  const resetPolicy = () => {
    policyMemory.reset();
    policyActive = false;
  };

  const teamIdx = (t) => (t === 1 ? 0 : t === 2 ? 1 : -1);

  // Como cliente (--join), entre el frame que vemos al decidir y el frame en que el host aplica
  // nuestro input pasan ~6-11 ticks (mediana 8) incluso con ping ~0: es estructural del cliente,
  // más el ping en salas remotas. Se mide en vivo (setKeyState -> onPlayerInputChange propio) y se
  // extrapola el estado por esa mediana. Como host no hay retraso y se usa el estado real.
  const delays = [];
  let pendingInput = null;
  const inputDelayTicks = () => {
    if (!delays.length) return 8;  // valor medido en runs/real/latency_*.json
    const d = [...delays].sort((a, b) => a - b);
    return d[d.length >> 1];
  };
  const disc = (p) => (useExt && p.disc.ext) || p.disc;
  let useExt = false;

  function snapshot() {
    const room = that.room;
    const me = room.currentPlayer;
    useExt = !room.isHost;
    if (useExt) {
      const ms = extrapMs != null ? extrapMs : Math.min(inputDelayTicks() * 1000 / 60, 300);
      room.extrapolate(ms);
    }
    const gs = (useExt && room.gameStateExt) || room.gameState;
    if (!gs || !me || !me.disc || teamIdx(me.team.id) < 0) return null;
    const ball = gs.physicsState.discs[0];
    const bpos = [ball.pos.x, ball.pos.y];
    const myTeam = teamIdx(me.team.id);
    const T = META.n_per_team;
    const inGame = room.state.players.filter((p) => p.disc && teamIdx(p.team.id) >= 0 && p.id !== me.id);
    const d2 = (p) => (disc(p).pos.x - bpos[0]) ** 2 + (disc(p).pos.y - bpos[1]) ** 2;
    // si hay más jugadores que en el entrenamiento, usar los más cercanos a la pelota
    const mates = inGame.filter((p) => teamIdx(p.team.id) === myTeam).sort((a, b) => d2(a) - d2(b)).slice(0, T - 1);
    const opps = inGame.filter((p) => teamIdx(p.team.id) !== myTeam).sort((a, b) => d2(a) - d2(b)).slice(0, T);
    const conv = (p) => ({
      team: teamIdx(p.team.id), pos: [disc(p).pos.x, disc(p).pos.y],
      vel: [disc(p).speed.x, disc(p).speed.y], canKick: true, touching: false
    });
    const players = [conv(me), ...mates.map(conv), ...opps.map(conv)];
    // rellenar con jugadores "fantasma" parados en su arco si faltan
    const ghost = (team) => ({ team, pos: [team === 0 ? -META.goal_x : META.goal_x, 0], vel: [0, 0], canKick: true, touching: false });
    while (players.filter((p) => p.team === myTeam).length < T) players.push(ghost(myTeam));
    while (players.filter((p) => p.team !== myTeam).length < T) players.push(ghost(1 - myTeam));
    const dist = Math.hypot(bpos[0] - players[0].pos[0], bpos[1] - players[0].pos[1]);
    players[0].touching = dist - R_PLAYER - R_BALL < KICK_REACH;
    return {
      ball: { pos: bpos, vel: [ball.speed.x, ball.speed.y] },
      players, myTeam,
      kickoff: gs.state === 0,
      kickoffTeam: teamIdx(gs.goalConcedingTeam ? gs.goalConcedingTeam.id : 1),
      tfrac: Math.min(ticksSinceKickoff / META.max_ticks, 1),
    };
  }

  // ---- obs universal: geometría del mapa actual + snapshot con TODOS los jugadores en cancha
  let geom = null, rules = null, geomFor = null, chargeTicks = 0;
  function ensureGeometry() {
    const st = that.room.stadium;
    if (geomFor === st) return geom !== null;
    geomFor = st;
    try {
      rules = rulesFor(st.name);
      assertObservationContract(META, rules);
      geom = stadiumGeometry(st);
      console.log(`mapa "${st.name}": cancha ${geom.field_half_w}x${geom.field_half_h}, reglas ` +
        `${rules.psOn ? "real (powershot + pelotas paradas)" : "sin script"}`);
      return true;
    } catch (e) {
      console.error("no pude calcular la geometría del mapa:", e.message);
      geom = null;
      if (e.code !== "OBSERVATION_CONTRACT") geomFor = null;
      that.room.setKeyState(0);
      lastKey = { dirX: 0, dirY: 0, kick: false };
      return false;
    }
  }

  function snapshotUniversal() {
    const room = that.room;
    const me = room.currentPlayer;
    useExt = !room.isHost;
    if (useExt) {
      const ms = extrapMs != null ? extrapMs : Math.min(inputDelayTicks() * 1000 / 60, 300);
      room.extrapolate(ms);
    }
    const gs = (useExt && room.gameStateExt) || room.gameState;
    if (!gs || !me || !me.disc || teamIdx(me.team.id) < 0 || !ensureGeometry()) return null;
    const ball = gs.physicsState.discs[0];
    const bd = (useExt && ball.ext) || ball;
    const bpos = [bd.pos.x, bd.pos.y];
    const others = room.state.players.filter((p) => p.disc && teamIdx(p.team.id) >= 0 && p.id !== me.id);
    const conv = (p) => ({
      team: teamIdx(p.team.id), pos: [disc(p).pos.x, disc(p).pos.y],
      vel: [disc(p).speed.x, disc(p).speed.y], canKick: true, touching: false
    });
    const players = [conv(me), ...others.map(conv)];
    const dist = Math.hypot(bpos[0] - players[0].pos[0], bpos[1] - players[0].pos[1]);
    players[0].touching = dist - geom.player_radius - geom.ball_radius < KICK_REACH;
    let ps = null;
    if (rules.psOn) {
      // el estado del script no se ve: se estima con lo observable (masa y gravedad de la pelota) y
      // cuánto tiempo venimos pegados a la pelota con la patada apretada (la carga del powershot)
      const cfg = META.ps_cfg;
      chargeTicks = players[0].touching && lastKey.kick ? chargeTicks + 1 : 0;
      const g = bd.gravity || { x: 0, y: 0 };
      const invb = (bd.invMass - geom.ball_invmass) / (cfg.power_inv - geom.ball_invmass);
      ps = {
        comba: g.x !== 0 || g.y !== 0 ? 1 : 0, prog: Math.min(chargeTicks / cfg.charge, 1),
        invb: Math.min(Math.max(invb, 0), 1), grav: [g.x / cfg.grav, g.y / cfg.grav],
        holder: chargeTicks > 0 ? 0 : -1
      };
    }
    const ballSpeed = Math.hypot(bd.speed.x, bd.speed.y);

    // Una vez que la pelota empieza a moverse,
    // ya terminó nuestro override manual del saque.
    if (forcedKickoff && ballSpeed > 0.05) {
      forcedKickoff = false;
      forcedKickoffTeam = -1;
    }

    const isKickoff = forcedKickoff || gs.state === 0;

    const detectedKickoffTeam =
      forcedKickoff
        ? forcedKickoffTeam
        : teamIdx(gs.goalConcedingTeam ? gs.goalConcedingTeam.id : 1);

    return {
      ball: { pos: bpos, vel: [bd.speed.x, bd.speed.y] },
      players, myTeam: teamIdx(me.team.id), ps,
      kickoff: isKickoff,
      kickoffTeam: detectedKickoffTeam,
      tfrac: Math.min(ticksSinceKickoff / META.max_ticks, 1),
    };
  }
  this.onStadiumChange = () => { geom = null; geomFor = null; chargeTicks = 0; resetPolicy(); };

  const resetControls = () => {
    kickCancel = false;
    pendingInput = null;
    lastKey = { dirX: 0, dirY: 0, kick: false };

    try {
      that.room.setKeyState(0);
    } catch (_) { }
  };

  this.onGameStart = () => {
    ticksSinceKickoff = 0;

    pendingKickoffTeam = -1;
    forcedKickoff = false;
    forcedKickoffTeam = -1;

    resetControls();
    resetPolicy();
  };

  this.onTeamGoal = (teamId) => {
    ticksSinceKickoff = 0;

    // teamId 1 = red, 2 = blue.
    // Saca el equipo que RECIBIÓ el gol.
    const scorer = teamIdx(teamId);

    if (scorer >= 0) {
      pendingKickoffTeam = 1 - scorer;
    }

    resetControls();
    resetPolicy();
  };

  this.onPositionsReset = () => {
    ticksSinceKickoff = 0;

    // Si venimos de un gol, éste es definitivamente un saque.
    if (pendingKickoffTeam >= 0) {
      forcedKickoff = true;
      forcedKickoffTeam = pendingKickoffTeam;
      pendingKickoffTeam = -1;
    }

    resetControls();
    resetPolicy();
  };

  this.onGameStop = () => {
    resetPolicy();

    if (that.room.isHost) {
      setTimeout(() => {
        if (!that.room.gameState) {
          that.room.startGame();
        }
      }, 1500);
    }
  };
  this.onPlayerTeamChange = resetPolicy;
  this.onPlayerLeave = resetPolicy;

  this.onPlayerInputChange = (id, value) => {
    const room = that.room;
    if (!pendingInput || id !== room.currentPlayerId || value !== pendingInput.state) return;
    const d = room.currentFrameNo - pendingInput.frame;
    pendingInput = null;
    if (d >= 0 && d < 120) {
      delays.push(d);
      if (delays.length > 60) delays.shift();
      if (delays.length === 10) console.log(`retraso de input medido: ${inputDelayTicks()} ticks`);
    }
  };

  this.onGameTick = () => {
    const s = UNIVERSAL ? snapshotUniversal() : snapshot();
    if (!s) {
      if (policyActive) resetPolicy();
      return;
    }
    policyActive = true;
    tick++;
    ticksSinceKickoff++;
    // kick_cancel del motor == (patear apretado && !isKicking), verificado con partidas reales
    const me = that.room.currentPlayer;
    kickCancel = lastKey.kick && !me.isKicking;
    s.players[0].canKick = !kickCancel;

    if (pendingInput && that.room.currentFrameNo - pendingInput.frame > 120) pendingInput = null;
    if (tick % META.frame_skip !== 0 || busy) return;
    busy = true;
    const obs = UNIVERSAL
      ? buildObsUniversal(s, 0, geom, { maxEntities: s.players.length - 1, ...rules })
      : buildObs(s, 0, META);
    const generation = policyMemory.generation;
    session.run({
      obs: new ort.Tensor("float32", Float32Array.from(obs), [1, obs.length]),
      ...policyMemory.feeds(ort)
    })
      .then((out) => {
        if (generation !== policyMemory.generation) return;
        const a = sample(Array.from(out.logits.data));
        const key = decodeAction(a, s.myTeam);
        const state = Utils.keyState(key.dirX, key.dirY, key.kick);
        const changed = state !== Utils.keyState(lastKey.dirX, lastKey.dirY, lastKey.kick);
        lastKey = key;
        if (changed && !that.room.isHost && !pendingInput) {
          pendingInput = { state, frame: that.room.currentFrameNo };
        }
        that.room.setKeyState(state);
        policyMemory.accept(out, a, generation);
      })
      .catch((e) => { resetPolicy(); console.error("inferencia:", e); })
      .finally(() => { busy = false; });
  };

  this.onPlayerJoin = (p) => {
    resetPolicy();
    const room = that.room;
    if (!room.isHost) return;
    room.sendChat(`Hola ${p.name}! Soy un bot entrenado con RL. Comandos: !bot red|blue|spec, !start, !greedy, !temp x`);
    const me = room.currentPlayer;
    if (me.team.id === 0) room.setPlayerTeam(me.id, 1);
    if (room.state.players.filter((q) => q.team.id === 2).length === 0) room.setPlayerTeam(p.id, 2);
    if (!room.gameState) {
      room.setTimeLimit(3);
      room.setScoreLimit(3);
      room.startGame();
    }
  };

  this.onPlayerChat = (id, msg) => {
    const room = that.room;
    const me = room.currentPlayer;
    const [cmd, val] = msg.trim().split(/\s+/);
    if (cmd === "!bot" && room.isHost) {
      room.setPlayerTeam(me.id, { red: 1, blue: 2, spec: 0 }[val] ?? me.team.id);
    } else if (cmd === "!greedy") {
      temperature = 0; room.sendChat("modo greedy (acción más probable)");
    } else if (cmd === "!temp") {
      temperature = parseFloat(val) || 1; room.sendChat(`temperatura ${temperature}`);
    } else if (cmd === "!start" && room.isHost) {
      if (room.gameState) room.stopGame();
      room.startGame();
    }
  };
}

(async () => {
  if (API.ready) await API.ready;
  const session = await ort.InferenceSession.create(MODEL);
  console.log(UNIVERSAL
    ? `modelo ${MODEL} (multi-tarea: cualquier mapa y formato; entrenado en ${(META.tasks || []).join(", ")})`
    : `modelo ${MODEL} (obs ${META.obs_dim}, ${META.n_per_team}v${META.n_per_team}, estadio ${META.stadium})`);
  const common = {
    storage: {
      player_name: arg("--player", "RL-Bot"),
      avatar: "8",
      geo: {
        lat: -34.6037,
        lon: -58.3816,
        flag: "ar"
      }
    },
    plugins: [new BotPlugin(session)],
    onOpen: (room) => {
      console.log("conectado a la sala:", room.name);
      room.onAfterRoomLink = (link) => console.log("link de la sala:", link);

      if (STADIUM_ARG) {
        const stadiumPath = path.join(REPO, "stadiums", `${STADIUM_ARG}.hbs`);

        if (!fs.existsSync(stadiumPath)) {
          console.error("No existe el mapa:", stadiumPath);
        } else {
          const stadiumText = fs.readFileSync(stadiumPath, "utf8");
          const stadium = Utils.parseStadium(stadiumText);
          room.setCurrentStadium(stadium);
          console.log("mapa cargado:", stadiumPath);
        }
      }
    },
    onClose: (e) => { console.log("sala cerrada", e?.toString?.() ?? ""); process.exit(0); },
  };
  if (joinId) {
    const [, authObj] = await Utils.generateAuth();
    Room.join({ id: joinId, authObj, password: arg("--password", undefined) }, common);
  } else {
    const token = process.env.HAXBALL_TOKEN || arg("--token", null);
    if (!token) {
      console.error("Falta el token: sacalo en https://www.haxball.com/headlesstoken y pasalo con HAXBALL_TOKEN=... o --token");
      process.exit(1);
    }
    Room.create({
      name: arg("--name", "HaxballRL bot"), password: arg("--password", undefined),
      showInRoomList: args.includes("--public"), maxPlayerCount: 8, token, noPlayer: false,
    }, common);
  }
})();
