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
const { PublicSignalTracker } = require("./public_signals");
const { sampleLogits, geometryFromText, ortSessionOptions } = require("./runtime");

// ------------------------------------------------------------------ args
const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const MANAGED = args.includes('--managed');
const managedSecrets = MANAGED ? JSON.parse(process.env.RS4_BOT_SECRETS || '{}') : {};
delete process.env.RS4_BOT_SECRETS;
let managedRoom = null;
const report = data => { if (MANAGED && process.connected) process.send(data); };
if (MANAGED) process.on('disconnect',()=>{try{managedRoom?.setKeyState(0);managedRoom?.leave();}finally{process.exit(0);}});
const MODEL = arg("--model", path.join(__dirname, "model.onnx"));
const META = JSON.parse(fs.readFileSync(MODEL.replace(/\.onnx$/, ".json"), "utf8"));
let temperature = parseFloat(arg("--temp", "0.0"));
const joinId = arg("--join", null);
// ms de extrapolación como cliente; por defecto se mide en vivo (ver inputDelayTicks)
const extrapMs = arg("--extrap", null) != null ? parseFloat(arg("--extrap")) : null;
const STADIUM_ARG = arg("--stadium", null);
const PUBLIC_CUES = {...(META.public_signals || {})};
PUBLIC_CUES.debug = args.includes("--debug-public-signals");
for (const [flag, key] of [["--rs4-barrier-discs", "barrier_discs"], ["--rs4-barrier-segments", "barrier_segments"], ["--rs4-barrier-joints", "barrier_joints"]]) {
  const supplied = arg(flag, null);
  if (supplied !== null) PUBLIC_CUES[key] = supplied === "" ? [] : supplied.split(",").map(Number);
}
if (args.includes("--rs4-auto-joints")) PUBLIC_CUES.auto_joints=true;
for (const [flag, key] of [["--rs4-red-colors", "red_colors"], ["--rs4-blue-colors", "blue_colors"]]) {
  const supplied = arg(flag, null);
  if (supplied !== null) PUBLIC_CUES[key] = supplied.split(",");
}

// Modelo multi-tarea (obs "universal", train/multitask.py): juega cualquier mapa y formato.
// La geometría del mapa de la sala se calcula con el mismo código Python del entrenamiento
// (export/stadium_geom.py). --rules plain|real|auto: si la sala tiene script de powershot + pelotas
// paradas (Real Soccer, Real Futsal, HaxEleven). auto = por el nombre del mapa.
const UNIVERSAL = META.layout === "universal";
// Modelo RS4-Z (export/to_onnx_rs4z.py): observación v2 armada con lo que se ve en la sala
// (deploy/rs4z/room_state.js, paridad con el simulador en deploy/test_rs4z_room_state.js). Entrenado
// con la latencia del cliente como entrada: no se extrapola el estado, se pasa el retraso medido.
const RS4Z = META.obs_version === "rs4z-obs-v2";
// Sala de prueba con varios bots (anfitrión): --lobby reparte a los bots por nombre y no arranca solo;
// !admin <clave> da admin a quien la escriba (clave por --admin-password o RS4_ADMIN_PASSWORD).
const LOBBY = args.includes("--lobby");
const BOT_PREFIX = arg("--bot-prefix", "RL-Bot");
const ADMIN_PASSWORD = process.env.RS4_ADMIN_PASSWORD || arg("--admin-password", null);
// RS4-Z: --hyst m conserva la acción anterior mientras su logit no quede más de m por debajo del mejor
// (antitemblor en greedy); --trace archivo.jsonl registra cada decisión (lo percibido y los logits).
const HYST = parseFloat(arg("--hyst", "0"));
const TRACE = arg("--trace", null) ? fs.createWriteStream(arg("--trace"), { flags: "a" }) : null;
const DUMP_STADIUM = arg("--dump-stadium", null);   // guarda el mapa de la sala (.hbs) al empezar a jugar
// Retraso total que se le informa a la red (ticks). Como cliente el bot ve el estado del host atrasado y su tecla
// llega tarde: en una grabación real actuaba ~15 ticks atrás (eco de la tecla ~3 + vista ~12). "auto" = eco
// medido + VIEW_LAG, con tope en el máximo de entrenamiento. Un número fija el valor.
const DELAY_ARG = arg("--delay", "auto");
const VIEW_LAG = parseFloat(arg("--view-lag", "12"));
// Como cliente: extrapolar el estado `ms` hacia adelante (como la opción de extrapolación del cliente oficial) para
// achicar el atraso de la vista; el retraso automático descuenta lo extrapolado.
const EXTRAP_MS = parseFloat(arg("--extrapolate", "0"));
const { RS4ZTracker, decodeOwnAction } = require("./rs4z/room_state");
const RULES = arg("--rules", "auto");
const REPO = path.resolve(__dirname, "..");

function stadiumGeometry(stadium) {
  return geometryFromText(Utils.exportStadium(stadium));
}

function rulesFor(name) {
  const real = RULES === "real" || (RULES === "auto" && /real|rs ?x|x6|x7|haxeleven|rsx/i.test(name || ""));
  return { psOn: real, outOfBounds: real };
}

const R_PLAYER = 15, R_BALL = 10, KICK_REACH = 4;

function sample(logits) {
  return sampleLogits(logits, temperature);
}

function BotPlugin(session, managed = false) {
  let managedEnabled = true;
  Object.setPrototypeOf(this, Plugin.prototype);
  Plugin.call(this, "rlBot", true, {
    version: "1.0", author: "HaxballRL", description: "Bot PPO self-play",
    allowFlags: AllowFlags.CreateRoom | AllowFlags.JoinRoom,
  });
  const that = this;
  let tick = 0;
  const publicTracker = META.public_signals ? new PublicSignalTracker(PUBLIC_CUES) : null;
  let publicPacket = null, publicFrame = null;
  let lastPublicColors = null;
  let ticksSinceKickoff = 0;
  let kickCancel = false;
  let lastKey = { dirX: 0, dirY: 0, kick: false };

  // Estado manual del saque después de un gol.
  let pendingKickoffTeam = -1;
  let forcedKickoff = false;
  let forcedKickoffTeam = -1;
  let busy = false;
  const policyMemory = new PolicyMemory(META);
  const rs4z = RS4Z ? new RS4ZTracker({ ...PUBLIC_CUES, ball_radii: META.ball_radii, kick_strengths: META.kick_strengths }) : null;
  let rs4zPrev = -1;
  let stadiumDumped = false;
  let policyActive = false;
  const resetPolicy = () => {
    policyMemory.reset();
    if (rs4z) rs4z.reset();
    rs4zPrev = -1;
    policyActive = false;
    if (publicTracker) { publicTracker.reset(); publicPacket = null; publicFrame = null; lastPublicColors = null; }
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
      if (publicTracker) publicTracker.configureStadium(st,geom);
      if (publicTracker && PUBLIC_CUES.debug) {
        console.log("RS4 public barrier candidates (diagnostic only):", JSON.stringify({
          discs: that.room.gameState.physicsState.discs.map((d, id) => ({id, color:d.color}))
            .filter(d => d.id > 0 && [0xFF0000,0x0000FF,0xE56E56,0x5689E5].includes(d.color)),
          segments: (that.room.gameState.physicsState.segments || st.segments || []).map((s, id) => ({id,color:s.color,vis:s.vis}))
            .filter(s => [0xFF0000,0x0000FF,0xE56E56,0x5689E5].includes(s.color)),
          auto_joints: publicTracker.jointIds
        }));
      }
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

    if (publicTracker) {
      const real = room.gameState.physicsState.discs[0];
      const frame = room.currentFrameNo;
      if (publicFrame === null || frame < publicFrame || frame - publicFrame >= META.frame_skip) {
        if (publicFrame !== null && frame < publicFrame) publicTracker.reset();
        publicPacket = publicTracker.sample({pos:[real.pos.x, real.pos.y], vel:[real.speed.x, real.speed.y], color:real.color},
          room.state.players.filter(p => p.disc && teamIdx(p.team.id) >= 0).map(p =>
            ({team:teamIdx(p.team.id),pos:[p.disc.pos.x,p.disc.pos.y]})),
          publicFrame === null || frame < publicFrame ? 0 : frame - publicFrame,
          geom.player_radius, geom.ball_radius,
          publicTracker.barrierColor(room.gameState.physicsState.discs,
            room.gameState.physicsState.segments || room.stadium.segments || [],
            room.gameState.physicsState.joints || room.stadium.joints || [],
            {pos:[real.pos.x,real.pos.y],vel:[real.speed.x,real.speed.y]}));
        publicFrame = frame;
        if (PUBLIC_CUES.debug) {
          const colors = JSON.stringify([publicPacket.ballColor, publicPacket.barrierColor]);
          if (colors !== lastPublicColors) {
            console.log("RS4 public cues:", colors, "contact estimate:", publicPacket.contactTeam);
            lastPublicColors = colors;
          }
        }
      }
    }
    return {
      ball: { pos: bpos, vel: [bd.speed.x, bd.speed.y] },
      ...(publicTracker ? {publicSignals: publicPacket} : {}),
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

    if (that.room.isHost && !managed) {
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

  function rs4zTick() {
    const room = that.room;
    const me = room.currentPlayer;
    const gs = room.gameState;
    if (!gs || !me || !me.disc || teamIdx(me.team.id) < 0) {
      if (policyActive) resetPolicy();
      return;
    }
    policyActive = true;
    const ext = EXTRAP_MS > 0 && !room.isHost;
    if (ext) room.extrapolate(EXTRAP_MS);
    const dsc = (d) => (ext && d.ext) || d;
    if (DUMP_STADIUM && !stadiumDumped) {
      stadiumDumped = true;
      try { fs.writeFileSync(DUMP_STADIUM, Utils.exportStadium(room.stadium)); } catch (e) { console.error("no pude guardar el mapa:", e.message); }
    }
    const ball = dsc(gs.physicsState.discs[0]);
    const frame = {
      state: gs.state, kickoffTeam: teamIdx(gs.goalConcedingTeam ? gs.goalConcedingTeam.id : 1),
      kickStrength: room.stadium.playerPhysics.kickStrength,
      ball: { x: ball.pos.x, y: ball.pos.y, vx: ball.speed.x, vy: ball.speed.y, r: ball.radius, color: gs.physicsState.discs[0].color },
      players: room.state.players.filter((p) => p.disc && teamIdx(p.team.id) >= 0).map((p) => ({
        id: p.id, team: teamIdx(p.team.id), x: dsc(p.disc).pos.x, y: dsc(p.disc).pos.y, vx: dsc(p.disc).speed.x,
        vy: dsc(p.disc).speed.y, input: p.input, isKicking: !!p.isKicking, invMass: p.disc.invMass })),
    };
    rs4z.update(frame);
    tick++;
    if (pendingInput && room.currentFrameNo - pendingInput.frame > 120) pendingInput = null;
    if (tick % META.frame_skip !== 0 || busy) return;
    const delayTicks = DELAY_ARG !== "auto" ? parseFloat(DELAY_ARG)
      : room.isHost ? 0 : Math.min(inputDelayTicks() + Math.max(0, VIEW_LAG - (ext ? EXTRAP_MS * 0.06 : 0)),
        META.max_delay || 24);
    const built = rs4z.build(frame, me.id, delayTicks);
    if (!built) return;
    busy = true;
    session.run({ obs: new ort.Tensor("float32", built.obs, [1, built.obs.length]) })
      .then((out) => {
        const logits = Array.from(out.logits.data);
        let a = sample(logits);
        if (HYST > 0 && temperature <= 0 && rs4zPrev >= 0 && logits[rs4zPrev] >= Math.max(...logits) - HYST) a = rs4zPrev;
        rs4zPrev = a;
        if (TRACE) TRACE.write(JSON.stringify({ frame: room.currentFrameNo, slot: built.slot, team: built.team,
          state: frame.state, restart: rs4z.restart, ballColor: frame.ball.color, kickStrength: frame.kickStrength,
          ballR: frame.ball.r, invMass: frame.players.find((p) => p.id === me.id).invMass, players: frame.players.length,
          action: a, logits: logits.map((v) => +v.toFixed(3)), obs: Array.from(built.obs, (v) => +v.toFixed(4)) }) + "\n");
        const key = decodeOwnAction(a, built.team);
        const state = Utils.keyState(key.dirX, key.dirY, key.kick);
        const changed = state !== Utils.keyState(lastKey.dirX, lastKey.dirY, lastKey.kick);
        lastKey = key;
        if (changed && !room.isHost && !pendingInput) pendingInput = { state, frame: room.currentFrameNo };
        room.setKeyState(state);
        rs4z.pushDecision(key.world);
      })
      .catch((e) => { resetPolicy(); console.error("inferencia:", e); })
      .finally(() => { busy = false; });
  }

  this.onGameTick = () => {
    if (!managedEnabled) return;
    if (RS4Z) return rs4zTick();
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
      ? buildObsUniversal(s, 0, geom, { maxEntities: s.players.length - 1, ...rules,
          publicSignalsVersion: META.public_signals?.version || 0, publicSignalConfig: PUBLIC_CUES })
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
    if (!room.isHost || managed) return;
    if (LOBBY) {
      // sala de prueba: los bots (nombre BOT_PREFIX) van al equipo con menos jugadores (rojo primero, máx. 4);
      // las personas quedan de espectadoras y el admin arma equipos y arranca el partido.
      const me = room.currentPlayer;
      if (me.team.id === 0) room.setPlayerTeam(me.id, 1);
      if (p.name.startsWith(BOT_PREFIX)) {
        const count = (t) => room.state.players.filter((q) => q.team.id === t).length;
        const team = count(1) < 4 && count(1) <= count(2) ? 1 : count(2) < 4 ? 2 : 0;
        room.setPlayerTeam(p.id, team);
      } else {
        room.sendChat(`Hola ${p.name}! Para administrar la sala escribí !admin <clave>.`, p.id);
      }
      return;
    }
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
    if (managed) return; // Managed instances accept commands only over local IPC.
    const room = that.room;
    const me = room.currentPlayer;
    const [cmd, val] = msg.trim().split(/\s+/);
    if (cmd === "!admin" && room.isHost) {
      if (ADMIN_PASSWORD && val === ADMIN_PASSWORD) {
        room.setPlayerAdmin(id, true);
        room.sendChat("admin concedido", id);
      } else {
        room.sendChat("clave de admin incorrecta", id);
      }
      return;
    }
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
  this.setManagedEnabled = enabled => { managedEnabled = enabled; resetPolicy(); resetControls(); };
}

(async () => {
  if (API.ready) await API.ready;
  if (MANAGED && !process.connected) return process.exit(0);
  // --ort-threads n: hilos de inferencia por bot (1 por defecto; ver ortSessionOptions)
  const session = await ort.InferenceSession.create(MODEL, ortSessionOptions(arg("--ort-threads", "1")));
  const plugin = new BotPlugin(session, MANAGED);
  console.log(RS4Z
    ? `modelo ${MODEL} (RS4-Z 4v4 Real Soccer ONE, obs v2, ${META.stage} ${(META.samples / 1e6).toFixed(0)}M muestras)`
    : UNIVERSAL
    ? `modelo ${MODEL} (multi-tarea: cualquier mapa y formato; entrenado en ${(META.tasks || []).join(", ")})`
    : `modelo ${MODEL} (obs ${META.obs_dim}, ${META.n_per_team}v${META.n_per_team}, estadio ${META.stadium})`);
  const common = {
    storage: {
      player_name: arg("--player", "RL-Bot"),
      avatar: arg('--avatar', '8'),
      geo: {
        lat: -34.6037,
        lon: -58.3816,
        flag: arg('--flag', 'ar')
      }
    },
    plugins: [plugin],
    onOpen: (room) => {
      console.log("conectado a la sala:", room.name);
      managedRoom = room;
      report({type:'connected',playerId:room.currentPlayerId,isHost:room.isHost});
      room.onAfterRoomLink = (link) => { console.log("link de la sala:", link); report({type:'link',link}); };
      if (MANAGED) {
        room.setAvatar(arg('--avatar','8'));
        if (room.isHost) {room.setTimeLimit(Number(arg('--time-limit','3')));room.setScoreLimit(Number(arg('--score-limit','3')));}
        room.setPlayerTeam(room.currentPlayerId,Number(arg('--team','0')));
      }

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
    onClose: (e) => { report({type:'closed',error:e?.toString?.() || ''}); console.log("sala cerrada", e?.toString?.() ?? ""); process.exit(0); },
  };
  if (joinId) {
    const [, authObj] = await Utils.generateAuth();
    Room.join({ id: joinId, authObj, password: managedSecrets.password || arg("--password", undefined) }, common);
  } else {
    const token = managedSecrets.token || process.env.HAXBALL_TOKEN || arg("--token", null);
    if (!token) {
      console.error("Falta el token: sacalo en https://www.haxball.com/headlesstoken y pasalo con HAXBALL_TOKEN=... o --token");
      process.exit(1);
    }
    Room.create({
      name: arg("--name", "HaxballRL bot"), password: managedSecrets.password || arg("--password", undefined),
      showInRoomList: args.includes("--public"), maxPlayerCount: Number(arg('--max-players','8')), token, noPlayer: false,
    }, common);
  }
  if (MANAGED) {
    const finish = () => { try {managedRoom?.setKeyState(0);managedRoom?.leave();} finally {process.exit(0);} };
    process.on('message',m=>{
      if (m?.type==='stop') return finish();
      if (!managedRoom) return;
      try {
        if (m.type==='enabled' && typeof m.enabled==='boolean') {plugin.setManagedEnabled(m.enabled);report({type:'enabled',enabled:m.enabled});}
        else if (m.type==='team' && [0,1,2].includes(m.team)) managedRoom.setPlayerTeam(m.playerId??managedRoom.currentPlayerId,m.team);
        else if (m.type==='game' && managedRoom.isHost) {
          if(m.action==='start') managedRoom.startGame();
          else if(m.action==='stop') managedRoom.stopGame();
          else if(m.action==='pause') managedRoom.pauseGame();
        }
      } catch {report({type:'commandError',error:'No se pudo aplicar el comando en la sala.'});}
    });
    setInterval(()=>{if(managedRoom)report({type:'state',team:managedRoom.currentPlayer?.team?.id??0,playing:!!managedRoom.gameState});},1000).unref();
  }
})().catch(e=>{report({type:'closed',error:e.message});console.error(e.message);process.exit(1);});
