// Bots de prueba para la sala HaxEleven (docs/PLAN_HAXELEVEN.md, S5 y S6). Entran como clientes, nunca escriben en
// el chat y anotan una traza JSONL con lo que veían al decidir. La grabación del host da la verdad para comparar.
//
//   node bridge/hx11_probe.js <link o id> --role lat|holder|presser [--name x] [--minutes 20] [--trace t.jsonl] [--gk]
//   --gk: escribe "!gk" una sola vez al entrar a un equipo (el script de la sala no arranca el partido sin arquero).
//
// Roles:
//   lat      S6: cada 45 ticks cambia de tecla (izquierda, nada, derecha, nada) cerca de donde está; no patea.
//   holder   S5: va a la pelota y se queda quieto con ella pegada (la vuelve a buscar sólo si se aleja > 6 px); en un saque
//            de su equipo (todos sin masa) patea y vuelve.
//   presser  S5: se pega al holder rival sin empujar (centros a <= 30,5 y quieto), con pausas de 4, 8 y 15 ticks dando un
//            paso atrás cada 60 de contacto, para medir el umbral de la falta y cuándo se reinicia la cuenta.
// Traza: una línea por cambio de tecla {t, frame, key, me:[x,y], ball:[x,y]} y por anuncio {ann}. Echo: {echo, frame}.

const fs = require("fs");
const path = require("path");
const API = require("./haxball")();
const { Utils, Room } = API;

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const target = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
if (!target) { console.error("uso: node bridge/hx11_probe.js <sala> --role lat|holder|presser"); process.exit(1); }
const m = target.match(/[?&]c=([\w-]+)/);
const roomId = m ? m[1] : target;
const ROLE = opt("--role", "lat");
const NAME = opt("--name", `[BOT] ${ROLE}`);
const MINUTES = parseFloat(opt("--minutes", "20"));
const GK = argv.includes("--gk");   // escribe "!gk" una vez al entrar a un equipo (la sala no arranca sin arquero)
const TRACE = opt("--trace", path.join(__dirname, "..", "runs", "hx11_room", `trace_${ROLE}_${Date.now()}.jsonl`));
fs.mkdirSync(path.dirname(TRACE), { recursive: true });
const out = fs.openSync(TRACE, "a");
const log = (o) => fs.writeSync(out, JSON.stringify(o) + "\n");

const UP = 1, DOWN = 2, LEFT = 4, RIGHT = 8, KICK = 16;
const toward = (dx, dy, dead = 3) =>
  (dy < -dead ? UP : 0) | (dy > dead ? DOWN : 0) | (dx < -dead ? LEFT : 0) | (dx > dead ? RIGHT : 0);

// programa de la presión: [ticks pegado, ticks de pausa] repetidos hasta la falta; cada fase dos veces
const PRESS_PLAN = [[999, 0], [999, 0], [60, 4], [60, 4], [60, 8], [60, 8], [60, 15], [60, 15]];

(async () => {
  await API.ready;
  const AUTH = path.join(__dirname, `.auth_key_${NAME.replace(/[^\w-]+/g, "_")}`);  // una identidad por bot
  let key, obj;
  if (fs.existsSync(AUTH)) { key = fs.readFileSync(AUTH, "utf8").trim(); obj = await Utils.authFromKey(key); }
  else { [key, obj] = await Utils.generateAuth(); fs.writeFileSync(AUTH, key); }
  let room = null;
  let ticks = 0, step = 0, lastKey = -1, pendingEcho = null;
  let phase = 0, phaseT = 0, pressing = true, prevBall = null, holding = false;
  const setKey = (k) => {
    if (k === lastKey) return;
    lastKey = k;
    const me = room.currentPlayer;
    const ball = room.gameState ? room.gameState.physicsState.discs[0] : null;
    log({ t: Math.round(performance.now()), frame: room.currentFrameNo, key: k,
          me: me && me.disc ? [+me.disc.pos.x.toFixed(2), +me.disc.pos.y.toFixed(2)] : null,
          ball: ball ? [+ball.pos.x.toFixed(2), +ball.pos.y.toFixed(2)] : null });
    pendingEcho = { k, frame: room.currentFrameNo };
    room.setKeyState(k);
  };
  const finish = (why) => {
    console.log(`${NAME}: fin (${why}); traza en ${TRACE}`);
    try { room && room.setKeyState(0); room && room.leave(); } catch (_) {}
    setTimeout(() => process.exit(0), 300);
  };
  setTimeout(() => finish("tiempo"), MINUTES * 60e3);
  process.on("SIGINT", () => finish("Ctrl+C"));
  console.log(`${NAME}: uniéndome a ${roomId}`);
  Room.join({ id: roomId, password: opt("--password", null), authObj: obj }, {
    storage: { player_name: NAME, avatar: ROLE === "lat" ? "⏱" : ROLE === "holder" ? "🅷" : "🅿", player_auth_key: key },
    onOpen: (r) => {
      room = r;
      console.log(`${NAME}: dentro de "${room.name}"`);
      log({ joined: room.name, stadium: room.state.stadium ? room.state.stadium.name : null, role: ROLE });
      room.onAnnouncement = (msg) => log({ ann: String(msg).slice(0, 160), frame: room.currentFrameNo });
      room.onStadiumChange = (st) => log({ stadium: st.name, frame: room.currentFrameNo });
      let gkSent = false;
      room.onPlayerTeamChange = (id, teamId) => {
        if (GK && !gkSent && id === room.currentPlayerId && teamId !== 0) {
          gkSent = true;
          setTimeout(() => { try { room.sendChat("!gk"); log({ chat: "!gk", frame: room.currentFrameNo }); } catch (_) {} }, 500);
        }
      };
      room.onPlayerInputChange = (id, value) => {
        if (id === room.currentPlayerId && pendingEcho && value === pendingEcho.k) {
          log({ echo: value, frame: room.currentFrameNo, decided: pendingEcho.frame });
          pendingEcho = null;
        }
      };
      room.onGameTick = () => {
        const me = room.currentPlayer;
        const gs = room.gameState;
        if (!gs || !me || !me.team || me.team.id === 0 || !me.disc) return;
        ticks++;
        const ball = gs.physicsState.discs[0];
        const mx = me.disc.pos.x, my = me.disc.pos.y;
        if (ROLE === "lat") {
          if (ticks % 45 !== 0) return;
          setKey([LEFT, 0, RIGHT, 0][step++ % 4]);
          return;
        }
        if (ROLE === "holder") {
          const dx = ball.pos.x - mx, dy = ball.pos.y - my;
          const edge = Math.hypot(dx, dy) - me.disc.radius - ball.radius;
          // saque en curso (el script deja a todos sin masa): patear cuando llega a la pelota
          const piece = me.disc.invMass > 1e5;
          if (piece && edge < 4 && lastKey !== KICK) { setKey(KICK); return; }
          // llega a la pelota y se queda quieto con ella pegada: sólo la vuelve a buscar si se le aleja
          if (edge > (holding ? 6 : 1.5)) { holding = false; setKey(toward(dx, dy, 1)); }
          else { holding = true; setKey(0); }
          return;
        }
        if (ROLE === "presser") {
          const rival = room.state.players.find((p) => p.team && p.team.id !== me.team.id && p.disc && p.name.includes("holder"));
          if (!rival) return;
          const [press, pause] = PRESS_PLAN[phase % PRESS_PLAN.length];
          phaseT++;
          if (pressing && phaseT >= press) { pressing = false; phaseT = 0; }
          else if (!pressing && phaseT >= pause) { pressing = true; phaseT = 0; }
          // colocación de la pelota por el script (salto de > 30 px): la falta se cobró → próxima fase
          const jump = prevBall && Math.hypot(ball.pos.x - prevBall[0], ball.pos.y - prevBall[1]) > 30;
          prevBall = [ball.pos.x, ball.pos.y];
          if (jump) { phase++; phaseT = 0; pressing = true; log({ placement: true, phase, frame: room.currentFrameNo }); }
          const dx = rival.disc.pos.x - mx, dy = rival.disc.pos.y - my;
          const dist = Math.hypot(dx, dy);
          // en un saque (sin masa) no presiona: lo frena la barrera igual
          if (me.disc.invMass > 1e5) { setKey(0); return; }
          // pegado sin empujar: se acerca hasta tocarlo (centros a <= 30,5) y se queda quieto; en las pausas da un paso atrás
          if (pressing) setKey(dist > 30.5 ? toward(dx, dy, 0) : 0);
          else setKey(toward(-dx, -dy, 0));
          if (ticks % 15 === 0) log({ press: pressing, phase, d: +dist.toFixed(2), frame: room.currentFrameNo });
        }
      };
    },
    onClose: (msg) => { console.log(`${NAME}: salí (${msg ? msg.toString() : ""})`); finish("desconectado"); },
  });
})().catch((e) => { console.error("error:", e && e.message ? e.message : e); process.exit(1); });
