// Partido entero de bots en una sala real: 22 clientes, uno por rol, repartidos en varios procesos de node, con UN
// cerebro (learn/hx11_cerebro.py en modo de varios bots, en el proceso principal) que decide por los 22 en un lote.
//
//   node deploy/hx11/partido.js --join <id o link> [--rojo 4-3-3] [--azul 4-4-2] [--temp 0.7] [--cada-ticks 3]
//        [--procesos 6] [--model runs/hx11_l0/l0_sinhist_ancho/model_final.pt] [--cada-ms 8000] [--reintento-ms 12000]
//        [--password x]
//
// Lo que se aprendió en la sala del usuario (2026-10-09; docs/TRASPASO_HAXELEVEN_D0_D3.md):
// * Cada cliente de node-haxball simula el partido entero y lo vuelve a simular con cada tecla atrasada que llega;
//   22 en un solo hilo lo saturaban (1,2 núcleos, teclas congeladas hasta 14 s, bots quietos). Van repartidos en
//   --procesos procesos; uno de ellos (el observador) arma el tick y lo manda al principal.
// * Un cerebro por proceso costaba 4,3 núcleos y dejaba la PC al 100% con el host en la misma máquina (ping 1500): el
//   cerebro es uno solo, en el principal (≈ 0,7 núcleos para los 22).
// * Cada bot entra con su propia clave de autenticación (el script de la sala echa a los que repiten datos), de a uno:
//   el servidor maestro de HaxBall limita los ingresos por IP ("Error de conexión Master").
// * Cada bot juega un rol de su formación medido en el banco (reports/hx11/pos_vocab.json → roles). Nombre del bot:
//   equipo + número + rol (R2 DEF-I, A10 DEL-CI…).
// * Equipos: si el admin de la sala le da admin a cualquiera de los bots, ese bot reparte a los 22 en su equipo.
//   Si no, los mueve el admin a mano según el nombre (R = rojo, A = azul).
// * Los arqueros escriben !gk al entrar a su equipo y lo repiten cada 8 s hasta ver el anuncio de la sala con su nombre
//   ("[✅] <nombre> es el Arquero del Red/Blue"): la sala no arranca sin arquero declarado.
// * La latencia de cada bot se mide en vivo con el eco de sus teclas. Cada 10 s se informa: ticks por segundo del
//   observador (tiene que dar 60), decisiones por segundo (60 / --cada-ticks), retraso del bucle de cada proceso.
// * Consola: "pausa", "sigue", "gk" (los arqueros vuelven a escribir !gk), "salir".
// Úsalo solo en salas propias o con permiso del dueño.
"use strict";

const readline = require("readline");
const { fork } = require("child_process");

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

const ROLES = {
  "4-3-3": ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MED-I", "MED-C", "MED-D", "DEL-I", "DEL-C", "DEL-D"],
  "4-4-2": ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MED-I", "MED-CI", "MED-CD", "MED-D", "DEL-CI", "DEL-CD"],
  "4-2-3-1": ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MCD-CI", "MCD-CD", "MED-I", "MED-C", "MED-D", "DEL-C"],
  "3-4-3": ["ARQ", "DEF-I", "DEF-C", "DEF-D", "MED-I", "MED-CI", "MED-CD", "MED-D", "DEL-I", "DEL-C", "DEL-D"],
  "3-5-2": ["ARQ", "DEF-I", "DEF-C", "DEF-D", "MED-I", "MED-CI", "MED-C", "MED-CD", "MED-D", "DEL-CI", "DEL-CD"],
  "4-1-4-1": ["ARQ", "DEF-I", "DEF-CI", "DEF-CD", "DEF-D", "MCD-C", "MED-I", "MED-CI", "MED-CD", "MED-D", "DEL-C"],
};
const D_MIN = 8, D_MAX = 36;

if (args.includes("--worker")) worker();
else parent();

// ---------------------------------------------------------------------------------------------------------------
function parent() {
  const { startCerebro } = require("./cerebro_proc");
  const joinArg = arg("--join", null);
  if (!joinArg) { console.error("uso: node deploy/hx11/partido.js --join <id o link> [--rojo 4-3-3] [--azul 4-4-2]"); process.exit(1); }
  const joinId = joinArg.includes("?c=") ? joinArg.split("?c=")[1].split("&")[0] : joinArg;
  const red = arg("--rojo", "4-3-3"), blue = arg("--azul", "4-4-2");
  for (const f of [red, blue]) if (!ROLES[f]) { console.error(`formación sin roles: ${f}`); process.exit(1); }
  const lineup = [
    ...ROLES[red].map((r, i) => ({ team: 1, name: `R${i + 1} ${r}`, key: `${red}:${r}`, gk: r === "ARQ" })),
    ...ROLES[blue].map((r, i) => ({ team: 2, name: `A${i + 1} ${r}`, key: `${blue}:${r}`, gk: r === "ARQ" })),
  ].map((l, idx) => ({ ...l, idx }));
  const nProc = Math.max(1, Math.min(22, Number(arg("--procesos", "6"))));
  const ids = new Map();          // idx -> id de jugador
  const live = new Map();         // id de jugador -> {inTeam, echo, k}: lo informa cada proceso
  const resolvers = {};
  const stats = {};
  let adminWorker = null, joinsDone = false, teamsDone = false, obsK = null, ready = false, paused = false;
  let nFrames = 0, nDec = 0;

  const cer = startCerebro({
    model: arg("--model", "runs/hx11_l0/l0_sinhist_ancho/model_final.pt"), pos: null, temp: parseFloat(arg("--temp", "0.7")),
    every: Number(arg("--cada-ticks", "3")), ens: Number(arg("--ens", "0")), ensM: parseFloat(arg("--ens-m", "0")),
    renovarPatada: parseFloat(arg("--renovar-patada", "0.33")),
    onReady: () => { ready = true; console.log("cerebro listo (varios bots)"); },
  });
  // el cerebro en modo varios bots responde {"t", "inputs": {id: bits}}: a cada proceso, las teclas de sus bots
  readline.createInterface({ input: cer.proc.stdout }).on("line", (line) => {
    let m; try { m = JSON.parse(line); } catch { return; }
    if (!m.inputs || paused) return;
    nDec++;
    const per = {};
    for (const [id, bits] of Object.entries(m.inputs)) {
      const s = live.get(Number(id));
      if (s) (per[s.k] = per[s.k] || {})[id] = bits;
    }
    for (const [k, keys] of Object.entries(per)) workers[k].connected && workers[k].send({ keys });
  });

  const workers = [];
  // el observador: el primer proceso con un bot en la sala; si lo pierde, otro
  const pickObserver = () => {
    if (obsK != null && stats[obsK]?.bots > 0) return;
    const k = Object.keys(stats).map(Number).find((kk) => stats[kk].bots > 0);
    if (k == null || k === obsK) return;
    if (obsK != null && workers[obsK].connected) workers[obsK].send({ observe: false });
    obsK = k;
    workers[k].send({ observe: true });
  };
  for (let k = 0; k < nProc; k++) {
    // repartidos en ronda: cada proceso mezcla rojos y azules de todas las líneas
    const mine = lineup.filter((l) => l.idx % nProc === k);
    const w = fork(__filename, ["--worker"], { stdio: ["ignore", "inherit", "inherit", "ipc"] });
    w.on("message", (m) => {
      if (m.frame) {
        if (!ready || k !== obsK) return;
        nFrames++;
        m.frame.bots = [...live.entries()].filter(([, s]) => s.inTeam).map(([id, s]) => ({ me: id, delay: s.echo }));
        cer.send(m.frame);
      } else if (m.state) {
        for (const s of m.state) live.set(s.id, { inTeam: s.inTeam, echo: s.echo, k });
      } else if (m.joined) {
        const r = resolvers[m.joined.idx]; delete resolvers[m.joined.idx];
        if (m.joined.ok) {
          cer.send({ register: { id: m.joined.id, pos: lineup[m.joined.idx].key } });
          live.set(m.joined.id, { inTeam: false, echo: 22, k });
          stats[k] = { ...(stats[k] || {}), bots: (stats[k]?.bots || 0) + 1 };
          pickObserver();
        }
        r && r(m.joined);
      } else if (m.left != null) {
        const id = ids.get(m.left);
        ids.delete(m.left);
        live.delete(id);
        if (stats[k]) stats[k].bots = Math.max(0, (stats[k].bots || 1) - 1);
        pickObserver();
      } else if (m.admin && !adminWorker) adminWorker = w;
      else if (m.teamsDone != null) {
        teamsDone = m.teamsDone;
        if (teamsDone) console.log(`equipos repartidos: rojo ${red}, azul ${blue}`);
      } else if (m.stats) stats[k] = m.stats;
    });
    w.on("exit", (c) => console.log(`proceso ${k} terminó (${c})`));
    w.send({ init: { k, bots: mine, joinId, password: arg("--password", undefined) } });
    workers.push(w);
  }
  const owner = (l) => workers[l.idx % nProc];

  setInterval(() => {
    if (teamsDone || !joinsDone || !adminWorker) return;
    adminWorker.send({ teams: lineup.filter((l) => ids.has(l.idx)).map((l) => ({ id: ids.get(l.idx), team: l.team })) });
  }, 2000);
  setInterval(() => {
    const ks = Object.keys(stats);
    if (!ks.length) return;
    const ecos = [...live.values()].filter((s) => s.inTeam).map((s) => s.echo).sort((a, b) => a - b);
    console.log(`observador p${obsK}: ${(nFrames / 10).toFixed(1)} ticks/s, ${(nDec / 10).toFixed(1)} dec/s, ` +
                `eco p50 ${ecos.length ? ecos[ecos.length >> 1] : "-"} | ` +
                ks.map((k) => `p${k}: ${stats[k].bots} bots, bucle p95 ${stats[k].lag_p95_ms ?? "-"} ms`).join(" | "));
    nFrames = 0; nDec = 0;
  }, 10000);

  readline.createInterface({ input: process.stdin }).on("line", (line) => {
    const cmd = line.trim();
    if (!["pausa", "sigue", "gk", "salir"].includes(cmd)) return;
    if (cmd === "pausa") paused = true;
    if (cmd === "sigue") paused = false;
    workers.forEach((w) => w.connected && w.send({ cmd }));
    if (cmd === "salir") { cer.stop(); setTimeout(() => process.exit(0), 1500); }
  });

  (async () => {
    const gap = Number(arg("--cada-ms", "8000"));
    // el maestro limita los ingresos por IP: después de uno exitoso, el siguiente necesita ~10 s
    const retryMs = Number(arg("--reintento-ms", "12000"));
    for (const l of lineup) {
      for (let k = 0; k < 6 && !ids.has(l.idx); k++) {
        const r = await new Promise((res) => { resolvers[l.idx] = res; owner(l).send({ join: l.idx }); });
        if (r.ok) ids.set(l.idx, r.id);
        else await sleep(retryMs);
      }
      await sleep(gap);
    }
    joinsDone = true;
    const faltan = lineup.filter((l) => !ids.has(l.idx)).map((l) => l.name);
    if (faltan.length) console.log(`no pudieron entrar: ${faltan.join(", ")}`);
    console.log("todos los bots intentaron entrar; dale admin a uno para que reparta los equipos, o movelos a mano");
  })();
}

// ---------------------------------------------------------------------------------------------------------------
function worker() {
  let API;
  try { API = require("../../bridge/haxball")(); } catch (e) { API = require("node-haxball")(); }
  const { Room, Utils, Plugin, AllowFlags } = API;
  const { FrameBuilder } = require("./frame");
  let bots = [], paused = false, observing = false, observer = null, fb = new FrameBuilder(), cfg = null;
  const lags = [];

  const echo = (b) => {
    if (!b.delays.length) return 22;   // S6: p50 21–25 ticks en la sala del usuario
    const d = [...b.delays].sort((x, y) => x - y);
    return Math.min(D_MAX, Math.max(D_MIN, d[d.length >> 1]));
  };
  const pickObserver = () => {
    if (!observing) { observer = null; return; }
    if (observer && observer.room) return;
    observer = bots.find((b) => b.room) || null;
    fb = new FrameBuilder();
  };

  function makePlugin(b) {
    function P() {
      Object.setPrototypeOf(this, Plugin.prototype);
      Plugin.call(this, `hx11_${b.name}`, true, { version: "1.0", author: "HaxballRL", description: "bot HaxEleven",
                                                  allowFlags: AllowFlags.JoinRoom });
    }
    const p = new P();
    const isObs = () => b === observer;
    p.onPlayerInputChange = (id, value) => {
      if (!b.pending || !b.room || id !== b.room.currentPlayerId || value !== b.pending.state) return;
      const d = b.room.currentFrameNo - b.pending.frame;
      b.pending = null;
      if (d >= 0 && d < 120) { b.delays.push(d); if (b.delays.length > 40) b.delays.shift(); }
    };
    p.onPlayerTeamChange = (id, teamId) => {
      if (b.room && id === b.room.currentPlayerId && b.gk && (teamId === 1 || teamId === 2) && !b.gkSent) {
        b.gkSent = true;
        b.gkAt = Date.now() + 800;
        setTimeout(() => b.room && b.room.sendChat("!gk"), 800);
      }
    };
    p.onGameTick = () => {
      if (!isObs() || !b.room) return;
      const fr = fb.build(b.room, b.room.currentFrameNo, b.id, 22);
      if (fr) process.send({ frame: fr });
    };
    p.onSetDiscProperties = (id, type, d1) => isObs() && fb.onSetDiscProperties(id, type, d1);
    p.onPlayerBallKick = (id) => isObs() && fb.onPlayerBallKick(id);
    p.onTeamGoal = (t) => isObs() && fb.onTeamGoal(t);
    p.onPlayerJoin = (pl) => isObs() && fb.onPlayerJoin(pl);
    p.onAnnouncement = (m) => {
      if (b.gk && typeof m === "string" && /es el Arquero/.test(m) && m.includes(b.name)) {
        if (!b.gkOk) console.log(`${b.name}: arquero confirmado por la sala`);
        b.gkOk = true;
      }
      if (isObs()) fb.onAnnouncement(m);
    };
    return p;
  }

  const joinOne = (b) => new Promise(async (resolve) => {
    let done = false;
    const end = (ok) => { if (!done) { done = true; resolve(ok); } };
    const [, authObj] = await Utils.generateAuth();
    Room.join({ id: cfg.joinId, authObj, password: cfg.password }, {
      storage: { player_name: b.name, avatar: b.gk ? "🧤" : String((b.idx % 11) + 1),
                 geo: { lat: -34.6, lon: -58.4, flag: "ar" } },
      plugins: [makePlugin(b)],
      onOpen: (r) => {
        b.room = r;
        b.id = r.currentPlayerId;
        pickObserver();
        console.log(`${b.name} conectado (id ${b.id})`);
        end(true);
      },
      onClose: (e) => {
        if (b.room) { console.log(`${b.name} salió`, e ? String(e) : ""); process.send({ left: b.idx }); }
        b.room = null;
        if (observer === b) { observer = null; pickObserver(); }
        end(false);
      },
    });
    setTimeout(() => end(false), 20000);
  });

  process.on("message", async (m) => {
    if (m.init) {
      cfg = m.init;
      bots = cfg.bots.map((l) => ({ ...l, id: null, room: null, delays: [], pending: null, gkSent: false }));
      // retraso del bucle de eventos (si crece, el proceso no da abasto)
      let expect = Date.now() + 100;
      setInterval(() => { const now = Date.now(); lags.push(now - expect); expect = now + 100; if (lags.length > 200) lags.shift(); }, 100);
      setInterval(() => {
        const l = [...lags].sort((x, y) => x - y);
        process.send({ stats: { bots: bots.filter((b) => b.room).length,
                                lag_p95_ms: l.length ? l[Math.floor(0.95 * (l.length - 1))] : null } });
      }, 10000);
      // equipo y eco de mis bots, para el cerebro del principal
      setInterval(() => {
        process.send({ state: bots.filter((b) => b.room && b.id != null).map((b) => ({
          id: b.id, inTeam: (b.room.currentPlayer?.team?.id ?? 0) > 0, echo: echo(b) })) });
      }, 1000);
      // arqueros: !gk mientras estén en un equipo y la sala no los haya anunciado (cada 8 s como mucho)
      setInterval(() => {
        for (const b of bots) {
          if (!b.gk || !b.room) continue;
          const t = b.room.currentPlayer?.team?.id;
          if (t !== 1 && t !== 2) { b.gkOk = false; continue; }
          if (b.gkOk || Date.now() - (b.gkAt || 0) < 8000) continue;
          b.gkAt = Date.now();
          b.room.sendChat("!gk");
          console.log(`${b.name}: !gk`);
        }
      }, 1000);
      // admin: si el admin de la sala se lo da a cualquiera de mis bots, aviso
      setInterval(() => { if (bots.some((b) => b.room?.currentPlayer?.isAdmin)) process.send({ admin: true }); }, 2000);
      if (API.ready) await API.ready;
    } else if (m.join != null) {
      if (API.ready) await API.ready;
      const b = bots.find((x) => x.idx === m.join);
      const ok = await joinOne(b);
      process.send({ joined: { idx: b.idx, ok, id: b.id } });
    } else if (m.observe != null) {
      observing = m.observe;
      observer = null;
      pickObserver();
    } else if (m.keys) {
      if (paused) return;
      for (const [id, bits] of Object.entries(m.keys)) {
        const b = bots.find((x) => x.id === Number(id));
        if (!b || !b.room) continue;
        if (!b.pending && bits !== b.room.currentPlayer?.input) b.pending = { state: bits, frame: b.room.currentFrameNo };
        b.room.setKeyState(bits);
      }
    } else if (m.teams) {
      const adm = bots.find((b) => b.room?.currentPlayer?.isAdmin);
      if (adm) for (const t of m.teams) adm.room.setPlayerTeam(t.id, t.team);
      process.send({ teamsDone: !!adm });
    } else if (m.cmd === "gk") {
      bots.forEach((b) => { if (b.gk) { b.gkOk = false; b.gkAt = 0; } });
    } else if (m.cmd === "pausa") { paused = true; bots.forEach((b) => b.room && b.room.setKeyState(0)); console.log("en pausa"); }
    else if (m.cmd === "sigue") { paused = false; }
    else if (m.cmd === "salir") {
      bots.forEach((b) => { try { b.room && b.room.leave(); } catch {} });
      setTimeout(() => process.exit(0), 500);
    }
  });
  process.on("disconnect", () => { bots.forEach((b) => { try { b.room && b.room.leave(); } catch {} }); process.exit(0); });
}
