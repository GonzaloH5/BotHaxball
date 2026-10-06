// El bot de sala real (deploy/bot.js) con un modelo RS4-Z, sin red: una sala falsa con la forma de
// node-haxball recorre cuadros de una grabación real (deploy/rs4z/fixture_room_state.json). Las teclas
// que aprieta el bot deben coincidir con las de una referencia que alimenta el rastreador validado
// (test_rs4z_room_state.js) con los mismos cuadros y el mismo modelo ONNX.
//   node deploy/test_bot_rs4z_runtime.js [deploy/rs4z]
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ort = require("onnxruntime-node");
const { RS4ZTracker, decodeOwnAction } = require("./rs4z/room_state");

const dir = process.argv[2] || path.join(__dirname, "rs4z");
const MODEL = path.join(dir, "model.onnx");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "rs4z", "fixture_room_state.json"), "utf8"));
const keyState = (dx, dy, kick) => (dy < 0 ? 1 : 0) | (dy > 0 ? 2 : 0) | (dx < 0 ? 4 : 0) | (dx > 0 ? 8 : 0) | (kick ? 16 : 0);
const argmax = (a) => a.indexOf(Math.max(...a));

(async () => {
  const frames = fixture.frames.slice(0, 1800);
  const ME = frames[0].players.find((p) => p.team === 1).id;   // un jugador azul (espejo)
  const pressed = [];
  // ---- sala falsa con la forma de node-haxball
  const toPlayer = (p) => ({ id: p.id, team: { id: p.team + 1 }, input: p.input, isKicking: p.isKicking,
    disc: { pos: { x: p.x, y: p.y }, speed: { x: p.vx, y: p.vy }, invMass: p.invMass } });
  const room = { isHost: true, name: "prueba", currentPlayerId: ME, currentFrameNo: 0, stadium: { playerPhysics: { kickStrength: 0 } },
    setKeyState: (v) => pressed.push(v), sendChat() {}, setPlayerTeam() {}, startGame() {}, state: { players: [] } };
  const load = (f) => {
    const ps = f.players.map(toPlayer);
    room.state.players = ps;
    room.currentPlayer = ps.find((p) => p.id === ME);
    room.currentFrameNo = f.frame;
    room.stadium.playerPhysics.kickStrength = f.kickStrength;
    room.gameState = { state: f.state, goalConcedingTeam: { id: f.kickoffTeam + 1 },
      physicsState: { discs: [{ pos: { x: f.ball.x, y: f.ball.y }, speed: { x: f.ball.vx, y: f.ball.vy }, radius: f.ball.r, color: f.ball.color }] } };
  };
  let plugin;
  const api = { Plugin: function () {}, AllowFlags: { CreateRoom: 1, JoinRoom: 2 },
    Utils: { keyState, generateAuth: async () => ["auth", {}], parseStadium: () => ({}) },
    Room: { create: (c, o) => { plugin = o.plugins[0]; plugin.room = room; } } };
  const proc = { argv: ["node", "bot.js", "--model", MODEL, "--token", "x"], env: {}, connected: false, exit() {}, on() {} };
  const context = { require: (id) => (id === "../bridge/haxball" ? () => api : require(id)), process: proc, __dirname,
    console: { log() {}, warn() {}, error: console.error }, setTimeout, clearTimeout, setInterval: () => ({ unref() {} }) };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "bot.js"), "utf8"), context);
  for (let i = 0; i < 50 && !plugin; i++) await new Promise((r) => setTimeout(r, 20));
  assert(plugin, "el bot no creó la sala");
  // ---- referencia: rastreador validado + mismo modelo
  const session = await ort.InferenceSession.create(MODEL);
  const ref = new RS4ZTracker();
  const expected = [];
  let tick = 0;
  for (const f of frames) {
    load(f);
    plugin.onGameTick();
    for (let k = 0; k < 5; k++) await new Promise((r) => setImmediate(r));
    ref.update(f);
    tick++;
    if (tick % 3 === 0) {
      const built = ref.build(f, ME, 0);
      const out = await session.run({ obs: new ort.Tensor("float32", built.obs, [1, built.obs.length]) });
      const key = decodeOwnAction(argmax(Array.from(out.logits.data)), built.team);
      expected.push(keyState(key.dirX, key.dirY, key.kick));
      ref.pushDecision(key.world);
    }
  }
  assert.equal(pressed.length, expected.length, `decisiones del bot ${pressed.length} vs referencia ${expected.length}`);
  const same = pressed.filter((v, i) => v === expected[i]).length;
  console.log(`bot RS4-Z en sala falsa: ${pressed.length} decisiones, ${same} iguales a la referencia`);
  assert.equal(same, expected.length);
  const kicks = pressed.filter((v) => v & 16).length;
  console.log(`PASS (${kicks} con patada; jugador azul ${ME})`);
})().catch((e) => { console.error(e); process.exitCode = 1; });
