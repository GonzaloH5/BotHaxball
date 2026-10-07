// Motor X4 del servicio de la extensión (deploy/hybrid/engine_x4.js).
//   node deploy/test_hybrid_x4.js
"use strict";
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");
const { X4Engine, validateX4Meta } = require("./hybrid/engine_x4");
const { RS4ZTracker, inputToAction, decodeOwnAction } = require("./rs4z/room_state");
const { OBS_DIM } = require("./rs4z/obs_v3");

const META = { obs_version: "x4-obs-v3", obs_dim: OBS_DIM, n_actions: 18, frame_skip: 3, default_delay: 10, temperature: 1,
  maps: ["sanguchito_rs_x4"] };
const SANGU = { name: "SANGUCHITO RS X4", playerPhysics: { kickStrength: 5.85 } };
const fakeOrt = { Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } } };
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "rs4z", "fixture_x4_room_state.json"), "utf8"));

// cuadro de la sala (formato de deploy/rs4z/room_state.js) → cuadro de la extensión (official_profile.snapshot)
function hybridFrame(fr, playerId, extra = {}) {
  const disc = (x, y, vx, vy, r, color, invMass) => ({ pos: [x, y], vel: [vx, vy], gravity: [0, 0], radius: r, color, invMass });
  // cMask de los jugadores como en la sala: en el saque inicial, la barrera del equipo que saca (redKO 8, blueKO 16)
  const cMask = 39 | (fr.state === 0 ? (fr.kickoffTeam === 0 ? 8 : 16) : 0);
  const players = fr.players.map((p) => ({ id: p.id, team: p.team, ...disc(p.x, p.y, p.vx, p.vy, 15, 0, p.invMass),
    cMask, input: p.input, isKicking: p.isKicking }));
  const b = fr.ball;
  return { frame: fr.frame, playerId, players, discs: [disc(b.x, b.y, b.vx, b.vy, b.r, b.color, 1), ...players], joints: [],
    score: [0, 0], phase: fr.state === 0 ? 0 : 1, paused: false, elapsed: fr.frame / 60, synchronized: true,
    capturedAt: 0, session: "s", epoch: 1, ...extra };
}

const tests = [];
const test = (name, fn) => tests.push({ name, fn });

test("valida el modelo y el mapa", () => {
  assert.throws(() => validateX4Meta({ training_program: "rs4_v3" }));
  const e = new X4Engine(META, null, fakeOrt);
  assert.throws(() => e.map({ name: "Real Soccer ONE" }), /sanguchito_rs_x4/);
  e.map(SANGU);
  assert.equal(e.mapName, "sanguchito_rs_x4");
});

test("la observación coincide con la de bot.js en cuadros reales de Sanguchito (cada 3 ticks)", () => {
  // referencia: el rastreador de bot.js, actualizado en cada tick (paridad con el dataset en test_x4_room_state.js)
  const ids = [0, 1].flatMap((t) => fixture.frames[0].players.filter((p) => p.team === t).sort((a, b) => a.id - b.id))
    .map((p) => p.id).slice(0, 8);
  let compared = 0, worst = 0;
  for (const id of [ids[0], ids[5]]) {
    const ref = new RS4ZTracker({ ...META, map: "sanguchito_rs_x4" });
    const eng = new X4Engine(META, null, fakeOrt);
    eng.map(SANGU);
    eng.control({ mode: "BOT", epoch: 1, session: "s" });
    for (const fr of fixture.frames) {
      ref.update(fr);
      if (fr.frame % 3 !== 0 || fr.players.length !== 8) continue;
      const me = fr.players.find((p) => p.id === id);
      if (!me) continue;
      const want = ref.build(fr, id, 10);
      const got = eng.observe(hybridFrame(fr, id));
      for (let j = 0; j < OBS_DIM; j++) worst = Math.max(worst, Math.abs(want.obs[j] - got.obs[j]));
      compared++;
      const a = inputToAction(me.nextInput);
      ref.pushDecision(a);
      eng.tracker.pushDecision(a);             // lo que hace ack() con la acción aceptada
    }
  }
  assert(compared > 100, `pocas observaciones: ${compared}`);
  // la edad del saque y del saque inicial se cuenta con el número de frame: igual que tick a tick
  assert(worst < 1e-6, `diferencia máxima ${worst}`);
});

test("cuadro → acción con la máscara de teclas → confirmación agrega la decisión a la historia", async () => {
  const fr = fixture.frames.find((f) => f.frame % 3 === 0 && f.players.length === 8 && f.state !== 0);
  const blue = fr.players.find((p) => p.team === 1);
  const session = { run: async (feeds) => {
    assert.equal(feeds.obs.dims[1], OBS_DIM);
    return { logits: { data: Float32Array.from({ length: 18 }, (_, i) => (i === 12 ? 50 : 0)) } };
  } };
  const e = new X4Engine(META, session, fakeOrt);
  e.map(SANGU);
  e.control({ mode: "BOT", epoch: 1, session: "s" });
  const sent = [];
  await e.frame(hybridFrame(fr, blue.id), (m) => sent.push(m));
  assert.equal(sent.length, 1);
  const key = decodeOwnAction(12, 1);               // 12 = patear + dirección 3 en el marco propio; azul espejado
  const mask = (key.dirY < 0 ? 1 : key.dirY > 0 ? 2 : 0) | (key.dirX < 0 ? 4 : key.dirX > 0 ? 8 : 0) | (key.kick ? 16 : 0);
  assert.equal(sent[0].action, 12);
  assert.equal(sent[0].mask, mask);
  assert.equal(inputToAction(mask), key.world);     // la máscara es la acción del mundo
  await e.ack({ session: "s", epoch: 1, frame: fr.frame, accepted: true, mask });
  assert.equal(e.tracker.hist[0], key.world);
  assert.equal(e.pending, null);
});

test("jugando el humano, sus teclas entran en la historia", () => {
  const fr = fixture.frames.find((f) => f.frame % 3 === 0 && f.players.length === 8 && f.players.some((p) => p.input));
  const me = fr.players.find((p) => p.input);
  const e = new X4Engine(META, null, fakeOrt);
  e.map(SANGU);
  e.control({ mode: "HUMANO", epoch: 1, session: "s" });
  e.observe(hybridFrame(fr, me.id));
  assert.equal(e.tracker.hist[0], inputToAction(me.input));
});

test("el servicio carga un modelo X4 y anuncia su versión y mapas", async () => {
  const model = ["x4_rl_c2000.onnx", "x4_rl_2500.onnx", "x4_rl.onnx"].map((f) => path.join(__dirname, "rs4z", f))
    .find((f) => fs.existsSync(f));
  if (!model) { console.log("  (sin modelo X4 local: se saltea)"); return; }
  const { loadModel } = require("./hybrid_server");
  const m = await loadModel(model);
  assert.equal(m.meta.obs_version, "x4-obs-v3");
  assert.deepEqual(m.meta.maps, ["sanguchito_rs_x4"]);
});

(async () => {
  let failed = 0;
  for (const t of tests) {
    try { await t.fn(); console.log(`ok   ${t.name}`); } catch (e) { failed++; console.error(`FALLA ${t.name}\n${e.stack}`); }
  }
  if (failed) process.exit(1);
  console.log("OK");
})();
