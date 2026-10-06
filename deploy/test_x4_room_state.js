// Paridad del bot de sala con observación v3 (X4): observación armada con lo que ve la sala
// (deploy/rs4z/room_state.js + obs_v3.js) contra la de Python (learn/x4_data.featurize) en los mismos
// cuadros de una grabación real de Sanguchito.
//   python -m export.x4_room_fixture && node deploy/test_x4_room_state.js
"use strict";
const fs = require("fs");
const path = require("path");
const { RS4ZTracker, inputToAction } = require("./rs4z/room_state");
const { OBS_DIM } = require("./rs4z/obs_v3");

const fixture = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "rs4z", "fixture_x4_room_state.json"), "utf8"));
const want = new Map();
for (const r of fixture.rows) want.set(`${r.frame}:${r.slot}`, r.obs);
const trackers = new Map();
let compared = 0, worst = 0, worstAt = null;
for (const frame of fixture.frames) {
  const order = [0, 1].flatMap((t) => frame.players.filter((p) => p.team === t).sort((a, b) => a.id - b.id));
  for (const p of order) {
    if (!trackers.has(p.id)) trackers.set(p.id, new RS4ZTracker({ obs_version: "x4-obs-v3", map: fixture.map }));
    trackers.get(p.id).update(frame);
  }
  if (frame.frame % 3 !== 0) continue;
  for (const p of order) {
    const tr = trackers.get(p.id);
    const built = tr.build(frame, p.id, 0);
    const ref = want.get(`${frame.frame}:${built.slot}`);
    if (ref) {
      if (ref.length !== OBS_DIM) throw new Error(`dimensión distinta: Python ${ref.length}, Node ${OBS_DIM}`);
      compared++;
      for (let j = 0; j < OBS_DIM; j++) {
        const d = Math.abs(ref[j] - built.obs[j]);
        if (d > worst) { worst = d; worstAt = [frame.frame, built.slot, j]; }
      }
    }
    tr.pushDecision(inputToAction(p.nextInput));
  }
}
console.log(`obs v3 de sala: ${compared} observaciones comparadas, diferencia máxima ${worst.toExponential(2)} en ${worstAt}`);
if (compared < 100) { console.error("FALLA: muy pocas observaciones comparadas"); process.exit(1); }
if (worst > 1e-4) { console.error("FALLA: paridad de observación v3"); process.exit(1); }
console.log("OK");
