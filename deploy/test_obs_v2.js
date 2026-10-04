// Paridad de la observación v2 de RS4-Z: Node (deploy/rs4z/obs_v2.js) contra Python (env/rs4z/obs_v2.py).
//   python -m export.rs4z_fixture && node deploy/test_obs_v2.js
"use strict";
const fs = require("fs");
const path = require("path");
const { buildObs, restartKind, OBS_DIM } = require("./rs4z/obs_v2");

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "rs4z", "fixture_obs_v2.json"), "utf8"));
if (fixture.obs_dim !== OBS_DIM) throw new Error(`dimensión distinta: Python ${fixture.obs_dim}, Node ${OBS_DIM}`);
let worst = 0, worstAt = null;
for (const [i, f] of fixture.frames.entries()) {
  const o = buildObs(f.state, f.slot);
  for (let j = 0; j < OBS_DIM; j++) {
    const d = Math.abs(o[j] - f.obs[j]);
    if (d > worst) { worst = d; worstAt = [i, j]; }
  }
}
if (!(restartKind(400, 688) === 1 && restartKind(-1140, 660) === 2 && restartKind(1030, -180) === 3 && restartKind(0, 0) === 0)) {
  throw new Error("decodificador de saques incorrecto");
}
console.log(`obs v2: ${fixture.frames.length} casos, diferencia máxima ${worst.toExponential(2)} en ${worstAt}`);
if (worst > 1e-5) { console.error("FALLA: paridad de observación"); process.exit(1); }
console.log("OK");
