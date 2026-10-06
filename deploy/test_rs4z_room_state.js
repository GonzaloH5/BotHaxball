// Paridad del bot de sala RS4-Z: observación armada con lo que ve la sala (deploy/rs4z/room_state.js)
// contra la del simulador en los mismos cuadros de una grabación real.
//   python -m export.rs4z_room_fixture && node deploy/test_rs4z_room_state.js
"use strict";
const fs = require("fs");
const path = require("path");
const { RS4ZTracker, inputToAction } = require("./rs4z/room_state");
const { SELF_DIM } = require("./rs4z/obs_v2");

const fixture = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "rs4z", "fixture_room_state.json"), "utf8"));
const NAMES = ["x", "y", "vx", "vy", "kick_armed", "kicking", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_dx",
  "ball_dy", "ball_dist", "ball_rvx", "ball_rvy", "line_opp", "line_own", "line_top", "line_bottom", "opp_goal_dx",
  "opp_goal_dy", "own_goal_dx", "own_goal_dy", "ball_opp_goal_dx", "ball_opp_goal_dy", "ball_own_goal_dx",
  "ball_own_goal_dy"];
const name = (j) => j < 27 ? NAMES[j] : j < 54 ? `hist_move${j - 27}` : j < 57 ? `hist_kick${j - 54}` :
  j < SELF_DIM ? ["delay", "n_mates", "n_rivals", "kick_strength", "ball_radius", "mass_phase", "restart_active",
    "restart_own", "restart_rival", "restart_lateral", "restart_corner", "restart_goal_kick", "restart_age",
    "kickoff", "kickoff_own", "kickoff_age"][j - 57] : `ent${Math.floor((j - SELF_DIM) / 9)}_${(j - SELF_DIM) % 9}`;

const want = new Map();
for (const r of fixture.rows) want.set(`${r.frame}:${r.slot}`, r.obs);
// un rastreador por jugador (cada bot lleva su propio historial)
const trackers = new Map();
let compared = 0, worst = 0;
const bad = new Map();
for (const frame of fixture.frames) {
  const order = [0, 1].flatMap((t) => frame.players.filter((p) => p.team === t).sort((a, b) => a.id - b.id));
  for (const p of order) {
    if (!trackers.has(p.id)) trackers.set(p.id, new RS4ZTracker());
    trackers.get(p.id).update(frame);
  }
  if (frame.frame % 3 !== 0) continue;
  for (const p of order) {
    const tr = trackers.get(p.id);
    const built = tr.build(frame, p.id, 0);
    const ref = want.get(`${frame.frame}:${built.slot}`);
    if (ref) {
      compared++;
      for (let j = 0; j < ref.length; j++) {
        const d = Math.abs(ref[j] - built.obs[j]);
        if (d > 1e-3) bad.set(j, (bad.get(j) || 0) + 1);
        if (j < 27 || j >= 57 + 16) worst = Math.max(worst, d);
      }
    }
    tr.pushDecision(inputToAction(p.nowInput));
  }
}
console.log(`comparadas ${compared} observaciones; peor diferencia (cinemática y entidades) ${worst.toExponential(2)}`);
const rows = [...bad.entries()].sort((a, b) => b[1] - a[1]);
for (const [j, n] of rows.slice(0, 15)) console.log(`  ${name(j).padEnd(18)} distinta en ${(100 * n / compared).toFixed(2)}%`);
const tolerated = rows.filter(([j, n]) => n / compared > 0.005);
if (!compared || tolerated.length) { console.error("FALLA: features con más de 0,5% de diferencias"); process.exit(1); }
console.log("OK");
