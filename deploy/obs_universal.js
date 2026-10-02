// Observación "universal" (cualquier mapa y formato), EXACTAMENTE igual que
// env/haxball_env.py::_observe_universal(). Se verifica con test_obs.js contra fixture.json.
//
// geom: salida de `python -m export.stadium_geom <mapa>` (medidas + obstáculos por tipo de disco)
// state = {
//   ball: {pos:[x,y], vel:[x,y]},
//   players: [{team: 0|1, pos, vel, canKick, touching}],   // todos los jugadores en cancha
//   kickoff, kickoffTeam, tfrac,
//   ps: null | {comba, prog, invb, grav:[x,y], holder}      // powershot (índice del jugador que carga o -1)
//   rules: null | [[...N_RULE_FEATS] por jugador]            // script Pegeche (env/pegeche.py::features)
//   players[i].expelled: true -> no se ve (en la sala real pasa a espectadores)
// }
// opts = {maxEntities, psOn, outOfBounds}

const POS_SCALE = 400, VEL_SCALE = 5;
const { publicFeatures } = require("./public_signals");
const N_RAYS = 8, RAY_MAX = 400;
const RAY_DIRS = Array.from({ length: N_RAYS }, (_, k) => [Math.cos(2 * Math.PI * k / N_RAYS), Math.sin(2 * Math.PI * k / N_RAYS)]);
const RAY_MIRROR = Array.from({ length: N_RAYS }, (_, k) => (((N_RAYS / 2 - k) % N_RAYS) + N_RAYS) % N_RAYS);
const U_ENT_DIM = 8;
const N_RULE_FEATS = 15;

// igual que env/geometry.py::_cast (float64 en JS)
function castRays(ox, oy, obs, radius) {
  const out = new Array(N_RAYS);
  const { seg_a, seg_b, circ_c, circ_r, pl_n, pl_d } = obs;
  for (let k = 0; k < N_RAYS; k++) {
    const dx = RAY_DIRS[k][0], dy = RAY_DIRS[k][1];
    let best = RAY_MAX + radius;
    for (let s = 0; s < seg_a.length; s++) {
      const ax = seg_a[s][0], ay = seg_a[s][1];
      const ex = seg_b[s][0] - ax, ey = seg_b[s][1] - ay;
      const den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-12) continue;
      const wx = ax - ox, wy = ay - oy;
      const t = (wx * ey - wy * ex) / den;
      const u = (wx * dy - wy * dx) / den;
      if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
    }
    for (let c = 0; c < circ_c.length; c++) {
      const fx = ox - circ_c[c][0], fy = oy - circ_c[c][1];
      const rr = circ_r[c] + radius;
      const b = fx * dx + fy * dy;
      const cc = fx * fx + fy * fy - rr * rr;
      const disc = b * b - cc;
      if (disc < 0) continue;
      const t = -b - Math.sqrt(disc);
      if (t > 0 && t + radius < best) best = t + radius;
    }
    for (let p = 0; p < pl_n.length; p++) {
      const nx = pl_n[p][0], ny = pl_n[p][1];
      const vn = dx * nx + dy * ny;
      if (vn >= 0) continue;
      const t = (pl_d[p] - (ox * nx + oy * ny)) / vn;
      if (t > 0 && t < best) best = t;
    }
    out[k] = Math.min(Math.max(best - radius, 0), RAY_MAX);
  }
  return out;
}

function mapFeatures(geom, T, opts) {
  return [geom.field_half_w / 1000, geom.field_half_h / 1000, geom.goal_half_height / 100,
          geom.goal_x / geom.field_half_w, geom.ball_radius / 10, geom.player_radius / 15,
          geom.kickoff_radius / 100, opts.psOn ? 1 : 0, opts.outOfBounds ? 1 : 0, T / 11];
}

function buildObsUniversal(state, p, geom, opts) {
  const me = state.players[p];
  const sgn = me.team === 0 ? 1 : -1;
  const own = (v) => [v[0] * sgn, v[1]];
  const W = geom.field_half_w, H = geom.field_half_h, gx = geom.goal_x;
  const pp = own(me.pos), pv = own(me.vel);
  const bp = own(state.ball.pos), bv = own(state.ball.vel);
  const rel = [bp[0] - pp[0], bp[1] - pp[1]];
  const nTeam = state.players.filter((q) => q.team === me.team).length;
  const T = Math.max(nTeam, state.players.length - nTeam);
  const o = [];
  const push = (...xs) => { for (const x of xs) o.push(x); };
  push(pp[0] / W, pp[1] / H, pv[0] / VEL_SCALE, pv[1] / VEL_SCALE);
  push(bp[0] / W, bp[1] / H, bv[0] / VEL_SCALE, bv[1] / VEL_SCALE);
  push(rel[0] / POS_SCALE, rel[1] / POS_SCALE, Math.hypot(rel[0], rel[1]) / POS_SCALE);
  push((gx - pp[0]) / W, (0 - pp[1]) / W, (-gx - pp[0]) / W, (0 - pp[1]) / W, (gx - bp[0]) / W, (0 - bp[1]) / W);
  push(me.canKick ? 1 : 0, state.kickoff ? 1 : 0, state.kickoff && state.kickoffTeam === me.team ? 1 : 0,
       me.touching ? 1 : 0, state.tfrac);
  if (opts.psOn && state.ps) {
    const ps = state.ps, g = own(ps.grav);
    push(ps.comba, ps.prog, ps.invb, g[0], g[1], ps.holder === p ? 1 : 0);
  } else {
    push(0, 0, 0, 0, 0, 0);
  }
  push(...mapFeatures(geom, T, opts));
  const mirror = (r) => (sgn < 0 ? RAY_MIRROR.map((i) => r[i]) : r);
  const rp = mirror(castRays(me.pos[0], me.pos[1], geom.obstacles[me.team === 0 ? "red" : "blue"], geom.player_radius));
  const rb = mirror(castRays(state.ball.pos[0], state.ball.pos[1], geom.obstacles.ball, geom.ball_radius));
  push(...rp.map((d) => d / RAY_MAX), ...rb.map((d) => d / RAY_MAX));
  push((T - 1) / 10, T / 10);
  // estado del script Pegeche: pelota parada (tipo, de quién, bloqueo, tiempo), slide propio (fase,
  // cooldown), X mantenida, falta para pedir, amarilla. TODO bot.js: estimarlo en la sala real.
  if (opts.publicSignalsVersion === 1) push(...publicFeatures(state, p, geom, opts.publicSignalConfig || {}));
  else if (state.rules) push(...state.rules[p]); else for (let j = 0; j < N_RULE_FEATS; j++) o.push(0);

  // entidades: compañeros primero, luego rivales, rellenadas hasta maxEntities
  const others = state.players.map((q, i) => i).filter((i) => i !== p && !state.players[i].expelled);
  others.sort((a, b) => (state.players[a].team !== me.team) - (state.players[b].team !== me.team));
  const E = opts.maxEntities;
  for (let k = 0; k < E; k++) {
    if (k >= others.length) { for (let j = 0; j < U_ENT_DIM; j++) o.push(0); continue; }
    const q = state.players[others[k]];
    const qp = own(q.pos), qv = own(q.vel);
    push(1, q.team !== me.team ? 1 : 0, (qp[0] - pp[0]) / POS_SCALE, (qp[1] - pp[1]) / POS_SCALE,
         qv[0] / VEL_SCALE, qv[1] / VEL_SCALE, (bp[0] - qp[0]) / POS_SCALE, (bp[1] - qp[1]) / POS_SCALE);
  }
  return o;
}

module.exports = { buildObsUniversal, castRays, U_ENT_DIM, N_RULE_FEATS, N_RAYS, RAY_MAX };
