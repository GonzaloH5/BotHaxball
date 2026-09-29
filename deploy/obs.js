// Construye la observación EXACTAMENTE igual que env/haxball_env.py::observe().
// Se verifica con test_obs.js contra fixture.json generado por export/to_onnx.py.
//
// state = {
//   ball: {pos:[x,y], vel:[x,y]},
//   players: [{team: 0|1 (0 rojo, 1 azul), pos, vel, canKick, touching}],  // orden estable
//   kickoff: bool, kickoffTeam: 0|1, tfrac: número en [0,1]
// }

// dirección por acción de movimiento (coords HaxBall, y hacia abajo) — igual que sim/physics.py
const MOVE_DIRS = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
const MIRROR_MOVE = [0, 1, 8, 7, 6, 5, 4, 3, 2];

function buildObs(state, p, meta) {
  const S = meta.pos_scale, V = meta.vel_scale, gx = meta.goal_x;
  const me = state.players[p];
  const sgn = me.team === 0 ? 1 : -1;
  const own = (v) => [v[0] * sgn, v[1]];
  const pp = own(me.pos), pv = own(me.vel);
  const bp = own(state.ball.pos), bv = own(state.ball.vel);
  const rel = [bp[0] - pp[0], bp[1] - pp[1]];
  const dist = Math.hypot(rel[0], rel[1]);
  const o = [];
  const push = (...xs) => { for (const x of xs) o.push(x); };
  push(pp[0] / S, pp[1] / S, pv[0] / V, pv[1] / V);
  push(bp[0] / S, bp[1] / S, bv[0] / V, bv[1] / V);
  push(rel[0] / S, rel[1] / S, dist / S);
  push((gx - pp[0]) / S, (0 - pp[1]) / S);
  push((-gx - pp[0]) / S, (0 - pp[1]) / S);
  push((gx - bp[0]) / S, (0 - bp[1]) / S);

  const mates = [], opps = [];
  state.players.forEach((q, i) => {
    if (i === p) return;
    (q.team === me.team ? mates : opps).push(q);
  });
  if (mates.length) {
    for (const q of mates) { const qp = own(q.pos); push((qp[0] - pp[0]) / S, (qp[1] - pp[1]) / S); }
    for (const q of mates) { const qv = own(q.vel); push(qv[0] / V, qv[1] / V); }
  }
  for (const q of opps) { const qp = own(q.pos); push((qp[0] - pp[0]) / S, (qp[1] - pp[1]) / S); }
  for (const q of opps) { const qv = own(q.vel); push(qv[0] / V, qv[1] / V); }
  for (const q of opps) { const qp = own(q.pos); push((bp[0] - qp[0]) / S, (bp[1] - qp[1]) / S); }

  push(me.canKick ? 1 : 0, state.kickoff ? 1 : 0,
       state.kickoff && state.kickoffTeam === me.team ? 1 : 0,
       me.touching ? 1 : 0, state.tfrac);
  return o;
}

// acción 0..17 (marco propio) -> {dirX, dirY, kick} en el mundo
function decodeAction(a, team) {
  let m = a % 9;
  const kick = a >= 9;
  if (team === 1) m = MIRROR_MOVE[m];
  const [dx, dy] = MOVE_DIRS[m];
  return { dirX: dx, dirY: dy, kick };
}

module.exports = { buildObs, decodeAction, MOVE_DIRS, MIRROR_MOVE };
