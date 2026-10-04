// Observación v2 de RS4-Z en Node: misma función que env/rs4z/obs_v2.py (build_obs), feature por feature.
// Sólo información pública de la sala: posiciones/velocidades, patada armada/visible, últimas 3 decisiones
// propias, latencia medida, variante del mapa, señales públicas de saque, saque inicial y fase de masa.
// Sin reloj ni marcador. La paridad con Python se prueba en deploy/test_obs_v2.js.
"use strict";

const SX = 1150, SY = 670, SR = 600, SVP = 3, SVB = 6, GOAL_X = 1150, N_HIST = 3, MAX_DELAY = 12;
const SELF_DIM = 27 + 9 * N_HIST + N_HIST + 16;
const ENT_DIM = 9, N_ENT = 7, N_MATES = 3;
const OBS_DIM = SELF_DIM + N_ENT * ENT_DIM;
const MIRROR_MOVE = [0, 1, 8, 7, 6, 5, 4, 3, 2];

function clip1(v) { return v > 1 ? 1 : (v < 0 ? 0 : v); }

/**
 * state = {
 *   ball: {x, y, vx, vy, r},
 *   players: [ {slot 0..7, team 0|1, active, x, y, vx, vy, kickCancel, applied (acción aplicada, mundo)} ] (8),
 *   actHist: [[a0,a1,a2] por slot] (decisiones en coordenadas del mundo, la más nueva primero),
 *   delay: [ticks por slot], kickStrength, restart: {team (-1 ninguno), kind 1..3, ticks},
 *   kickoff: {active, team, ticks}, massPhase 0|1
 * }
 * Devuelve Float32Array(OBS_DIM) para el lugar `p`.
 */
function buildObs(state, p) {
  const row = new Float32Array(OBS_DIM);
  const pl = state.players;
  const me = pl[p];
  if (!me.active) return row;
  let nRed = 0, nBlue = 0;
  for (const q of pl) if (q.active) { if (q.team === 0) nRed++; else nBlue++; }
  const s = me.team === 0 ? 1 : -1;
  const px = me.x * s, py = me.y, pvx = me.vx * s, pvy = me.vy;
  const b = state.ball;
  const bx = b.x * s, by = b.y, bvx = b.vx * s, bvy = b.vy;
  const dx = bx - px, dy = by - py;
  row[0] = px / SX; row[1] = py / SY; row[2] = pvx / SVP; row[3] = pvy / SVP;
  row[4] = me.kickCancel ? 0 : 1;
  row[5] = (me.applied >= 9 && !me.kickCancel) ? 1 : 0;
  row[6] = bx / SX; row[7] = by / SY; row[8] = bvx / SVB; row[9] = bvy / SVB;
  row[10] = dx / SR; row[11] = dy / SR; row[12] = Math.sqrt(dx * dx + dy * dy) / SR;
  row[13] = (bvx - pvx) / SVB; row[14] = (bvy - pvy) / SVB;
  row[15] = (GOAL_X - px) / SX; row[16] = (px + GOAL_X) / SX; row[17] = (SY + py) / SY; row[18] = (SY - py) / SY;
  row[19] = (GOAL_X - px) / SX; row[20] = -py / SY; row[21] = (-GOAL_X - px) / SX; row[22] = -py / SY;
  row[23] = (GOAL_X - bx) / SX; row[24] = -by / SY; row[25] = (-GOAL_X - bx) / SX; row[26] = -by / SY;
  let j = 27;
  const hist = state.actHist[p];
  for (let h = 0; h < N_HIST; h++) {
    let m = hist[h] % 9;
    if (s < 0) m = MIRROR_MOVE[m];
    row[j + h * 9 + m] = 1;
  }
  j += 9 * N_HIST;
  for (let h = 0; h < N_HIST; h++) row[j + h] = hist[h] >= 9 ? 1 : 0;
  j += N_HIST;
  row[j] = state.delay[p] / MAX_DELAY;
  const ownN = me.team === 0 ? nRed : nBlue, rivN = me.team === 0 ? nBlue : nRed;
  row[j + 1] = (ownN - 1) / 3; row[j + 2] = rivN / 4;
  row[j + 3] = (state.kickStrength - 5.8) / 0.1;
  row[j + 4] = (b.r - 8.1625) / 0.1625;
  row[j + 5] = state.massPhase;
  const r = state.restart;
  if (r.team >= 0) {
    row[j + 6] = 1; row[j + 7] = r.team === me.team ? 1 : 0; row[j + 8] = r.team !== me.team ? 1 : 0;
    row[j + 9] = r.kind === 1 ? 1 : 0; row[j + 10] = r.kind === 2 ? 1 : 0; row[j + 11] = r.kind === 3 ? 1 : 0;
    row[j + 12] = clip1(r.ticks / 600);
  }
  const ko = state.kickoff;
  if (ko.active) {
    row[j + 13] = 1; row[j + 14] = ko.team === me.team ? 1 : 0; row[j + 15] = clip1(ko.ticks / 600);
  }
  let eMate = 0, eRiv = 0;
  for (let q = 0; q < pl.length; q++) {
    const o = pl[q];
    if (q === p || !o.active) continue;
    let base;
    if (o.team === me.team) {
      if (eMate >= N_MATES) continue;
      base = SELF_DIM + eMate * ENT_DIM; eMate++;
    } else {
      base = SELF_DIM + (N_MATES + eRiv) * ENT_DIM; eRiv++;
    }
    const qx = o.x * s, qy = o.y;
    row[base] = 1;
    row[base + 1] = (qx - px) / SR; row[base + 2] = (qy - py) / SR;
    row[base + 3] = o.vx * s / SVP; row[base + 4] = o.vy / SVP;
    row[base + 5] = (bx - qx) / SR; row[base + 6] = (by - qy) / SR;
    row[base + 7] = Math.sqrt((bx - qx) ** 2 + (by - qy) ** 2) / SR;
    row[base + 8] = (o.applied >= 9 && !o.kickCancel) ? 1 : 0;
  }
  return row;
}

// Decodificador público del tipo de saque por el punto exacto del contrato (F8): lateral y=±688,
// córner (±1140, ±660), saque de arco (±1030, ±180).
function restartKind(ballX, ballY) {
  const x = Math.abs(ballX), y = Math.abs(ballY);
  if (Math.abs(y - 688) < 2) return 1;
  if (Math.abs(x - 1140) < 2 && Math.abs(y - 660) < 2) return 2;
  if (Math.abs(x - 1030) < 2 && Math.abs(y - 180) < 2) return 3;
  return 0;
}

module.exports = { buildObs, restartKind, OBS_DIM, SELF_DIM, ENT_DIM };
