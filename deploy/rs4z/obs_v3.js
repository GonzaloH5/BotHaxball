// Observación v3 de X4 en Node: misma función que env/rs4z/obs_v3.py (_featurize), feature por feature.
// Respecto de v2: 5 decisiones propias (latencia de sala de 9–12 ticks), geometría del mapa (lateral y
// línea de gol), el mapa como one-hot y el margen de contacto con la pelota (propio y de cada entidad). La paridad con Python se prueba en deploy/test_obs_v3.js.
"use strict";

const SX = 1150, SY = 670, SR = 600, SVP = 3, SVB = 6, N_HIST = 5, MAX_DELAY = 15;
const PLAYER_R = 15, KICK_REACH = 4, GAP_SCALE = 8;
const MAP_NAMES = ["rs_one", "sanguchito_rs_x4", "haxarg_2k23"];
const N_MAPS = MAP_NAMES.length;
// línea lateral y línea de gol de cada mapa (env/rs4z/contract.py)
const MAP_GEOMETRY = { rs_one: [670, 1150], sanguchito_rs_x4: [670, 1150], haxarg_2k23: [600, 1150] };
const SELF_DIM = 27 + 9 * N_HIST + N_HIST + 16 + N_MAPS + 2;
const ENT_DIM = 10, N_ENT = 7, N_MATES = 3;
const OBS_DIM = SELF_DIM + N_ENT * ENT_DIM;
const MIRROR_MOVE = [0, 1, 8, 7, 6, 5, 4, 3, 2];

function clip1(v) { return v > 1 ? 1 : (v < 0 ? 0 : v); }
// margen de contacto con la pelota (distancia − radios) / 8 px, recortado a [−1, 4]
function gapFeat(g) { const v = g / GAP_SCALE; return v < -1 ? -1 : (v > 4 ? 4 : v); }

/**
 * state = obs v2 (deploy/rs4z/obs_v2.js) con actHist de 5 decisiones por lugar y además
 *   map: "rs_one" | "sanguchito_rs_x4" | "haxarg_2k23".
 * players[q].kicking: patada aplicada en el último tick y no cancelada (en v2 salía de `applied`).
 */
function buildObs(state, p) {
  const row = new Float32Array(OBS_DIM);
  const pl = state.players;
  const me = pl[p];
  if (!me.active) return row;
  const mapIdx = MAP_NAMES.indexOf(state.map);
  const [lineH, goalX] = MAP_GEOMETRY[state.map] || MAP_GEOMETRY.rs_one;
  let nRed = 0, nBlue = 0;
  for (const q of pl) if (q.active) { if (q.team === 0) nRed++; else nBlue++; }
  const s = me.team === 0 ? 1 : -1;
  const px = me.x * s, py = me.y, pvx = me.vx * s, pvy = me.vy;
  const b = state.ball;
  const bx = b.x * s, by = b.y, bvx = b.vx * s, bvy = b.vy;
  const dx = bx - px, dy = by - py;
  const kicking = (o) => (o.kicking !== undefined ? !!o.kicking : (o.applied >= 9 && !o.kickCancel));
  row[0] = px / SX; row[1] = py / SY; row[2] = pvx / SVP; row[3] = pvy / SVP;
  row[4] = me.kickCancel ? 0 : 1;
  row[5] = kicking(me) ? 1 : 0;
  row[6] = bx / SX; row[7] = by / SY; row[8] = bvx / SVB; row[9] = bvy / SVB;
  row[10] = dx / SR; row[11] = dy / SR; row[12] = Math.sqrt(dx * dx + dy * dy) / SR;
  row[13] = (bvx - pvx) / SVB; row[14] = (bvy - pvy) / SVB;
  row[15] = (goalX - px) / SX; row[16] = (px + goalX) / SX; row[17] = (lineH + py) / SY; row[18] = (lineH - py) / SY;
  row[19] = (goalX - px) / SX; row[20] = -py / SY; row[21] = (-goalX - px) / SX; row[22] = -py / SY;
  row[23] = (goalX - bx) / SX; row[24] = -by / SY; row[25] = (-goalX - bx) / SX; row[26] = -by / SY;
  let j = 27;
  const hist = state.actHist[p];
  for (let h = 0; h < N_HIST; h++) {
    let m = (hist[h] || 0) % 9;
    if (s < 0) m = MIRROR_MOVE[m];
    row[j + h * 9 + m] = 1;
  }
  j += 9 * N_HIST;
  for (let h = 0; h < N_HIST; h++) row[j + h] = (hist[h] || 0) >= 9 ? 1 : 0;
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
  if (mapIdx >= 0) row[j + 16 + mapIdx] = 1;
  const gap = Math.sqrt(dx * dx + dy * dy) - PLAYER_R - b.r;
  row[j + 16 + N_MAPS] = gapFeat(gap);
  row[j + 17 + N_MAPS] = gap < KICK_REACH ? 1 : 0;
  let eMate = 0, eRiv = 0;
  for (let q = 0; q < pl.length; q++) {
    const o = pl[q];
    if (q === p || !o.active) continue;
    let base;
    if (o.team === me.team) {
      if (eMate >= N_MATES) continue;
      base = SELF_DIM + eMate * ENT_DIM; eMate++;
    } else {
      if (eRiv >= 4) continue;
      base = SELF_DIM + (N_MATES + eRiv) * ENT_DIM; eRiv++;
    }
    const qx = o.x * s, qy = o.y;
    row[base] = 1;
    row[base + 1] = (qx - px) / SR; row[base + 2] = (qy - py) / SR;
    row[base + 3] = o.vx * s / SVP; row[base + 4] = o.vy / SVP;
    row[base + 5] = (bx - qx) / SR; row[base + 6] = (by - qy) / SR;
    const dq = Math.sqrt((bx - qx) ** 2 + (by - qy) ** 2);
    row[base + 7] = dq / SR;
    row[base + 8] = kicking(o) ? 1 : 0;
    row[base + 9] = gapFeat(dq - PLAYER_R - b.r);
  }
  return row;
}

// Mapa por nombre del estadio de la sala.
function mapFromStadiumName(name) {
  const n = (name || "").toUpperCase();
  if (n.includes("SANGUCHITO")) return "sanguchito_rs_x4";
  if (n.includes("2K23") || n.includes("HAXARG")) return "haxarg_2k23";
  return "rs_one";
}

module.exports = { buildObs, mapFromStadiumName, OBS_DIM, SELF_DIM, ENT_DIM, N_HIST, MAX_DELAY, MAP_NAMES };
