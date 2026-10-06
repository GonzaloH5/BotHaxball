// Estado RS4-Z (observación v2) del jugador del bot a partir de lo que se ve en una sala real.
//
// Todo es información pública de la sala:
// * posiciones y velocidades; input e isKicking de cada jugador (HaxBall los sincroniza a todos los
//   clientes). Patada cancelada = tecla de patear apretada && !isKicking, como en el motor.
// * saques del script RS ONE: la pelota toma el color del equipo que saca y se coloca en el punto
//   del contrato (`restartKind`); vuelve a blanco al liberarse. Mismo criterio que el dataset humano.
// * saque inicial: estado 0 del juego y equipo que saca.
// * fase de masa: invMass del propio disco (0,5 tras reposicionar; 0,3 o pieza después).
// * últimas 3 decisiones propias (mundo) y retraso de input medido.
//
// Se alimenta tick a tick con `update(frame)` y se pide la observación con `build(frame, meId, delay)`.
// frame = {state, kickoffTeam (0 rojo, 1 azul), kickStrength,
//          ball: {x, y, vx, vy, r, color},
//          players: [{id, team (0|1), x, y, vx, vy, input, isKicking, invMass}]}
"use strict";

const { buildObs, restartKind } = require("./obs_v2");
const { colorTeam } = require("../public_signals");

const WHITE = 0xFFFFFF;
const MOVE_DIRS = [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
const MIRROR_MOVE = [0, 1, 8, 7, 6, 5, 4, 3, 2];

// input de HaxBall (bits 1 arriba, 2 abajo, 4 izquierda, 8 derecha, 16 patear) -> acción del mundo 0..17
function inputToAction(input) {
  const dx = ((input >> 3) & 1) - ((input >> 2) & 1);
  const dy = ((input >> 1) & 1) - (input & 1);
  const m = MOVE_DIRS.findIndex(([x, y]) => x === dx && y === dy);
  return m + (input & 16 ? 9 : 0);
}

// acción del marco propio (salida de la red) -> {dirX, dirY, kick, world}
function decodeOwnAction(a, team) {
  let m = a % 9;
  if (team === 1) m = MIRROR_MOVE[m];
  const kick = a >= 9;
  const [dirX, dirY] = MOVE_DIRS[m];
  return { dirX, dirY, kick, world: m + (kick ? 9 : 0) };
}

// Tipo de saque por zona cuando el punto no es el exacto de RS ONE. Sirve para RS ONE y HAXARG 2K23
// (lateral y=±688 / ±618; córner (±1140, ±660) / (±1140, ±590); saque de arco (±1030, ±180) / (±1060, 0)).
function restartZone(x, y) {
  const ax = Math.abs(x), ay = Math.abs(y);
  if (ax >= 1100 && ay >= 450) return 2;
  if (ay >= 600) return 1;
  return 3;
}

class RS4ZTracker {
  constructor(config = {}) { this.config = config; this.reset(); }

  reset() {
    this.tick = 0;
    this.hist = [0, 0, 0];
    this.restart = { team: -1, kind: 0, start: 0 };
    this.koStart = -1;
  }

  update(frame) {
    this.tick++;
    const c = frame.ball.color;
    const team = c === undefined || c === WHITE ? -1 : colorTeam(c, this.config);
    if (team < 0) {
      this.restart = { team: -1, kind: 0, start: 0 };
    } else if (this.restart.team !== team) {
      const kind = restartKind(frame.ball.x, frame.ball.y) || restartZone(frame.ball.x, frame.ball.y);
      this.restart = { team, kind, start: this.tick };
    }
    if (frame.state === 0) {
      if (this.koStart < 0) this.koStart = this.tick;
    } else {
      this.koStart = -1;
    }
  }

  pushDecision(worldAction) {
    this.hist = [worldAction, this.hist[0], this.hist[1]];
  }

  // Ocho lugares como el simulador: rojos 0..3 y azules 4..7 por id; con más de 4 por equipo se
  // quedan los más cercanos a la pelota (siempre incluye al bot).
  slots(frame, meId) {
    const b = frame.ball;
    const out = [];
    for (const t of [0, 1]) {
      let team = frame.players.filter((p) => p.team === t);
      if (team.length > 4) {
        const d2 = (p) => (p.x - b.x) ** 2 + (p.y - b.y) ** 2;
        const me = team.find((p) => p.id === meId);
        team = team.filter((p) => p.id !== meId).sort((p, q) => d2(p) - d2(q)).slice(0, me ? 3 : 4);
        if (me) team.push(me);
      }
      team.sort((p, q) => p.id - q.id);
      for (let k = 0; k < 4; k++) out.push(team[k] || null);
    }
    return out;
  }

  build(frame, meId, delayTicks = 0) {
    const slots = this.slots(frame, meId);
    const slot = slots.findIndex((p) => p && p.id === meId);
    if (slot < 0) return null;
    const players = slots.map((p, k) => p ? {
      team: k < 4 ? 0 : 1, active: true, x: p.x, y: p.y, vx: p.vx, vy: p.vy,
      kickCancel: !!(p.input & 16) && !p.isKicking, applied: inputToAction(p.input),
    } : { team: k < 4 ? 0 : 1, active: false, x: 0, y: 0, vx: 0, vy: 0, kickCancel: false, applied: 0 });
    const actHist = slots.map(() => [0, 0, 0]);
    actHist[slot] = this.hist.slice();
    const delay = slots.map(() => 0);
    delay[slot] = delayTicks;
    const me = slots[slot];
    const massPhase = Math.abs(me.invMass - 0.5) < 0.05 ? 0 : 1;
    const r = this.restart;
    // Pelota y patada fuera del rango con que se entrenó el modelo (p. ej. un modelo de RS ONE en HAXARG 2K23:
    // pelota 9, patada 5,65) dejaban estas entradas a 5 desvíos de lo visto: se recortan a ese rango.
    // El rango viene del modelo (META ball_radii / kick_strengths); por defecto, el de RS ONE.
    const clip = (v, range) => Math.min(Math.max(v, Math.min(...range)), Math.max(...range));
    const state = {
      ball: { x: frame.ball.x, y: frame.ball.y, vx: frame.ball.vx, vy: frame.ball.vy,
        r: clip(frame.ball.r, this.config.ball_radii || [8, 8.325]) },
      players, actHist, delay, kickStrength: clip(frame.kickStrength, this.config.kick_strengths || [5.75, 5.85]), massPhase,
      restart: r.team >= 0 ? { team: r.team, kind: r.kind, ticks: this.tick - r.start } : { team: -1, kind: 0, ticks: 0 },
      kickoff: frame.state === 0 ? { active: true, team: frame.kickoffTeam, ticks: this.tick - this.koStart }
        : { active: false, team: -1, ticks: 0 },
    };
    return { obs: buildObs(state, slot), slot, team: slot < 4 ? 0 : 1 };
  }
}

module.exports = { RS4ZTracker, inputToAction, decodeOwnAction };
