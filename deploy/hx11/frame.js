// Tick de HaxEleven para el cerebro (learn/hx11_cerebro.py), armado igual desde una sala en vivo o desde una
// repetición: node-haxball comparte los eventos (onSetDiscProperties, onPlayerBallKick, onTeamGoal, onAnnouncement),
// que son los mismos que usa D0 al procesar las grabaciones.
"use strict";

const GK_ANN = /^\[✅\]\s+(.+?)\s+es el Arquero del .*?(Red|Blue)/;

class FrameBuilder {
  constructor() {
    this.placement = null;   // el script colocó la pelota desde el último tick
    this.kicks = [];
    this.goal = false;
    this.gk = [-1, -1];
    this.offside = true;
    this.names = new Map();  // nombre -> id (para los !gk anunciados)
  }

  // ganchos (la sala y la repetición tienen los mismos nombres de callback)
  onSetDiscProperties(id, type, d1) {
    // data1: [x, y, vx, vy, gx, gy, radius, bCoef, invMass, damping]; colocación = posición de la pelota (disco 0)
    if (!type && id === 0 && d1 && d1[0] != null) this.placement = [d1[0], d1[1]];
  }
  onPlayerBallKick(playerId) { this.kicks.push(playerId); }
  onTeamGoal() { this.goal = true; }
  onPlayerJoin(p) { if (p && p.name) this.names.set(p.name.trim(), p.id); }
  onAnnouncement(msg) {
    const g = GK_ANN.exec(msg || "");
    if (g) {
      const id = this.names.get(g[1].trim());
      if (id != null) this.gk[g[2] === "Red" ? 0 : 1] = id;
    }
    if (/Offside/.test(msg || "")) {
      if (/desactivad|Offside \[❌\]/.test(msg)) this.offside = false;
      else if (/activad|Offside \[✅\]/.test(msg)) this.offside = true;
    }
  }

  // src: la sala (Room) o el lector de la repetición; los dos tienen state y gameState
  build(src, t, meId, delay) {
    const gs = src.gameState;
    if (!gs) return null;
    for (const p of src.state.players) if (p.name) this.names.set(p.name.trim(), p.id);
    const ball = gs.physicsState.discs[0];
    const players = [];
    const present = new Set();
    for (const p of src.state.players) {
      if (!p.team || p.team.id === 0 || !p.disc) continue;
      present.add(p.id);
      players.push([p.id, p.team.id === 1 ? 0 : 1, p.disc.pos.x, p.disc.pos.y, p.disc.speed.x, p.disc.speed.y,
                    p.input | 0]);
    }
    for (let k = 0; k < 2; k++) if (this.gk[k] >= 0 && !present.has(this.gk[k])) this.gk[k] = -1;
    const fr = {
      t, state: gs.state, score: [gs.redScore, gs.blueScore],
      ball: [ball.pos.x, ball.pos.y, ball.speed.x, ball.speed.y, ball.invMass,
             ball.gravity ? ball.gravity.x : 0, ball.gravity ? ball.gravity.y : 0],
      players, placement: this.placement, kicks: this.kicks, goal: this.goal, gk: [...this.gk],
      offside: this.offside, me: meId, delay,
    };
    this.placement = null;
    this.kicks = [];
    this.goal = false;
    return fr;
  }
}

module.exports = { FrameBuilder };
