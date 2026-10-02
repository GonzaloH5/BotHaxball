// Public RS4 signals. No private host/script state and no oracle last toucher.
// Joint discovery only accepts the paired RS4 auxiliary-line topology.
const {discoverJointBarriers,visibleJointColors}=require('./joint_barriers');
const RED = [0xFF0000, 0xE56E56, 0xEC7458, 0xFF3F34], BLUE = [0x0000FF, 0x5689E5, 0x48BEF9, 0x0FBCF9];
function rgb(value) {
  if (Array.isArray(value) && value.length === 3) return (value[0] << 16) | (value[1] << 8) | value[2];
  if (typeof value === "string") {
    const text = value.replace(/^#/, "");
    if (/^[a-f0-9]{6}$/i.test(text)) return parseInt(text, 16);
  }
  return Number.isInteger(value) ? value : -1;
}
function colorTeam(value, config = {}) {
  const c = rgb(value);
  return (config.red_colors || RED).map(rgb).includes(c) ? 0
    : (config.blue_colors || BLUE).map(rgb).includes(c) ? 1 : -1;
}
function cues(packet, velocity, config = {}) {
  const a = Math.hypot(...velocity) <= .05 ? colorTeam(packet.ballColor, config) : -1;
  const b = packet.barrierColor === -2 ? -2 : colorTeam(packet.barrierColor, config);
  const conflict = b === -2 || (a >= 0 && b >= 0 && a !== b);
  return {a, b, conflict, owner: conflict ? -1 : a >= 0 ? a : b};
}
function publicFeatures(state, player, geom, config = {}) {
  const packet = state.publicSignals;
  if (!packet) throw new Error("public-v1 model requires a public signal packet (unknown is allowed)");
  const {a, b, conflict, owner} = cues(packet, state.ball.vel, config);
  const visible = a >= 0 || b >= 0 || b === -2;
  const [bx, by] = state.ball.pos.map(Math.abs);
  const kind = !visible ? 0 : by >= .85 * geom.field_half_h && bx >= .85 * geom.goal_x ? 2
    : by >= .85 * geom.field_half_h ? 1 : bx >= .65 * geom.goal_x ? 3 : 0;
  const age = Math.max(0, packet.contactAge || 0);
  const confidence = Math.max(0, Math.min(1, packet.contactConfidence || 0)) * Math.max(0, 1 - age / 600);
  const team = state.players[player].team;
  return [Number(visible), Number(owner === team), Number(owner >= 0 && owner !== team), Number(conflict),
    Number(a >= 0), Number(b >= 0 || b === -2), Number(kind === 1), Number(kind === 2), Number(kind === 3),
    Number(packet.contactTeam === team && confidence > 0),
    Number(packet.contactTeam >= 0 && packet.contactTeam !== team && confidence > 0), confidence,
    Math.min(1, age / 600), Math.min(1, (packet.restartAge || 0) / 600), 1];
}
class PublicSignalTracker {
  constructor(config = {}) { this.config = config; this.jointIds=[]; this.jointGeom=null; this.reset(); }
  configureStadium(stadium, geom) {
    this.jointGeom=geom;
    this.jointIds=[...new Set([...(this.config.barrier_joints || []),
      ...(this.config.auto_joints?discoverJointBarriers(stadium,geom):[])])];
    return this.jointIds;
  }
  reset() { this.contactTeam = -1; this.contactAge = 600; this.contactConfidence = 0;
    this.restartOwner = -1; this.restartAge = 0; this.previousBall = null; }
  barrierColor(discs, segments = [], joints = [], ball = null) {
    const colors = [];
    for (const id of this.config.barrier_discs || []) {
      if (!Number.isInteger(id) || id <= 0) throw new Error("Barrier disc IDs must exclude the ball (0)");
      if (discs[id]) colors.push(discs[id].color);
    }
    for (const id of this.config.barrier_segments || []) {
      if (!Number.isInteger(id) || id < 0) throw new Error("Invalid barrier segment ID");
      const line = segments[id];
      if (line && line.vis !== false && line.vis !== 0) colors.push(line.color);
    }
    colors.push(...visibleJointColors(discs,joints,this.jointIds,ball,this.jointGeom));
    const teams = new Set(colors.map(c => colorTeam(c, this.config)).filter(t => t >= 0));
    return teams.size > 1 ? -2 : teams.size === 1 ? [...teams][0] === 0
      ? (this.config.red_colors || RED)[0] : (this.config.blue_colors || BLUE)[0] : -1;
  }
  sample(ball, players, dt, radius = 15, ballRadius = 10, barrierColor = -1) {
    this.contactAge += dt;
    const jump = this.previousBall && Math.hypot(ball.pos[0] - this.previousBall[0], ball.pos[1] - this.previousBall[1]) > 80 + dt * 15;
    if (jump) { this.contactTeam = -1; this.contactAge = 600; this.contactConfidence = 0; }
    const close = [false, false];
    for (const p of players) {
      if (p.team === 0 || p.team === 1) close[p.team] ||= Math.hypot(p.pos[0] - ball.pos[0], p.pos[1] - ball.pos[1]) <= radius + ballRadius + 3;
    }
    if (!jump && close[0] && close[1]) { this.contactTeam = -1; this.contactAge = 0; this.contactConfidence = 0; }
    else if (!jump && (close[0] || close[1])) { this.contactTeam = close[0] ? 0 : 1; this.contactAge = 0; this.contactConfidence = .6; }
    const packet = {ballColor: rgb(ball.color), barrierColor: rgb(barrierColor)};
    const {owner} = cues(packet, ball.vel, this.config);
    this.restartAge = owner >= 0 && owner === this.restartOwner ? this.restartAge + dt : 0;
    this.restartOwner = owner;
    this.previousBall = [...ball.pos];
    return {...packet, contactTeam: this.contactTeam, contactAge: this.contactAge,
      contactConfidence: this.contactConfidence, restartAge: this.restartAge};
  }
}
module.exports = {publicFeatures, PublicSignalTracker, colorTeam};
