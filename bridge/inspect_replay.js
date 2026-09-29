// Inspecciona un .hbr2 con el motor original y devuelve JSON por stdout.
// Uso: node bridge/inspect_replay.js <replay.hbr2> [--team-size 3]

const fs = require("fs");
const API = require("./haxball")({}, { language: false });
const { Replay } = API;

class Inspection {
  constructor(teamSize = null) {
    this.teamSize = teamSize;
    this.currentStadium = "";
    this.stadiums = new Set();
    this.teamSizeStadiums = new Set();
  }

  stadium(name) {
    this.currentStadium = String(name || "").trim();
    if (this.currentStadium) this.stadiums.add(this.currentStadium);
  }

  players(players) {
    if (this.teamSize == null) return;
    let red = 0;
    let blue = 0;
    for (const player of players || []) {
      const team = player.team?.id ?? player.team;
      if (team === 1) red++;
      else if (team === 2) blue++;
    }
    if (red === this.teamSize && blue === this.teamSize && this.currentStadium) {
      this.teamSizeStadiums.add(this.currentStadium);
    }
  }

  result() {
    return {
      stadiums: [...this.stadiums],
      teamSize: this.teamSize,
      teamSizeMatched: this.teamSize == null || this.teamSizeStadiums.size > 0,
      teamSizeStadiums: [...this.teamSizeStadiums],
    };
  }
}

function inspect(file, teamSize) {
  const inspection = new Inspection(teamSize);
  const data = new Uint8Array(fs.readFileSync(file));
  let reader = null;
  let finished = false;
  let lastFrame = -1;
  let still = 0;

  function finish(error = null) {
    if (finished) return;
    finished = true;
    try { reader?.setSpeed(0); } catch (_) {}
    if (error) {
      process.stderr.write(String(error?.stack || error) + "\n");
      process.exit(1);
    }
    process.stdout.write(JSON.stringify(inspection.result()));
    process.exit(0);
  }

  try {
    reader = Replay.read(data, {
      onGameTick: () => {
        const stadium = reader.state.stadium?.name;
        if (stadium && stadium !== inspection.currentStadium) inspection.stadium(stadium);
        inspection.players(reader.state.players);
      },
      onStadiumChange: (stadium) => inspection.stadium(stadium?.name),
      onEnd: () => finish(),
    }, {
      requestAnimationFrame: (cb) => setImmediate(() => cb(performance.now())),
      cancelAnimationFrame: (id) => clearImmediate(id),
    });
    inspection.stadium(reader.state.stadium?.name);
    reader.setSpeed(1000);
  } catch (error) {
    finish(error);
    return;
  }

  setInterval(() => {
    try {
      const frame = reader.getCurrentFrameNo();
      if (frame >= reader.maxFrameNo - 1) return finish();
      still = frame === lastFrame ? still + 1 : 0;
      lastFrame = frame;
      if (still >= 5) finish();
    } catch (error) {
      finish(error);
    }
  }, 1000);
  setTimeout(() => finish(new Error("timeout inspeccionando el replay")), 2 * 60e3);
}

module.exports = { Inspection };

if (require.main === module) {
  const args = process.argv.slice(2);
  const file = args.find((arg, index) => !arg.startsWith("--") && args[index - 1] !== "--team-size");
  const teamIndex = args.indexOf("--team-size");
  const teamSize = teamIndex >= 0 ? Number(args[teamIndex + 1]) : null;
  if (!file || (teamSize != null && (!Number.isInteger(teamSize) || teamSize < 1))) {
    console.error("uso: node bridge/inspect_replay.js <replay.hbr2> [--team-size 3]");
    process.exit(2);
  }
  inspect(file, teamSize);
}
