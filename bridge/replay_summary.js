// Resumen de una carpeta de replays (.hbr2): modo (estadio), duración y jugadores, agrupado por mapa.
// Uso: node bridge/replay_summary.js [carpeta=~/Downloads]
const fs = require("fs");
const os = require("os");
const path = require("path");
const API = require("./haxball")({}, { language: false });
const { Replay } = API;

const dir = process.argv[2] || path.join(os.homedir(), "Downloads");
const files = fs.readdirSync(dir).filter((f) => /\.hbr2?$/i.test(f));
const groups = {};
for (const f of files) {
  let data;
  try {
    data = Replay.readAll(new Uint8Array(fs.readFileSync(path.join(dir, f))));
  } catch (e) {
    console.log(`  (no se pudo leer ${f}: ${e.message})`);
    continue;
  }
  const room = data.roomData;
  const name = (room.stadium && room.stadium.name) || "?";
  const minutes = data.totalFrames / 3600;
  const inTeams = (room.players || []).filter((p) => p.team && p.team.id !== 0).length;
  const g = (groups[name] = groups[name] || { n: 0, minutes: 0, playerMinutes: 0, maxPlayers: 0, files: [] });
  g.n++;
  g.minutes += minutes;
  g.playerMinutes += minutes * inTeams;
  g.maxPlayers = Math.max(g.maxPlayers, inTeams);
  g.files.push(f);
}
console.log(`${files.length} replays en ${dir}\n`);
console.log(`${"mapa".padEnd(44)} ${"replays".padStart(7)} ${"minutos".padStart(8)} ${"jug·min".padStart(8)} ${"máx jug".padStart(7)}`);
for (const [name, g] of Object.entries(groups).sort((a, b) => b[1].minutes - a[1].minutes)) {
  console.log(`${name.slice(0, 44).padEnd(44)} ${String(g.n).padStart(7)} ${g.minutes.toFixed(0).padStart(8)} ` +
              `${g.playerMinutes.toFixed(0).padStart(8)} ${String(g.maxPlayers).padStart(7)}`);
}
process.exit(0);
